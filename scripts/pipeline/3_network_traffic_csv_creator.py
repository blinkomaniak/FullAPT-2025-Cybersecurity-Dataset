#!/usr/bin/env python3
"""
Network Traffic Flow JSONL to CSV Converter - Multi-Threaded Version

Converted from notebook: 3_elastic_network-traffic-flow-ds_csv_creator.ipynb
Transforms Elasticsearch network traffic flow data from JSONL format into structured CSV datasets for ML analysis.
Features multi-threading support for high-capacity servers with dozens of CPUs and hundreds of GB RAM.

USAGE EXAMPLES:
    # Use with input/output only (no config required)
    python 3_network_traffic_csv_creator.py --input network.jsonl --output network.csv
    
    # Process specific APT run directory with auto-detection
    python 3_network_traffic_csv_creator.py --apt-dir apt-1/apt-1-run-04
    
    # Use config.yaml settings (if file exists)
    python 3_network_traffic_csv_creator.py --config config_restructured.yaml
    
    # Skip validation for faster processing  
    python 3_network_traffic_csv_creator.py --input network.jsonl --output network.csv --no-validate
    
    # High-performance server example (config.yaml):
    script_03_network_csv_creator:
      max_workers: auto        # Uses all CPU cores
      chunk_size: 50000        # Large chunks for high-memory servers

MULTI-THREADING CONFIGURATION:
    max_workers: auto          # Auto-detect CPU cores, or specify number
    chunk_size: 10000          # JSONL lines per chunk (increase for more RAM)

KEY FEATURES:
    - Multi-threaded JSON parsing and field extraction
    - Timestamp standardization: timestamp converted to epoch format for ML compatibility
    - Proper network flow grouping by network_traffic_flow_id
    - Accurate traffic statistics (no double-counting)
    - Thread-safe statistics aggregation
    - Enhanced timeline analysis with flow duration metrics
    - Structured JSON logging: log-netflow-JSONL-to-csv-run-X.json
    - Support for config_restructured.yaml format

Dependencies: pandas, numpy, pyyaml
"""

import argparse
import json
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import multiprocessing as mp
import threading

try:
    import numpy as np
    import pandas as pd
    import yaml
except ImportError as e:
    print(f"❌ Error: Required library not installed: {e}")
    print("   Install with: pip install pandas numpy pyyaml")
    sys.exit(1)


class NetworkTrafficCSVCreator:
    """
    Professional Network Traffic JSONL to CSV converter for cybersecurity datasets.
    
    Features:
    - Handles nested JSON structures with safe value extraction
    - Manages both scalar and array fields appropriately  
    - Preserves all network flow metadata for analysis
    - Comprehensive exploratory data analysis
    - Based on field analysis from notebooks 3a and 3b
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize with optional configuration."""
        self.config = self._load_config(config_file) if config_file and os.path.exists(config_file) else {}
        self.logger = self._setup_logging()
        
        # Multi-threading configuration (support both old and new config formats)
        network_config = self.config.get('network_traffic_processor', {}) or self.config.get('script_03_network_csv_creator', {})
        max_workers_config = network_config.get('max_workers', 'auto')
        self.max_workers = mp.cpu_count() if max_workers_config == 'auto' else int(max_workers_config)
        self.chunk_size = network_config.get('chunk_size', 10000)
        self.progress_lock = Lock()
        self.stats_lock = Lock()
        
        # Shared statistics across threads
        self.shared_stats = {
            'total_processed': 0,
            'total_errors': 0,
            'flow_ids': set(),
            'port_conversions': 0
        }
        
        # Field mapping: (JSON_path, CSV_column_name)
        # Based on field analysis from notebooks 3a (exploratory) and 3b (structure consistency)
        self.fields = [
            # === TEMPORAL FIELDS ===
            ('@timestamp', 'timestamp'),                    # Always present (100%)
            ('event.start', 'event_start'),                    # Always present (100%)
            ('event.end', 'event_end'),                    # Always present (100%)
            
            # === DESTINATION FIELDS ===
            ('destination.bytes', 'destination_bytes'),     # Always present (100%)
            ('destination.ip', 'destination_ip'),           # Always present (100%)
            ('destination.mac', 'destination_mac'),         # Always present (100%)
            ('destination.packets', 'destination_packets'), # Always present (100%)
            ('destination.port', 'destination_port'),       # Always present (100%)
            ('destination.process.args', 'destination_process_args'), # Rare field (~2.8% presence) - kept for analysis
            ('destination.process.pid', 'destination_process_pid'), # 
            ('destination.process.executable', 'destination_process_executable'), # 
            ('destination.process.ppid', 'destination_process_ppid'), # 

            # === EVENT FIELDS ===
            ('event.action', 'event_action'),               # Always present (100%)
            ('event.duration', 'event_duration'),           # Always present (100%)
            ('event.type[0]', 'event_type'),               # Always present (100%) - extract first element
            
            # === HOST FIELDS ===
            ('host.hostname', 'host_hostname'),             # Always present (100%)
            ('host.ip', 'host_ip'),                        # Always present (100%) - keep all IPs as list
            ('host.mac[0]', 'host_mac'),                   # Always present (100%) - extract first MAC
            ('host.os.platform', 'host_os_platform'),      # Always present (100%)
            
            # === NETWORK FIELDS ===
            ('network.bytes', 'network_bytes'),             # Always present (100%)
            ('network.packets', 'network_packets'),         # Always present (100%)
            ('network.transport', 'network_transport'),     # Always present (100%) - tcp/udp
            ('network.type', 'network_type'),               # Always present (100%) - ipv4/ipv6
            ('network.community_id', 'network_community_id'),         # Always present (100%)

            # === NETWORK TRAFFIC FIELDS ===
            ('network_traffic.flow.id', 'network_traffic_flow_id'), # Always present (100%)
            ('network_traffic.flow.final', 'network_traffic_flow_final'), # Always present (100%)
            
            # === PROCESS FIELDS ===
            ('process.args', 'process_args'),               # Conditional (~64% presence) - keep as list
            ('process.executable', 'process_executable'),   # Conditional (~64% presence)
            ('process.name', 'process_name'),               # Conditional (~64% presence)
            ('process.parent.pid', 'process_parent_pid'),   # Conditional (~64% presence)
            ('process.pid', 'process_pid'),                 # Conditional (~64% presence)
            
            # === SOURCE FIELDS ===
            ('source.bytes', 'source_bytes'),               # Always present (100%)
            ('source.ip', 'source_ip'),                     # Always present (100%)
            ('source.mac', 'source_mac'),                   # Always present (100%)
            ('source.packets', 'source_packets'),           # Always present (100%) - fixed typo from 'packet'
            ('source.port', 'source_port'),                 # Always present (100%)
            
            # === SOURCE PROCESS FIELDS ===
            ('source.process.args', 'source_process_args'), # Conditional (~61% presence) - keep as list
            ('source.process.executable', 'source_process_executable'), # Conditional (~61% presence)
            ('source.process.name', 'source_process_name'), # Conditional (~61% presence)
            ('source.process.pid', 'source_process_pid'),   # Conditional (~61% presence)
            ('source.process.ppid', 'source_process_ppid')  # Conditional (~61% presence)
        ]
        
        # Define which fields should be treated as arrays (kept as lists)
        self.array_fields = ['process.args', 'source.process.args', 'host.ip', 'destination.process.args']
        
        # Define port fields that need integer conversion (fix Elasticsearch float issue)
        self.port_fields = ['destination.port', 'source.port']
        
    def _load_config(self, config_file: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"⚠️  Warning: Could not load config file {config_file}: {e}")
            print("🔧 Using default configuration...")
            return {}
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger(__name__)
        
        # Support both old and new config formats
        network_config = self.config.get('network_traffic_processor', {}) or self.config.get('script_03_network_csv_creator', {})
        if network_config.get('enable_logging', True):
            logger.setLevel(logging.INFO)
            
            # Console handler
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def get_nested_value(self, doc: dict, path: str) -> Any:
        """
        Safely retrieve nested values from document structure using dot notation.
        
        Args:
            doc (dict): The source JSON document
            path (str): Dot-separated path to the desired field (e.g., 'host.os.platform')
            
        Returns:
            The value at the specified path, or None if path doesn't exist
            
        Examples:
            get_nested_value({'host': {'os': {'platform': 'windows'}}}, 'host.os.platform') → 'windows'
            get_nested_value({'event': {'type': ['connection']}}, 'event.type[0]') → 'connection'
        """
        keys = path.split('.')
        current = doc
        
        for key in keys:
            if isinstance(current, dict):
                # Handle array indexing (e.g., 'type[0]')
                if '[' in key and ']' in key:
                    field_name, index_part = key.split('[', 1)
                    index = int(index_part.rstrip(']'))
                    array_value = current.get(field_name)
                    if isinstance(array_value, list) and index < len(array_value):
                        current = array_value[index]
                    else:
                        return None
                else:
                    # Navigate through dictionary structure
                    current = current.get(key)
            elif isinstance(current, list) and key.isdigit():
                # Handle direct array indexing
                try:
                    current = current[int(key)] if int(key) < len(current) else None
                except (ValueError, IndexError):
                    return None
            else:
                # Path doesn't exist in current structure
                return None
                
            # Stop if we hit a dead end
            if current is None:
                return None
                
        return current
    
    def _save_processing_log(self, jsonl_path: str, total_lines: int, valid_records: int, 
                           error_count: int, df: pd.DataFrame):
        """Save structured JSON processing log"""
        self.logger.info("📊 Starting to generate network traffic processing log...")
        try:
            end_time = datetime.now()
            
            # Enhanced dataset statistics
            dataset_stats = {
                "total_events_in_dataset": len(df),
                "total_unique_network_flow_ids": df['network_traffic_flow_id'].nunique() if 'network_traffic_flow_id' in df.columns else 0,
                "events_per_flow_id_avg": len(df) / df['network_traffic_flow_id'].nunique() if 'network_traffic_flow_id' in df.columns and df['network_traffic_flow_id'].nunique() > 0 else 0
            }
            
            # Analyze flow statistics  
            flow_stats = {}
            if 'network_traffic_flow_id' in df.columns:
                unique_flows = df['network_traffic_flow_id'].nunique()
                events_per_flow = len(df) / unique_flows if unique_flows > 0 else 0
                flow_stats = {
                    "unique_flows": unique_flows,
                    "events_per_flow_avg": events_per_flow,
                    "total_events": len(df)
                }
            
            # Analyze traffic volume properly (no double-counting)
            traffic_stats = {}
            if 'network_bytes' in df.columns and 'network_traffic_flow_id' in df.columns:
                flow_bytes = df.groupby('network_traffic_flow_id')['network_bytes'].first()
                traffic_stats = {
                    "total_traffic_gb": flow_bytes.sum() / 1024**3,
                    "mean_flow_size_bytes": flow_bytes.mean(),
                    "max_flow_size_bytes": flow_bytes.max(),
                    "min_flow_size_bytes": flow_bytes.min()
                }
            
            # Enhanced temporal statistics with human-readable timestamps
            temporal_stats = {}
            if 'timestamp' in df.columns:
                valid_timestamps = df['timestamp'].dropna()  
                if len(valid_timestamps) > 0:
                    # Check if timestamps are epoch format (integers) or datetime objects
                    if pd.api.types.is_integer_dtype(valid_timestamps) or pd.api.types.is_float_dtype(valid_timestamps):
                        # Convert epoch milliseconds to datetime for human-readable format
                        datetime_timestamps = pd.to_datetime(valid_timestamps, unit='ms')
                        min_time = datetime_timestamps.min()
                        max_time = datetime_timestamps.max()
                        min_epoch = int(valid_timestamps.min())
                        max_epoch = int(valid_timestamps.max())
                    else:
                        # Already datetime objects
                        min_time = valid_timestamps.min()
                        max_time = valid_timestamps.max()
                        min_epoch = int(min_time.timestamp())
                        max_epoch = int(max_time.timestamp())
                    
                    duration = max_time - min_time
                    temporal_stats = {
                        "minimum_timestamp_epoch": min_epoch,
                        "maximum_timestamp_epoch": max_epoch,
                        "minimum_timestamp_human_readable": min_time.strftime('%Y-%m-%d %H:%M:%S UTC'),
                        "maximum_timestamp_human_readable": max_time.strftime('%Y-%m-%d %H:%M:%S UTC'),
                        "timeline_start": min_time.isoformat(),
                        "timeline_end": max_time.isoformat(),
                        "timeline_duration_hours": duration.total_seconds() / 3600,
                        "timeline_duration_formatted": str(duration),
                        "chronologically_sorted": True,
                        "total_events_with_timestamps": len(valid_timestamps)
                    }
                    
                    # Add flow timeline analysis
                    if 'network_traffic_flow_id' in df.columns:
                        flow_timeline_stats = df.groupby('network_traffic_flow_id')['timestamp'].agg(['min', 'max', 'count'])
                        avg_flow_duration = (flow_timeline_stats['max'] - flow_timeline_stats['min']).mean()
                        # Convert from nanoseconds to seconds for numpy.timedelta64 objects
                        avg_flow_duration_seconds = float(avg_flow_duration) / 1e9 if pd.api.types.is_timedelta64_dtype(avg_flow_duration) else avg_flow_duration.total_seconds() if hasattr(avg_flow_duration, 'total_seconds') else float(avg_flow_duration)
                        temporal_stats.update({
                            "avg_flow_duration_seconds": avg_flow_duration_seconds,
                            "avg_events_per_flow": flow_timeline_stats['count'].mean()
                        })
            
            # Create log data structure
            log_data = {
                "processing_metadata": {
                    "timestamp": end_time.isoformat(),
                    "operation_type": "network_traffic_jsonl_to_csv_conversion",
                    "script_version": "3.0_enhanced_flow_grouping",
                    "input_file": jsonl_path,
                    "input_file_size_bytes": Path(jsonl_path).stat().st_size if Path(jsonl_path).exists() else None
                },
                "processing_statistics": {
                    "total_jsonl_lines_processed": total_lines,
                    "valid_records_extracted": valid_records,
                    "json_parsing_errors": error_count,
                    "success_rate_percent": (valid_records / total_lines) * 100 if total_lines > 0 else 0
                },
                "dataset_statistics": dataset_stats,
                "network_flow_analysis": flow_stats,
                "traffic_volume_analysis": traffic_stats,
                "temporal_analysis": temporal_stats,
                "field_mapping": {
                    "total_fields_mapped": len(self.fields),
                    "array_fields": self.array_fields,
                    "port_fields": self.port_fields,
                    "output_columns": len(df.columns)
                },
                "data_quality": {
                    "flow_grouping_enabled": 'network_traffic_flow_id' in df.columns,
                    "process_attribution_available": 'process_pid' in df.columns or 'source_process_pid' in df.columns,
                    "temporal_correlation_ready": 'timestamp' in df.columns,
                    "chronologically_sorted": temporal_stats.get("chronologically_sorted", False)
                },
                "session_info": {
                    "hostname": os.uname().nodename if hasattr(os, 'uname') else "unknown",
                    "working_directory": str(Path.cwd()),
                    "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
                }
            }
            
            # Generate output filename: log-netflow-JSONL-to-csv-run-X.json
            run_number = self._extract_run_number(jsonl_path)
            json_log_file = Path(jsonl_path).parent / f"log-netflow-JSONL-to-csv-run-{run_number}.json"
            print(f"🔍 DEBUG: Generated log file path: {json_log_file}")
            print(f"🔍 DEBUG: Log file parent directory: {json_log_file.parent}")
            print(f"🔍 DEBUG: Parent exists: {json_log_file.parent.exists()}")
            print(f"🔍 DEBUG: Parent writable: {os.access(json_log_file.parent, os.W_OK)}")
            
            # Check if log file already exists and remove it to ensure replacement
            if json_log_file.exists():
                try:
                    json_log_file.unlink()  # Delete existing log file
                    self.logger.info(f"🗑️ Removed existing log file: {json_log_file.name}")
                except Exception as e:
                    self.logger.warning(f"⚠️ Could not remove existing log file: {e}")
            
            self.logger.info(f"💾 Writing network traffic log file: {json_log_file}")
            
            # Save JSON log with explicit write mode and error handling
            try:
                with open(json_log_file, 'w', encoding='utf-8') as f:
                    json.dump(log_data, f, indent=2, default=str, ensure_ascii=False)
                
                # Verify file was written successfully
                if json_log_file.exists() and json_log_file.stat().st_size > 0:
                    self.logger.info(f"✅ Network traffic processing log saved: {json_log_file.name} ({json_log_file.stat().st_size} bytes)")
                    print(f"📊 Network traffic log created: {json_log_file}")  # Extra visibility
                else:
                    raise Exception("Log file was not created or is empty")
                    
            except Exception as write_error:
                self.logger.error(f"❌ Failed to write log file: {write_error}")
                # Try alternative location if main write fails
                fallback_log = Path(jsonl_path).parent / f"log-netflow-fallback-{run_number}-{datetime.now().strftime('%H%M%S')}.json"
                try:
                    with open(fallback_log, 'w', encoding='utf-8') as f:
                        json.dump(log_data, f, indent=2, default=str, ensure_ascii=False)
                    self.logger.info(f"📋 Fallback log saved: {fallback_log.name}")
                except Exception as fallback_error:
                    self.logger.error(f"❌ Fallback log write also failed: {fallback_error}")
                    raise write_error
            
        except Exception as e:
            self.logger.error(f"❌ Error saving processing log: {str(e)}")
    
    def _extract_run_number(self, jsonl_path: str) -> str:
        """Extract run number from jsonl filename or directory path"""
        import re
        
        print(f"🔍 DEBUG: Extracting run number from: {jsonl_path}")
        
        # Try to extract from filename first
        filename = Path(jsonl_path).name
        print(f"🔍 DEBUG: Filename: {filename}")
        run_match = re.search(r'run-(\d+)', filename)
        if run_match:
            run_number = run_match.group(1)
            print(f"🔍 DEBUG: Found run number in filename: {run_number}")
            return run_number
        
        # Try to extract from directory path
        path_str = str(jsonl_path)
        print(f"🔍 DEBUG: Full path: {path_str}")
        run_match = re.search(r'run-(\d+)', path_str)
        if run_match:
            run_number = run_match.group(1)
            print(f"🔍 DEBUG: Found run number in path: {run_number}")
            return run_number
        
        # Fallback to timestamp if no run number found
        fallback = datetime.now().strftime('%Y%m%d_%H%M%S')
        print(f"🔍 DEBUG: No run number found, using fallback: {fallback}")
        return fallback
    
    def read_jsonl_in_chunks(self, jsonl_path: str) -> List[List[str]]:
        """Read JSONL file and split into chunks for multi-threading."""
        self.logger.info(f"📖 Reading and chunking JSONL file: {jsonl_path}")
        
        chunks = []
        current_chunk = []
        
        with open(jsonl_path, 'r') as f:
            for line_number, line in enumerate(f, 1):
                current_chunk.append(line.strip())
                
                if len(current_chunk) >= self.chunk_size:
                    chunks.append(current_chunk)
                    current_chunk = []
                    
                # Progress reporting
                if line_number % 100000 == 0:
                    self.logger.info(f"📊 Read {line_number:,} lines, created {len(chunks)} chunks")
            
            # Add remaining lines
            if current_chunk:
                chunks.append(current_chunk)
        
        self.logger.info(f"📦 Created {len(chunks)} chunks of max size {self.chunk_size}")
        return chunks
    
    def process_chunk(self, chunk_lines: List[str], chunk_id: int) -> Tuple[List[Dict], Dict]:
        """Process a chunk of JSONL lines in a separate thread."""
        records = []
        chunk_stats = {
            'processed': 0,
            'errors': 0,
            'flow_ids': set(),
            'port_conversions': 0
        }
        
        for line_idx, line in enumerate(chunk_lines):
            try:
                if not line.strip():
                    continue
                    
                doc = json.loads(line)
                record = {}
                
                # Extract each mapped field
                for path, column in self.fields:
                    # Get value using safe nested extraction
                    value = self.get_nested_value(doc, path)
                    
                    # Handle array fields consistently - keep as lists
                    if path in self.array_fields:
                        if isinstance(value, list):
                            record[column] = value              # Keep existing list
                        else:
                            record[column] = [value] if value is not None else []  # Wrap single value or empty list
                    
                    # Handle port fields - convert floats to integers (fix Elasticsearch issue)
                    elif path in self.port_fields and value is not None:
                        try:
                            # Convert float ports to integers (e.g., 443.0 → 443)
                            record[column] = int(float(value))  # Handle both int and float inputs
                            chunk_stats['port_conversions'] += 1
                        except (ValueError, TypeError):
                            # If conversion fails, use NaN
                            record[column] = np.nan
                            
                    else:
                        # Handle other scalar fields - use NaN for missing values
                        record[column] = value if value is not None else np.nan
                
                # Track flow IDs for statistics
                if 'network_traffic_flow_id' in record and record['network_traffic_flow_id'] is not np.nan:
                    chunk_stats['flow_ids'].add(record['network_traffic_flow_id'])
                
                records.append(record)
                chunk_stats['processed'] += 1
                
            except json.JSONDecodeError:
                chunk_stats['errors'] += 1
                # Thread-safe error logging
                with self.progress_lock:
                    self.logger.error(f"Chunk {chunk_id}, Line {line_idx}: JSON parsing error")
            except Exception as e:
                chunk_stats['errors'] += 1
                # Thread-safe error logging
                with self.progress_lock:
                    self.logger.error(f"Chunk {chunk_id}, Line {line_idx}: {str(e)}")
        
        # Thread-safe progress reporting
        with self.progress_lock:
            self.logger.info(f"✅ Chunk {chunk_id}: {chunk_stats['processed']} processed, {chunk_stats['errors']} errors")
        
        return records, chunk_stats
    
    def merge_chunk_stats(self, chunk_stats_list: List[Dict]):
        """Merge statistics from all chunks into shared stats."""
        with self.stats_lock:
            for chunk_stats in chunk_stats_list:
                self.shared_stats['total_processed'] += chunk_stats['processed']
                self.shared_stats['total_errors'] += chunk_stats['errors']
                self.shared_stats['port_conversions'] += chunk_stats['port_conversions']
                
                # Merge flow IDs (union of sets)
                self.shared_stats['flow_ids'].update(chunk_stats['flow_ids'])
    
    def process_jsonl_file(self, input_file: str) -> pd.DataFrame:
        """Multi-threaded processing of JSONL file and convert network traffic flows to structured DataFrame."""
        start_time = datetime.now()
        self.logger.info(f"🔄 Starting multi-threaded JSONL to CSV conversion...")
        self.logger.info(f"⚙️ Using {self.max_workers} worker threads, chunk size: {self.chunk_size:,}")
        self.logger.info(f"📊 Processing {len(self.fields)} fields per record")
        self.logger.info(f"🕐 Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Read and chunk the file
        chunks = self.read_jsonl_in_chunks(input_file)
        
        if not chunks:
            self.logger.error("No data chunks created - file may be empty")
            return pd.DataFrame()
        
        all_records = []
        chunk_stats_list = []
        
        # Process chunks in parallel
        self.logger.info(f"🚀 Processing {len(chunks)} chunks with {self.max_workers} threads")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all chunks for processing
            future_to_chunk = {
                executor.submit(self.process_chunk, chunk, chunk_id): chunk_id 
                for chunk_id, chunk in enumerate(chunks)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_chunk):
                chunk_id = future_to_chunk[future]
                try:
                    chunk_records, chunk_stats = future.result()
                    all_records.extend(chunk_records)
                    chunk_stats_list.append(chunk_stats)
                    
                    # Progress update
                    completed_chunks = len(chunk_stats_list)
                    progress_pct = (completed_chunks / len(chunks)) * 100
                    self.logger.info(f"📈 Progress: {completed_chunks}/{len(chunks)} chunks ({progress_pct:.1f}%)")
                    
                except Exception as e:
                    self.logger.error(f"❌ Chunk {chunk_id} failed: {str(e)}")
        
        # Merge statistics from all chunks
        self.merge_chunk_stats(chunk_stats_list)
        
        # Calculate processing time and performance metrics
        end_time = datetime.now()
        processing_duration = end_time - start_time
        total_lines = self.shared_stats['total_processed'] + self.shared_stats['total_errors']
        lines_per_second = total_lines / processing_duration.total_seconds() if processing_duration.total_seconds() > 0 else 0
        
        # Log final statistics with timing
        self.logger.info(f"✅ Multi-threaded processing complete!")
        self.logger.info(f"   🕐 Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"   🕐 End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"   ⏱️ Processing duration: {processing_duration}")
        self.logger.info(f"   📊 Total JSONL lines processed: {total_lines:,}")
        self.logger.info(f"   ✅ Valid records extracted: {self.shared_stats['total_processed']:,}")
        self.logger.info(f"   ❌ JSON parsing errors: {self.shared_stats['total_errors']:,}")
        self.logger.info(f"   📈 Success rate: {(self.shared_stats['total_processed']/total_lines)*100:.1f}%" if total_lines > 0 else "   📈 Success rate: N/A")
        self.logger.info(f"   🚀 Processing speed: {lines_per_second:.1f} lines/second")
        self.logger.info(f"   🔧 Port fields converted: {self.shared_stats['port_conversions']:,}")
        self.logger.info(f"   🔄 Unique flow IDs detected: {len(self.shared_stats['flow_ids']):,}")
        
        # Create DataFrame and apply temporal sorting
        self.logger.info(f"🏗️ Creating DataFrame from {len(all_records):,} records")
        df_result = pd.DataFrame(all_records)
        
        # Apply temporal sorting and timestamp standardization for ML compatibility
        df_result = self._apply_temporal_sorting_and_standardization(df_result)
        
        # Save comprehensive processing log
        print(f"🔄 Generating network traffic processing statistics...")
        print(f"🔍 DEBUG: About to create log for input file: {input_file}")
        try:
            self._save_processing_log(input_file, total_lines, self.shared_stats['total_processed'], 
                                     self.shared_stats['total_errors'], df_result)
            print(f"✅ Network traffic processing statistics complete!")
        except Exception as log_error:
            print(f"❌ CRITICAL: Log creation failed with error: {log_error}")
            print(f"🔍 DEBUG: Input file path: {input_file}")
            print(f"🔍 DEBUG: Input file exists: {os.path.exists(input_file)}")
            print(f"🔍 DEBUG: Input file parent: {Path(input_file).parent}")
            print(f"🔍 DEBUG: Parent directory exists: {Path(input_file).parent.exists()}")
            print(f"🔍 DEBUG: Parent directory writable: {os.access(Path(input_file).parent, os.W_OK)}")
            # Continue execution even if logging fails
            self.logger.error(f"Log creation failed: {log_error}")
        
        return df_result
    
    def _apply_temporal_sorting_and_standardization(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply temporal sorting and convert timestamps to epoch format for ML compatibility."""
        network_config = self.config.get('network_traffic_processor', {}) or self.config.get('script_03_network_csv_creator', {})
        enable_temporal_sorting = network_config.get('enable_temporal_sorting', True)
        
        # Define all timestamp fields to process
        timestamp_fields = ['timestamp', 'event_start', 'event_end']
        processed_fields = []
        
        for field in timestamp_fields:
            if field in df.columns:
                self.logger.info(f"🕒 Converting {field} to ML-compatible epoch format")
                try:
                    # Convert timestamp field to datetime for processing
                    df[field] = pd.to_datetime(df[field], errors='coerce')
                    
                    # Count and log invalid timestamps
                    invalid_timestamps = df[field].isnull().sum()
                    if invalid_timestamps > 0:
                        self.logger.warning(f"⚠️ Found {invalid_timestamps} invalid {field} timestamps")
                    
                    # CONVERT TO EPOCH TIMESTAMP FOR ML COMPATIBILITY
                    # Convert to epoch with millisecond precision preserved as INTEGER
                    df[field] = (df[field].astype('int64') // 10**6).astype('int64')  # nanoseconds to milliseconds (integer)
                    
                    processed_fields.append(field)
                    
                    # Log conversion statistics
                    valid_epochs = df[field].dropna()
                    if len(valid_epochs) > 0:
                        self.logger.info(f"✅ Converted {len(valid_epochs):,} {field} values to epoch format")
                        
                except Exception as e:
                    self.logger.error(f"❌ Error during {field} processing: {e}")
                    self.logger.warning(f"⚠️ Continuing without {field} standardization")
        
        if not processed_fields:
            self.logger.warning("⚠️ No timestamp fields found - timestamps will not be processed")
            return df
        
        # Apply temporal sorting if enabled (use main timestamp field if available)
        if 'timestamp' in processed_fields and enable_temporal_sorting:
            self.logger.info("🔄 Sorting network events chronologically by main timestamp")
            df = df.sort_values('timestamp', na_position='last').reset_index(drop=True)
        
        # Log comprehensive timestamp analysis using main timestamp field
        if 'timestamp' in processed_fields:
            try:
                valid_timestamps = df['timestamp'].dropna()
                if len(valid_timestamps) > 0:
                    # Convert back to human-readable for verification and logging
                    min_time = pd.to_datetime(valid_timestamps.min(), unit='ms')
                    max_time = pd.to_datetime(valid_timestamps.max(), unit='ms')
                    duration = max_time - min_time
                    
                    self.logger.info(f"📅 Network Timeline: {min_time} → {max_time}")
                    self.logger.info(f"⏱️ Network Duration: {duration.total_seconds()/3600:.2f} hours")
                    self.logger.info(f"📊 Main timestamp epoch range: {valid_timestamps.min()} to {valid_timestamps.max()}")
                    
                    # Log flow timeline statistics
                    if 'network_traffic_flow_id' in df.columns:
                        # Convert epoch back to datetime for duration calculations
                        df_temp = df.copy()
                        df_temp['timestamp_dt'] = pd.to_datetime(df_temp['timestamp'], unit='ms')
                        flow_timeline_stats = df_temp.groupby('network_traffic_flow_id')['timestamp_dt'].agg(['min', 'max', 'count'])
                        avg_flow_duration = (flow_timeline_stats['max'] - flow_timeline_stats['min']).mean()
                        avg_events_per_flow = flow_timeline_stats['count'].mean()
                        
                        # Handle different types of duration objects
                        if pd.api.types.is_timedelta64_dtype(avg_flow_duration):
                            avg_duration_seconds = float(avg_flow_duration) / 1e9
                        elif hasattr(avg_flow_duration, 'total_seconds'):
                            avg_duration_seconds = avg_flow_duration.total_seconds()
                        else:
                            avg_duration_seconds = float(avg_flow_duration)
                        
                        self.logger.info(f"🔄 Average flow duration: {avg_duration_seconds:.1f} seconds")
                        self.logger.info(f"📊 Average events per flow: {avg_events_per_flow:.1f}")
                        
            except Exception as e:
                self.logger.error(f"❌ Error during timestamp analysis: {e}")
        
        # Log summary of processed timestamp fields
        if processed_fields:
            self.logger.info(f"⚡ Timestamp fields processed: {', '.join(processed_fields)}")
            self.logger.info("⚡ All timestamp precision: Milliseconds preserved as integer for exact correlation matching")
        
        return df
    
    def perform_exploratory_analysis(self, df: pd.DataFrame) -> None:
        """Perform comprehensive exploratory data analysis on the dataset."""
        network_config = self.config.get('network_traffic_processor', {}) or self.config.get('script_03_network_csv_creator', {})
        if not network_config.get('enable_exploratory_analysis', True):
            return
            
        self.logger.info("🔬 Starting comprehensive exploratory data analysis...")
        
        # Dataset Overview
        self.logger.info("=" * 60)
        self.logger.info("📊 DATASET OVERVIEW")
        self.logger.info("=" * 60)
        self.logger.info(f"• Total Records: {len(df):,}")
        self.logger.info(f"• Total Columns: {len(df.columns)}")
        self.logger.info(f"• Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        # Data Types
        dtype_counts = df.dtypes.value_counts()
        self.logger.info(f"📋 DATA TYPES:")
        for dtype, count in dtype_counts.items():
            self.logger.info(f"   • {str(dtype):15s}: {count:2d} columns")
        
        # Missing Data Analysis
        missing_counts = df.isnull().sum()
        missing_data = missing_counts[missing_counts > 0].sort_values(ascending=False)
        if len(missing_data) > 0:
            self.logger.info(f"📊 MISSING DATA - Top 10 fields:")
            for field, count in missing_data.head(10).items():
                percentage = (count / len(df)) * 100
                self.logger.info(f"   • {field:30s}: {count:,} ({percentage:.1f}%)")
        
        # Network Protocol Analysis
        if 'network_transport' in df.columns:
            self.logger.info(f"🚦 TRANSPORT PROTOCOLS:")
            transport_dist = df['network_transport'].value_counts().head(5)
            for protocol, count in transport_dist.items():
                percentage = (count / len(df)) * 100
                self.logger.info(f"   • {str(protocol).upper():8s}: {count:,} ({percentage:.1f}%)")
        
        # Port Analysis
        if 'destination_port' in df.columns:
            self.logger.info(f"🎯 TOP DESTINATION PORTS:")
            top_ports = df['destination_port'].value_counts().head(10)
            for port, count in top_ports.items():
                percentage = (count / len(df)) * 100
                self.logger.info(f"   • Port {port:0f}: {count:,} ({percentage:.2f}%)")
        
        # Traffic Volume Analysis - FIXED: Proper flow-based statistics
        if 'network_bytes' in df.columns and 'network_traffic_flow_id' in df.columns:
            self.logger.info(f"💾 TRAFFIC VOLUME ANALYSIS:")
            
            # Group by flow ID to avoid double-counting traffic within same flow
            flow_stats = df.groupby('network_traffic_flow_id')['network_bytes'].first()  # Take first occurrence per flow
            
            total_gb = flow_stats.sum() / 1024**3
            bytes_stats = flow_stats.describe()
            
            self.logger.info(f"   🔄 Total unique flows: {len(flow_stats):,}")
            self.logger.info(f"   📊 Total traffic (unique flows): {total_gb:.2f} GB")
            self.logger.info(f"   📈 Mean flow size: {bytes_stats['mean']:,.0f} bytes")
            self.logger.info(f"   📊 Max flow size: {bytes_stats['max']:,.0f} bytes")
            self.logger.info(f"   📊 Min flow size: {bytes_stats['min']:,.0f} bytes")
            
            # Also show raw event stats for comparison
            raw_total_gb = df['network_bytes'].sum() / 1024**3
            events_per_flow = len(df) / len(flow_stats)
            self.logger.info(f"   ⚠️  Raw event total (includes duplicates): {raw_total_gb:.2f} GB")
            self.logger.info(f"   📋 Average events per flow: {events_per_flow:.1f}")
            
        elif 'network_bytes' in df.columns:
            # Fallback if no flow ID available
            bytes_stats = df['network_bytes'].describe()
            total_gb = df['network_bytes'].sum() / 1024**3
            self.logger.info(f"💾 TRAFFIC VOLUME (WARNING: No flow grouping):")
            self.logger.info(f"   • Total traffic: {total_gb:.2f} GB")
            self.logger.info(f"   • Mean event size: {bytes_stats['mean']:,.0f} bytes")
            self.logger.info(f"   • Max event size: {bytes_stats['max']:,.0f} bytes")
        
        # Host Analysis
        if 'host_hostname' in df.columns:
            unique_hosts = df['host_hostname'].nunique()
            self.logger.info(f"🏠 INFRASTRUCTURE:")
            self.logger.info(f"   • Unique hostnames: {unique_hosts:,}")
            if 'source_ip' in df.columns:
                self.logger.info(f"   • Unique source IPs: {df['source_ip'].nunique():,}")
            if 'destination_ip' in df.columns:
                self.logger.info(f"   • Unique destination IPs: {df['destination_ip'].nunique():,}")
        
        self.logger.info("🎉 Exploratory analysis complete!")
    
    def backup_existing_file(self, output_path: str) -> Optional[str]:
        """Create backup of existing output file if it exists."""
        if not os.path.exists(output_path):
            return None
        
        # Support both old and new config formats
        network_config = self.config.get('network_traffic_processor', {}) or self.config.get('script_03_network_csv_creator', {})
        backup_dir = Path(network_config.get('backup_dir', './backups'))
        backup_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_filename = f"{Path(output_path).stem}_backup_{timestamp}.csv"
        backup_path = backup_dir / backup_filename
        
        shutil.copy2(output_path, backup_path)
        self.logger.info(f"📁 Backup created: {backup_path}")
        
        return str(backup_path)
    
    def compare_outputs(self, original_path: str, new_path: str) -> bool:
        """Compare original and new CSV files for validation."""
        if not original_path or not os.path.exists(original_path):
            self.logger.info("No original file to compare with")
            return True
        
        self.logger.info(f"🔍 Comparing outputs...")
        
        try:
            # Load both files
            original_df = pd.read_csv(original_path)
            new_df = pd.read_csv(new_path)
            
            # Basic comparison
            if original_df.shape != new_df.shape:
                self.logger.warning(f"Shape mismatch: Original {original_df.shape} vs New {new_df.shape}")
                return False
            
            # Sort both for consistent comparison
            if 'timestamp' in original_df.columns:
                original_sorted = original_df.sort_values(['timestamp']).reset_index(drop=True)
                new_sorted = new_df.sort_values(['timestamp']).reset_index(drop=True)
            else:
                original_sorted = original_df.reset_index(drop=True)
                new_sorted = new_df.reset_index(drop=True)
            
            # Compare data
            differences = 0
            for col in original_sorted.columns:
                if col in new_sorted.columns:
                    if not original_sorted[col].equals(new_sorted[col]):
                        differences += 1
                        self.logger.warning(f"Column '{col}' differs between files")
            
            if differences == 0:
                self.logger.info("✅ Files are identical!")
                return True
            else:
                self.logger.warning(f"❌ Found {differences} differing columns")
                return False
        
        except Exception as e:
            self.logger.error(f"Error comparing files: {e}")
            return False
    
    def run(self, input_file: Optional[str] = None, output_file: Optional[str] = None, 
            validate: bool = True) -> bool:
        """Run the complete CSV creation process."""
        # Use config values if not provided (support both old and new config formats)
        network_config = self.config.get('network_traffic_processor', {}) or self.config.get('script_03_network_csv_creator', {})
        if not input_file:
            input_file = network_config['input_file']
        if not output_file:
            output_file = network_config['output_file']
        
        self.logger.info("🚀 Starting Network Traffic CSV creation process")
        self.logger.info(f"Input: {input_file}")
        self.logger.info(f"Output: {output_file}")
        
        # Check input file exists
        if not os.path.exists(input_file):
            self.logger.error(f"❌ Input file not found: {input_file}")
            return False
        
        # Log input file size
        file_size_mb = os.path.getsize(input_file) / (1024 * 1024)
        self.logger.info(f"📁 Input file size: {file_size_mb:.1f} MB")
        
        try:
            # Backup existing output file
            backup_path = None
            if validate:
                backup_path = self.backup_existing_file(output_file)
            
            # Process JSONL file
            df = self.process_jsonl_file(input_file)
            
            # Perform exploratory analysis
            self.perform_exploratory_analysis(df)
            
            # Export to CSV
            self.logger.info(f"💾 Exporting to CSV: {output_file}")
            df.to_csv(output_file, index=False)
            
            # Verify output file
            if os.path.exists(output_file):
                output_size_mb = os.path.getsize(output_file) / (1024 * 1024)
                self.logger.info(f"📁 Output file size: {output_size_mb:.1f} MB")
            
            # Validate output
            if validate and backup_path:
                identical = self.compare_outputs(backup_path, output_file)
                if identical:
                    self.logger.info("🎉 Validation successful - outputs are identical!")
                else:
                    self.logger.warning("⚠️ Validation found differences - manual review recommended")
            
            self.logger.info("✅ CSV creation completed successfully!")
            self.logger.info("🎯 Dataset ready for machine learning analysis!")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Processing failed: {e}")
            return False


def auto_detect_files(apt_dir: str, base_dir: str = '.') -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Auto-detect input JSONL, output CSV, and config files in APT directory.
    
    Args:
        apt_dir: APT run directory (e.g., 'apt-1/apt-1-05-04-run-05')
        base_dir: Base directory for APT runs
    
    Returns:
        Tuple of (input_jsonl_path, output_csv_path, config_file_path)
    """
    import glob
    
    # Construct full path
    full_apt_path = os.path.join(base_dir, apt_dir)
    
    if not os.path.exists(full_apt_path):
        raise FileNotFoundError(f"APT directory not found: {full_apt_path}")
    
    print(f"🔍 Auto-detecting files in: {full_apt_path}")
    
    # Detect Network Traffic JSONL file
    network_patterns = [
        "*network*.jsonl",
        "*network_traffic*.jsonl", 
        "*ds-logs-network_traffic*.jsonl"
    ]
    
    input_file = None
    for pattern in network_patterns:
        matches = glob.glob(os.path.join(full_apt_path, pattern))
        if matches:
            input_file = matches[0]  # Take first match
            break
    
    if not input_file:
        print("⚠️  No Network Traffic JSONL file found")
        return None, None, None
    
    # Detect config file first
    config_patterns = ["config.yaml", "config.yml"]
    config_file = None
    for pattern in config_patterns:
        config_path = os.path.join(full_apt_path, pattern)
        if os.path.exists(config_path):
            config_file = config_path
            break
    
    # Generate output CSV filename - prioritize config.yaml, then fallback to auto-generation
    output_file = None
    if config_file:
        try:
            import yaml
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # Get output filename from config (support both old and new config formats)
            network_config = config.get('network_traffic_processor', {}) or config.get('script_03_network_csv_creator', {})
            if 'output_file' in network_config:
                config_output = network_config['output_file']
                output_file = os.path.join(full_apt_path, config_output)
                print(f"📋 Using config-specified output: {config_output}")
        except Exception as e:
            print(f"⚠️  Error reading config file: {e}")
    
    # Fallback to auto-generation if config reading failed
    if not output_file:
        input_basename = os.path.basename(input_file)
        if "network" in input_basename:
            import re
            # PRIORITY 1: Extract run number from input file (e.g., run-04 from ds-logs-network_traffic-flow-default-run-04.jsonl)
            run_match = re.search(r'run-(\d+)', input_basename)
            if run_match:
                run_number = run_match.group(1)
                output_file = os.path.join(full_apt_path, f"netflow-run-{run_number}.csv")
                print(f"📝 Using run-number based naming: netflow-run-{run_number}.csv")
            else:
                # PRIORITY 2: Try date pattern as backup
                date_match = re.search(r'(\d{4}-\d{2}-\d{2})', input_basename)
                if date_match:
                    date_str = date_match.group(1)
                    output_file = os.path.join(full_apt_path, f"network_traffic_flow-{date_str}-000001.csv")
                    print(f"📝 Using date-based naming: network_traffic_flow-{date_str}-000001.csv")
                else:
                    # PRIORITY 3: Final fallback naming
                    output_file = os.path.join(full_apt_path, "network_traffic_flow-output.csv")
                    print(f"📝 Using fallback naming: network_traffic_flow-output.csv")
        else:
            output_file = input_file.replace('.jsonl', '.csv')
            print(f"📝 Using filename replacement: {os.path.basename(output_file)}")
    
    print(f"📥 Detected input: {os.path.relpath(input_file)}")
    print(f"📤 Target output: {os.path.relpath(output_file)}")
    if config_file:
        print(f"⚙️  Detected config: {os.path.relpath(config_file)}")
    else:
        print("⚙️  No config file detected in APT directory")
    
    return input_file, output_file, config_file


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Network Traffic Flow JSONL to CSV Converter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use config.yaml settings (default)
  python 3_network_traffic_csv_creator.py
  
  # Specify custom input/output files
  python 3_network_traffic_csv_creator.py --input custom.jsonl --output custom.csv
  
  # Skip validation (faster)
  python 3_network_traffic_csv_creator.py --no-validate
  
  # Disable exploratory analysis
  python 3_network_traffic_csv_creator.py --no-analysis
  
  # Use custom config file
  python 3_network_traffic_csv_creator.py --config my_config.yaml
        """
    )
    
    # Batch processing parameters
    parser.add_argument('--apt-dir', 
                       help='APT run directory (e.g., apt-1/apt-1-05-04-run-05) - enables auto-detection')
    parser.add_argument('--base-dir', default='.',
                       help='Base directory for APT runs (default: current directory)')
    
    # Traditional parameters
    parser.add_argument('--config', '-c', default='config.yaml',
                       help='Configuration file path (optional, uses defaults if not found)')
    parser.add_argument('--input', '-i',
                       help='Input JSONL file path (overrides config and auto-detection)')
    parser.add_argument('--output', '-o', 
                       help='Output CSV file path (overrides config and auto-detection)')
    parser.add_argument('--no-validate', action='store_true',
                       help='Skip validation against existing output')
    parser.add_argument('--no-analysis', action='store_true',
                       help='Skip exploratory data analysis')
    
    args = parser.parse_args()
    
    try:
        # Handle batch processing mode
        if args.apt_dir:
            input_file, output_file, config_file = auto_detect_files(args.apt_dir, args.base_dir)
            
            # Check if auto-detection failed
            if input_file is None:
                print(f"❌ Error: No Network Traffic JSONL file found in directory: {args.apt_dir}")
                print(f"   Searched patterns: *network*.jsonl, *network_traffic*.jsonl, *ds-logs-network_traffic*.jsonl")
                print(f"   You can specify a file manually with --input <file.jsonl>")
                sys.exit(1)
            
            # Use detected config if available, otherwise fall back to provided config
            if config_file and os.path.exists(config_file):
                converter = NetworkTrafficCSVCreator(config_file)
                print(f"🔧 Using detected config: {config_file}")
            else:
                converter = NetworkTrafficCSVCreator(args.config)
                print(f"🔧 Using default config: {args.config}")
            
            print(f"📂 Processing APT directory: {args.apt_dir}")
            print(f"📥 Auto-detected input: {input_file}")
            print(f"📤 Auto-detected output: {output_file}")
            
            # Override with explicit parameters if provided
            final_input = args.input if args.input else input_file
            final_output = args.output if args.output else output_file
            
            # CRITICAL FIX: Ensure paths are absolute to prevent working directory issues
            final_input = os.path.abspath(final_input)
            final_output = os.path.abspath(final_output)
            print(f"🔧 Resolved absolute input: {final_input}")
            print(f"🔧 Resolved absolute output: {final_output}")
            
        else:
            # Traditional mode - only use config if it exists and --input/--output are not provided
            config_to_use = args.config if os.path.exists(args.config) and not (args.input and args.output) else None
            converter = NetworkTrafficCSVCreator(config_to_use)
            final_input = args.input
            final_output = args.output
            
            # CRITICAL FIX: Ensure paths are absolute in traditional mode too
            if final_input:
                final_input = os.path.abspath(final_input)
                print(f"🔧 Resolved absolute input: {final_input}")
            if final_output:
                final_output = os.path.abspath(final_output)
                print(f"🔧 Resolved absolute output: {final_output}")
            
            if config_to_use:
                print(f"🔧 Traditional mode using config: {args.config}")
            else:
                print(f"🔧 Traditional mode using defaults (no config file)")
        
        # Override config settings if requested
        if args.no_analysis:
            if not hasattr(converter, 'config') or not converter.config:
                converter.config = {}
            converter.config.setdefault('network_traffic_processor', {})['enable_exploratory_analysis'] = False
        
        # Run conversion
        success = converter.run(
            input_file=final_input,
            output_file=final_output,
            validate=not args.no_validate
        )
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n⏹️  Processing cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()