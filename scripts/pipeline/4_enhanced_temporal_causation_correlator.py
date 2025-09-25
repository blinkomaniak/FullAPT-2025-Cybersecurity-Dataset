#!/usr/bin/env python3
"""
Enhanced Temporal Causation NetFlow-Sysmon Correlator v3.0
==========================================================

DESCRIPTION:
    Advanced dual-domain correlation script implementing comprehensive temporal 
    causation analysis between network flows and Sysmon host events with detailed
    statistics collection for histogram plotting and scenario analysis.

PREREQUISITES:
    - Raw APT datasets in: data-raw/apt-X/apt-X-run-XX/ containing:
      * sysmon-run-XX.csv (or sysmon-run-XX-OLD.csv)
      * netflow-run-XX.csv 
    - Python packages: pandas, numpy, matplotlib, seaborn, argparse

USAGE:
    # Run from project root directory (/home/researcher/Downloads/research/)
    cd /home/researcher/Downloads/research/
    
    # Single APT run analysis
    python3 dataset/scripts/pipeline/4_enhanced_temporal_causation_correlator.py --apt-type apt-1 --run-id 10
    
    # Batch processing (high-performing runs ≥90% attribution)
    python3 dataset/scripts/pipeline/4_enhanced_temporal_causation_correlator.py --batch-high-performing --workers 8
    
    # Process all runs (all APT types and run IDs)
    python3 dataset/scripts/pipeline/4_enhanced_temporal_causation_correlator.py --batch-all --workers 8
    
    # Specific run range
    python3 dataset/scripts/pipeline/4_enhanced_temporal_causation_correlator.py --apt-type apt-1 --run-range 09-15

COMMAND LINE OPTIONS:
    --apt-type      APT campaign type (apt-1, apt-2, apt-3, apt-4, apt-5, apt-6)
    --run-id        Specific run ID (01, 02, 03, ..., 51)
    --run-range     Range of runs (e.g., 09-15)
    --batch-high-performing   Process all runs with ≥90% attribution rates
    --batch-all     Process all available APT runs across all types
    --workers       Number of parallel workers for batch processing (default: 4)
    --sample        Sample size for testing (limits dataset size)

INPUT REQUIREMENTS:
    - Sysmon CSV: ProcessCreate(1), ProcessTerminate(5), and other Sysmon events
    - NetFlow CSV: Network community ID grouped flows with process attribution
    - Directory structure: data-raw/apt-X/apt-X-run-XX/[sysmon|netflow]-run-XX.csv

OUTPUT GENERATED:
    analysis/correlation-analysis-v3/apt-X/run-XX/
    ├── enhanced_temporal_correlation_results.json    # Comprehensive statistics
    ├── attribution_summary_flows-run-XX.png          # Flow attribution chart
    ├── attribution_summary_events-run-XX.png         # Event attribution chart  
    ├── process_lifecycle_breakdown.png               # Process types distribution
    ├── scenario_distribution_flows.png               # Temporal scenarios (flows)
    ├── scenario_distribution_events.png              # Temporal scenarios (events)
    ├── start_end_timing_analysis.png                 # 6×3 timing analysis grid
    ├── no_end_timing_analysis.png                    # 3×2 timing analysis grid
    ├── no_start_timing_analysis.png                  # 3×2 timing analysis grid
    ├── no_bounds_timing_analysis.png                 # NetFlow duration analysis
    └── [corresponding .npz files for batch processing]

EXPECTED RUNTIME:
    - Single run: 2-5 minutes (depends on dataset size)
    - High-performing batch: 30-60 minutes (21 runs)
    - Full batch: 2-4 hours (all ~50 APT runs)

TEMPORAL SCENARIOS ANALYZED:
    - Start-End Process: 6 cases (4 overlap + 2 non-overlap scenarios)
    - No-End Process: 3 temporal relationship scenarios  
    - No-Start Process: 3 temporal relationship scenarios
    - No-Start-No-End Process: Always attributed (no temporal constraints)

KEY FEATURES:
    - Complete temporal scenario coverage for all process lifecycle types
    - Comprehensive statistics collection for timing analysis
    - Scenario frequency counting and time gap measurements  
    - Overlap ratio calculations and duration distributions
    - Enhanced visualization with histogram plots and count legends
    - Detailed temporal causation validation with positive gap calculations
    - Green/red color coding for attribution success visualization

EXAMPLE WORKFLOWS:
    # Test single high-performing run
    python3 dataset/scripts/pipeline/4_enhanced_temporal_causation_correlator.py --apt-type apt-1 --run-id 10
    
    # Process all high-performing runs for comprehensive analysis
    python3 dataset/scripts/pipeline/4_enhanced_temporal_causation_correlator.py --batch-high-performing --workers 8
    
    # Generate summary after processing multiple runs
    python3 dataset/scripts/exploratory/4_complete_correlation_summary.py

TROUBLESHOOTING:
    - "File not found": Check data-raw/ directory structure and CSV file names
    - "Memory error": Reduce --workers or use --sample for testing
    - "No attribution": Verify Sysmon/NetFlow data quality and process information
    - "JSON error": Ensure sufficient disk space for large result files
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from pathlib import Path
import argparse
import glob
import ast
import threading
import multiprocessing
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for threading compatibility
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

class TemporalStatistics:
    """Class to collect comprehensive temporal analysis statistics"""
    
    def __init__(self):
        # Attribution categories (core framework)
        self.attribution_categories = {
            'successfully_attributed': 0,
            'temporal_mismatch': 0,
            'no_sysmon_match': 0,
            'missing_pid': 0,
            'inconsistent_overlap': 0
        }
        
        # Flow vs Event tracking
        self.flow_level_results = []
        self.event_level_results = []
        
        # Original input dataset sizes
        self.original_sysmon_events = 0
        self.original_netflow_events = 0
        self.total_flows_for_attribution = 0
        
        # Scenario counters (enhanced temporal analysis)
        self.scenario_counts = {
            'missing_netflow_timestamps': 0,
            'missing_sysmon_timestamps': 0,
            'start_end_process_count': 0,
            'no_end_process_count': 0, 
            'no_start_process_count': 0,
            'no_start_no_end_process_count': 0,
            'unknown_process_type': 0
        }
        
        # Individual case tracking for detailed analysis
        self.start_end_cases = {
            'case_1': 0,  # sysmon_start ≤ netflow_start AND netflow_end ≤ sysmon_end
            'case_2': 0,  # netflow_start ≤ sysmon_start AND netflow_end ≤ sysmon_end
            'case_3': 0,  # sysmon_start ≤ netflow_start AND sysmon_end ≤ netflow_end
            'case_4': 0,  # sysmon_start ≥ netflow_start AND sysmon_end ≤ netflow_end
            'case_5': 0,  # 0 ≤ post_sysmon_end_gap (netflow after sysmon ends)
            'case_6': 0   # 0 ≤ pre_sysmon_start_gap (netflow before sysmon starts)
        }
        
        # No-End Process cases
        self.no_end_cases = {
            'case_1': 0,  # netflow_start ≥ sysmon_start
            'case_2': 0,  # netflow_end ≥ sysmon_start AND sysmon_start ≥ netflow_start
            'case_3': 0   # sysmon_start ≥ netflow_end
        }
        
        # No-Start Process cases
        self.no_start_cases = {
            'case_1': 0,  # sysmon_end ≥ netflow_end
            'case_2': 0,  # netflow_end ≥ sysmon_end AND sysmon_end ≥ netflow_start
            'case_3': 0   # netflow_start ≥ sysmon_end
        }
        
        # No-Start-No-End Process (only one case)
        self.no_bounds_cases = {
            'case_1': 0   # Always attributed (no temporal constraints)
        }
        
        # Case-specific timing statistics (organized by individual cases)
        self.timing_stats = {
            # Start-End Process Cases (6 cases)
            'start_end_case_1': {
                'pre_gaps': [],      # netflow_start - sysmon_start 
                'post_gaps': [],     # sysmon_end - netflow_end
                'overlap_ratios': [] # netflow_duration / sysmon_duration
            },
            'start_end_case_2': {
                'pre_gaps': [],      # netflow_start - sysmon_start
                'post_gaps': [],     # sysmon_end - netflow_end  
                'overlap_ratios': [] # netflow_duration / sysmon_duration
            },
            'start_end_case_3': {
                'pre_gaps': [],      # netflow_start - sysmon_start
                'post_gaps': [],     # sysmon_end - netflow_end
                'overlap_ratios': [] # netflow_duration / sysmon_duration
            },
            'start_end_case_4': {
                'pre_gaps': [],      # netflow_start - sysmon_start
                'post_gaps': [],     # sysmon_end - netflow_end
                'overlap_ratios': [] # netflow_duration / sysmon_duration
            },
            'start_end_case_5': {
                'pre_gaps': [],      # No pre-gaps (netflow after sysmon ends)
                'post_gaps': [],     # post_sysmon_end_gap = netflow_start - sysmon_end
                'overlap_ratios': [] # No overlap ratios (no overlap)
            },
            'start_end_case_6': {
                'pre_gaps': [],      # pre_sysmon_start_gap = sysmon_start - netflow_end  
                'post_gaps': [],     # No post-gaps (netflow before sysmon starts)
                'overlap_ratios': [] # No overlap ratios (no overlap)
            },
            
            # No-End Process Cases (3 cases)
            'no_end_case_1': {
                'pre_gaps': [],      # netflow_start - sysmon_start (≥0)
                'post_gaps': [],     # N/A (no sysmon end)
                'overlap_ratios': [], # N/A (unbounded end)
                'netflow_durations': [] # netflow_end - netflow_start
            },
            'no_end_case_2': {
                'pre_gaps': [],      # netflow_start - sysmon_start (≤0)
                'post_gaps': [],     # N/A (no sysmon end)
                'overlap_ratios': [], # N/A (unbounded end)
                'netflow_durations': [] # netflow_end - netflow_start
            },
            'no_end_case_3': {
                'pre_gaps': [],      # sysmon_start - netflow_end (trigger scenario)
                'post_gaps': [],     # N/A (no sysmon end)
                'overlap_ratios': [], # N/A (unbounded end)
                'netflow_durations': [] # netflow_end - netflow_start
            },
            
            # No-Start Process Cases (3 cases)
            'no_start_case_1': {
                'pre_gaps': [],      # N/A (no sysmon start)
                'post_gaps': [],     # sysmon_end - netflow_end (≥0)
                'overlap_ratios': [], # N/A (unbounded start)
                'netflow_durations': [] # netflow_end - netflow_start
            },
            'no_start_case_2': {
                'pre_gaps': [],      # N/A (no sysmon start)  
                'post_gaps': [],     # sysmon_end - netflow_end (≤0)
                'overlap_ratios': [], # N/A (unbounded start)
                'netflow_durations': [] # netflow_end - netflow_start
            },
            'no_start_case_3': {
                'pre_gaps': [],      # N/A (no sysmon start)
                'post_gaps': [],     # netflow_start - sysmon_end (post-termination)
                'overlap_ratios': [], # N/A (unbounded start)
                'netflow_durations': [] # netflow_end - netflow_start
            },
            
            # No-Start-No-End Process Cases (1 case)
            'no_bounds_case_1': {
                'pre_gaps': [],      # N/A (no boundaries)
                'post_gaps': [],     # N/A (no boundaries)
                'overlap_ratios': [], # N/A (no boundaries)
                'netflow_durations': [] # netflow_end - netflow_start
            }
        }
        
        # Attribution results tracking (legacy format)
        self.attribution_results = []
        
        # Detailed results for CSV export
        self.detailed_flow_results = []
        self.detailed_event_results = []

class EnhancedTemporalCorrelator:
    """Enhanced NetFlow-Sysmon correlator with comprehensive temporal analysis"""
    
    def __init__(self, max_workers=None, sample_size=None):
        self.base_dir = Path("/home/researcher/Downloads/research")
        self.dataset_dir = self.base_dir / "dataset"
        self.results_dir = self.base_dir / "analysis" / "correlation-analysis-v3"
        
        # Threading configuration
        self.max_workers = max_workers or min(32, multiprocessing.cpu_count())
        self.progress_lock = threading.Lock()
        self.processed_chunks = 0
        self.total_chunks = 0
        
        # Sampling for testing (if specified, limit processing to N rows)
        self.sample_size = sample_size
        
        # Dynamic mappings (extracted per run)
        self.computer_ip_mapping = {}
        self.ip_computer_mapping = {}
        
        # Statistics collection
        self.temporal_stats = TemporalStatistics()
    
    def find_dataset_files(self, apt_type: str, run_id: str) -> Tuple[Path, Path]:
        """Find sysmon and netflow dataset files"""
        apt_dir = self.dataset_dir / apt_type / f"{apt_type}-run-{run_id}"
        
        # Try multiple naming patterns for sysmon files
        sysmon_paths = [
            apt_dir / f"sysmon-run-{run_id}.csv",
            apt_dir / f"sysmon-run-{run_id}-OLD.csv"
        ]
        
        # Try multiple patterns for netflow files  
        netflow_paths = [
            apt_dir / f"netflow-run-{run_id}.csv"
        ]
        
        sysmon_file = None
        for path in sysmon_paths:
            if path.exists():
                sysmon_file = path
                break
                
        # Fallback: glob pattern for sysmon files
        if not sysmon_file:
            pattern = str(apt_dir / f"sysmon*{run_id}*.csv")
            candidates = glob.glob(pattern)
            if candidates:
                sysmon_file = Path(candidates[0])
        
        netflow_file = None
        for path in netflow_paths:
            if path.exists():
                netflow_file = path
                break
        
        if not sysmon_file or not netflow_file:
            raise FileNotFoundError(
                f"Missing dataset files for {apt_type}-run-{run_id}\n"
                f"Sysmon: {sysmon_file}\nNetflow: {netflow_file}"
            )
        
        return sysmon_file, netflow_file
    
    def load_datasets(self, apt_type: str, run_id: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load sysmon and netflow datasets for correlation"""
        print(f"🔍 Loading datasets for {apt_type.upper()}-Run-{run_id}")
        
        sysmon_file, netflow_file = self.find_dataset_files(apt_type, run_id)
        
        print(f"📥 Loading Sysmon events: {sysmon_file.name}")
        try:
            sysmon_df = pd.read_csv(sysmon_file, low_memory=False)
        except Exception as e:
            print(f"⚠️  Trying alternate CSV engine for Sysmon: {e}")
            sysmon_df = pd.read_csv(sysmon_file, engine='python', low_memory=False)
        
        print(f"📥 Loading NetFlow events: {netflow_file.name}")
        try:
            netflow_df = pd.read_csv(netflow_file, low_memory=False)
        except Exception as e:
            print(f"⚠️  Trying alternate CSV engine for NetFlow: {e}")
            netflow_df = pd.read_csv(netflow_file, engine='python', low_memory=False)
        
        # Apply sampling if specified for testing
        if self.sample_size:
            original_sysmon_len = len(sysmon_df)
            original_netflow_len = len(netflow_df)
            sysmon_df = sysmon_df.head(self.sample_size)
            netflow_df = netflow_df.head(self.sample_size)
            print(f"🔬 Sampling applied: {original_sysmon_len:,} → {len(sysmon_df):,} Sysmon, {original_netflow_len:,} → {len(netflow_df):,} NetFlow")
        
        print(f"✅ Loaded {len(sysmon_df):,} Sysmon events")
        print(f"✅ Loaded {len(netflow_df):,} NetFlow events")
        
        return sysmon_df, netflow_df
    
    def build_dynamic_computer_ip_mapping(self, sysmon_df: pd.DataFrame, netflow_df: pd.DataFrame):
        """Build dynamic computer-IP mapping from dataset contents"""
        print("🔍 Building dynamic computer-IP mapping...")
        
        # Extract hostnames from sysmon
        sysmon_hostnames = sysmon_df['Computer'].str.split('.', n=1).str[0].unique()
        sysmon_hostnames = [h for h in sysmon_hostnames if pd.notna(h)]
        
        print(f"   Found {len(sysmon_hostnames)} unique hostnames in Sysmon: {list(sysmon_hostnames)}")
        
        # Extract network info from netflow
        if 'host_hostname' in netflow_df.columns and 'host_ip' in netflow_df.columns:
            netflow_mapping = netflow_df[['host_hostname', 'host_ip', 'host_mac']].drop_duplicates()
            
            print(f"   Found {len(netflow_mapping)} unique host mappings in NetFlow")
            
            # Build mapping dictionaries
            self.computer_ip_mapping = {}
            self.ip_computer_mapping = {}
            
            for _, row in netflow_mapping.iterrows():
                hostname = row['host_hostname']
                if pd.isna(hostname):
                    continue
                    
                # Parse IP addresses (can be lists)
                try:
                    if isinstance(row['host_ip'], str):
                        if row['host_ip'].startswith('['):
                            # Parse list format: "['ip1', 'ip2']"
                            ip_list = ast.literal_eval(row['host_ip'])
                        else:
                            # Single IP
                            ip_list = [row['host_ip']]
                    else:
                        continue
                    
                    # Map hostname to IPs (both IPv4 and IPv6)
                    for ip in ip_list:
                        if isinstance(ip, str) and ip.strip():
                            # Create bidirectional mapping
                            if hostname not in self.computer_ip_mapping:
                                self.computer_ip_mapping[hostname] = []
                            self.computer_ip_mapping[hostname].append(ip)
                            self.ip_computer_mapping[ip] = hostname
                            
                except Exception as e:
                    continue
            
            print(f"   ✅ Built mapping: {len(self.computer_ip_mapping)} hostnames → {len(self.ip_computer_mapping)} IPs")
        else:
            print("   ⚠️  NetFlow missing hostname/IP columns - using fallback mapping")
            
            # Fallback: extract IPs from source/dest fields
            unique_ips = set()
            for col in ['source_ip', 'destination_ip']:
                if col in netflow_df.columns:
                    unique_ips.update(netflow_df[col].dropna().unique())
            
            print(f"   Found {len(unique_ips)} unique IPs in NetFlow")
            
            # Simple mapping (limited without hostname info)
            for hostname in sysmon_hostnames:
                self.computer_ip_mapping[hostname] = list(unique_ips)[:5]  # Limit to first 5 IPs
                for ip in self.computer_ip_mapping[hostname]:
                    self.ip_computer_mapping[ip] = hostname

    def analyze_sysmon_process_lifecycles(self, sysmon_df: pd.DataFrame) -> Dict[str, Dict]:
        """Analyze Sysmon process lifecycles with comprehensive classification"""
        print("🔄 Analyzing Sysmon process lifecycles...")
        
        # Group by process tuple: (ProcessId, Image, Computer)
        process_groups = sysmon_df.groupby(['ProcessId', 'Image', 'Computer'])
        
        process_lifecycles = {}
        
        for (pid, image, computer), group in process_groups:
            # Convert timestamp column to datetime if it's string
            if 'timestamp' in group.columns:
                group = group.copy()
                group['timestamp'] = pd.to_datetime(group['timestamp'])
            
            # Classify events by type
            create_events = group[group['EventID'] == 1]  # Process Create
            terminate_events = group[group['EventID'] == 5]  # Process Terminate
            other_events = group[~group['EventID'].isin([1, 5])]  # Other process events
            
            # Determine process lifecycle type
            has_start = len(create_events) > 0
            has_end = len(terminate_events) > 0
            has_activity = len(other_events) > 0
            
            if has_start and has_end:
                lifecycle_type = "Start-End Process"
                start_time = create_events['timestamp'].min()
                end_time = terminate_events['timestamp'].max()
            elif has_start and not has_end:
                lifecycle_type = "No-End Process"
                start_time = create_events['timestamp'].min()
                end_time = pd.NaT  # No end time (process still running)
            elif not has_start and has_end:
                lifecycle_type = "No-Start Process"
                start_time = pd.NaT  # No start time (pre-existing process)
                end_time = terminate_events['timestamp'].max()
            elif has_activity:
                lifecycle_type = "No-Start-No-End Process"
                start_time = pd.NaT  # Unknown start
                end_time = pd.NaT    # Unknown end
            else:
                # Empty process tuple - skip
                continue
            
            # Create process identifier
            process_key = f"{pid}_{image}_{computer}"
            
            process_lifecycles[process_key] = {
                'ProcessId': pid,
                'Image': image,
                'Computer': computer,
                'lifecycle_type': lifecycle_type,
                'start_time': start_time,
                'end_time': end_time,
                'create_events_count': len(create_events),
                'terminate_events_count': len(terminate_events),
                'other_events_count': len(other_events),
                'total_events': len(group)
            }
        
        # Print statistics
        lifecycle_counts = {}
        for process_data in process_lifecycles.values():
            ltype = process_data['lifecycle_type']
            lifecycle_counts[ltype] = lifecycle_counts.get(ltype, 0) + 1
        
        print(f"✅ Analyzed {len(process_lifecycles)} unique processes:")
        for ltype, count in lifecycle_counts.items():
            print(f"   {ltype}: {count:,} processes")
        
        return process_lifecycles

    def group_netflow_by_community_id(self, netflow_df: pd.DataFrame) -> Dict[str, List[Dict]]:
        """Group NetFlow by network_community_id and identify individual flows"""
        print("🔗 Grouping NetFlow by network_community_id...")
        
        if 'network_community_id' not in netflow_df.columns:
            print("⚠️  No network_community_id column found")
            return {}
        
        # Sort by timestamp for proper flow analysis
        if 'timestamp' in netflow_df.columns:
            ip_traffic_sorted = netflow_df.sort_values('timestamp')
        else:
            print("⚠️  No timestamp column found - using original order")
            ip_traffic_sorted = netflow_df
        
        community_flows = {}
        edge_case_stats = {
            'inconsistent_network_flow_overlap': 0,
            'inconsistent_network_flow_pid_missing': 0,
            'warning_process_pid_assignment_delay': 0,
            'total_flows': 0
        }
        
        # Group by network_community_id
        grouped = ip_traffic_sorted.groupby('network_community_id')
        
        for community_id, group_events in grouped:
            flows = self._analyze_community_id_flows(community_id, group_events, edge_case_stats)
            if flows:
                community_flows[community_id] = flows
        
        print(f"✅ Identified flows from {len(grouped)} community IDs:")
        print(f"   Total flows: {edge_case_stats['total_flows']}")
        if edge_case_stats['total_flows'] > 0:
            print(f"   Edge cases:")
            for case, count in edge_case_stats.items():
                if case != 'total_flows' and count > 0:
                    print(f"     {case}: {count}")
        
        return community_flows

    def _analyze_community_id_flows(self, community_id: str, events: pd.DataFrame, edge_case_stats: Dict) -> List[Dict]:
        """Analyze individual flows within a network_community_id group"""
        flows = []
        current_flow_events = []
        current_process_pid = None
        flow_sequence = 0
        
        for idx, event in events.iterrows():
            process_pid = event.get('process_pid')
            process_executable = event.get('process_executable')
            flow_final = event.get('network_traffic_flow_final', False)
            
            # Convert flow_final to boolean
            if isinstance(flow_final, str):
                flow_final = flow_final.lower() == 'true'
            
            # Handle process_pid assignment
            if pd.notna(process_pid):
                if current_process_pid is None:
                    # First non-null PID in the group
                    current_process_pid = process_pid
                    if len(current_flow_events) > 0:
                        # PID assignment delay - first event had null PID
                        edge_case_stats['warning_process_pid_assignment_delay'] += 1
                elif current_process_pid != process_pid:
                    # Different PID before flow end - overlap issue
                    edge_case_stats['inconsistent_network_flow_overlap'] += 1
                    # End current flow prematurely and start new one
                    if current_flow_events:
                        flows.append(self._create_flow_dict(
                            community_id, flow_sequence, current_flow_events, 
                            current_process_pid, 'Inconsistent Overlap'
                        ))
                        flow_sequence += 1
                        edge_case_stats['total_flows'] += 1
                    current_flow_events = []
                    current_process_pid = process_pid
            
            # Add event to current flow
            current_flow_events.append(event)
            
            # Check if flow ends
            if flow_final:
                if current_process_pid is not None:
                    flows.append(self._create_flow_dict(
                        community_id, flow_sequence, current_flow_events, 
                        current_process_pid, 'Normal Flow'
                    ))
                    edge_case_stats['total_flows'] += 1
                else:
                    # Flow with no process_pid at all
                    edge_case_stats['inconsistent_network_flow_pid_missing'] += 1
                
                # Reset for next flow
                current_flow_events = []
                current_process_pid = None
                flow_sequence += 1
        
        # Handle any remaining events (flow didn't end properly)
        if current_flow_events:
            if current_process_pid is not None:
                flows.append(self._create_flow_dict(
                    community_id, flow_sequence, current_flow_events, 
                    current_process_pid, 'Incomplete Flow'
                ))
                edge_case_stats['total_flows'] += 1
            else:
                edge_case_stats['inconsistent_network_flow_pid_missing'] += 1
        
        return flows

    def _create_flow_dict(self, community_id: str, flow_sequence: int, events: List, 
                         process_pid, flow_status: str) -> Dict:
        """Create a flow dictionary from events"""
        events_df = pd.DataFrame(events)
        
        # Extract flow characteristics
        source_ips = events_df['source_ip'].dropna().unique().tolist()
        dest_ips = events_df['destination_ip'].dropna().unique().tolist()
        source_ports = events_df['source_port'].dropna().unique().tolist()
        dest_ports = events_df['destination_port'].dropna().unique().tolist()
        
        # Get process executable (take first non-null)
        process_executable = events_df['process_executable'].dropna().iloc[0] if len(events_df['process_executable'].dropna()) > 0 else None
        
        # Time span - Use actual network flow timespan, not recording timestamps
        if 'event_start' in events_df.columns and 'event_end' in events_df.columns:
            # Use actual network communication timespan
            event_start = pd.to_datetime(events_df['event_start'].min())
            event_end = pd.to_datetime(events_df['event_end'].max())
        elif 'timestamp' in events_df.columns:
            # Fallback to recording timestamps if flow times not available
            event_start = pd.to_datetime(events_df['timestamp'].min())
            event_end = pd.to_datetime(events_df['timestamp'].max())
        else:
            event_start = None
            event_end = None
        
        return {
            'flow_id': f"{community_id}_{flow_sequence}",
            'community_id': community_id,
            'flow_sequence': flow_sequence,
            'process_pid': process_pid,
            'process_executable': process_executable,
            'source_ips': source_ips,
            'dest_ips': dest_ips,
            'source_ports': source_ports,
            'dest_ports': dest_ports,
            'event_start': event_start,
            'event_end': event_end,
            'event_count': len(events),
            'flow_status': flow_status,
            'events': events_df.to_dict('records')  # Store original events for analysis
        }

    def analyze_temporal_overlap_enhanced(self, flow: Dict, sysmon_process: Dict) -> Dict:
        """
        Enhanced temporal causation analysis implementing comprehensive scenario coverage
        Based on the fixed pseudocode with complete temporal relationship analysis
        """
        # Extract temporal information
        netflow_start = flow.get('event_start')
        netflow_end = flow.get('event_end')
        sysmon_start = sysmon_process.get('start_time')
        sysmon_end = sysmon_process.get('end_time')
        process_type = sysmon_process.get('lifecycle_type')
        
        # Initialize result structure
        result = {
            'has_overlap': False,
            'reason': '',
            'scenario': '',
            'timing_stats': {},
            'process_type': process_type
        }
        
        # Check for missing NetFlow timestamps
        if pd.isna(netflow_start) or pd.isna(netflow_end):
            self.temporal_stats.scenario_counts['missing_netflow_timestamps'] += 1
            result.update({
                'has_overlap': False,
                'reason': 'Missing NetFlow timestamps',
                'scenario': 'missing_netflow_timestamps'
            })
            return result
        
        # Convert to datetime if needed
        if not isinstance(netflow_start, pd.Timestamp):
            netflow_start = pd.to_datetime(netflow_start)
        if not isinstance(netflow_end, pd.Timestamp):
            netflow_end = pd.to_datetime(netflow_end)
        
        # Calculate netflow duration
        netflow_duration_ms = (netflow_end - netflow_start).total_seconds() * 1000
        
        # Process type-specific temporal analysis
        if process_type == "Start-End Process":
            return self._analyze_start_end_process(
                netflow_start, netflow_end, netflow_duration_ms,
                sysmon_start, sysmon_end, result
            )
        
        elif process_type == "No-End Process":
            return self._analyze_no_end_process(
                netflow_start, netflow_end, netflow_duration_ms,
                sysmon_start, result
            )
        
        elif process_type == "No-Start Process":
            return self._analyze_no_start_process(
                netflow_start, netflow_end, netflow_duration_ms,
                sysmon_end, result
            )
        
        elif process_type == "No-Start-No-End Process":
            return self._analyze_no_bounds_process(
                netflow_start, netflow_end, netflow_duration_ms, result
            )
        
        else:
            # Unknown process type
            self.temporal_stats.scenario_counts['unknown_process_type'] += 1
            result.update({
                'has_overlap': False,
                'reason': 'Unknown process type',
                'scenario': 'unknown_process_type'
            })
            return result

    def _analyze_start_end_process(self, netflow_start, netflow_end, netflow_duration_ms,
                                 sysmon_start, sysmon_end, result):
        """Analyze Start-End Process temporal scenarios"""
        self.temporal_stats.scenario_counts['start_end_process_count'] += 1
        
        # Check for missing Sysmon timestamps
        if pd.isna(sysmon_start) or pd.isna(sysmon_end):
            self.temporal_stats.scenario_counts['missing_sysmon_timestamps'] += 1
            result.update({
                'has_overlap': False,
                'reason': 'Missing Sysmon Process timestamps',
                'scenario': 'missing_sysmon_timestamps'
            })
            return result
        
        # Convert to datetime if needed
        if not isinstance(sysmon_start, pd.Timestamp):
            sysmon_start = pd.to_datetime(sysmon_start)
        if not isinstance(sysmon_end, pd.Timestamp):
            sysmon_end = pd.to_datetime(sysmon_end)
        
        # Calculate timing statistics
        sysmon_duration_ms = (sysmon_end - sysmon_start).total_seconds() * 1000
        
        # Case 1: Netflow span inside Sysmon process span
        if sysmon_start <= netflow_start and netflow_end <= sysmon_end:
            self.temporal_stats.start_end_cases['case_1'] += 1
            pre_gap_ms = (netflow_start - sysmon_start).total_seconds() * 1000
            post_gap_ms = (sysmon_end - netflow_end).total_seconds() * 1000
            overlap_ratio = netflow_duration_ms / sysmon_duration_ms if sysmon_duration_ms > 0 else 0
            
            # Store case-specific timing statistics
            self.temporal_stats.timing_stats['start_end_case_1']['pre_gaps'].append(pre_gap_ms)
            self.temporal_stats.timing_stats['start_end_case_1']['post_gaps'].append(post_gap_ms)
            self.temporal_stats.timing_stats['start_end_case_1']['overlap_ratios'].append(overlap_ratio)
            
            result.update({
                'has_overlap': True,
                'reason': 'Overlap: Netflow span inside Sysmon process span',
                'scenario': 'start_end_case_1',
                'timing_stats': {
                    'pre_gap_ms': pre_gap_ms,
                    'post_gap_ms': post_gap_ms,
                    'overlap_ratio': overlap_ratio,
                    'netflow_duration_ms': netflow_duration_ms,
                    'sysmon_duration_ms': sysmon_duration_ms
                }
            })
            return result
        
        # Case 2: Netflow span started before Sysmon process span
        elif netflow_start <= sysmon_start and netflow_end <= sysmon_end:
            self.temporal_stats.start_end_cases['case_2'] += 1
            pre_gap_ms = (sysmon_start - netflow_start).total_seconds() * 1000
            overlap_duration_ms = (netflow_end - sysmon_start).total_seconds() * 1000
            overlap_ratio = overlap_duration_ms / sysmon_duration_ms if sysmon_duration_ms > 0 else 0
            post_gap_ms = (sysmon_end - netflow_end).total_seconds() * 1000
            
            # Store case-specific timing statistics
            self.temporal_stats.timing_stats['start_end_case_2']['pre_gaps'].append(pre_gap_ms)  # Always positive
            self.temporal_stats.timing_stats['start_end_case_2']['post_gaps'].append(post_gap_ms)
            self.temporal_stats.timing_stats['start_end_case_2']['overlap_ratios'].append(overlap_ratio)
            
            result.update({
                'has_overlap': True,
                'reason': 'Overlap: Netflow span started before Sysmon process span',
                'scenario': 'start_end_case_2',
                'timing_stats': {
                    'pre_gap_ms': pre_gap_ms,
                    'overlap_ratio': overlap_ratio,
                    'post_gap_ms': post_gap_ms,
                    'netflow_duration_ms': netflow_duration_ms,
                    'sysmon_duration_ms': sysmon_duration_ms
                }
            })
            return result
        
        # Case 3: Netflow span started after Sysmon Process span 
        elif sysmon_start <= netflow_start and sysmon_end <= netflow_end:
            self.temporal_stats.start_end_cases['case_3'] += 1
            pre_gap_ms = (netflow_start - sysmon_start).total_seconds() * 1000
            overlap_duration_ms = (sysmon_end - netflow_start).total_seconds() * 1000
            overlap_ratio = overlap_duration_ms / sysmon_duration_ms if sysmon_duration_ms > 0 else 0
            post_gap_ms = (netflow_end - sysmon_end).total_seconds() * 1000
            
            # Store case-specific timing statistics
            self.temporal_stats.timing_stats['start_end_case_3']['pre_gaps'].append(pre_gap_ms)
            self.temporal_stats.timing_stats['start_end_case_3']['post_gaps'].append(post_gap_ms)  # Always positive
            self.temporal_stats.timing_stats['start_end_case_3']['overlap_ratios'].append(overlap_ratio)
            
            result.update({
                'has_overlap': True,
                'reason': 'Overlap: Netflow span started after Sysmon Process span',
                'scenario': 'start_end_case_3',
                'timing_stats': {
                    'pre_gap_ms': pre_gap_ms,
                    'overlap_ratio': overlap_ratio,
                    'post_gap_ms': post_gap_ms,
                    'netflow_duration_ms': netflow_duration_ms,
                    'sysmon_duration_ms': sysmon_duration_ms
                }
            })
            return result
        
        # Case 4: Netflow span started before and ends after Sysmon Process span (completely contains)
        elif sysmon_start >= netflow_start and sysmon_end <= netflow_end:
            self.temporal_stats.start_end_cases['case_4'] += 1
            pre_gap_ms = (sysmon_start - netflow_start).total_seconds() * 1000
            containment_ratio = sysmon_duration_ms / netflow_duration_ms if netflow_duration_ms > 0 else 0
            post_gap_ms = (netflow_end - sysmon_end).total_seconds() * 1000
            
            # Store case-specific timing statistics
            self.temporal_stats.timing_stats['start_end_case_4']['pre_gaps'].append(pre_gap_ms)  # Always positive
            self.temporal_stats.timing_stats['start_end_case_4']['post_gaps'].append(post_gap_ms)  # Always positive
            self.temporal_stats.timing_stats['start_end_case_4']['overlap_ratios'].append(containment_ratio)
            
            result.update({
                'has_overlap': True,
                'reason': 'Overlap: Netflow span started before and ends after Sysmon Process span',
                'scenario': 'start_end_case_4',
                'timing_stats': {
                    'pre_gap_ms': pre_gap_ms,
                    'containment_ratio': containment_ratio,
                    'post_gap_ms': post_gap_ms,
                    'netflow_duration_ms': netflow_duration_ms,
                    'sysmon_duration_ms': sysmon_duration_ms
                }
            })
            return result
        
        # Non-overlap scenarios
        post_sysmon_end_gap = (netflow_start - sysmon_end).total_seconds() * 1000
        pre_sysmon_start_gap = (sysmon_start - netflow_end).total_seconds() * 1000
        
        # Case 5: Netflow occurs after Sysmon process ended
        if post_sysmon_end_gap >= 0:
            self.temporal_stats.start_end_cases['case_5'] += 1
            # Store case-specific timing statistics (Case 5: No overlap, post-termination)
            self.temporal_stats.timing_stats['start_end_case_5']['post_gaps'].append(post_sysmon_end_gap)
            
            result.update({
                'has_overlap': False,
                'reason': 'Netflow occurs after Sysmon process ended',
                'scenario': 'start_end_case_5',
                'timing_stats': {
                    'separation_gap_ms': post_sysmon_end_gap,
                    'netflow_duration_ms': netflow_duration_ms
                }
            })
            return result
        
        # Case 6: Netflow occurs before Sysmon process started
        if pre_sysmon_start_gap >= 0:
            self.temporal_stats.start_end_cases['case_6'] += 1
            # Store case-specific timing statistics (Case 6: No overlap, pre-start)
            self.temporal_stats.timing_stats['start_end_case_6']['pre_gaps'].append(pre_sysmon_start_gap)
            
            result.update({
                'has_overlap': False,
                'reason': 'Netflow occurs before Sysmon process started',
                'scenario': 'start_end_case_6',
                'timing_stats': {
                    'separation_gap_ms': pre_sysmon_start_gap,
                    'netflow_duration_ms': netflow_duration_ms
                }
            })
            return result
        
        # Should not reach here, but safety fallback
        result.update({
            'has_overlap': False,
            'reason': 'Unhandled Start-End temporal scenario',
            'scenario': 'start_end_unhandled'
        })
        return result

    def _analyze_no_end_process(self, netflow_start, netflow_end, netflow_duration_ms,
                              sysmon_start, result):
        """Analyze No-End Process temporal scenarios"""
        self.temporal_stats.scenario_counts['no_end_process_count'] += 1
        
        # Check for missing Sysmon start time
        if pd.isna(sysmon_start):
            result.update({
                'has_overlap': False,
                'reason': 'Missing Sysmon Process start time',
                'scenario': 'missing_sysmon_start_time'
            })
            return result
        
        # Convert to datetime if needed
        if not isinstance(sysmon_start, pd.Timestamp):
            sysmon_start = pd.to_datetime(sysmon_start)
        
        # Case 1: NetFlow starts after Sysmon Process start
        if netflow_start >= sysmon_start:
            self.temporal_stats.no_end_cases['case_1'] += 1
            start_gap_ms = (netflow_start - sysmon_start).total_seconds() * 1000
            
            # Store case-specific timing statistics
            self.temporal_stats.timing_stats['no_end_case_1']['pre_gaps'].append(start_gap_ms)
            self.temporal_stats.timing_stats['no_end_case_1']['netflow_durations'].append(netflow_duration_ms)
            
            result.update({
                'has_overlap': True,
                'reason': 'NetFlow starts after Sysmon Process start',
                'scenario': 'no_end_case_1',
                'timing_stats': {
                    'start_gap_ms': start_gap_ms,
                    'netflow_duration_ms': netflow_duration_ms
                }
            })
            return result
        
        # Case 2: NetFlow starts before and ends after Sysmon Process start
        elif netflow_end >= sysmon_start and sysmon_start >= netflow_start:
            self.temporal_stats.no_end_cases['case_2'] += 1
            pre_start_gap_ms = (sysmon_start - netflow_start).total_seconds() * 1000
            
            # Store case-specific timing statistics
            self.temporal_stats.timing_stats['no_end_case_2']['pre_gaps'].append(pre_start_gap_ms)  # Always positive
            self.temporal_stats.timing_stats['no_end_case_2']['netflow_durations'].append(netflow_duration_ms)
            
            result.update({
                'has_overlap': True,
                'reason': 'NetFlow starts before and ends after Sysmon Process start',
                'scenario': 'no_end_case_2',
                'timing_stats': {
                    'pre_start_gap_ms': pre_start_gap_ms,
                    'netflow_duration_ms': netflow_duration_ms
                }
            })
            return result
        
        # Case 3: NetFlow starts and ends before Sysmon Process start
        elif sysmon_start >= netflow_end:
            self.temporal_stats.no_end_cases['case_3'] += 1
            trigger_gap_ms = (sysmon_start - netflow_end).total_seconds() * 1000
            
            # Store case-specific timing statistics (this could be a trigger scenario)
            self.temporal_stats.timing_stats['no_end_case_3']['pre_gaps'].append(trigger_gap_ms)
            self.temporal_stats.timing_stats['no_end_case_3']['netflow_durations'].append(netflow_duration_ms)
            
            result.update({
                'has_overlap': True,  # Still considered valid attribution for No-End processes
                'reason': 'NetFlow starts and ends before Sysmon Process start',
                'scenario': 'no_end_case_3',
                'timing_stats': {
                    'trigger_gap_ms': trigger_gap_ms,
                    'netflow_duration_ms': netflow_duration_ms
                }
            })
            return result
        
        # Should not reach here, but safety fallback
        result.update({
            'has_overlap': False,
            'reason': 'Unhandled No-End temporal scenario',
            'scenario': 'no_end_unhandled'
        })
        return result

    def _analyze_no_start_process(self, netflow_start, netflow_end, netflow_duration_ms,
                                sysmon_end, result):
        """Analyze No-Start Process temporal scenarios"""
        self.temporal_stats.scenario_counts['no_start_process_count'] += 1
        
        # Check for missing Sysmon end time
        if pd.isna(sysmon_end):
            result.update({
                'has_overlap': False,
                'reason': 'Missing Sysmon Process end time',
                'scenario': 'missing_sysmon_end_time'
            })
            return result
        
        # Convert to datetime if needed
        if not isinstance(sysmon_end, pd.Timestamp):
            sysmon_end = pd.to_datetime(sysmon_end)
        
        # Case 1: NetFlow starts and ends before Sysmon Process end
        if sysmon_end >= netflow_end:
            self.temporal_stats.no_start_cases['case_1'] += 1
            end_gap_ms = (sysmon_end - netflow_end).total_seconds() * 1000
            
            # Store case-specific timing statistics
            self.temporal_stats.timing_stats['no_start_case_1']['post_gaps'].append(end_gap_ms)
            self.temporal_stats.timing_stats['no_start_case_1']['netflow_durations'].append(netflow_duration_ms)
            
            result.update({
                'has_overlap': True,
                'reason': 'NetFlow starts and ends before Sysmon Process end',
                'scenario': 'no_start_case_1',
                'timing_stats': {
                    'end_gap_ms': end_gap_ms,
                    'netflow_duration_ms': netflow_duration_ms
                }
            })
            return result
        
        # Case 2: NetFlow starts before and ends after Sysmon Process end
        elif netflow_end >= sysmon_end and sysmon_end >= netflow_start:
            self.temporal_stats.no_start_cases['case_2'] += 1
            post_end_gap_ms = (netflow_end - sysmon_end).total_seconds() * 1000
            
            # Store case-specific timing statistics
            self.temporal_stats.timing_stats['no_start_case_2']['post_gaps'].append(post_end_gap_ms)  # Always positive
            self.temporal_stats.timing_stats['no_start_case_2']['netflow_durations'].append(netflow_duration_ms)
            
            result.update({
                'has_overlap': True,
                'reason': 'NetFlow starts before and ends after Sysmon Process end',
                'scenario': 'no_start_case_2',
                'timing_stats': {
                    'post_end_gap_ms': post_end_gap_ms,
                    'netflow_duration_ms': netflow_duration_ms
                }
            })
            return result
        
        # Case 3: NetFlow starts after Sysmon Process end
        elif netflow_start >= sysmon_end:
            self.temporal_stats.no_start_cases['case_3'] += 1
            post_gap_ms = (netflow_start - sysmon_end).total_seconds() * 1000
            
            # Store case-specific timing statistics (this could still be valid for No-Start processes)
            self.temporal_stats.timing_stats['no_start_case_3']['post_gaps'].append(post_gap_ms)
            self.temporal_stats.timing_stats['no_start_case_3']['netflow_durations'].append(netflow_duration_ms)
            
            result.update({
                'has_overlap': True,  # Still considered valid attribution for No-Start processes
                'reason': 'NetFlow starts after Sysmon Process end',
                'scenario': 'no_start_case_3',
                'timing_stats': {
                    'post_gap_ms': post_gap_ms,
                    'netflow_duration_ms': netflow_duration_ms
                }
            })
            return result
        
        # Should not reach here, but safety fallback
        result.update({
            'has_overlap': False,
            'reason': 'Unhandled No-Start temporal scenario',
            'scenario': 'no_start_unhandled'
        })
        return result

    def _analyze_no_bounds_process(self, netflow_start, netflow_end, netflow_duration_ms, result):
        """Analyze No-Start-No-End Process temporal scenarios"""
        self.temporal_stats.scenario_counts['no_start_no_end_process_count'] += 1
        
        # Case 1: Always attributed (no temporal constraints)
        self.temporal_stats.no_bounds_cases['case_1'] += 1
        
        # Store case-specific timing statistics (No timing constraints for this case)
        # Note: No pre_gaps, post_gaps, or overlap_ratios for unbounded processes
        self.temporal_stats.timing_stats['no_bounds_case_1']['netflow_durations'].append(netflow_duration_ms)
        
        result.update({
            'has_overlap': True,  # Always attributed (no temporal constraints)
            'reason': 'Netflow matching Sysmon Process with No-Start and No-End',
            'scenario': 'no_bounds_case_1',
            'timing_stats': {
                'netflow_duration_ms': netflow_duration_ms
            }
        })
        return result

    def check_computer_ip_match(self, flow: Dict, sysmon_computer: str) -> bool:
        """Check if flow IPs match sysmon computer using dynamic mapping"""
        # Collect all IPs from the flow
        flow_ips = set()
        
        if flow.get('source_ips'):
            flow_ips.update(flow['source_ips'])
        if flow.get('dest_ips'):
            flow_ips.update(flow['dest_ips'])
        
        # Extract hostname from sysmon computer
        if '.' in sysmon_computer:
            hostname = sysmon_computer.split('.')[0].lower()
        else:
            hostname = sysmon_computer.lower()
        
        # Check if any flow IP maps to this hostname
        for ip in flow_ips:
            if ip in self.ip_computer_mapping:
                if self.ip_computer_mapping[ip].lower() == hostname:
                    return True
        
        return False

    def perform_enhanced_correlation(self, apt_type: str, run_id: str):
        """Perform enhanced temporal correlation analysis with standard attribution framework"""
        print(f"\n🚀 Starting Enhanced Temporal Attribution Analysis for {apt_type.upper()}-Run-{run_id}")
        start_time = datetime.now()
        
        try:
            # Initialize statistics for this run
            self.temporal_stats = TemporalStatistics()
            
            # Load datasets
            sysmon_df, netflow_df = self.load_datasets(apt_type, run_id)
            
            # Store original input dataset sizes 
            self.temporal_stats.original_sysmon_events = len(sysmon_df)
            self.temporal_stats.original_netflow_events = len(netflow_df)
            
            # Build computer-IP mapping
            self.build_dynamic_computer_ip_mapping(sysmon_df, netflow_df)
            
            # Analyze Sysmon process lifecycles
            process_lifecycles = self.analyze_sysmon_process_lifecycles(sysmon_df)
            
            # Group NetFlow by community ID
            community_flows = self.group_netflow_by_community_id(netflow_df)
            
            # Perform comprehensive attribution analysis
            print("🔗 Performing enhanced temporal attribution analysis...")
            
            total_flows = sum(len(flows) for flows in community_flows.values())
            processed_flows = 0
            
            # Store total flows that should be processed for attribution
            self.temporal_stats.total_flows_for_attribution = total_flows
            
            for community_id, flows in community_flows.items():
                for flow in flows:
                    # Process each flow through attribution framework
                    self._process_flow_attribution(flow, process_lifecycles)
                    
                    processed_flows += 1
                    
                    # Progress update
                    if processed_flows % 1000 == 0:
                        progress = (processed_flows / total_flows) * 100
                        print(f"   Progress: {processed_flows:,}/{total_flows:,} flows ({progress:.1f}%)")
            
            # Calculate flow-level and event-level statistics
            self._calculate_attribution_statistics()
            
            # Calculate final statistics
            total_flows_attributed = self.temporal_stats.attribution_categories['successfully_attributed']
            total_flows_analyzed = len(self.temporal_stats.flow_level_results)
            flow_attribution_rate = (total_flows_attributed / total_flows_analyzed) * 100 if total_flows_analyzed > 0 else 0
            
            total_events_attributed = sum(1 for r in self.temporal_stats.event_level_results if r['attribution_status'] == 'Successfully Attributed')
            total_events_analyzed = len(self.temporal_stats.event_level_results)
            event_attribution_rate = (total_events_attributed / total_events_analyzed) * 100 if total_events_analyzed > 0 else 0
            
            print(f"✅ Enhanced Temporal Attribution completed:")
            print(f"   Total flows analyzed: {total_flows_analyzed:,}")
            print(f"   Successfully attributed flows: {total_flows_attributed:,} ({flow_attribution_rate:.2f}%)")
            print(f"   Total events analyzed: {total_events_analyzed:,}")
            print(f"   Successfully attributed events: {total_events_attributed:,} ({event_attribution_rate:.2f}%)")
            
            # Save results and create visualizations
            self.save_enhanced_results(apt_type, run_id, start_time)
            self.create_enhanced_visualizations(apt_type, run_id)
            self.create_standard_attribution_plots(apt_type, run_id)
            self.export_detailed_results(apt_type, run_id)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            print(f"🎯 Analysis completed in {processing_time:.1f} seconds")
            
            return {
                'apt_type': apt_type,
                'run_id': run_id,
                'total_flows': total_flows_analyzed,
                'attributed_flows': total_flows_attributed,
                'flow_attribution_rate': flow_attribution_rate,
                'total_events': total_events_analyzed,
                'attributed_events': total_events_attributed,
                'event_attribution_rate': event_attribution_rate,
                'processing_time': processing_time
            }
            
        except Exception as e:
            print(f"❌ Error in attribution analysis: {e}")
            raise

    def _process_flow_attribution(self, flow: Dict, process_lifecycles: Dict):
        """Process single flow through comprehensive attribution framework"""
        flow_pid = flow.get('process_pid')
        flow_executable = flow.get('process_executable')
        flow_events = flow.get('events', [])
        
        # Check for missing PID
        if pd.isna(flow_pid):
            self.temporal_stats.attribution_categories['missing_pid'] += 1
            
            # Record flow-level result
            flow_result = self._create_flow_attribution_result(flow, 'Missing PID', None, None)
            self.temporal_stats.flow_level_results.append(flow_result)
            
            # Also populate legacy attribution_results for backward compatibility
            self.temporal_stats.attribution_results.append(flow_result)
            
            # Record event-level results
            for event in flow_events:
                event_result = self._create_event_attribution_result(flow, event, 'Missing PID', None, None)
                self.temporal_stats.event_level_results.append(event_result)
            
            return
        
        # Search for matching sysmon processes by PID
        matching_processes = []
        for process_key, process_data in process_lifecycles.items():
            if process_data['ProcessId'] == flow_pid:
                # Check computer-IP match
                if self.check_computer_ip_match(flow, process_data['Computer']):
                    # Check executable match if both are available
                    if flow_executable and process_data.get('Image'):
                        flow_exe_lower = str(flow_executable).lower()
                        sysmon_exe_lower = str(process_data['Image']).lower()
                        # Fuzzy executable matching (contains)
                        if flow_exe_lower in sysmon_exe_lower or sysmon_exe_lower in flow_exe_lower:
                            matching_processes.append(process_data)
                    else:
                        # No executable info to compare, accept IP match
                        matching_processes.append(process_data)
        
        # Check if no matching process found
        if not matching_processes:
            self.temporal_stats.attribution_categories['no_sysmon_match'] += 1
            
            # Record flow-level result
            flow_result = self._create_flow_attribution_result(flow, 'No Sysmon Match', None, None)
            self.temporal_stats.flow_level_results.append(flow_result)
            
            # Also populate legacy attribution_results for backward compatibility
            self.temporal_stats.attribution_results.append(flow_result)
            
            # Record event-level results
            for event in flow_events:
                event_result = self._create_event_attribution_result(flow, event, 'No Sysmon Match', None, None)
                self.temporal_stats.event_level_results.append(event_result)
            
            return
        
        # Use best matching process (first one for now, could be enhanced)
        sysmon_process = matching_processes[0]
        
        # Perform enhanced temporal analysis
        temporal_result = self.analyze_temporal_overlap_enhanced(flow, sysmon_process)
        
        # Determine attribution status
        if temporal_result['has_overlap']:
            attribution_status = 'Successfully Attributed'
            self.temporal_stats.attribution_categories['successfully_attributed'] += 1
        else:
            attribution_status = 'Temporal Mismatch'
            self.temporal_stats.attribution_categories['temporal_mismatch'] += 1
        
        # Record flow-level result
        flow_result = self._create_flow_attribution_result(flow, attribution_status, sysmon_process, temporal_result)
        self.temporal_stats.flow_level_results.append(flow_result)
        
        # Also populate legacy attribution_results for backward compatibility
        self.temporal_stats.attribution_results.append(flow_result)
        
        # Record event-level results
        for event in flow_events:
            event_result = self._create_event_attribution_result(flow, event, attribution_status, sysmon_process, temporal_result)
            self.temporal_stats.event_level_results.append(event_result)

    def _create_flow_attribution_result(self, flow: Dict, attribution_status: str, 
                                      sysmon_process: Optional[Dict], temporal_result: Optional[Dict]) -> Dict:
        """Create flow-level attribution result"""
        result = {
            'flow_id': flow['flow_id'],
            'community_id': flow['community_id'],
            'attribution_status': attribution_status,
            'process_pid': flow.get('process_pid'),
            'process_executable': flow.get('process_executable'),
            'event_count': flow.get('event_count', 0),
            'event_start': flow.get('event_start'),
            'event_end': flow.get('event_end'),
            'source_ips': flow.get('source_ips', []),
            'dest_ips': flow.get('dest_ips', []),
            'temporal_analysis': temporal_result
        }
        
        if sysmon_process:
            result.update({
                'sysmon_process_id': sysmon_process['ProcessId'],
                'sysmon_image': sysmon_process['Image'],
                'sysmon_computer': sysmon_process['Computer'],
                'sysmon_lifecycle_type': sysmon_process['lifecycle_type'],
                'sysmon_start_time': sysmon_process.get('start_time'),
                'sysmon_end_time': sysmon_process.get('end_time')
            })
        
        return result

    def _create_event_attribution_result(self, flow: Dict, event: Dict, attribution_status: str,
                                       sysmon_process: Optional[Dict], temporal_result: Optional[Dict]) -> Dict:
        """Create event-level attribution result"""
        result = {
            'flow_id': flow['flow_id'],
            'community_id': flow['community_id'],
            'event_timestamp': event.get('timestamp'),
            'attribution_status': attribution_status,
            'process_pid': event.get('process_pid'),
            'process_executable': event.get('process_executable'),
            'source_ip': event.get('source_ip'),
            'destination_ip': event.get('destination_ip'),
            'source_port': event.get('source_port'),
            'destination_port': event.get('destination_port'),
            'network_transport': event.get('network_transport'),
            'temporal_analysis': temporal_result
        }
        
        if sysmon_process:
            result.update({
                'sysmon_process_id': sysmon_process['ProcessId'],
                'sysmon_image': sysmon_process['Image'],
                'sysmon_computer': sysmon_process['Computer'],
                'sysmon_lifecycle_type': sysmon_process['lifecycle_type']
            })
        
        return result

    def _calculate_attribution_statistics(self):
        """Calculate comprehensive attribution statistics"""
        # Already calculated in _process_flow_attribution
        # This method can be extended for additional statistics
        pass

    def create_standard_attribution_plots(self, apt_type: str, run_id: str):
        """Create standard attribution plots matching previous script outputs"""
        print("📊 Creating standard attribution plots...")
        
        output_dir = self.results_dir / apt_type / f"run-{run_id}"
        
        # Set style for better plots
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Flow-level attribution summary
        self._create_flow_attribution_summary_plot(output_dir, run_id)
        
        # 2. Event-level attribution summary
        self._create_event_attribution_summary_plot(output_dir, run_id)
        
        print("✅ Standard attribution plots created")

    def _create_flow_attribution_summary_plot(self, output_dir: Path, run_id: str):
        """Create flow-level attribution summary plot"""
        try:
            # Prepare flow attribution data
            flow_categories = {}
            for result in self.temporal_stats.flow_level_results:
                status = result['attribution_status']
                flow_categories[status] = flow_categories.get(status, 0) + 1
            
            if not flow_categories:
                print("⚠️  No flow data to plot")
                return
            
            # Create single bar chart with percentages
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            
            categories = list(flow_categories.keys())
            counts = list(flow_categories.values())
            
            # Assign colors: Green for 'Successfully Attributed', Red for others
            colors = []
            for category in categories:
                if 'Successfully Attributed' in category:
                    colors.append('#2E8B57')  # Green
                else:
                    colors.append('#DC143C')  # Red
            
            total_flows = sum(counts)
            percentages = [(count / total_flows) * 100 for count in counts]
            
            bars = ax.bar(categories, counts, color=colors)
            ax.set_ylabel('Number of Flows')
            ax.set_title(f'NetFlow-Sysmon Flow Attribution Analysis - Run {run_id}', 
                        fontsize=14, fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', alpha=0.3)
            
            # Add total flows legend/info box
            ax.text(0.02, 0.98, f'Total Flows: {total_flows:,}', transform=ax.transAxes, 
                   fontsize=12, verticalalignment='top', fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
            
            # Add value and percentage labels on bars
            for bar, count, pct in zip(bars, counts, percentages):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(counts) * 0.01,
                       f'{count:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout()
            
            # Save plot (PNG + NPZ)
            plot_file = output_dir / f'attribution_summary_flows-run-{run_id}.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            
            # Save NPZ format for batch processing
            npz_file = output_dir / f'attribution_summary_flows-run-{run_id}.npz'
            np.savez(npz_file, categories=categories, counts=counts, colors=colors, run_id=run_id)
            
            plt.close()
            
            print(f"   📊 Flow attribution summary saved: {plot_file.name} + NPZ")
            
        except Exception as e:
            print(f"⚠️  Error creating flow attribution summary: {e}")

    def _create_event_attribution_summary_plot(self, output_dir: Path, run_id: str):
        """Create event-level attribution summary plot"""
        try:
            # Prepare event attribution data
            event_categories = {}
            for result in self.temporal_stats.event_level_results:
                status = result['attribution_status']
                event_categories[status] = event_categories.get(status, 0) + 1
            
            if not event_categories:
                print("⚠️  No event data to plot")
                return
            
            # Create single bar chart with percentages
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            
            categories = list(event_categories.keys())
            counts = list(event_categories.values())
            
            # Assign colors: Green for 'Successfully Attributed', Red for others
            colors = []
            for category in categories:
                if 'Successfully Attributed' in category:
                    colors.append('#2E8B57')  # Green
                else:
                    colors.append('#DC143C')  # Red
            
            total_events = sum(counts)
            percentages = [(count / total_events) * 100 for count in counts]
            
            bars = ax.bar(categories, counts, color=colors)
            ax.set_ylabel('Number of Events')
            ax.set_title(f'NetFlow-Sysmon Event Attribution Analysis - Run {run_id}', 
                        fontsize=14, fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', alpha=0.3)
            
            # Add total events legend/info box
            ax.text(0.02, 0.98, f'Total Events: {total_events:,}', transform=ax.transAxes, 
                   fontsize=12, verticalalignment='top', fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
            
            # Add value and percentage labels on bars
            for bar, count, pct in zip(bars, counts, percentages):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(counts) * 0.01,
                       f'{count:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout()
            
            # Save plot (PNG + NPZ)
            plot_file = output_dir / f'attribution_summary_events-run-{run_id}.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            
            # Save NPZ format for batch processing
            npz_file = output_dir / f'attribution_summary_events-run-{run_id}.npz'
            np.savez(npz_file, categories=categories, counts=counts, colors=colors, 
                    percentages=percentages, run_id=run_id)
            
            plt.close()
            
            print(f"   📊 Event attribution summary saved: {plot_file.name} + NPZ")
            
        except Exception as e:
            print(f"⚠️  Error creating event attribution summary: {e}")

    def _create_temporal_overlap_analysis_plot(self, output_dir: Path, run_id: str):
        """Create temporal overlap analysis plot"""
        try:
            # Collect temporal scenario data from successfully attributed flows
            attributed_flows = [r for r in self.temporal_stats.flow_level_results 
                              if r['attribution_status'] == 'Successfully Attributed']
            
            if not attributed_flows:
                print("⚠️  No attributed flows for temporal analysis plot")
                return
            
            # Prepare temporal scenario data
            temporal_scenarios = {}
            process_type_counts = {}
            
            for flow_result in attributed_flows:
                temporal_analysis = flow_result.get('temporal_analysis', {})
                if temporal_analysis:
                    scenario = temporal_analysis.get('scenario', 'unknown')
                    process_type = temporal_analysis.get('process_type', 'unknown')
                    
                    temporal_scenarios[scenario] = temporal_scenarios.get(scenario, 0) + 1
                    process_type_counts[process_type] = process_type_counts.get(process_type, 0) + 1
            
            # Create plot
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # 1. Temporal scenarios distribution
            if temporal_scenarios:
                scenarios = list(temporal_scenarios.keys())
                scenario_counts = list(temporal_scenarios.values())
                
                ax1.barh(scenarios, scenario_counts, color='skyblue')
                ax1.set_xlabel('Number of Flows')
                ax1.set_title('Temporal Scenario Distribution')
                
                # Add value labels
                for i, count in enumerate(scenario_counts):
                    ax1.text(count + max(scenario_counts) * 0.01, i, str(count), va='center')
            
            # 2. Process lifecycle types distribution
            if process_type_counts:
                types = list(process_type_counts.keys())
                type_counts = list(process_type_counts.values())
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(types)]
                
                ax2.pie(type_counts, labels=types, colors=colors, autopct='%1.1f%%')
                ax2.set_title('Process Lifecycle Types')
            
            # 3. Timing gaps histogram (Start-End processes)
            start_end_gaps = self.temporal_stats.timing_stats.get('start_end_pre_gaps', [])
            if start_end_gaps:
                # Filter outliers for better visualization
                gaps_array = np.array(start_end_gaps)
                q95 = np.percentile(gaps_array, 95)
                q05 = np.percentile(gaps_array, 5)
                filtered_gaps = gaps_array[(gaps_array >= q05) & (gaps_array <= q95)]
                
                ax3.hist(filtered_gaps, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
                ax3.set_xlabel('Time Gap (ms)')
                ax3.set_ylabel('Frequency')
                ax3.set_title('Start-End Process Pre-Gaps Distribution')
                ax3.grid(alpha=0.3)
            
            # 4. Overlap ratios histogram
            overlap_ratios = self.temporal_stats.timing_stats.get('start_end_overlap_ratios', [])
            if overlap_ratios:
                ax4.hist(overlap_ratios, bins=30, alpha=0.7, color='coral', edgecolor='black')
                ax4.set_xlabel('Overlap Ratio')
                ax4.set_ylabel('Frequency')
                ax4.set_title('Temporal Overlap Ratios Distribution')
                ax4.grid(alpha=0.3)
            
            plt.suptitle(f'Temporal Overlap Analysis - Run {run_id}', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # Save plot
            plot_file = output_dir / f'temporal_overlap_analysis-run-{run_id}.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"   📊 Temporal overlap analysis saved: {plot_file.name}")
            
        except Exception as e:
            print(f"⚠️  Error creating temporal overlap analysis: {e}")

    def export_detailed_results(self, apt_type: str, run_id: str):
        """Export detailed results to CSV and JSON formats"""
        print("📤 Exporting detailed results...")
        
        output_dir = self.results_dir / apt_type / f"run-{run_id}"
        
        try:
            # Export flow-level results to CSV
            if self.temporal_stats.flow_level_results:
                flow_df = pd.DataFrame(self.temporal_stats.flow_level_results)
                
                # Flatten temporal_analysis data for CSV
                if 'temporal_analysis' in flow_df.columns:
                    temporal_data = []
                    for _, row in flow_df.iterrows():
                        temp_analysis = row['temporal_analysis']
                        if temp_analysis and isinstance(temp_analysis, dict):
                            temporal_data.append({
                                'temporal_has_overlap': temp_analysis.get('has_overlap'),
                                'temporal_reason': temp_analysis.get('reason'),
                                'temporal_scenario': temp_analysis.get('scenario'),
                                'temporal_process_type': temp_analysis.get('process_type')
                            })
                        else:
                            temporal_data.append({
                                'temporal_has_overlap': None,
                                'temporal_reason': None,
                                'temporal_scenario': None,
                                'temporal_process_type': None
                            })
                    
                    temporal_df = pd.DataFrame(temporal_data)
                    flow_df = flow_df.drop('temporal_analysis', axis=1)
                    flow_df = pd.concat([flow_df, temporal_df], axis=1)
                
                # Convert lists to strings for CSV compatibility
                for col in ['source_ips', 'dest_ips']:
                    if col in flow_df.columns:
                        flow_df[col] = flow_df[col].astype(str)
                
                csv_file = output_dir / f'detailed_correlation_results-run-{run_id}.csv'
                flow_df.to_csv(csv_file, index=False)
                print(f"   📊 Flow results CSV exported: {csv_file.name}")
            
            # Export comprehensive JSON results (enhanced format)
            comprehensive_results = {
                'analysis_metadata': {
                    'apt_type': apt_type,
                    'run_id': run_id,
                    'analysis_version': 'v3.0-enhanced-temporal-with-standard-attribution',
                    'analysis_timestamp': datetime.now().isoformat(),
                    'total_sysmon_events_loaded': self.temporal_stats.original_sysmon_events,
                    'total_netflow_events_loaded': self.temporal_stats.original_netflow_events,
                    'total_flows_for_attribution': self.temporal_stats.total_flows_for_attribution,
                    'flows_actually_processed': len(self.temporal_stats.flow_level_results),
                    'events_actually_processed': len(self.temporal_stats.event_level_results),
                },
                'attribution_summary': {
                    'flow_level': {
                        'total_flows': self.temporal_stats.total_flows_for_attribution,
                        'attribution_breakdown': self.temporal_stats.attribution_categories.copy(),
                        'attribution_rate_percent': (self.temporal_stats.attribution_categories['successfully_attributed'] / 
                                                   self.temporal_stats.total_flows_for_attribution * 100) 
                                                   if self.temporal_stats.total_flows_for_attribution > 0 else 0
                    },
                    'event_level': {
                        'total_events': self.temporal_stats.original_netflow_events,
                        'successfully_attributed': sum(1 for r in self.temporal_stats.event_level_results 
                                                     if r['attribution_status'] == 'Successfully Attributed'),
                        'attribution_rate_percent': (sum(1 for r in self.temporal_stats.event_level_results 
                                                        if r['attribution_status'] == 'Successfully Attributed') / 
                                                    self.temporal_stats.original_netflow_events * 100) 
                                                    if self.temporal_stats.original_netflow_events > 0 else 0
                    }
                },
                'enhanced_temporal_analysis': {
                    'scenario_counts': self.temporal_stats.scenario_counts,
                    'start_end_cases': self.temporal_stats.start_end_cases,
                    'no_end_cases': self.temporal_stats.no_end_cases,
                    'no_start_cases': self.temporal_stats.no_start_cases,
                    'no_bounds_cases': self.temporal_stats.no_bounds_cases,
                    'case_specific_timing_statistics': {
                        # Start-End Process Cases (6 cases) 
                        'start_end_case_1': {
                            'pre_gaps_stats': self._calculate_timing_stats(self.temporal_stats.timing_stats['start_end_case_1']['pre_gaps']),
                            'post_gaps_stats': self._calculate_timing_stats(self.temporal_stats.timing_stats['start_end_case_1']['post_gaps']),
                            'overlap_ratios_stats': self._calculate_timing_stats(self.temporal_stats.timing_stats['start_end_case_1']['overlap_ratios'])
                        },
                        'start_end_case_2': {
                            'pre_gaps_stats': self._calculate_timing_stats(self.temporal_stats.timing_stats['start_end_case_2']['pre_gaps']),
                            'post_gaps_stats': self._calculate_timing_stats(self.temporal_stats.timing_stats['start_end_case_2']['post_gaps']),
                            'overlap_ratios_stats': self._calculate_timing_stats(self.temporal_stats.timing_stats['start_end_case_2']['overlap_ratios'])
                        },
                        'start_end_case_3': {
                            'pre_gaps_stats': self._calculate_timing_stats(self.temporal_stats.timing_stats['start_end_case_3']['pre_gaps']),
                            'post_gaps_stats': self._calculate_timing_stats(self.temporal_stats.timing_stats['start_end_case_3']['post_gaps']),
                            'overlap_ratios_stats': self._calculate_timing_stats(self.temporal_stats.timing_stats['start_end_case_3']['overlap_ratios'])
                        },
                        'start_end_case_4': {
                            'pre_gaps_stats': self._calculate_timing_stats(self.temporal_stats.timing_stats['start_end_case_4']['pre_gaps']),
                            'post_gaps_stats': self._calculate_timing_stats(self.temporal_stats.timing_stats['start_end_case_4']['post_gaps']),
                            'overlap_ratios_stats': self._calculate_timing_stats(self.temporal_stats.timing_stats['start_end_case_4']['overlap_ratios'])
                        },
                        'start_end_case_5': {
                            'post_gaps_stats': self._calculate_timing_stats(self.temporal_stats.timing_stats['start_end_case_5']['post_gaps'])
                        },
                        'start_end_case_6': {
                            'pre_gaps_stats': self._calculate_timing_stats(self.temporal_stats.timing_stats['start_end_case_6']['pre_gaps'])
                        },
                        
                        # No-End Process Cases (3 cases)
                        'no_end_case_1': {
                            'pre_gaps_stats': self._calculate_timing_stats(self.temporal_stats.timing_stats['no_end_case_1']['pre_gaps'])
                        },
                        'no_end_case_2': {
                            'pre_gaps_stats': self._calculate_timing_stats(self.temporal_stats.timing_stats['no_end_case_2']['pre_gaps'])
                        },
                        'no_end_case_3': {
                            'pre_gaps_stats': self._calculate_timing_stats(self.temporal_stats.timing_stats['no_end_case_3']['pre_gaps'])
                        },
                        
                        # No-Start Process Cases (3 cases)
                        'no_start_case_1': {
                            'post_gaps_stats': self._calculate_timing_stats(self.temporal_stats.timing_stats['no_start_case_1']['post_gaps'])
                        },
                        'no_start_case_2': {
                            'post_gaps_stats': self._calculate_timing_stats(self.temporal_stats.timing_stats['no_start_case_2']['post_gaps'])
                        },
                        'no_start_case_3': {
                            'post_gaps_stats': self._calculate_timing_stats(self.temporal_stats.timing_stats['no_start_case_3']['post_gaps'])
                        },
                        
                        # No-Start-No-End Process Cases (1 case)  
                        'no_bounds_case_1': {
                            'note': 'No timing constraints for unbounded processes'
                        }
                    }
                },
                'detailed_results_grouped': {
                    'successfully_attributed': [r for r in self.temporal_stats.flow_level_results 
                                              if r['attribution_status'] == 'Successfully Attributed'],
                    'temporal_mismatch': [r for r in self.temporal_stats.flow_level_results 
                                        if r['attribution_status'] == 'Temporal Mismatch'],
                    'no_sysmon_match': [r for r in self.temporal_stats.flow_level_results 
                                      if r['attribution_status'] == 'No Sysmon Match'],
                    'missing_pid': [r for r in self.temporal_stats.flow_level_results 
                                  if r['attribution_status'] == 'Missing PID'],
                    'inconsistent_overlap': [r for r in self.temporal_stats.flow_level_results 
                                           if r['attribution_status'] == 'Inconsistent Overlap']
                }
            }
            
            json_file = output_dir / f'enhanced_netflow_sysmon_correlation-run-{run_id}.json'
            with open(json_file, 'w') as f:
                json.dump(comprehensive_results, f, indent=2, default=str)
            
            print(f"   📊 Comprehensive JSON exported: {json_file.name}")
            
        except Exception as e:
            print(f"⚠️  Error exporting detailed results: {e}")

    def save_enhanced_results(self, apt_type: str, run_id: str, start_time: datetime):
        """Save enhanced temporal correlation results"""
        print("💾 Saving enhanced correlation results...")
        
        # Create output directory
        output_dir = self.results_dir / apt_type / f"run-{run_id}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Compile comprehensive results
        results = {
            'analysis_metadata': {
                'apt_type': apt_type,
                'run_id': run_id,
                'analysis_version': 'v3.0-enhanced-temporal',
                'analysis_timestamp': datetime.now().isoformat(),
                'processing_time_seconds': (datetime.now() - start_time).total_seconds(),
                'total_sysmon_events_loaded': self.temporal_stats.original_sysmon_events,
                'total_netflow_events_loaded': self.temporal_stats.original_netflow_events,
                'total_flows_for_attribution': self.temporal_stats.total_flows_for_attribution,
                'flows_actually_processed': len(self.temporal_stats.flow_level_results),
                'events_actually_processed': len(self.temporal_stats.event_level_results),
            },
            'attribution_summary': {
                'flow_level': {
                    'total_flows': self.temporal_stats.total_flows_for_attribution,
                    'attribution_breakdown': {
                        'successfully_attributed': sum(1 for r in self.temporal_stats.flow_level_results 
                                                     if r['attribution_status'] == 'Successfully Attributed'),
                        'temporal_mismatch': sum(1 for r in self.temporal_stats.flow_level_results 
                                               if r['attribution_status'] == 'Temporal Mismatch'),
                        'missing_pid': sum(1 for r in self.temporal_stats.flow_level_results 
                                         if r['attribution_status'] == 'Missing PID'),
                        'no_sysmon_match': sum(1 for r in self.temporal_stats.flow_level_results 
                                             if r['attribution_status'] == 'No Sysmon Match'),
                    },
                    'attribution_rate_percent': (sum(1 for r in self.temporal_stats.flow_level_results 
                                                   if r['attribution_status'] == 'Successfully Attributed') / 
                                               self.temporal_stats.total_flows_for_attribution * 100) 
                                               if self.temporal_stats.total_flows_for_attribution > 0 else 0
                },
                'event_level': {
                    'total_events': len(self.temporal_stats.event_level_results),
                    'successfully_attributed': sum(1 for r in self.temporal_stats.event_level_results 
                                                 if r['attribution_status'] == 'Successfully Attributed'),
                    'attribution_rate_percent': (sum(1 for r in self.temporal_stats.event_level_results 
                                                    if r['attribution_status'] == 'Successfully Attributed') / 
                                                len(self.temporal_stats.event_level_results) * 100) 
                                                if len(self.temporal_stats.event_level_results) > 0 else 0
                }
            },
            'temporal_scenario_statistics': {
                'scenario_counts': self.temporal_stats.scenario_counts,
                'start_end_cases': self.temporal_stats.start_end_cases,
                'no_end_cases': self.temporal_stats.no_end_cases,
                'no_start_cases': self.temporal_stats.no_start_cases,
                'no_bounds_cases': self.temporal_stats.no_bounds_cases
            },
            'case_specific_timing_statistics': {
                # Start-End Process Cases (6 cases) 
                'start_end_case_1': {
                    'pre_gaps_stats': self._calculate_timing_stats(self.temporal_stats.timing_stats['start_end_case_1']['pre_gaps']),
                    'post_gaps_stats': self._calculate_timing_stats(self.temporal_stats.timing_stats['start_end_case_1']['post_gaps']),
                    'overlap_ratios_stats': self._calculate_timing_stats(self.temporal_stats.timing_stats['start_end_case_1']['overlap_ratios'])
                },
                'start_end_case_2': {
                    'pre_gaps_stats': self._calculate_timing_stats(self.temporal_stats.timing_stats['start_end_case_2']['pre_gaps']),
                    'post_gaps_stats': self._calculate_timing_stats(self.temporal_stats.timing_stats['start_end_case_2']['post_gaps']),
                    'overlap_ratios_stats': self._calculate_timing_stats(self.temporal_stats.timing_stats['start_end_case_2']['overlap_ratios'])
                },
                'start_end_case_3': {
                    'pre_gaps_stats': self._calculate_timing_stats(self.temporal_stats.timing_stats['start_end_case_3']['pre_gaps']),
                    'post_gaps_stats': self._calculate_timing_stats(self.temporal_stats.timing_stats['start_end_case_3']['post_gaps']),
                    'overlap_ratios_stats': self._calculate_timing_stats(self.temporal_stats.timing_stats['start_end_case_3']['overlap_ratios'])
                },
                'start_end_case_4': {
                    'pre_gaps_stats': self._calculate_timing_stats(self.temporal_stats.timing_stats['start_end_case_4']['pre_gaps']),
                    'post_gaps_stats': self._calculate_timing_stats(self.temporal_stats.timing_stats['start_end_case_4']['post_gaps']),
                    'overlap_ratios_stats': self._calculate_timing_stats(self.temporal_stats.timing_stats['start_end_case_4']['overlap_ratios'])
                },
                'start_end_case_5': {
                    'post_gaps_stats': self._calculate_timing_stats(self.temporal_stats.timing_stats['start_end_case_5']['post_gaps'])
                },
                'start_end_case_6': {
                    'pre_gaps_stats': self._calculate_timing_stats(self.temporal_stats.timing_stats['start_end_case_6']['pre_gaps'])
                },
                
                # No-End Process Cases (3 cases)
                'no_end_case_1': {
                    'pre_gaps_stats': self._calculate_timing_stats(self.temporal_stats.timing_stats['no_end_case_1']['pre_gaps'])
                },
                'no_end_case_2': {
                    'pre_gaps_stats': self._calculate_timing_stats(self.temporal_stats.timing_stats['no_end_case_2']['pre_gaps'])
                },
                'no_end_case_3': {
                    'pre_gaps_stats': self._calculate_timing_stats(self.temporal_stats.timing_stats['no_end_case_3']['pre_gaps'])
                },
                
                # No-Start Process Cases (3 cases)
                'no_start_case_1': {
                    'post_gaps_stats': self._calculate_timing_stats(self.temporal_stats.timing_stats['no_start_case_1']['post_gaps'])
                },
                'no_start_case_2': {
                    'post_gaps_stats': self._calculate_timing_stats(self.temporal_stats.timing_stats['no_start_case_2']['post_gaps'])
                },
                'no_start_case_3': {
                    'post_gaps_stats': self._calculate_timing_stats(self.temporal_stats.timing_stats['no_start_case_3']['post_gaps'])
                },
                
                # No-Start-No-End Process Cases (1 case)  
                'no_bounds_case_1': {
                    'note': 'No timing constraints for unbounded processes'
                }
            },
            'detailed_attribution_results': self.temporal_stats.flow_level_results
        }
        
        # Note: Attribution rates are already calculated in the nested structure above
        
        # Save JSON results
        json_file = output_dir / 'enhanced_temporal_correlation_results.json'
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"✅ Results saved to: {json_file}")

    def _calculate_timing_stats(self, timing_list: List[float]) -> Dict:
        """Calculate timing statistics for a list of timing values"""
        if not timing_list:
            return {
                'count': 0,
                'mean': 0,
                'median': 0,
                'std': 0,
                'min': 0,
                'max': 0,
                'percentiles': {}
            }
        
        timing_array = np.array(timing_list)
        
        return {
            'count': len(timing_list),
            'mean': float(np.mean(timing_array)),
            'median': float(np.median(timing_array)),
            'std': float(np.std(timing_array)),
            'min': float(np.min(timing_array)),
            'max': float(np.max(timing_array)),
            'percentiles': {
                'p25': float(np.percentile(timing_array, 25)),
                'p75': float(np.percentile(timing_array, 75)),
                'p90': float(np.percentile(timing_array, 90)),
                'p95': float(np.percentile(timing_array, 95)),
                'p99': float(np.percentile(timing_array, 99))
            }
        }

    def create_enhanced_visualizations(self, apt_type: str, run_id: str):
        """Create comprehensive visualizations for enhanced temporal analysis"""
        print("📊 Creating enhanced temporal visualizations...")
        
        output_dir = self.results_dir / apt_type / f"run-{run_id}"
        
        # Set style for better plots
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Temporal Scenario Distribution (both flow and event level)
        self._create_scenario_distribution_plot(output_dir)
        
        # 2. Process Lifecycle Breakdown (bar chart only with percentages)
        self._create_lifecycle_breakdown_plot(output_dir)
        
        # 3. Individual Timing Analysis per Process Type
        self._create_individual_timing_plots(output_dir)
        
        print("✅ Enhanced visualizations created")

    def _create_scenario_distribution_plot(self, output_dir: Path):
        """Create comprehensive temporal scenario distribution plot with individual cases"""
        try:
            # Define all possible cases (including zeros)
            all_cases = [
                ('Start-End Process: Case 1', 'start_end_cases', 'case_1'),
                ('Start-End Process: Case 2', 'start_end_cases', 'case_2'), 
                ('Start-End Process: Case 3', 'start_end_cases', 'case_3'),
                ('Start-End Process: Case 4', 'start_end_cases', 'case_4'),
                ('Start-End Process: Case 5', 'start_end_cases', 'case_5'),
                ('Start-End Process: Case 6', 'start_end_cases', 'case_6'),
                ('No-End Process: Case 1', 'no_end_cases', 'case_1'),
                ('No-End Process: Case 2', 'no_end_cases', 'case_2'),
                ('No-End Process: Case 3', 'no_end_cases', 'case_3'),
                ('No-Start Process: Case 1', 'no_start_cases', 'case_1'),
                ('No-Start Process: Case 2', 'no_start_cases', 'case_2'),
                ('No-Start Process: Case 3', 'no_start_cases', 'case_3'),
                ('No-Start-No-End Process', 'no_bounds_cases', 'case_1')
            ]
            
            # Get total Successfully Attributed flows for percentage calculation
            total_attributed = self.temporal_stats.attribution_categories['successfully_attributed']
            
            # Collect counts for all cases
            scenario_labels = []
            scenario_counts = []
            scenario_percentages = []
            
            for label, category, case in all_cases:
                count = getattr(self.temporal_stats, category)[case]
                percentage = (count / total_attributed * 100) if total_attributed > 0 else 0
                
                scenario_labels.append(label)
                scenario_counts.append(count) 
                scenario_percentages.append(percentage)
            
            # Create plot for flows
            fig, ax = plt.subplots(figsize=(16, 10))
            
            # Color by process type
            colors = []
            for label in scenario_labels:
                if 'Start-End' in label:
                    colors.append('#1f77b4')  # Blue
                elif 'No-End' in label:
                    colors.append('#ff7f0e')  # Orange  
                elif 'No-Start Process:' in label:  # Avoid matching "No-Start-No-End"
                    colors.append('#2ca02c')  # Green
                elif 'No-Start-No-End' in label:
                    colors.append('#d62728')  # Red
            
            # Create horizontal bar plot
            bars = ax.barh(range(len(scenario_labels)), scenario_counts, color=colors)
            
            # Customize labels to wrap long text
            wrapped_labels = []
            for label in scenario_labels:
                if len(label) > 30:
                    words = label.split(' ')
                    mid = len(words) // 2
                    wrapped = ' '.join(words[:mid]) + '\n' + ' '.join(words[mid:])
                    wrapped_labels.append(wrapped)
                else:
                    wrapped_labels.append(label)
            
            ax.set_yticks(range(len(scenario_labels)))
            ax.set_yticklabels(wrapped_labels, fontsize=9)
            
            # Add value and percentage labels on bars
            for i, (bar, count, pct) in enumerate(zip(bars, scenario_counts, scenario_percentages)):
                if count > 0:
                    ax.text(count + max(scenario_counts) * 0.01, i, 
                           f'{count:,} ({pct:.1f}%)', va='center', fontsize=9)
                else:
                    ax.text(max(scenario_counts) * 0.01, i, 
                           '0 (0.0%)', va='center', fontsize=9)
            
            ax.set_xlabel('Number of Flows', fontsize=12)
            ax.set_title('Flow-Level Temporal Scenario Distribution\n(Percentages relative to Successfully Attributed flows)', 
                        fontsize=14, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            # Add total count legend/info box
            total_flows = sum(scenario_counts)
            info_text = f'Total Attributed Flows: {total_attributed:,}\nTotal Scenario Flows: {total_flows:,}'
            ax.text(0.7, 0.98, info_text, transform=ax.transAxes, fontsize=10, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            # Invert y-axis to have Start-End at top
            ax.invert_yaxis()
            
            plt.tight_layout()
            
            # Save plot (PNG + NPZ)
            plot_file = output_dir / 'temporal_scenario_distribution_flows.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            
            # Save NPZ format
            npz_file = output_dir / 'temporal_scenario_distribution_flows.npz'
            np.savez(npz_file, labels=scenario_labels, counts=scenario_counts, 
                    percentages=scenario_percentages, colors=colors, total_attributed=total_attributed)
            
            plt.close()
            
            print(f"   📊 Flow scenario distribution plot saved: {plot_file.name} + NPZ")
            
            # Create event-level variant
            self._create_event_scenario_distribution_plot(output_dir, all_cases)
            
        except Exception as e:
            print(f"⚠️  Error creating scenario distribution plot: {e}")

    def _create_event_scenario_distribution_plot(self, output_dir: Path, all_cases: List):
        """Create event-level temporal scenario distribution plot"""
        try:
            # Count events by scenario
            event_scenario_counts = {case: 0 for _, _, case in all_cases}
            total_attributed_events = 0
            
            # Count events for each scenario
            for result in self.temporal_stats.event_level_results:
                if result['attribution_status'] == 'Successfully Attributed':
                    total_attributed_events += 1
                    temporal_analysis = result.get('temporal_analysis', {})
                    if temporal_analysis:
                        scenario = temporal_analysis.get('scenario', '')
                        # Map scenario to case
                        if scenario.startswith('start_end_case_'):
                            case_num = scenario.split('_')[-1]
                            event_scenario_counts[case_num] += 1
                        elif scenario.startswith('no_end_case_'):
                            case_num = scenario.split('_')[-1]  
                            event_scenario_counts[case_num] += 1
                        elif scenario.startswith('no_start_case_'):
                            case_num = scenario.split('_')[-1]
                            event_scenario_counts[case_num] += 1
                        elif scenario == 'no_bounds_case_1':
                            event_scenario_counts['case_1'] += 1
            
            # Prepare data similar to flow plot
            scenario_labels = []
            scenario_counts = []
            scenario_percentages = []
            
            case_index = 0
            for label, category, case in all_cases:
                if category == 'start_end_cases':
                    count = event_scenario_counts.get(case, 0)
                elif category == 'no_end_cases':
                    count = event_scenario_counts.get(case, 0)
                elif category == 'no_start_cases':
                    count = event_scenario_counts.get(case, 0)
                elif category == 'no_bounds_cases':
                    count = event_scenario_counts.get(case, 0)
                else:
                    count = 0
                
                percentage = (count / total_attributed_events * 100) if total_attributed_events > 0 else 0
                
                scenario_labels.append(label)
                scenario_counts.append(count)
                scenario_percentages.append(percentage)
            
            # Create plot for events
            fig, ax = plt.subplots(figsize=(16, 10))
            
            # Color by process type (same as flows)
            colors = []
            for label in scenario_labels:
                if 'Start-End' in label:
                    colors.append('#1f77b4')  # Blue
                elif 'No-End' in label:
                    colors.append('#ff7f0e')  # Orange  
                elif 'No-Start Process:' in label:
                    colors.append('#2ca02c')  # Green
                elif 'No-Start-No-End' in label:
                    colors.append('#d62728')  # Red
            
            # Create horizontal bar plot
            bars = ax.barh(range(len(scenario_labels)), scenario_counts, color=colors)
            
            # Customize labels to wrap long text
            wrapped_labels = []
            for label in scenario_labels:
                if len(label) > 30:
                    words = label.split(' ')
                    mid = len(words) // 2
                    wrapped = ' '.join(words[:mid]) + '\n' + ' '.join(words[mid:])
                    wrapped_labels.append(wrapped)
                else:
                    wrapped_labels.append(label)
            
            ax.set_yticks(range(len(scenario_labels)))
            ax.set_yticklabels(wrapped_labels, fontsize=9)
            
            # Add value and percentage labels on bars
            for i, (bar, count, pct) in enumerate(zip(bars, scenario_counts, scenario_percentages)):
                if count > 0:
                    ax.text(count + max(scenario_counts) * 0.01, i, 
                           f'{count:,} ({pct:.1f}%)', va='center', fontsize=9)
                else:
                    ax.text(max(scenario_counts) * 0.01 if max(scenario_counts) > 0 else 1, i, 
                           '0 (0.0%)', va='center', fontsize=9)
            
            ax.set_xlabel('Number of Events', fontsize=12)
            ax.set_title('Event-Level Temporal Scenario Distribution\n(Percentages relative to Successfully Attributed events)', 
                        fontsize=14, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            # Add total count legend/info box  
            total_events = sum(scenario_counts)
            info_text = f'Total Attributed Events: {total_attributed_events:,}\nTotal Scenario Events: {total_events:,}'
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
            
            # Invert y-axis to have Start-End at top
            ax.invert_yaxis()
            
            plt.tight_layout()
            
            # Save plot (PNG + NPZ)
            plot_file = output_dir / 'temporal_scenario_distribution_events.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            
            # Save NPZ format
            npz_file = output_dir / 'temporal_scenario_distribution_events.npz'
            np.savez(npz_file, labels=scenario_labels, counts=scenario_counts, 
                    percentages=scenario_percentages, colors=colors, total_attributed=total_attributed_events)
            
            plt.close()
            
            print(f"   📊 Event scenario distribution plot saved: {plot_file.name} + NPZ")
            
        except Exception as e:
            print(f"⚠️  Error creating event scenario distribution plot: {e}")

    def _create_timing_histograms(self, output_dir: Path):
        """Create timing gaps histograms"""
        try:
            # Prepare timing data
            timing_datasets = []
            
            # Start-End timing data
            if self.temporal_stats.timing_stats['start_end_pre_gaps']:
                timing_datasets.append(('Start-End Pre-Gaps (ms)', self.temporal_stats.timing_stats['start_end_pre_gaps']))
            if self.temporal_stats.timing_stats['start_end_post_gaps']:
                timing_datasets.append(('Start-End Post-Gaps (ms)', self.temporal_stats.timing_stats['start_end_post_gaps']))
            if self.temporal_stats.timing_stats['start_end_separation_gaps']:
                timing_datasets.append(('Start-End Separation Gaps (ms)', self.temporal_stats.timing_stats['start_end_separation_gaps']))
            
            # No-End timing data
            if self.temporal_stats.timing_stats['no_end_start_gaps']:
                timing_datasets.append(('No-End Start Gaps (ms)', self.temporal_stats.timing_stats['no_end_start_gaps']))
            if self.temporal_stats.timing_stats['no_end_trigger_gaps']:
                timing_datasets.append(('No-End Trigger Gaps (ms)', self.temporal_stats.timing_stats['no_end_trigger_gaps']))
            
            # No-Start timing data
            if self.temporal_stats.timing_stats['no_start_end_gaps']:
                timing_datasets.append(('No-Start End Gaps (ms)', self.temporal_stats.timing_stats['no_start_end_gaps']))
            if self.temporal_stats.timing_stats['no_start_post_gaps']:
                timing_datasets.append(('No-Start Post Gaps (ms)', self.temporal_stats.timing_stats['no_start_post_gaps']))
            
            if not timing_datasets:
                print("⚠️  No timing data to plot")
                return
            
            # Create subplots
            n_plots = len(timing_datasets)
            cols = min(3, n_plots)
            rows = (n_plots + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
            if n_plots == 1:
                axes = [axes]
            elif rows == 1:
                axes = [axes] if n_plots == 1 else list(axes)
            else:
                axes = axes.flatten()
            
            # Create histograms
            for i, (title, data) in enumerate(timing_datasets):
                if i < len(axes):
                    ax = axes[i]
                    
                    # Convert to numpy array and remove extreme outliers
                    data_array = np.array(data)
                    q95 = np.percentile(data_array, 95)
                    q05 = np.percentile(data_array, 5)
                    filtered_data = data_array[(data_array >= q05) & (data_array <= q95)]
                    
                    ax.hist(filtered_data, bins=30, alpha=0.7, edgecolor='black')
                    ax.set_title(title, fontsize=10)
                    ax.set_xlabel('Time (ms)')
                    ax.set_ylabel('Frequency')
                    ax.grid(alpha=0.3)
                    
                    # Add statistics text
                    stats_text = f'Count: {len(data)}\nMean: {np.mean(data_array):.1f}ms\nMedian: {np.median(data_array):.1f}ms'
                    ax.text(0.8, 0.8, stats_text, transform=ax.transAxes, 
                           verticalalignment='top', fontsize=8, 
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Hide unused subplots
            for i in range(n_plots, len(axes)):
                axes[i].set_visible(False)
            
            plt.suptitle('Timing Gaps Distribution by Process Lifecycle Type', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # Save plot
            plot_file = output_dir / 'timing_gaps_histograms.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"   📊 Timing histograms saved: {plot_file.name}")
            
        except Exception as e:
            print(f"⚠️  Error creating timing histograms: {e}")

    def _create_overlap_ratios_plot(self, output_dir: Path):
        """Create overlap ratios distribution plot"""
        try:
            overlap_data = self.temporal_stats.timing_stats['start_end_overlap_ratios']
            
            if not overlap_data:
                print("⚠️  No overlap ratio data to plot")
                return
            
            # Create plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Histogram
            ax1.hist(overlap_data, bins=30, alpha=0.7, edgecolor='black', color='skyblue')
            ax1.set_xlabel('Overlap Ratio (NetFlow Duration / Sysmon Duration)')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Distribution of Temporal Overlap Ratios')
            ax1.grid(alpha=0.3)
            
            # Add statistics
            stats_text = f'Count: {len(overlap_data)}\nMean: {np.mean(overlap_data):.3f}\nMedian: {np.median(overlap_data):.3f}\nStd: {np.std(overlap_data):.3f}'
            ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
                    verticalalignment='top', fontsize=10, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Box plot
            ax2.boxplot(overlap_data, vert=True)
            ax2.set_ylabel('Overlap Ratio')
            ax2.set_title('Overlap Ratios Box Plot')
            ax2.grid(alpha=0.3)
            
            plt.suptitle('Temporal Overlap Analysis for Start-End Processes', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # Save plot
            plot_file = output_dir / 'overlap_ratios_distribution.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"   📊 Overlap ratios plot saved: {plot_file.name}")
            
        except Exception as e:
            print(f"⚠️  Error creating overlap ratios plot: {e}")

    def _create_lifecycle_breakdown_plot(self, output_dir: Path):
        """Create process lifecycle breakdown plot (bar chart only with percentages)"""
        try:
            # Prepare lifecycle data
            lifecycle_data = {
                'Start-End Process': self.temporal_stats.scenario_counts['start_end_process_count'],
                'No-End Process': self.temporal_stats.scenario_counts['no_end_process_count'],
                'No-Start Process': self.temporal_stats.scenario_counts['no_start_process_count'],
                'No-Start-No-End Process': self.temporal_stats.scenario_counts['no_start_no_end_process_count']
            }
            
            # Include zero counts for completeness
            labels = list(lifecycle_data.keys())
            sizes = list(lifecycle_data.values())
            
            if sum(sizes) == 0:
                print("⚠️  No lifecycle data to plot")
                return
            
            # Calculate percentages
            total_processes = sum(sizes)
            percentages = [(size / total_processes) * 100 for size in sizes]
            
            # Create single bar chart
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(labels)]
            
            bars = ax.bar(labels, sizes, color=colors)
            ax.set_ylabel('Number of Processes')
            ax.set_title('Process Lifecycle Analysis Breakdown', fontsize=14, fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', alpha=0.3)
            
            # Add total processes legend/info box
            ax.text(0.02, 0.98, f'Total Processes: {total_processes:,}', transform=ax.transAxes, 
                   fontsize=12, verticalalignment='top', fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
            
            # Add value and percentage labels on bars
            for bar, size, pct in zip(bars, sizes, percentages):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(sizes) * 0.01,
                       f'{size:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout()
            
            # Save plot (PNG + NPZ)
            plot_file = output_dir / 'process_lifecycle_breakdown.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            
            # Save NPZ format
            npz_file = output_dir / 'process_lifecycle_breakdown.npz'
            np.savez(npz_file, labels=labels, counts=sizes, percentages=percentages, colors=colors)
            
            plt.close()
            
            print(f"   📊 Lifecycle breakdown plot saved: {plot_file.name} + NPZ")
            
        except Exception as e:
            print(f"⚠️  Error creating lifecycle breakdown plot: {e}")

    def _create_individual_timing_plots(self, output_dir: Path):
        """Create separate timing plots per process type with subplot grids"""
        try:
            print("   📊 Creating individual case-specific timing plots...")
            
            # Define process types and their cases
            process_types = {
                'start_end': {
                    'name': 'Start-End Process',
                    'cases': ['case_1', 'case_2', 'case_3', 'case_4', 'case_5', 'case_6'],
                    'timing_types': ['pre_gaps', 'post_gaps', 'overlap_ratios'],
                    'grid_size': (6, 3)
                },
                'no_end': {
                    'name': 'No-End Process',
                    'cases': ['case_1', 'case_2', 'case_3'],
                    'timing_types': ['pre_gaps', 'netflow_durations'],  # Pre_gaps + NetFlow durations
                    'grid_size': (3, 2)
                },
                'no_start': {
                    'name': 'No-Start Process',
                    'cases': ['case_1', 'case_2', 'case_3'],
                    'timing_types': ['post_gaps', 'netflow_durations'],  # Post_gaps + NetFlow durations
                    'grid_size': (3, 2)
                },
                'no_bounds': {
                    'name': 'No-Start-No-End Process',
                    'cases': ['case_1'],
                    'timing_types': ['netflow_durations'],  # Only NetFlow durations
                    'grid_size': (1, 1)
                }
            }
            
            for process_key, process_info in process_types.items():
                if process_key == 'no_bounds':
                    # Special case: No timing data for unbounded processes
                    self._create_no_bounds_timing_plot(output_dir, process_info)
                else:
                    self._create_process_timing_subplot_grid(output_dir, process_key, process_info)
            
            print("   ✅ Individual timing plots completed")
            
        except Exception as e:
            print(f"⚠️  Error creating individual timing plots: {e}")
    
    def _create_process_timing_subplot_grid(self, output_dir: Path, process_key: str, process_info: dict):
        """Create subplot grid for a specific process type"""
        try:
            cases = process_info['cases']
            timing_types = process_info['timing_types']
            grid_rows, grid_cols = process_info['grid_size']
            process_name = process_info['name']
            
            # Calculate total flows for this process type
            total_flows_for_process = 0
            for case in cases:
                case_key = f'{process_key}_{case}'
                for timing_type in timing_types:
                    timing_data = self.temporal_stats.timing_stats[case_key][timing_type]
                    if timing_data:
                        total_flows_for_process += len(timing_data)
                        break  # Count each case only once
            
            # Create figure with subplots
            fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(5*grid_cols, 4*grid_rows))
            fig.suptitle(f'{process_name} - Case-Specific Timing Analysis\n(Total Flows: {total_flows_for_process:,})', 
                        fontsize=16, fontweight='bold')
            
            # Handle single subplot case
            if grid_rows == 1 and grid_cols == 1:
                axes = np.array([[axes]])
            elif grid_rows == 1 or grid_cols == 1:
                axes = axes.reshape(grid_rows, grid_cols)
            
            # Process each case
            for case_idx, case in enumerate(cases):
                case_key = f'{process_key}_{case}'
                
                for timing_idx, timing_type in enumerate(timing_types):
                    if timing_idx < grid_cols:  # Ensure we don't exceed grid
                        ax = axes[case_idx, timing_idx]
                        
                        # Get timing data for this specific case and timing type
                        timing_data = self.temporal_stats.timing_stats[case_key][timing_type]
                        
                        if timing_data:
                            # Create histogram
                            ax.hist(timing_data, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
                            ax.set_title(f'{case.replace("_", " ").title()}\n{timing_type.replace("_", " ").title()}')
                            ax.set_xlabel('Time (ms)')
                            ax.set_ylabel('Frequency')
                            ax.grid(alpha=0.3)
                            
                            # Add statistics and count information  
                            if timing_data:
                                mean_val = np.mean(timing_data)
                                median_val = np.median(timing_data)
                                count_data = len(timing_data)
                                ax.axvline(mean_val, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.1f}ms')
                                ax.axvline(median_val, color='orange', linestyle='--', alpha=0.7, label=f'Median: {median_val:.1f}ms')
                                ax.legend(fontsize=8)
                                
                                # Add count info in subplot
                                ax.text(0.85, 0.80, f'N={count_data:,}', transform=ax.transAxes, fontsize=8,
                                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                        else:
                            # No data for this case/timing combination
                            ax.text(0.5, 0.5, f'No data\n{case.replace("_", " ").title()}\n{timing_type.replace("_", " ").title()}', 
                                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
                            ax.set_xticks([])
                            ax.set_yticks([])
                
                # Hide unused subplots in the row
                for timing_idx in range(len(timing_types), grid_cols):
                    if timing_idx < grid_cols:
                        axes[case_idx, timing_idx].set_visible(False)
            
            plt.tight_layout()
            
            # Save plot (PNG + NPZ)
            plot_file = output_dir / f'{process_key}_timing_analysis.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            
            # Save NPZ format with timing data
            npz_data = {}
            for case in cases:
                case_key = f'{process_key}_{case}'
                npz_data[case] = {}
                for timing_type in timing_types:
                    timing_data = self.temporal_stats.timing_stats[case_key][timing_type]
                    npz_data[case][timing_type] = np.array(timing_data) if timing_data else np.array([])
            
            npz_file = output_dir / f'{process_key}_timing_analysis.npz'
            np.savez_compressed(npz_file, **npz_data)
            
            plt.close()
            
            print(f"   📊 {process_name} timing plot saved: {plot_file.name} + NPZ")
            
        except Exception as e:
            print(f"⚠️  Error creating {process_key} timing plot: {e}")
    
    def _create_no_bounds_timing_plot(self, output_dir: Path, process_info: dict):
        """Create timing plot for No-Start-No-End processes with NetFlow durations"""
        try:
            process_name = process_info['name']
            
            # Get NetFlow duration data for no_bounds_case_1
            netflow_durations = self.temporal_stats.timing_stats['no_bounds_case_1']['netflow_durations']
            
            # Create figure
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            if netflow_durations and len(netflow_durations) > 0:
                # Create NetFlow duration histogram
                ax.hist(netflow_durations, bins=30, alpha=0.7, color='lightcoral', 
                       edgecolor='black', linewidth=1)
                
                ax.set_xlabel('NetFlow Duration (ms)', fontsize=12, fontweight='bold')
                ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
                ax.set_title(f'{process_name} - NetFlow Duration Distribution', fontsize=14, fontweight='bold')
                ax.grid(alpha=0.3)
                
                # Add statistics text
                mean_duration = np.mean(netflow_durations)
                median_duration = np.median(netflow_durations)
                std_duration = np.std(netflow_durations)
                count_data = len(netflow_durations)
                
                stats_text = f'Count: {count_data:,}\nMean: {mean_duration:.1f}ms\nMedian: {median_duration:.1f}ms\nStd: {std_duration:.1f}ms'
                ax.text(0.8, 0.8, stats_text, transform=ax.transAxes, fontsize=10,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                # Add note about no timing constraints
                ax.text(0.98, 0.98, 'No Temporal Constraints\n(Always attributed)', 
                       transform=ax.transAxes, fontsize=10,
                       verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            else:
                # No data available - show informational message
                ax.text(0.5, 0.5, f'{process_name}\n\nNo NetFlow Duration Data Available\n\nProcesses are always attributed\nregardless of timing', 
                       ha='center', va='center', transform=ax.transAxes, 
                       fontsize=14, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
                
                ax.set_xticks([])
                ax.set_yticks([])
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)
            
            plt.tight_layout()
            
            # Save plot (PNG + NPZ)
            plot_file = output_dir / 'no_bounds_timing_analysis.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            
            # Save NPZ with actual data or note
            npz_file = output_dir / 'no_bounds_timing_analysis.npz'
            if netflow_durations:
                np.savez_compressed(npz_file, netflow_durations=netflow_durations, 
                                  process_type='no_bounds', case='case_1')
            else:
                np.savez_compressed(npz_file, note="No NetFlow duration data available")
            
            plt.close()
            
            print(f"   📊 {process_name} plot saved: {plot_file.name} + NPZ")
            
        except Exception as e:
            print(f"⚠️  Error creating no-bounds timing plot: {e}")

def main():
    """Main execution function with command line argument parsing"""
    parser = argparse.ArgumentParser(description='Enhanced Temporal Causation NetFlow-Sysmon Correlator v3.0')
    
    # Basic arguments
    parser.add_argument('--apt-type', type=str, help='APT type (apt-1, apt-2, etc.)')
    parser.add_argument('--run-id', type=str, help='Run ID (01, 02, etc.)')
    parser.add_argument('--run-range', type=str, help='Run range (e.g., 09-15)')
    
    # Batch processing
    parser.add_argument('--batch-high-performing', action='store_true',
                       help='Process all high-performing runs (≥90% attribution)')
    parser.add_argument('--batch-all', action='store_true',
                       help='Process all available APT runs')
    
    # Performance options
    parser.add_argument('--workers', type=int, default=16,
                       help='Number of worker threads (default: 16)')
    parser.add_argument('--sample-size', type=int,
                       help='Sample size for testing (limits dataset size)')
    
    args = parser.parse_args()
    
    # Initialize correlator
    correlator = EnhancedTemporalCorrelator(
        max_workers=args.workers,
        sample_size=args.sample_size
    )
    
    # High-performing runs (≥90% event attribution from previous analysis)
    high_performing_runs = {
        'apt-1': ['09', '10', '11', '12', '15', '51'],
        'apt-2': ['22', '29', '30'],
        'apt-3': ['37', '38'],
        'apt-4': ['39', '41', '42', '43', '44'],
        'apt-5': ['46', '47'],
        'apt-6': ['48', '49', '50']
    }
    
    results_summary = []
    
    try:
        if args.batch_high_performing:
            # Process all high-performing runs
            print("🚀 Starting batch processing of high-performing APT runs...")
            
            for apt_type, run_ids in high_performing_runs.items():
                for run_id in run_ids:
                    try:
                        print(f"\n" + "="*60)
                        result = correlator.perform_enhanced_correlation(apt_type, run_id)
                        results_summary.append(result)
                        print(f"✅ Completed {apt_type}-run-{run_id}: {result['flow_attribution_rate']:.2f}% flow attribution")
                    except Exception as e:
                        print(f"❌ Failed {apt_type}-run-{run_id}: {e}")
                        results_summary.append({
                            'apt_type': apt_type,
                            'run_id': run_id,
                            'error': str(e)
                        })
        
        elif args.batch_all:
            # Process all available runs
            print("🚀 Starting batch processing of all APT runs...")
            
            all_runs = {
                'apt-1': [f"{i:02d}" for i in range(1, 21)] + ['51'],
                'apt-2': [f"{i:02d}" for i in range(21, 31)],
                'apt-3': [f"{i:02d}" for i in range(31, 39)],
                'apt-4': [f"{i:02d}" for i in range(39, 45)],
                'apt-5': [f"{i:02d}" for i in range(45, 48)],
                'apt-6': [f"{i:02d}" for i in range(48, 51)]
            }
            
            for apt_type, run_ids in all_runs.items():
                for run_id in run_ids:
                    try:
                        print(f"\n" + "="*60)
                        result = correlator.perform_enhanced_correlation(apt_type, run_id)
                        results_summary.append(result)
                        print(f"✅ Completed {apt_type}-run-{run_id}: {result['flow_attribution_rate']:.2f}% flow attribution")
                    except Exception as e:
                        print(f"❌ Failed {apt_type}-run-{run_id}: {e}")
                        results_summary.append({
                            'apt_type': apt_type,
                            'run_id': run_id,
                            'error': str(e)
                        })
        
        elif args.run_range and args.apt_type:
            # Process run range
            start_run, end_run = map(int, args.run_range.split('-'))
            run_ids = [f"{i:02d}" for i in range(start_run, end_run + 1)]
            
            print(f"🚀 Starting range processing: {args.apt_type} runs {start_run}-{end_run}")
            
            for run_id in run_ids:
                try:
                    print(f"\n" + "="*60)
                    result = correlator.perform_enhanced_correlation(args.apt_type, run_id)
                    results_summary.append(result)
                    print(f"✅ Completed {args.apt_type}-run-{run_id}: {result['flow_attribution_rate']:.2f}% flow attribution")
                except Exception as e:
                    print(f"❌ Failed {args.apt_type}-run-{run_id}: {e}")
                    results_summary.append({
                        'apt_type': args.apt_type,
                        'run_id': run_id,
                        'error': str(e)
                    })
        
        elif args.apt_type and args.run_id:
            # Process single run
            print(f"🚀 Starting single run analysis: {args.apt_type}-run-{args.run_id}")
            result = correlator.perform_enhanced_correlation(args.apt_type, args.run_id)
            results_summary.append(result)
            print(f"✅ Completed {args.apt_type}-run-{args.run_id}: {result['flow_attribution_rate']:.2f}% flow attribution")
        
        else:
            print("❌ Error: Please specify either:")
            print("   --apt-type and --run-id for single run")
            print("   --apt-type and --run-range for range processing")
            print("   --batch-high-performing for high-performing runs")
            print("   --batch-all for all runs")
            return
        
        # Print summary
        if len(results_summary) > 1:
            print(f"\n" + "="*60)
            print(f"📊 BATCH PROCESSING SUMMARY")
            print(f"="*60)
            
            successful_results = [r for r in results_summary if 'flow_attribution_rate' in r]
            failed_results = [r for r in results_summary if 'error' in r]
            
            print(f"✅ Successful runs: {len(successful_results)}")
            print(f"❌ Failed runs: {len(failed_results)}")
            
            if successful_results:
                avg_attribution = np.mean([r['flow_attribution_rate'] for r in successful_results])
                print(f"📈 Average attribution rate: {avg_attribution:.2f}%")
                
                best_run = max(successful_results, key=lambda x: x['flow_attribution_rate'])
                print(f"🏆 Best performing run: {best_run['apt_type']}-{best_run['run_id']} ({best_run['flow_attribution_rate']:.2f}%)")
            
            if failed_results:
                print(f"\n❌ Failed runs:")
                for result in failed_results:
                    print(f"   {result['apt_type']}-{result['run_id']}: {result['error']}")
        
        print(f"\n🎯 Enhanced temporal correlation analysis completed!")
        print(f"📁 Results saved in: analysis/correlation-analysis-v3/")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        raise

if __name__ == "__main__":
    main()