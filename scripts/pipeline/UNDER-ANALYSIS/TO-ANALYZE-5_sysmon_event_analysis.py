#!/usr/bin/env python3
"""
Clean Event Tracer v2 - Production Implementation
Unified detection system for cybersecurity event analysis with batch processing support.

Supports:
- Type 1: Terminal command detection (EventID 1 - CommandLine matching)
- Type 2A: File creation detection (EventID 11 - TargetFilename matching) 
- Type 2B: File deletion detection (EventID 23 - TargetFilename matching)
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import json
import gc
import re
from datetime import datetime, timedelta
import yaml
import time
import logging
from contextlib import contextmanager
import psutil
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import partial
import warnings
import argparse

# Global configuration will be loaded from config.yaml or command line arguments
config = None
DEBUG_MODE = False  # Will be overridden by config

# Global variables for command line arguments
CLI_ARGS = None

# ==================== CUSTOM EXCEPTIONS ====================

class EventTracerException(Exception):
    """Base exception for event tracer errors"""
    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self):
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} (Details: {details_str})"
        return self.message

class ConfigurationError(EventTracerException):
    """Configuration file or settings issues"""
    pass

class DataValidationError(EventTracerException):
    """Data quality or integrity issues"""
    pass

class FileProcessingError(EventTracerException):
    """File loading or processing issues"""
    pass

class AnalysisError(EventTracerException):
    """Analysis or computation issues"""
    pass

# ==================== UTILITY FUNCTIONS ====================

def safe_operation_with_retry(func, *args, max_retries: int = 3, 
                            base_delay: float = 0.1, **kwargs):
    """
    Execute operation with retry logic and exponential backoff
    
    Args:
        func: Function to execute
        *args: Function arguments
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds for exponential backoff
        **kwargs: Function keyword arguments
    
    Returns:
        Function result
        
    Raises:
        EventTracerException: If all retries fail
    """
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            if attempt == max_retries - 1:
                # Final attempt failed
                raise EventTracerException(
                    f"Operation failed after {max_retries} attempts",
                    details={
                        "function": func.__name__,
                        "final_error": str(e),
                        "attempts": max_retries
                    }
                ) from e
            
            # Calculate delay with exponential backoff
            delay = base_delay * (2 ** attempt)
            if DEBUG_MODE:
                print(f"[RETRY] Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s...")
            time.sleep(delay)
    
    # This should never be reached, but just in case
    raise EventTracerException("Unexpected retry logic error") from last_exception

@contextmanager
def error_context(operation: str, **context_data):
    """
    Context manager for enhanced error reporting
    
    Args:
        operation: Description of the operation being performed
        **context_data: Additional context information
    """
    try:
        if DEBUG_MODE:
            print(f"[OPERATION] Starting: {operation}")
        yield
        if DEBUG_MODE:
            print(f"[OPERATION] Completed: {operation}")
    except Exception as e:
        error_msg = f"Failed during {operation}"
        raise EventTracerException(error_msg, details=context_data) from e

def validate_file_exists(file_path: str, file_type: str = "file") -> None:
    """
    Validate that a file exists and is accessible
    
    Args:
        file_path: Path to the file
        file_type: Type description for error messages
        
    Raises:
        FileProcessingError: If file doesn't exist or isn't accessible
    """
    if not file_path:
        raise FileProcessingError(f"Empty {file_type} path provided")
    
    if not os.path.exists(file_path):
        raise FileProcessingError(
            f"{file_type.title()} not found: {file_path}",
            details={"file_path": file_path, "file_type": file_type}
        )
    
    if not os.path.isfile(file_path):
        raise FileProcessingError(
            f"Path exists but is not a file: {file_path}",
            details={"file_path": file_path, "file_type": file_type}
        )
    
    # Check if file is readable
    try:
        with open(file_path, 'r') as f:
            f.read(1)  # Try to read first character
    except PermissionError:
        raise FileProcessingError(
            f"No permission to read {file_type}: {file_path}",
            details={"file_path": file_path, "file_type": file_type}
        )
    except Exception as e:
        raise FileProcessingError(
            f"Cannot access {file_type}: {file_path}",
            details={"file_path": file_path, "file_type": file_type, "error": str(e)}
        )

# ==================== PERFORMANCE & MEMORY FUNCTIONS ====================

def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage statistics"""
    memory = psutil.virtual_memory()
    return {
        "total_gb": memory.total / (1024**3),
        "available_gb": memory.available / (1024**3),
        "used_gb": memory.used / (1024**3),
        "percent_used": memory.percent,
        "free_gb": memory.free / (1024**3)
    }

def monitor_memory_usage(operation: str = "", threshold_percent: float = 80.0) -> bool:
    """
    Monitor memory usage and warn if approaching limits
    
    Args:
        operation: Description of current operation
        threshold_percent: Warning threshold for memory usage
        
    Returns:
        bool: True if memory usage is below threshold, False otherwise
    """
    memory_stats = get_memory_usage()
    memory_percent = memory_stats["percent_used"]
    
    if memory_percent > threshold_percent:
        warning_msg = f"⚠️ High memory usage: {memory_percent:.1f}% ({memory_stats['used_gb']:.1f}GB used)"
        if operation:
            warning_msg += f" during {operation}"
        print(warning_msg)
        
        # Force garbage collection
        collected = gc.collect()
        if DEBUG_MODE:
            print(f"   Garbage collected {collected} objects")
        
        # Check if we're approaching critical levels
        if memory_percent > 90.0:
            print(f"   🚨 Critical memory usage! Consider using chunked processing.")
            return False
    
    elif DEBUG_MODE and operation:
        print(f"[MEMORY] {operation}: {memory_percent:.1f}% used ({memory_stats['used_gb']:.1f}GB)")
    
    return True

def optimize_dataframe_memory(df: pd.DataFrame, operation: str = "") -> pd.DataFrame:
    """
    Optimize DataFrame memory usage through better data types
    
    Args:
        df: DataFrame to optimize
        operation: Description for logging
        
    Returns:
        Optimized DataFrame
    """
    if len(df) == 0:
        return df
    
    original_memory = df.memory_usage(deep=True).sum() / (1024**2)  # MB
    
    # Optimize categorical columns
    categorical_candidates = ['Computer', 'Protocol', 'EventType', 'User']
    for col in categorical_candidates:
        if col in df.columns:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio < 0.5:  # Less than 50% unique values
                df[col] = df[col].astype('category')
    
    # Optimize integer columns - use smallest possible integer type
    try:
        int_cols = df.select_dtypes(include=['int64']).columns
        for col in int_cols:
            if col in df.columns:
                try:
                    col_min = df[col].min()
                    col_max = df[col].max()
                    
                    # Skip if contains nulls or invalid values
                    if pd.isna(col_min) or pd.isna(col_max):
                        continue
                    
                    if col_min >= 0:  # Unsigned integers
                        if col_max < 255:
                            df[col] = df[col].astype('uint8')
                        elif col_max < 65535:
                            df[col] = df[col].astype('uint16')
                        elif col_max < 4294967295:
                            df[col] = df[col].astype('uint32')
                    else:  # Signed integers
                        if col_min >= -128 and col_max < 127:
                            df[col] = df[col].astype('int8')
                        elif col_min >= -32768 and col_max < 32767:
                            df[col] = df[col].astype('int16')
                        elif col_min >= -2147483648 and col_max < 2147483647:
                            df[col] = df[col].astype('int32')
                except Exception as e:
                    if DEBUG_MODE:
                        print(f"[DEBUG] Could not optimize integer column {col}: {e}")
                    continue
    except Exception as e:
        if DEBUG_MODE:
            print(f"[DEBUG] Error during integer optimization: {e}")
    
    # Optimize object columns - convert to string category where beneficial
    try:
        object_cols = df.select_dtypes(include=['object']).columns
        for col in object_cols:
            if col in df.columns and col not in ['CommandLine', 'TargetFilename']:  # Skip long text fields
                try:
                    unique_ratio = df[col].nunique() / len(df)
                    if unique_ratio < 0.3:  # Less than 30% unique values
                        df[col] = df[col].astype('category')
                except Exception as e:
                    if DEBUG_MODE:
                        print(f"[DEBUG] Could not optimize object column {col}: {e}")
                    continue
    except Exception as e:
        if DEBUG_MODE:
            print(f"[DEBUG] Error during object column optimization: {e}")
    
    optimized_memory = df.memory_usage(deep=True).sum() / (1024**2)  # MB
    memory_reduction = ((original_memory - optimized_memory) / original_memory) * 100
    
    if DEBUG_MODE and operation:
        print(f"[MEMORY] {operation}: Optimized from {original_memory:.1f}MB to {optimized_memory:.1f}MB "
              f"({memory_reduction:.1f}% reduction)")
    
    return df

def estimate_chunk_size(file_path: str, target_memory_mb: float = 500.0) -> int:
    """
    Estimate optimal chunk size for processing large files
    
    Args:
        file_path: Path to the file
        target_memory_mb: Target memory usage per chunk in MB
        
    Returns:
        Estimated optimal chunk size in rows
    """
    try:
        # Get file size
        file_size_mb = os.path.getsize(file_path) / (1024**2)
        
        # Estimate rows (assuming average row is ~200 bytes for CSV)
        estimated_rows = int((file_size_mb * 1024 * 1024) / 200)
        
        # Calculate chunk size to stay within target memory
        if file_size_mb <= target_memory_mb:
            return estimated_rows  # Process entire file if small enough
        
        chunk_ratio = target_memory_mb / file_size_mb
        chunk_size = max(1000, int(estimated_rows * chunk_ratio))  # Minimum 1000 rows
        
        if DEBUG_MODE:
            print(f"[MEMORY] File size: {file_size_mb:.1f}MB, estimated rows: {estimated_rows}, "
                  f"chunk size: {chunk_size}")
        
        return chunk_size
        
    except Exception as e:
        if DEBUG_MODE:
            print(f"[MEMORY] Could not estimate chunk size: {e}, using default 10000")
        return 10000  # Default fallback

def create_dataframe_indices(df: pd.DataFrame, index_columns: List[str] = None) -> pd.DataFrame:
    """
    Create optimized indices for faster DataFrame operations
    
    Args:
        df: DataFrame to index
        index_columns: Columns to create indices on
        
    Returns:
        DataFrame with optimized indices
    """
    if index_columns is None:
        # Default indices for Sysmon data
        index_columns = ['Computer', 'EventID']
    
    # Only create indices if columns exist and DataFrame is large enough
    if len(df) < 1000:
        return df  # Too small to benefit from indexing
    
    available_columns = [col for col in index_columns if col in df.columns]
    
    if available_columns:
        try:
            # Create multi-level index for faster filtering
            df_indexed = df.set_index(available_columns)
            
            if DEBUG_MODE:
                print(f"[PERFORMANCE] Created index on columns: {available_columns}")
            
            return df_indexed
        except Exception as e:
            if DEBUG_MODE:
                print(f"[PERFORMANCE] Could not create index: {e}")
            return df
    
    return df

@contextmanager
def memory_profiler(operation: str):
    """Context manager for memory profiling operations"""
    initial_memory = get_memory_usage()
    start_time = time.time()
    
    try:
        if DEBUG_MODE:
            print(f"[PROFILER] Starting {operation} - Memory: {initial_memory['percent_used']:.1f}%")
        yield
        
    finally:
        end_time = time.time()
        final_memory = get_memory_usage()
        duration = end_time - start_time
        memory_delta = final_memory['used_gb'] - initial_memory['used_gb']
        
        if DEBUG_MODE or abs(memory_delta) > 0.1:  # Show if significant memory change or debug mode
            print(f"[PROFILER] Completed {operation} in {duration:.2f}s - "
                  f"Memory: {final_memory['percent_used']:.1f}% "
                  f"({memory_delta:+.2f}GB change)")

def cleanup_memory():
    """Force memory cleanup and garbage collection"""
    collected = gc.collect()
    if DEBUG_MODE:
        memory_stats = get_memory_usage()
        print(f"[CLEANUP] Collected {collected} objects, "
              f"Memory: {memory_stats['percent_used']:.1f}% used")

def load_sysmon_data_chunked(file_path: str, chunk_size: int = None) -> pd.DataFrame:
    """
    Load large Sysmon CSV files using chunked processing for memory efficiency
    
    Args:
        file_path: Path to the Sysmon CSV file
        chunk_size: Number of rows per chunk (auto-calculated if None)
        
    Returns:
        Complete DataFrame loaded efficiently
    """
    if chunk_size is None:
        target_memory = get_config_value('performance.chunk_size', 500)  # MB from config
        chunk_size = estimate_chunk_size(file_path, target_memory)
    
    print(f"📊 Loading large file in chunks of {chunk_size:,} rows...")
    
    # Define data types for optimization (only for columns that exist)
    dtype_spec = {
        'ProcessId': 'Int64',
        'SourcePort': 'Int64', 
        'DestinationPort': 'Int64',
        'SourceProcessId': 'Int64',
        'ParentProcessId': 'Int64',
        'SourceThreadId': 'Int64',
        'TargetProcessId': 'Int64',
        'ProcessGuid': 'string',
        'SourceProcessGUID': 'string',
        'TargetProcessGUID': 'string', 
        'ParentProcessGuid': 'string',
        'Computer': 'category',
        'Protocol': 'category',
        'EventType': 'category'
    }
    
    chunks = []
    total_rows = 0
    
    with memory_profiler("chunked CSV loading"):
        try:
            # First, read a few rows to determine which columns actually exist
            sample_df = pd.read_csv(file_path, nrows=1)
            available_columns = set(sample_df.columns)
            
            # Filter dtype_spec to only include columns that exist
            filtered_dtype_spec = {
                col: dtype for col, dtype in dtype_spec.items() 
                if col in available_columns
            }
            
            if DEBUG_MODE and len(filtered_dtype_spec) < len(dtype_spec):
                missing_cols = set(dtype_spec.keys()) - available_columns
                print(f"[DEBUG] Skipping dtype for missing columns: {missing_cols}")
            
            chunk_reader = pd.read_csv(
                file_path,
                chunksize=chunk_size,
                low_memory=False,
                dtype=filtered_dtype_spec
            )
            
            for i, chunk in enumerate(chunk_reader):
                # Monitor memory usage
                if not monitor_memory_usage(f"processing chunk {i+1}", threshold_percent=85.0):
                    print(f"⚠️ Memory usage too high, reducing chunk size for remaining data")
                    chunk_size = max(1000, chunk_size // 2)
                    break
                
                # Process timestamp conversion for this chunk
                chunk['UtcTime'] = pd.to_datetime(chunk['UtcTime'], errors='coerce')
                
                # Apply canonical normalization to CommandLine
                chunk['cmd_norm'] = chunk['CommandLine'].map(canonical)
                
                # Optimize memory for this chunk
                chunk = optimize_dataframe_memory(chunk, f"chunk {i+1}")
                
                chunks.append(chunk)
                total_rows += len(chunk)
                
                if DEBUG_MODE:
                    print(f"   Processed chunk {i+1}: {len(chunk):,} rows (total: {total_rows:,})")
                
                # Cleanup between chunks
                if i % 5 == 0:  # Every 5 chunks
                    cleanup_memory()
            
            # Combine all chunks
            if chunks:
                print(f"🔗 Combining {len(chunks)} chunks into final DataFrame...")
                combined_df = pd.concat(chunks, ignore_index=True)
                
                # Final memory optimization
                combined_df = optimize_dataframe_memory(combined_df, "final combined dataset")
                
                # Cleanup chunk list
                chunks.clear()
                cleanup_memory()
                
                return combined_df
            else:
                raise DataValidationError("No data chunks were successfully processed")
                
        except MemoryError:
            raise DataValidationError(
                f"Insufficient memory to process file even with chunking. "
                f"Try reducing chunk size in config or processing on a machine with more RAM."
            )
        except Exception as e:
            raise FileProcessingError(f"Failed during chunked loading: {e}")

def should_use_chunked_loading(file_path: str, memory_threshold_mb: float = 1000.0) -> bool:
    """
    Determine if chunked loading should be used based on file size and available memory
    
    Args:
        file_path: Path to the file
        memory_threshold_mb: File size threshold for chunked loading
        
    Returns:
        bool: True if chunked loading is recommended
    """
    try:
        file_size_mb = os.path.getsize(file_path) / (1024**2)
        memory_stats = get_memory_usage()
        available_memory_mb = memory_stats["available_gb"] * 1024
        
        # Use chunked loading if:
        # 1. File is larger than threshold, OR
        # 2. File size is more than 25% of available memory
        should_chunk = (
            file_size_mb > memory_threshold_mb or 
            file_size_mb > (available_memory_mb * 0.25)
        )
        
        if DEBUG_MODE:
            print(f"[MEMORY] File: {file_size_mb:.1f}MB, Available: {available_memory_mb:.1f}MB, "
                  f"Use chunking: {should_chunk}")
        
        return should_chunk
        
    except Exception as e:
        if DEBUG_MODE:
            print(f"[MEMORY] Could not determine file size: {e}, defaulting to chunked loading")
        return True  # Default to chunked loading if uncertain

def process_entry_parallel(tracer_state: Dict, entry_config: 'EntryConfig', 
                          include_mask4: bool, show_arrows: bool) -> 'DetectionResult':
    """
    Process a single entry in parallel (worker function)
    
    Args:
        tracer_state: Serialized tracer state (sysmon_df, caldera_entries, etc.)
        entry_config: Entry configuration to process
        include_mask4: Whether to include mask 4
        show_arrows: Whether to show arrows in plots
        
    Returns:
        DetectionResult for this entry
    """
    try:
        # Recreate tracer from state (for multiprocessing)
        from event_tracer import EventTracer
        
        tracer = EventTracer(
            sysmon_df=tracer_state['sysmon_df'],
            caldera_entries=tracer_state['caldera_entries'],
            debug=tracer_state.get('debug', False)
        )
        
        # Process this entry
        result = tracer.detect_zero_level_event(entry_config)
        
        if result.success:
            # Apply appropriate analysis
            if entry_config.method_type == "type1":
                plot_filename = tracer.apply_type1_analysis(result, include_mask4, show_arrows)
            else:  # type2a or type2b
                plot_filename = tracer.apply_type2_analysis(result, show_arrows)
            
            # Add plot filename to result
            result.plot_filename = plot_filename
        
        return result
        
    except Exception as e:
        # Create failed result
        from event_tracer import DetectionResult
        return DetectionResult(
            entry_config=entry_config,
            success=False,
            error_message=f"Parallel processing error: {str(e)}"
        )

def get_optimal_worker_count() -> int:
    """
    Determine optimal number of worker processes/threads
    
    Returns:
        Optimal number of workers
    """
    cpu_count = multiprocessing.cpu_count()
    memory_stats = get_memory_usage()
    
    # Base worker count on CPU cores
    optimal_workers = max(1, cpu_count - 1)  # Leave one core for main process
    
    # Adjust based on available memory (each worker needs ~500MB)
    memory_per_worker_gb = 0.5
    max_workers_by_memory = int(memory_stats["available_gb"] / memory_per_worker_gb)
    
    # Use the lower of CPU-based or memory-based limit
    optimal_workers = min(optimal_workers, max_workers_by_memory)
    
    # Configuration override
    config_workers = get_config_value('performance.max_workers', None)
    if config_workers:
        optimal_workers = min(optimal_workers, config_workers)
    
    # Ensure at least 1 worker
    optimal_workers = max(1, optimal_workers)
    
    if DEBUG_MODE:
        print(f"[PARALLEL] CPUs: {cpu_count}, Memory-based max: {max_workers_by_memory}, "
              f"Chosen: {optimal_workers} workers")
    
    return optimal_workers

def process_entries_parallel(tracer: 'EventTracer', configs: List['EntryConfig'], 
                           include_mask4: bool, show_arrows: bool,
                           use_processes: bool = False) -> List['DetectionResult']:
    """
    Process multiple entries in parallel using threads or processes
    
    Args:
        tracer: EventTracer instance
        configs: List of entry configurations
        include_mask4: Whether to include mask 4
        show_arrows: Whether to show arrows
        use_processes: Use processes instead of threads (slower startup, better for CPU-bound)
        
    Returns:
        List of DetectionResults
    """
    if len(configs) <= 1:
        # Not worth parallelizing for single entry
        return tracer.process_entry_batch(configs, include_mask4, show_arrows)
    
    optimal_workers = get_optimal_worker_count()
    
    # Don't use more workers than entries
    num_workers = min(optimal_workers, len(configs))
    
    print(f"🚀 Processing {len(configs)} entries using {num_workers} parallel workers...")
    
    results = []
    
    try:
        with memory_profiler("parallel entry processing"):
            if use_processes:
                # Process-based parallelism (good for CPU-intensive tasks)
                # Need to serialize tracer state for multiprocessing
                tracer_state = {
                    'sysmon_df': tracer.sysmon_df,
                    'caldera_entries': tracer.caldera_entries,
                    'debug': tracer.debug
                }
                
                process_func = partial(
                    process_entry_parallel,
                    tracer_state, 
                    include_mask4=include_mask4,
                    show_arrows=show_arrows
                )
                
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    results = list(executor.map(process_func, configs))
            
            else:
                # Thread-based parallelism (good for I/O-bound tasks, lower overhead)
                def process_single_entry(config):
                    try:
                        result = tracer.detect_zero_level_event(config)
                        
                        if result.success:
                            # Apply appropriate analysis
                            if config.method_type == "type1":
                                plot_filename = tracer.apply_type1_analysis(result, include_mask4, show_arrows)
                            else:  # type2a or type2b
                                plot_filename = tracer.apply_type2_analysis(result, show_arrows)
                        
                        return result
                        
                    except Exception as e:
                        from event_tracer import DetectionResult
                        return DetectionResult(
                            entry_config=config,
                            success=False,
                            error_message=f"Thread processing error: {str(e)}"
                        )
                
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    # Submit all tasks
                    future_to_config = {
                        executor.submit(process_single_entry, config): config 
                        for config in configs
                    }
                    
                    # Collect results as they complete
                    for i, future in enumerate(as_completed(future_to_config), 1):
                        config = future_to_config[future]
                        try:
                            result = future.result()
                            results.append(result)
                            
                            if DEBUG_MODE or i % 5 == 0:  # Progress updates
                                print(f"   Completed {i}/{len(configs)} entries...")
                            
                            # Memory monitoring during parallel processing
                            if i % 3 == 0:  # Check every 3 completions
                                monitor_memory_usage("parallel processing")
                            
                        except Exception as e:
                            print(f"⚠️ Entry {config.entry_id} failed: {e}")
                            # Create failed result
                            from event_tracer import DetectionResult
                            failed_result = DetectionResult(
                                entry_config=config,
                                success=False,
                                error_message=f"Future processing error: {str(e)}"
                            )
                            results.append(failed_result)
    
    except Exception as e:
        print(f"❌ Parallel processing failed: {e}")
        print("   Falling back to sequential processing...")
        # Fallback to sequential processing
        return tracer.process_entry_batch(configs, include_mask4, show_arrows)
    
    # Sort results by entry_id to maintain order
    results.sort(key=lambda r: r.entry_config.entry_id)
    
    print(f"✅ Parallel processing completed: {len(results)} results")
    cleanup_memory()
    
    return results

def validate_sysmon_data(df: pd.DataFrame) -> Dict[str, any]:
    """
    Comprehensive validation of Sysmon data quality
    
    Args:
        df: Sysmon DataFrame to validate
        
    Returns:
        dict: Validation results and statistics
        
    Raises:
        DataValidationError: If critical data quality issues are found
    """
    validation_results = {
        "total_rows": len(df),
        "warnings": [],
        "errors": [],
        "statistics": {}
    }
    
    # Check if DataFrame is empty
    if len(df) == 0:
        raise DataValidationError("Sysmon DataFrame is empty")
    
    # Check required columns
    required_columns = ['EventID', 'Computer', 'UtcTime', 'ProcessGuid']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise DataValidationError(
            f"Missing required columns: {missing_columns}",
            details={"missing_columns": missing_columns, "available_columns": list(df.columns)}
        )
    
    # Check for excessive null values
    null_percentages = df.isnull().sum() / len(df) * 100
    high_null_threshold = 50.0  # 50% null values threshold
    
    problematic_columns = null_percentages[null_percentages > high_null_threshold]
    if not problematic_columns.empty:
        validation_results["warnings"].append({
            "type": "high_null_percentage",
            "message": f"High null percentage in columns: {problematic_columns.to_dict()}",
            "columns": problematic_columns.to_dict()
        })
    
    # Validate EventID ranges
    if 'EventID' in df.columns:
        valid_event_ids = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26}
        unique_event_ids = set(df['EventID'].dropna().unique())
        invalid_event_ids = unique_event_ids - valid_event_ids
        
        if invalid_event_ids:
            validation_results["warnings"].append({
                "type": "invalid_event_ids",
                "message": f"Found invalid EventIDs: {invalid_event_ids}",
                "invalid_ids": list(invalid_event_ids)
            })
        
        validation_results["statistics"]["event_id_distribution"] = df['EventID'].value_counts().to_dict()
    
    # Validate timestamp format and range
    if 'UtcTime' in df.columns:
        try:
            df['UtcTime'] = pd.to_datetime(df['UtcTime'], errors='coerce')
            null_timestamps = df['UtcTime'].isnull().sum()
            
            if null_timestamps > 0:
                null_percentage = (null_timestamps / len(df)) * 100
                if null_percentage > 5.0:  # More than 5% invalid timestamps
                    validation_results["errors"].append({
                        "type": "invalid_timestamps",
                        "message": f"High percentage of invalid timestamps: {null_percentage:.1f}%",
                        "invalid_count": null_timestamps
                    })
                else:
                    validation_results["warnings"].append({
                        "type": "some_invalid_timestamps",
                        "message": f"Some invalid timestamps found: {null_timestamps} rows ({null_percentage:.1f}%)",
                        "invalid_count": null_timestamps
                    })
            
            # Check timestamp range reasonableness
            valid_timestamps = df['UtcTime'].dropna()
            if len(valid_timestamps) > 0:
                min_time = valid_timestamps.min()
                max_time = valid_timestamps.max()
                time_span = max_time - min_time
                
                # Check if timestamps are in a reasonable range (not too old or in future)
                now = pd.Timestamp.now()
                very_old = now - pd.Timedelta(days=365*5)  # 5 years ago
                future = now + pd.Timedelta(days=1)  # 1 day in future
                
                old_timestamps = valid_timestamps < very_old
                future_timestamps = valid_timestamps > future
                
                if old_timestamps.any():
                    validation_results["warnings"].append({
                        "type": "very_old_timestamps",
                        "message": f"Found {old_timestamps.sum()} timestamps older than 5 years",
                        "count": old_timestamps.sum()
                    })
                
                if future_timestamps.any():
                    validation_results["warnings"].append({
                        "type": "future_timestamps", 
                        "message": f"Found {future_timestamps.sum()} timestamps in the future",
                        "count": future_timestamps.sum()
                    })
                
                validation_results["statistics"]["time_range"] = {
                    "min_time": str(min_time),
                    "max_time": str(max_time),
                    "span_hours": time_span.total_seconds() / 3600
                }
                
        except Exception as e:
            validation_results["errors"].append({
                "type": "timestamp_processing_error",
                "message": f"Error processing timestamps: {str(e)}",
                "error": str(e)
            })
    
    # Validate Computer field consistency
    if 'Computer' in df.columns:
        unique_computers = df['Computer'].dropna().unique()
        validation_results["statistics"]["unique_computers"] = len(unique_computers)
        validation_results["statistics"]["computer_list"] = list(unique_computers)
        
        if len(unique_computers) == 0:
            validation_results["errors"].append({
                "type": "no_computer_names",
                "message": "No valid computer names found"
            })
    
    # Check ProcessGuid format consistency
    if 'ProcessGuid' in df.columns:
        non_null_guids = df['ProcessGuid'].dropna()
        if len(non_null_guids) > 0:
            # Basic GUID format check (should contain hyphens and hex characters)
            sample_guid = non_null_guids.iloc[0]
            if not isinstance(sample_guid, str) or len(sample_guid) < 10:
                validation_results["warnings"].append({
                    "type": "suspicious_guid_format",
                    "message": f"ProcessGuid format may be invalid. Sample: {sample_guid}",
                    "sample": str(sample_guid)
                })
    
    # Raise error if critical issues found
    if validation_results["errors"]:
        error_messages = [error["message"] for error in validation_results["errors"]]
        raise DataValidationError(
            f"Critical data validation errors found: {'; '.join(error_messages)}",
            details=validation_results
        )
    
    return validation_results

def validate_configuration_values(config_dict: dict) -> None:
    """
    Validate configuration values for correctness
    
    Args:
        config_dict: Configuration dictionary to validate
        
    Raises:
        ConfigurationError: If configuration values are invalid
    """
    errors = []
    
    # Validate time windows
    try:
        primary_window = config_dict.get('time_windows', {}).get('primary_seconds')
        fallback_window = config_dict.get('time_windows', {}).get('fallback_seconds')
        
        if primary_window is not None:
            if not isinstance(primary_window, (int, float)) or primary_window <= 0:
                errors.append("time_windows.primary_seconds must be a positive number")
            
        if fallback_window is not None:
            if not isinstance(fallback_window, (int, float)) or fallback_window <= 0:
                errors.append("time_windows.fallback_seconds must be a positive number")
            
            if primary_window and fallback_window and fallback_window <= primary_window:
                errors.append("time_windows.fallback_seconds should be greater than primary_seconds")
    
    except Exception as e:
        errors.append(f"Error validating time windows: {str(e)}")
    
    # Validate output settings
    try:
        plot_dpi = config_dict.get('output', {}).get('plot_dpi')
        if plot_dpi is not None:
            if not isinstance(plot_dpi, int) or plot_dpi < 50 or plot_dpi > 1000:
                errors.append("output.plot_dpi must be an integer between 50 and 1000")
        
        figure_size = config_dict.get('output', {}).get('figure_size')
        if figure_size is not None:
            if not isinstance(figure_size, list) or len(figure_size) != 2:
                errors.append("output.figure_size must be a list of two numbers [width, height]")
            elif not all(isinstance(x, (int, float)) and x > 0 for x in figure_size):
                errors.append("output.figure_size values must be positive numbers")
    
    except Exception as e:
        errors.append(f"Error validating output settings: {str(e)}")
    
    # Validate analysis settings
    try:
        active_masks = config_dict.get('analysis', {}).get('active_masks')
        if active_masks is not None:
            if not isinstance(active_masks, list):
                errors.append("analysis.active_masks must be a list")
            elif not all(isinstance(x, int) and 1 <= x <= 4 for x in active_masks):
                errors.append("analysis.active_masks must contain integers between 1 and 4")
        
        max_depth = config_dict.get('analysis', {}).get('max_recursion_depth')
        if max_depth is not None:
            if not isinstance(max_depth, int) or max_depth < 1 or max_depth > 50:
                errors.append("analysis.max_recursion_depth must be an integer between 1 and 50")
    
    except Exception as e:
        errors.append(f"Error validating analysis settings: {str(e)}")
    
    if errors:
        raise ConfigurationError(
            f"Configuration validation failed: {'; '.join(errors)}",
            details={"validation_errors": errors}
        )

def canonical(cmd):
    """Canonical normalization of command strings (from original clean_event_tracer.py)"""
    if pd.isna(cmd) or cmd is None:
        return ""
    
    cmd = str(cmd).strip()
    
    # Remove wrapper commands
    WRAPPER_RX = re.compile(
        r"""(?ix)
        ^(?:cmd\.exe\s+/c\s+|
            powershell\.exe\s+(?:-executionpolicy\s+\w+\s+)?(?:-c|-command)\s+")
        """
    )
    
    cmd = WRAPPER_RX.sub("", cmd)
    cmd = cmd.rstrip('"')
    
    # Clean up quotes and spaces
    cmd = (cmd
           .replace('"', '')
           .replace("'", "")
           .replace("  ", " ")
           )
    
    return cmd.rstrip(';')

def load_config(config_file: str = "config.yaml") -> Dict:
    """Load and validate configuration from YAML file with enhanced error handling"""
    
    with error_context("configuration loading", config_file=config_file):
        # Validate file exists and is accessible
        validate_file_exists(config_file, "configuration file")
        
        # Load YAML with retry logic
        config_data = safe_operation_with_retry(
            _load_yaml_file, 
            config_file,
            max_retries=2
        )
        
        # Validate configuration values
        validate_configuration_values(config_data)
        
        print(f"✅ Configuration loaded and validated from {config_file}")
        return config_data

def _load_yaml_file(config_file: str) -> Dict:
    """Helper function to load YAML file (used by retry logic)"""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        if config_data is None:
            raise ConfigurationError("Configuration file is empty or contains only comments")
        
        if not isinstance(config_data, dict):
            raise ConfigurationError("Configuration file must contain a YAML dictionary/object")
        
        return config_data
        
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Invalid YAML syntax: {e}")
    except UnicodeDecodeError as e:
        raise ConfigurationError(f"Configuration file encoding error: {e}")
    except Exception as e:
        raise FileProcessingError(f"Failed to read configuration file: {e}")

def get_config_value(key_path: str, default=None):
    """Get configuration value using dot notation (e.g., 'time_windows.primary_seconds')"""
    global config
    if config is None:
        # Try to load config if not loaded yet
        try:
            config = load_config()
        except:
            return default
    
    keys = key_path.split('.')
    value = config
    
    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default

@dataclass
class EntryConfig:
    """Configuration for a single entry analysis"""
    entry_id: int
    computer_short: str  # Will be expanded to .boombox.local
    totem_eventid: int  # 1=ProcessCreate, 11=FileCreate, 23=FileDelete
    
    @property
    def computer_full(self) -> str:
        """Full computer name with domain"""
        return f"{self.computer_short}.boombox.local"
    
    @property
    def method_type(self) -> str:
        """Determine method type from totem_eventid"""
        if self.totem_eventid == 1:
            return "type1"
        elif self.totem_eventid == 11:
            return "type2a"
        elif self.totem_eventid == 23:
            return "type2b"
        else:
            raise ValueError(f"Unsupported totem_eventid: {self.totem_eventid}")
    
    @property
    def totem_column(self) -> str:
        """Column to search for totem identifier"""
        if self.totem_eventid == 1:
            return "cmd_norm"  # Use canonical normalized column for CommandLine
        elif self.totem_eventid in [11, 23]:
            return "TargetFilename"
        else:
            raise ValueError(f"Unsupported totem_eventid: {self.totem_eventid}")

@dataclass
class DetectionResult:
    """Result of zero-level event detection"""
    entry_config: EntryConfig
    success: bool
    zero_level_event: Optional[pd.Series] = None
    matches_found: int = 0
    all_matches: Optional[pd.DataFrame] = None
    error_message: Optional[str] = None
    detection_time_ms: float = 0.0
    
    @property
    def method_type(self) -> str:
        return self.entry_config.method_type

class EventTracer:
    """Main event tracing and analysis class"""
    
    def __init__(self, sysmon_df: pd.DataFrame, caldera_entries: Dict, debug: bool = False):
        self.sysmon_df = sysmon_df
        self.caldera_entries = caldera_entries
        self.debug = debug or get_config_value('debug.enable_debug_mode', False)
        
        # Ensure output directory exists
        if CLI_ARGS and CLI_ARGS.output_dir:
            output_dir = CLI_ARGS.output_dir
        else:
            output_dir = get_config_value('output.directory', '5_entry-events-plots')
        Path(output_dir).mkdir(exist_ok=True)
        
        # Convert UtcTime to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(self.sysmon_df['UtcTime']):
            self.sysmon_df['UtcTime'] = pd.to_datetime(self.sysmon_df['UtcTime'])
    
    def debug_log(self, message: str, level: str = "INFO"):
        """Conditional debug logging"""
        if self.debug:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            print(f"[DEBUG] [{timestamp}] [{level}] {message}")
    
    def load_batch_configuration(self, csv_file: str = None) -> List[EntryConfig]:
        """Load batch configuration from CSV file"""
        # Use config file if not specified
        if csv_file is None:
            csv_file = get_config_value('data_sources.entry_config_file', 'entry_config.csv')
        
        self.debug_log(f"Loading batch configuration from {csv_file}")
        
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"Configuration file not found: {csv_file}")
        
        df = pd.read_csv(csv_file, header=None, names=['entry_id', 'computer_short', 'totem_eventid'])
        
        configs = []
        for _, row in df.iterrows():
            config = EntryConfig(
                entry_id=int(row['entry_id']),
                computer_short=str(row['computer_short']),
                totem_eventid=int(row['totem_eventid'])
            )
            configs.append(config)
            self.debug_log(f"Loaded config: Entry {config.entry_id} -> {config.computer_full} ({config.method_type})")
        
        return configs
    
    def get_entry_reference_data(self, entry_id: int) -> Tuple[datetime, str]:
        """Get entry reference timestamp and command"""
        # Convert to string key for JSON access
        entry_key = str(entry_id)
        if entry_key not in self.caldera_entries:
            raise ValueError(f"Entry {entry_id} not found in Caldera data")
        
        entry_data = self.caldera_entries[entry_key]
        finished_timestamp = entry_data.get('finished_timestamp', None)
        
        if finished_timestamp is None:
            raise ValueError(f"Entry {entry_id} has no finished_timestamp")
        
        entry_time = pd.to_datetime(finished_timestamp)
        
        # Get command/totem identifier and apply canonical normalization
        raw_command = entry_data.get('new_command', '')
        
        return entry_time, canonical(raw_command)
    
    def find_totem_matches(self, config: EntryConfig, totem_identifier: str) -> pd.DataFrame:
        """Find events matching totem identifier with time window filtering"""
        self.debug_log(f"Searching for totem matches: EventID {config.totem_eventid}, Computer {config.computer_full}")
        
        # Get entry reference time for time window filtering
        entry_time, _ = self.get_entry_reference_data(config.entry_id)
        
        # Define time window around entry timestamp (from config)
        time_window_seconds = get_config_value('time_windows.primary_seconds', 180)
        search_start = entry_time - pd.Timedelta(seconds=time_window_seconds)
        search_end = entry_time + pd.Timedelta(seconds=time_window_seconds)
        
        self.debug_log(f"Time window filtering: {search_start} to {search_end}")
        
        # Base filters with time window
        mask = (
            (self.sysmon_df['EventID'] == config.totem_eventid) &
            (self.sysmon_df['Computer'] == config.computer_full) &
            (pd.to_datetime(self.sysmon_df['UtcTime']) >= search_start) &
            (pd.to_datetime(self.sysmon_df['UtcTime']) <= search_end)
        )
        
        filtered_events = self.sysmon_df[mask].copy()
        self.debug_log(f"Events after computer/eventid/time filter: {len(filtered_events)}")
        
        if len(filtered_events) == 0:
            self.debug_log(f"No events found in time window, expanding search...")
            # Fallback: expand time window if no matches (from config)
            expanded_window = get_config_value('time_windows.fallback_seconds', 300)
            search_start = entry_time - pd.Timedelta(seconds=expanded_window)
            search_end = entry_time + pd.Timedelta(seconds=expanded_window)
            
            mask = (
                (self.sysmon_df['EventID'] == config.totem_eventid) &
                (self.sysmon_df['Computer'] == config.computer_full) &
                (pd.to_datetime(self.sysmon_df['UtcTime']) >= search_start) &
                (pd.to_datetime(self.sysmon_df['UtcTime']) <= search_end)
            )
            
            filtered_events = self.sysmon_df[mask].copy()
            self.debug_log(f"Events after expanded time window: {len(filtered_events)}")
        
        if len(filtered_events) == 0:
            return pd.DataFrame()
        
        # Totem identifier matching
        totem_column = config.totem_column
        if totem_column not in filtered_events.columns:
            self.debug_log(f"Warning: Column {totem_column} not found in filtered events")
            return pd.DataFrame()
        
        # Search for totem identifier in the specified column
        totem_mask = filtered_events[totem_column].str.contains(
            totem_identifier, case=False, na=False, regex=False
        )
        
        matches = filtered_events[totem_mask].copy()
        self.debug_log(f"Totem matches found: {len(matches)}")
        
        return matches
    
    def select_zero_level_event(self, config: EntryConfig, matches: pd.DataFrame, 
                              entry_time: datetime) -> Tuple[Optional[pd.Series], str]:
        """Select zero-level event from matches based on method type"""
        
        if len(matches) == 0:
            return None, "No matches found"
        
        if len(matches) == 1:
            self.debug_log("Single match found - auto-selecting")
            return matches.iloc[0], "Single match auto-selected"
        
        # Multiple matches handling
        if config.totem_eventid == 1:  # Type 1: Select oldest event (within time window)
            # Since we now have time window filtering, select the oldest event within that window
            # This gives us the zero-level/root cause event that started the sequence
            oldest_event = matches.loc[matches['UtcTime'].idxmin()]
            oldest_time = pd.to_datetime(oldest_event['UtcTime'])
            time_diff = (oldest_time - entry_time).total_seconds()
            
            self.debug_log(f"Multiple Type 1 matches - selected oldest within time window: {oldest_event['UtcTime']} ({time_diff:+.1f}s from entry)")
            return oldest_event, f"Oldest within time window of {len(matches)} matches selected"
        
        else:  # Type 2A/2B: Print all matches for manual selection
            self.debug_log("Multiple Type 2 matches - printing for manual selection")
            
            print(f"\n📋 MULTIPLE MATCHES FOUND FOR ENTRY {config.entry_id}")
            print(f"   Entry finished_timestamp: {entry_time}")
            print(f"   Target: {config.computer_full} EventID {config.totem_eventid}")
            print(f"   Found {len(matches)} matching events:")
            print(f"   {'Index':<8} {'Timestamp':<20} {'Time Diff (s)':<12} {config.totem_column}")
            print(f"   {'-' * 80}")
            
            for idx, event in matches.iterrows():
                event_time = pd.to_datetime(event['UtcTime'])
                time_diff = (event_time - entry_time).total_seconds()
                totem_value = str(event.get(config.totem_column, 'N/A'))[:50]
                
                print(f"   {idx:<8} {event_time.strftime('%H:%M:%S.%f')[:-3]:<20} {time_diff:+8.1f}s    {totem_value}")
            
            print(f"   Manual selection required - continuing to next entry...")
            return None, f"Multiple matches ({len(matches)}) - manual selection required"
    
    def detect_zero_level_event(self, config: EntryConfig) -> DetectionResult:
        """Main detection method for zero-level events"""
        start_time = datetime.now()
        
        try:
            self.debug_log(f"Starting detection for Entry {config.entry_id}")
            
            # Get reference data
            entry_time, totem_identifier = self.get_entry_reference_data(config.entry_id)
            self.debug_log(f"Entry {config.entry_id}: {entry_time}, totem: '{totem_identifier[:50]}...'")
            
            # Find matches
            matches = self.find_totem_matches(config, totem_identifier)
            
            # Select zero-level event
            zero_level_event, selection_message = self.select_zero_level_event(
                config, matches, entry_time
            )
            
            # Calculate detection time
            detection_time = (datetime.now() - start_time).total_seconds() * 1000
            
            success = zero_level_event is not None
            
            result = DetectionResult(
                entry_config=config,
                success=success,
                zero_level_event=zero_level_event,
                matches_found=len(matches),
                all_matches=matches if len(matches) > 0 else None,
                error_message=None if success else selection_message,
                detection_time_ms=detection_time
            )
            
            self.debug_log(f"Detection completed: {success}, {len(matches)} matches, {detection_time:.1f}ms")
            return result
            
        except Exception as e:
            detection_time = (datetime.now() - start_time).total_seconds() * 1000
            error_message = f"Detection failed: {str(e)}"
            self.debug_log(f"Detection error: {error_message}")
            
            return DetectionResult(
                entry_config=config,
                success=False,
                error_message=error_message,
                detection_time_ms=detection_time
            )
    
    def apply_type1_analysis(self, result: DetectionResult, 
                           include_mask4: bool, show_arrows: bool) -> Optional[str]:
        """Apply Type 1 analysis with comprehensive error handling"""
        if not result.success:
            return None
        
        entry_id = result.entry_config.entry_id
        
        with error_context("Type 1 analysis", entry_id=entry_id):
            self.debug_log(f"Applying Type 1 analysis for Entry {entry_id}")
            
            zero_event = result.zero_level_event
            config = result.entry_config
            
            # Validate zero-level event data
            try:
                process_guid = zero_event.get('ProcessGuid')
                process_id = zero_event.get('ProcessId')
                
                if pd.isna(process_guid) or pd.isna(process_id):
                    raise AnalysisError(
                        f"Invalid ProcessGuid/ProcessId in zero-level event",
                        details={"ProcessGuid": process_guid, "ProcessId": process_id}
                    )
                
                zero_level_timestamp = pd.to_datetime(zero_event['UtcTime'])
                target_computer = config.computer_full
                
            except Exception as e:
                raise AnalysisError(f"Failed to extract zero-level event details: {e}")
            
            # Apply mask analysis with error handling
            try:
                m1 = self.mask_to_find_child_process(process_guid, process_id, target_computer)
                m2 = self.mask_to_find_child_processcreate(process_guid, process_id, target_computer)
                m3 = self.mask_to_find_child_eventid_8_or_10(process_guid, process_id, target_computer)
            except Exception as e:
                raise AnalysisError(f"Failed to apply mask analysis: {e}")
            
            # Apply temporal filtering (events after zero-level) with error handling
            try:
                m1_events = self.sysmon_df[m1]
                m2_events = self.sysmon_df[m2]
                m3_events = self.sysmon_df[m3]
                
                m1_filtered = m1_events[m1_events['UtcTime'] > zero_level_timestamp]
                m2_filtered = m2_events[m2_events['UtcTime'] > zero_level_timestamp]
                m3_filtered = m3_events[m3_events['UtcTime'] > zero_level_timestamp]
                
                m1_temporal = self.sysmon_df.index.isin(m1_filtered.index)
                m2_temporal = self.sysmon_df.index.isin(m2_filtered.index)
                m3_temporal = self.sysmon_df.index.isin(m3_filtered.index)
                
                self.debug_log(f"Type 1 masks: M1={m1_temporal.sum()}, M2={m2_temporal.sum()}, M3={m3_temporal.sum()}")
                
                # Calculate Level 1 scenario (masks 1-3)
                scenario1 = m1_temporal | m2_temporal | m3_temporal
                combined_events = self.sysmon_df[scenario1]  # Start with Level 1 events
                
                print(f"   Level 1 events found: {scenario1.sum()} events")
                
            except Exception as e:
                raise AnalysisError(f"Failed during temporal filtering: {e}")
            
            # Recursive tracing with error handling
            try:
                # RECURSIVE TRACING (ALWAYS RUNS - like original script)
                print(f"\n🔄 RECURSIVE EVENT TRACING:")
                print(f"   Finding all spawned events recursively...")
                
                # Find spawning-capable events from Level 1 (EventID 1, 8, 10)
                level1_spawners = combined_events[combined_events['EventID'].isin([1, 8, 10])]
            except Exception as e:
                raise AnalysisError(f"Failed during recursive tracing setup: {e}")
            
            # Complete the analysis (simplified for now - keeping original logic)
        
        if len(level1_spawners) > 0:
            print(f"   🎯 Starting with {len(level1_spawners)} spawning-capable events from Level 1")
            
            # Perform recursive tracing with temporal filtering and system filtering
            all_recursive_events = self.recursive_event_tracing(
                level1_spawners, 
                zero_level_timestamp=zero_level_timestamp, 
                target_computer=target_computer
            )
            
            if len(all_recursive_events) > 0:
                print(f"   ✅ Recursive tracing found {len(all_recursive_events)} additional events across all levels")
                
                # Combine level 1 + recursive events for complete analysis
                complete_events = pd.concat([combined_events, all_recursive_events], ignore_index=True).drop_duplicates()
                
                print(f"\n📊 COMPLETE RECURSIVE ANALYSIS:")
                print(f"   Level 0 (zero-level): 1 event")
                print(f"   Level 1 (direct children): {len(combined_events)} events")
                print(f"   Levels 2+ (recursive): {len(all_recursive_events)} events")
                print(f"   Total unique events: {len(complete_events)} events")
                
                print(f"\n📋 COMPLETE RECURSIVE EVENT BREAKDOWN:")
                complete_breakdown = complete_events['EventID'].value_counts().sort_index()
                for event_id, count in complete_breakdown.items():
                    print(f"      EventID {event_id}: {count} events")
                
                # Update combined_events to include recursive results for plotting (ORIGINAL BEHAVIOR)
                combined_events = complete_events
            else:
                print(f"   🛑 No additional events found through recursion")
        else:
            print(f"   🛑 No spawning-capable events in Level 1 - skipping recursion")
        
        # Apply parent mask (mask 4) if requested - SEPARATE from recursive tracing
        if include_mask4:
            print(f"\n📦 MASK 4: PARENT EVENTS")
            # Note: mask4 in original is about parent events, not recursive
            # For now, we'll skip this since it's a different feature
            print(f"   🔒 Parent mask not implemented in v2 yet")
        else:
            print(f"   🔒 Parent mask disabled")
        
        # VERIFICATION: Check what's actually being passed to plotting
        print(f"\n🔍 VERIFICATION BEFORE PLOTTING:")
        print(f"   Events being sent to plot: {len(combined_events)}")
        if len(combined_events) > 0:
            event_breakdown = combined_events['EventID'].value_counts().sort_index()
            print(f"   Event breakdown:")
            for event_id, count in event_breakdown.items():
                print(f"      EventID {event_id}: {count} events")
            
            # Show time range of events
            min_time = combined_events['UtcTime'].min()
            max_time = combined_events['UtcTime'].max()
            time_span = (max_time - min_time).total_seconds()
            print(f"   Time span: {time_span:.3f}s ({min_time} to {max_time})")
        else:
            print(f"   ⚠️ WARNING: No events to plot!")
        
        # Generate plot
        plot_filename = self.plot_zero_level_events(
            config.entry_id, zero_event, combined_events, show_arrows, result.method_type
        )
        
        self.debug_log(f"Type 1 analysis completed: {len(combined_events)} events, plot: {plot_filename}")
        return plot_filename
    
    def apply_type2_analysis(self, result: DetectionResult, show_arrows: bool) -> Optional[str]:
        """Apply Type 2 analysis (simplified file operation detection)"""
        if not result.success:
            return None
        
        self.debug_log(f"Applying Type 2 analysis for Entry {result.entry_config.entry_id}")
        
        zero_event = result.zero_level_event
        config = result.entry_config
        
        # For Type 2, the zero-level event IS the file operation event
        # No additional spawned events - just plot the single event
        single_event_df = pd.DataFrame([zero_event])
        
        # Generate plot
        plot_filename = self.plot_zero_level_events(
            config.entry_id, zero_event, single_event_df, show_arrows, result.method_type
        )
        
        self.debug_log(f"Type 2 analysis completed: single event plot: {plot_filename}")
        return plot_filename
    
    def mask_to_find_child_process(self, process_guid: str, process_pid: int, 
                                 target_computer: Optional[str] = None) -> pd.Series:
        """Find child events by ProcessGuid/ProcessId (excluding specific EventIDs)"""
        mask = (
            (self.sysmon_df['ProcessGuid'] == process_guid) & 
            (self.sysmon_df['ProcessId'] == process_pid) &
            (~self.sysmon_df['EventID'].isin([1, 8, 10]))
        )
        if target_computer:
            mask = mask & (self.sysmon_df['Computer'] == target_computer)
        return mask
    
    def mask_to_find_child_processcreate(self, process_guid: str, process_pid: int,
                                       target_computer: Optional[str] = None) -> pd.Series:
        """Find child ProcessCreate events by ParentProcessGuid/ParentProcessId"""
        mask = (
            (self.sysmon_df['EventID'] == 1) &
            (self.sysmon_df['ParentProcessGuid'] == process_guid) &
            (self.sysmon_df['ParentProcessId'] == process_pid)
        )
        if target_computer:
            mask = mask & (self.sysmon_df['Computer'] == target_computer)
        return mask
    
    def mask_to_find_child_eventid_8_or_10(self, process_guid: str, process_pid: int,
                                         target_computer: Optional[str] = None) -> pd.Series:
        """Find CreateRemoteThread/ProcessAccess events by SourceProcessGuid/SourceProcessId"""
        mask = (
            (self.sysmon_df['EventID'].isin([8, 10])) &
            (self.sysmon_df['SourceProcessGUID'] == process_guid) &
            (self.sysmon_df['SourceProcessId'] == process_pid)
        )
        if target_computer:
            mask = mask & (self.sysmon_df['Computer'] == target_computer)
        return mask
    
    def recursive_event_tracing(self, spawning_events: pd.DataFrame, processed_guids: Optional[set] = None,
                              depth: int = 0, zero_level_timestamp: Optional[pd.Timestamp] = None,
                              target_computer: Optional[str] = None) -> pd.DataFrame:
        """
        Recursively trace all spawned events from spawning-capable events (from original script).
        
        Args:
            spawning_events: DataFrame of events that can spawn children (EventID 1, 8, 10)
            processed_guids: Set of already processed ProcessGuid_ProcessId combinations
            depth: Current recursion depth (for debugging)
            zero_level_timestamp: Timestamp of zero-level event for temporal filtering
            target_computer: Target computer name for system-level filtering
            
        Returns:
            DataFrame of all recursively found events
        """
        self.debug_log(f"Starting recursive tracing at depth {depth} with {len(spawning_events)} spawning events")
        
        if processed_guids is None:
            processed_guids = set()
        
        if len(spawning_events) == 0:
            self.debug_log(f"No spawning events at depth {depth} - terminating recursion")
            return pd.DataFrame()
        
        all_children = []
        new_spawning_candidates = []
        
        for idx, event in spawning_events.iterrows():
            event_id = event['EventID']
            
            # Determine parent identifiers based on event type
            if event_id == 1:  # ProcessCreate
                parent_guid = event.get('ProcessGuid')
                parent_pid = event.get('ProcessId')
                spawn_type = "ProcessCreate"
            # elif event_id in [8, 10]:  # CreateRemoteThread, ProcessAccess
                # parent_guid = event.get('TargetProcessGUID')
                # parent_pid = event.get('TargetProcessId')
                # spawn_type = f"EventID_{event_id}"
            else:
                continue  # Skip non-spawning events
            
            # Skip if missing required identifiers
            if pd.isna(parent_guid) or pd.isna(parent_pid):
                self.debug_log(f"Skipping event with missing identifiers: {parent_guid}_{parent_pid}")
                continue
                
            # Create unique identifier for deduplication
            guid_pid_key = f"{parent_guid}_{parent_pid}"
            
            if guid_pid_key in processed_guids:
                self.debug_log(f"Skipping already processed: {guid_pid_key}")
                continue
                
            processed_guids.add(guid_pid_key)
            
            print(f"   🔄 Depth {depth}: Tracing from {spawn_type} {guid_pid_key}")
            
            # Apply the three masks to find children
            m1 = self.mask_to_find_child_process(parent_guid, parent_pid, target_computer)
            m2 = self.mask_to_find_child_processcreate(parent_guid, parent_pid, target_computer)
            m3 = self.mask_to_find_child_eventid_8_or_10(parent_guid, parent_pid, target_computer)
            
            # Combine all masks
            combined_mask = m1 | m2 | m3
            children = self.sysmon_df[combined_mask]
            
            # Apply temporal filtering if zero_level_timestamp is provided
            if zero_level_timestamp is not None and len(children) > 0:
                children_before_filter = len(children)
                children = children[children['UtcTime'] > zero_level_timestamp]
                children_filtered = len(children)
                if children_before_filter > children_filtered:
                    print(f"      └─ Temporal filter: {children_before_filter} → {children_filtered} events (removed {children_before_filter - children_filtered} pre-zero events)")
            
            if len(children) > 0:
                print(f"      └─ Found {len(children)} children ({m1.sum()}, {m2.sum()}, {m3.sum()})")
                all_children.append(children)
                
                # Find events in children that can spawn more events (EventID 1, 8, 10)
                potential_spawners = children[children['EventID'].isin([1, 8, 10])]
                if len(potential_spawners) > 0:
                    new_spawning_candidates.append(potential_spawners)
                    print(f"      └─ {len(potential_spawners)} can spawn more events")
            else:
                print(f"      └─ No children found")
        
        # Combine all children found at this level
        if all_children:
            current_level_events = pd.concat(all_children, ignore_index=True).drop_duplicates()
        else:
            current_level_events = pd.DataFrame()
        
        # Combine all new spawning candidates
        if new_spawning_candidates:
            next_level_spawners = pd.concat(new_spawning_candidates, ignore_index=True).drop_duplicates()
            
            # Recursively trace the next level
            deeper_events = self.recursive_event_tracing(next_level_spawners, processed_guids, depth + 1, zero_level_timestamp, target_computer)
            
            # Combine current level with deeper levels
            if len(deeper_events) > 0:
                all_recursive_events = pd.concat([current_level_events, deeper_events], ignore_index=True).drop_duplicates()
            else:
                all_recursive_events = current_level_events
        else:
            all_recursive_events = current_level_events
            print(f"   🛑 Depth {depth}: No more spawning candidates - recursion ends")
        
        self.debug_log(f"Depth {depth} complete: {len(current_level_events)} current + {len(all_recursive_events) - len(current_level_events)} deeper = {len(all_recursive_events)} total")
        return all_recursive_events
    
    def plot_zero_level_events(self, entry_id: int, zero_level_event: pd.Series,
                             combined_events: pd.DataFrame, show_arrows: bool,
                             method_type: str) -> str:
        """Generate timeline visualization (matching original methodology)"""
        self.debug_log(f"Generating plot for Entry {entry_id}")
        
        # Convert timestamps to datetime if they aren't already
        combined_events = combined_events.copy()
        combined_events['UtcTime'] = pd.to_datetime(combined_events['UtcTime'])
        zero_level_time = pd.to_datetime(zero_level_event['UtcTime'])
        
        # Add the zero-level event itself to the events (it's not included in mask results)
        zero_level_df = pd.DataFrame([zero_level_event])
        zero_level_df['UtcTime'] = pd.to_datetime(zero_level_df['UtcTime'])
        
        # Combine with mask results and remove duplicates
        all_events_with_zero = pd.concat([combined_events, zero_level_df], ignore_index=True).drop_duplicates()
        
        # VERIFICATION: What events are we actually plotting?
        print(f"   🔍 PLOT VERIFICATION:")
        print(f"      Combined events input: {len(combined_events)}")
        print(f"      Zero-level events: {len(zero_level_df)}")
        print(f"      Total after merge: {len(all_events_with_zero)}")
        
        if len(combined_events) > 0:
            combined_breakdown = combined_events['EventID'].value_counts().sort_index()
            print(f"      Combined events breakdown:")
            for event_id, count in combined_breakdown.items():
                print(f"         EventID {event_id}: {count} events")
        
        # Calculate smart time window based on actual event timestamps (ORIGINAL METHOD)
        min_time = all_events_with_zero['UtcTime'].min()  # Usually zero-level event time
        max_time = all_events_with_zero['UtcTime'].max()  # Latest spawned event time
        
        # Calculate padding of the total time span (from config)
        time_span = max_time - min_time
        padding_percentage = get_config_value('visualization.padding_percentage', 0.05)
        
        # Handle case where all events have the same timestamp
        if time_span.total_seconds() == 0:
            padding = pd.Timedelta(seconds=1)  # Default 1 second padding
        else:
            padding = time_span * padding_percentage
        
        # Define smart adaptive time window
        window_start = min_time - padding
        window_end = max_time + padding
        
        # All events are already within window since we calculated window from them
        focused_events = all_events_with_zero.copy()
        
        if len(focused_events) == 0:
            print(f"   ⚠️ No events found in adaptive time window")
            return None
        
        # Sort events by time
        focused_events_sorted = focused_events.sort_values('UtcTime')
        
        # Calculate and display time span information
        total_span_seconds = time_span.total_seconds()
        padding_seconds = padding.total_seconds()
        
        print(f"   📊 Plotting {len(focused_events)} events in adaptive time window")
        print(f"   📊 Event span: {total_span_seconds:.3f}s with {padding_seconds:.3f}s padding ({padding_percentage*100:.1f}%)")
        print(f"   📊 Window: {window_start.strftime('%H:%M:%S.%f')[:-3]} to {window_end.strftime('%H:%M:%S.%f')[:-3]}")
        print(f"   🏹 Arrow drawing: {'ENABLED' if show_arrows else 'DISABLED'}")
        
        # Create plot with configurable styling
        figure_size = get_config_value('output.figure_size', [16, 8])
        fig, ax = plt.subplots(figsize=figure_size)
        
        # Event colors from configuration
        default_colors = {
            1: '#E74C3C', 3: '#95A5A6', 5: '#1ABC9C', 7: '#F39C12', 8: '#8E44AD',
            9: '#D35400', 10: '#F1C40F', 11: '#E91E63', 12: '#3498DB', 13: '#9B59B6',
            15: '#16A085', 17: '#2ECC71', 18: '#FF69B4', 22: '#34495E', 23: '#2C3E50',
            24: '#27AE60', 25: '#C0392B'
        }
        event_colors = get_config_value('visualization.event_colors', default_colors)
        
        zero_level_marked = False
        
        # Plot all events with black borders (original style)
        for idx, event in focused_events_sorted.iterrows():
            event_id = event['EventID']
            timestamp = event['UtcTime']
            color = event_colors.get(event_id, '#95A5A6')  # Default gray
            
            # Determine if this is the exact zero-level event (more precise matching)
            is_zero_level = (pd.notna(event['ProcessGuid']) and pd.notna(zero_level_event['ProcessGuid']) and
                        event['ProcessGuid'] == zero_level_event['ProcessGuid'] and 
                        event['ProcessId'] == zero_level_event['ProcessId'] and
                        event['EventID'] == zero_level_event['EventID'] and  # Must be same EventID
                        abs((timestamp - zero_level_time).total_seconds()) < 0.1)  # Within 100ms
            
            if is_zero_level and not zero_level_marked:
                # Mark the zero-level event specially (black star, only once)
                ax.scatter(timestamp, event_id, c=color, s=300, marker='*', 
                          edgecolors='black', linewidth=2, zorder=6, alpha=0.9)
                zero_level_marked = True
            else:
                # Regular event with black border
                ax.scatter(timestamp, event_id, c=color, s=100, alpha=0.8, 
                          edgecolors='black', linewidth=1, zorder=5)
        
        # Draw arrows if requested
        if show_arrows and len(focused_events) > 1:
            self.debug_log("Drawing connection arrows")
            self.draw_simple_connection_arrows(ax, focused_events_sorted, zero_level_event)
        
        # Create event breakdown statistics for the plot
        event_counts = focused_events_sorted['EventID'].value_counts().sort_index()
        total_events = len(focused_events_sorted)
        
        # Create breakdown text
        breakdown_lines = [f"Total Events: {total_events}"]
        event_descriptions = {
            1: 'Process Create', 2: 'File Creation Time', 3: 'Network Connection',
            4: 'Sysmon State Change', 5: 'Process Terminate', 6: 'Driver Load',
            7: 'Image Load', 8: 'Create Remote Thread', 9: 'Raw Access Read',
            10: 'Process Access', 11: 'File Create', 12: 'Registry Event',
            13: 'Registry Event', 15: 'File Create Stream Hash', 17: 'Pipe Create',
            18: 'Pipe Connect', 22: 'DNS Event', 23: 'File Delete', 24: 'Clipboard Change',
            25: 'Process Tampering'
        }
        
        for event_id, count in event_counts.items():
            desc = event_descriptions.get(event_id, 'Unknown')
            breakdown_lines.append(f"EventID {event_id}: {count} ({desc})")
        
        # Limit to top 8 most frequent events to avoid overcrowding
        if len(breakdown_lines) > 9:  # Total + 8 event types
            breakdown_lines = breakdown_lines[:9] + [f"... and {len(breakdown_lines)-9} more types"]
        
        breakdown_text = '\n'.join(breakdown_lines)
        
        # Customize the plot (original style)
        ax.set_xlabel('Time (HH:MM:SS.mmm)', fontsize=12)
        ax.set_ylabel('Event ID', fontsize=12)
        
        # Create title based on method type
        if method_type in ['type2a', 'type2b']:  # File operations
            title_command = f"File Create: {zero_level_event.get('TargetFilename', 'Unknown')}"
        else:  # Type 1 - terminal commands
            title_command = f"Command: {str(zero_level_event.get('CommandLine', 'Unknown'))[:80]}..."
        
        ax.set_title(f'Zero-Level Events Timeline - Entry {entry_id} (Adaptive Window: {total_span_seconds:.3f}s + 5% padding)\n'
                    f'Process: {zero_level_event["ProcessGuid"]}_{zero_level_event["ProcessId"]}\n'
                    f'{title_command}', 
                    fontsize=14, pad=20)
        
        # Set focused time window for X-axis
        ax.set_xlim(window_start, window_end)
        
        # Format time axis with appropriate precision based on adaptive window size
        total_window_seconds = (window_end - window_start).total_seconds()
        if total_window_seconds <= 10:
            # For very short windows, show milliseconds
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S.%f'))
            ax.xaxis.set_major_locator(mdates.SecondLocator(interval=1))
        elif total_window_seconds <= 60:
            # For medium windows, show seconds  
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            ax.xaxis.set_major_locator(mdates.SecondLocator(interval=5))
        else:
            # For longer windows, show minutes
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            ax.xaxis.set_major_locator(mdates.SecondLocator(interval=10))
        
        plt.xticks(rotation=45)
        
        # Set Y-axis to show only EventIDs that exist in focused events
        unique_event_ids = sorted(focused_events_sorted['EventID'].unique())
        ax.set_yticks(unique_event_ids)
        ax.set_ylim(min(unique_event_ids) - 0.5, max(unique_event_ids) + 0.5)
        
        # Add vertical line at the zero-level event time (RED REFERENCE LINE)
        ax.axvline(x=zero_level_time, color='red', linestyle='--', alpha=0.7, linewidth=2, 
                   label='Zero-Level Event Time')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Create legend for EventID colors
        legend_elements = []
        
        for event_id in unique_event_ids:
            color = event_colors.get(event_id, '#95A5A6')
            desc = event_descriptions.get(event_id, 'Unknown')
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                            markerfacecolor=color, markersize=8,
                                            markeredgecolor='black', markeredgewidth=1,
                                            label=f'EventID {event_id}: {desc}'))
        
        # Add zero-level marker to legend
        legend_elements.insert(0, plt.Line2D([0], [0], marker='*', color='w', 
                                           markerfacecolor='black', markersize=12,
                                           markeredgecolor='black', markeredgewidth=1,
                                           label='Zero-Level Event'))
        
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.7))
        
        # Add event breakdown text box outside the plot (below the legend)
        props = dict(boxstyle='round', facecolor='lightblue', alpha=0.9)
        ax.text(1.02, 0.3, breakdown_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"entry_{entry_id}_{method_type}_zero_level_{'arrows_' if show_arrows else ''}adaptive_{total_span_seconds:.3f}s.png"
        if CLI_ARGS and CLI_ARGS.output_dir:
            output_dir = CLI_ARGS.output_dir
        else:
            output_dir = get_config_value('output.directory', '5_entry-events-plots')
        plot_path = os.path.join(output_dir, plot_filename)
        plot_dpi = get_config_value('output.plot_dpi', 150)
        plt.savefig(plot_path, dpi=plot_dpi, bbox_inches='tight')
        plt.close()
        
        # Cleanup
        gc.collect()
        
        self.debug_log(f"Plot saved: {plot_path}")
        return plot_filename
    
    def draw_simple_connection_arrows(self, ax, focused_events_sorted: pd.DataFrame, zero_level_event: pd.Series):
        """Draw simple connection arrows from zero-level to all other events"""
        zero_time = pd.to_datetime(zero_level_event['UtcTime'])
        zero_eventid = zero_level_event['EventID']
        
        # Simple connection: zero-level to all other events
        other_events = focused_events_sorted[
            abs((pd.to_datetime(focused_events_sorted['UtcTime']) - zero_time).dt.total_seconds()) > 0.1
        ]
        
        arrow_count = 0
        for _, event in other_events.iterrows():
            event_time = pd.to_datetime(event['UtcTime'])
            event_id = event['EventID']
            
            # Draw green arrow from zero-level to this event
            ax.annotate('', xy=(event_time, event_id), xytext=(zero_time, zero_eventid),
                        arrowprops=dict(arrowstyle='->', color='darkgreen', alpha=0.6, lw=1.5))
            arrow_count += 1
        
        if arrow_count > 0:
            print(f"   🏹 Drew {arrow_count} connection arrows from zero-level to spawned events")
    
    def process_entry_batch(self, configs: List[EntryConfig], 
                          include_mask4: bool, show_arrows: bool) -> List[DetectionResult]:
        """Process batch of entries with optional parallel processing"""
        results = []
        
        print(f"\n🚀 PROCESSING BATCH OF {len(configs)} ENTRIES")
        print(f"   Mask 4 (recursive): {'ENABLED' if include_mask4 else 'DISABLED'}")
        print(f"   Connection arrows: {'ENABLED' if show_arrows else 'DISABLED'}")
        print(f"   Debug mode: {'ENABLED' if self.debug else 'DISABLED'}")
        if CLI_ARGS and CLI_ARGS.output_dir:
            output_dir = CLI_ARGS.output_dir
        else:
            output_dir = get_config_value('output.directory', '5_entry-events-plots')
        print(f"   Output directory: {output_dir}")
        
        # Check if parallel processing is enabled and beneficial
        enable_parallel = get_config_value('performance.enable_parallel_processing', True)
        use_multiprocessing = get_config_value('performance.use_multiprocessing', False)
        
        if enable_parallel and len(configs) > 1:
            print(f"   Parallel processing: ENABLED ({'processes' if use_multiprocessing else 'threads'})")
            try:
                return process_entries_parallel(self, configs, include_mask4, show_arrows, use_multiprocessing)
            except Exception as e:
                print(f"⚠️ Parallel processing failed: {e}")
                print(f"   Falling back to sequential processing...")
                # Fall through to sequential processing
        else:
            print(f"   Parallel processing: DISABLED (entries: {len(configs)})")
        
        # Sequential processing (fallback or when parallel is disabled)
        for i, config in enumerate(configs, 1):
            print(f"\n{'='*60}")
            print(f"🔍 PROCESSING ENTRY {config.entry_id} ({i}/{len(configs)})")
            print(f"   Computer: {config.computer_full}")
            print(f"   Method: {config.method_type} (EventID {config.totem_eventid})")
            print(f"{'='*60}")
            
            # Detect zero-level event
            result = self.detect_zero_level_event(config)
            
            if result.success:
                print(f"✅ Detection successful: {result.matches_found} match(es) found")
                
                # Apply appropriate analysis
                if config.method_type == "type1":
                    plot_filename = self.apply_type1_analysis(result, include_mask4, show_arrows)
                else:  # type2a or type2b
                    plot_filename = self.apply_type2_analysis(result, show_arrows)
                
                if plot_filename:
                    print(f"📊 Plot generated: {plot_filename}")
                else:
                    print(f"⚠️  Plot generation failed")
                    
            else:
                print(f"❌ Detection failed: {result.error_message}")
            
            results.append(result)
            
            # Memory cleanup
            gc.collect()
        
        return results
    
    def generate_summary_report(self, results: List[DetectionResult]):
        """Generate summary report of batch processing"""
        print(f"\n{'='*80}")
        print(f"📊 BATCH PROCESSING SUMMARY")
        print(f"{'='*80}")
        
        total_entries = len(results)
        successful = sum(1 for r in results if r.success)
        failed = total_entries - successful
        
        # Overall statistics
        print(f"Total entries processed: {total_entries}")
        print(f"Successful detections: {successful}")
        print(f"Failed detections: {failed}")
        print(f"Success rate: {successful/total_entries*100:.1f}%")
        
        # Method type breakdown
        method_stats = {}
        for result in results:
            method = result.method_type
            if method not in method_stats:
                method_stats[method] = {'total': 0, 'success': 0}
            method_stats[method]['total'] += 1
            if result.success:
                method_stats[method]['success'] += 1
        
        print(f"\n📋 Method type breakdown:")
        for method, stats in method_stats.items():
            success_rate = stats['success']/stats['total']*100 if stats['total'] > 0 else 0
            print(f"   {method}: {stats['success']}/{stats['total']} ({success_rate:.1f}%)")
        
        # Detailed results
        if successful > 0:
            print(f"\n✅ SUCCESSFUL DETECTIONS:")
            for result in results:
                if result.success:
                    print(f"   Entry {result.entry_config.entry_id}: {result.matches_found} match(es), "
                          f"{result.detection_time_ms:.1f}ms ({result.method_type})")
        
        if failed > 0:
            print(f"\n❌ FAILED DETECTIONS:")
            for result in results:
                if not result.success:
                    print(f"   Entry {result.entry_config.entry_id}: {result.error_message} ({result.method_type})")
        
        # Performance statistics
        avg_detection_time = np.mean([r.detection_time_ms for r in results])
        print(f"\n⏱️  Average detection time: {avg_detection_time:.1f}ms")


def load_sysmon_data() -> pd.DataFrame:
    """Load Sysmon data from CSV with comprehensive validation and error handling"""
    # Use command line argument if provided, otherwise fall back to config
    if CLI_ARGS and CLI_ARGS.sysmon_csv:
        sysmon_file = CLI_ARGS.sysmon_csv
    else:
        sysmon_file = get_config_value('data_sources.sysmon_file', 'sysmon-2025-05-04-000001.csv')
    
    with error_context("Sysmon data loading", file_path=sysmon_file):
        if DEBUG_MODE:
            print(f"[DEBUG] Loading Sysmon data from {sysmon_file}")
        
        # Validate file exists and is accessible
        validate_file_exists(sysmon_file, "Sysmon CSV file")
        
        # Define data types for optimization (only for columns that exist)
        dtype_spec = {
            'ProcessId': 'Int64',
            'SourcePort': 'Int64', 
            'DestinationPort': 'Int64',
            'SourceProcessId': 'Int64',
            'ParentProcessId': 'Int64',
            'SourceThreadId': 'Int64',
            'TargetProcessId': 'Int64',
            'ProcessGuid': 'string',
            'SourceProcessGUID': 'string',
            'TargetProcessGUID': 'string', 
            'ParentProcessGuid': 'string',
            'Computer': 'category',
            'Protocol': 'category',
            'EventType': 'category'
        }
        
        # Determine loading strategy based on file size and memory
        threshold_mb = get_config_value('performance.memory_threshold_mb', 1000)
        use_chunked = should_use_chunked_loading(sysmon_file, threshold_mb)
        
        if use_chunked and get_config_value('performance.memory_optimization', True):
            print(f"📊 Large file detected, using chunked loading for memory efficiency")
            df = load_sysmon_data_chunked(sysmon_file)
        else:
            # Standard loading with retry logic and error handling
            df = safe_operation_with_retry(
                _load_csv_file,
                sysmon_file,
                dtype_spec,
                max_retries=2
            )
            
            # Convert timestamps with error handling
            try:
                df['UtcTime'] = pd.to_datetime(df['UtcTime'], errors='coerce')
            except Exception as e:
                raise DataValidationError(f"Failed to process timestamps: {e}")
            
            # Apply canonical normalization to CommandLine for Type 1 matching
            if 'CommandLine' in df.columns:
                try:
                    if DEBUG_MODE:
                        print(f"[DEBUG] Applying canonical normalization to CommandLine ({len(df)} rows)")
                        sample_commands = df['CommandLine'].dropna().head(3).tolist()
                        print(f"[DEBUG] Sample commands: {sample_commands}")
                    df['cmd_norm'] = df['CommandLine'].map(canonical)
                    if DEBUG_MODE:
                        print(f"[DEBUG] Canonical normalization completed successfully")
                except Exception as e:
                    if DEBUG_MODE:
                        print(f"[DEBUG] Error during canonical normalization: {type(e).__name__}: {e}")
                    raise DataValidationError(f"Failed to normalize command lines: {e}")
            else:
                if DEBUG_MODE:
                    print("[DEBUG] CommandLine column not found, skipping normalization")
                df['cmd_norm'] = ""
            
            # Optimize memory usage
            if get_config_value('performance.memory_optimization', True):
                if DEBUG_MODE:
                    print("[DEBUG] Starting memory optimization")
                df = optimize_dataframe_memory(df, "Sysmon data loading")
                if DEBUG_MODE:
                    print("[DEBUG] Memory optimization completed")
        
        # Validate data quality (for both loading methods)
        if DEBUG_MODE:
            print("[DEBUG] Starting data validation")
        validation_results = validate_sysmon_data(df)
        if DEBUG_MODE:
            print("[DEBUG] Data validation completed")
        
        # Print results with validation summary
        print(f"✅ Sysmon data loaded: {len(df)} rows")
        if 'UtcTime' in df.columns:
            print(f"   Time range: {df['UtcTime'].min()} to {df['UtcTime'].max()}")
        if 'EventID' in df.columns:
            print(f"   Event types: {sorted(df['EventID'].unique())}")
        
        # Report validation warnings if any
        if validation_results.get("warnings"):
            print(f"   ⚠️  Data quality warnings: {len(validation_results['warnings'])}")
            for warning in validation_results["warnings"][:3]:  # Show first 3 warnings
                print(f"      - {warning['message']}")
            if len(validation_results["warnings"]) > 3:
                print(f"      - ... and {len(validation_results['warnings']) - 3} more warnings")
        
        return df

def _load_csv_file(file_path: str, dtype_spec: dict) -> pd.DataFrame:
    """Helper function to load CSV file (used by retry logic)"""
    try:
        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            raise DataValidationError(f"CSV file is empty: {file_path}")
        
        if file_size > 5 * 1024 * 1024 * 1024:  # 5GB limit
            raise DataValidationError(
                f"CSV file too large: {file_size / (1024**3):.1f}GB. Consider using chunked processing."
            )
        
        # First, read a few rows to determine which columns actually exist
        sample_df = pd.read_csv(file_path, nrows=1)
        available_columns = set(sample_df.columns)
        
        # Filter dtype_spec to only include columns that exist
        filtered_dtype_spec = {
            col: dtype for col, dtype in dtype_spec.items() 
            if col in available_columns
        }
        
        if DEBUG_MODE and len(filtered_dtype_spec) < len(dtype_spec):
            missing_cols = set(dtype_spec.keys()) - available_columns
            print(f"[DEBUG] Skipping dtype for missing columns: {missing_cols}")
        
        # Load CSV with filtered dtype specification
        df = pd.read_csv(file_path, low_memory=False, dtype=filtered_dtype_spec)
        
        if len(df) == 0:
            raise DataValidationError("CSV file contains no data rows")
        
        return df
        
    except pd.errors.EmptyDataError:
        raise DataValidationError("CSV file is empty or contains no valid data")
    except pd.errors.ParserError as e:
        raise DataValidationError(f"CSV parsing error: {e}")
    except MemoryError:
        raise DataValidationError("Insufficient memory to load CSV file. Consider using chunked processing.")
    except Exception as e:
        raise FileProcessingError(f"Failed to load CSV file: {e}")


def load_caldera_data() -> Dict:
    """Load Caldera entries data with comprehensive error handling"""
    # Use command line argument if provided, otherwise fall back to config
    if CLI_ARGS and CLI_ARGS.caldera_json:
        caldera_file = CLI_ARGS.caldera_json
    else:
        caldera_file = get_config_value('data_sources.caldera_file', "apt34-05-04-test-1_event-logs_extracted_information.json")
    
    with error_context("Caldera data loading", file_path=caldera_file):
        if DEBUG_MODE:
            print(f"[DEBUG] Loading Caldera data from {caldera_file}")
        
        # Validate file exists and is accessible
        validate_file_exists(caldera_file, "Caldera JSON file")
        
        # Load JSON with retry logic and validation
        data = safe_operation_with_retry(
            _load_json_file,
            caldera_file,
            max_retries=2
        )
        
        # Validate Caldera data structure
        _validate_caldera_data(data)
        
        print(f"✅ Caldera data loaded: {len(data)} entries")
        print(f"   Entry IDs: {sorted(data.keys())}")
        
        return data

def _load_json_file(file_path: str) -> Dict:
    """Helper function to load JSON file (used by retry logic)"""
    try:
        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            raise DataValidationError(f"JSON file is empty: {file_path}")
        
        if file_size > 100 * 1024 * 1024:  # 100MB limit for JSON
            raise DataValidationError(
                f"JSON file too large: {file_size / (1024**2):.1f}MB"
            )
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not data:
            raise DataValidationError("JSON file contains no data")
        
        return data
        
    except json.JSONDecodeError as e:
        raise DataValidationError(f"Invalid JSON format: {e}")
    except UnicodeDecodeError as e:
        raise DataValidationError(f"JSON file encoding error: {e}")
    except Exception as e:
        raise FileProcessingError(f"Failed to load JSON file: {e}")

def _validate_caldera_data(data: Dict) -> None:
    """Validate Caldera data structure and content"""
    if not isinstance(data, dict):
        raise DataValidationError("Caldera data must be a dictionary/object")
    
    if len(data) == 0:
        raise DataValidationError("Caldera data contains no entries")
    
    # Validate structure of first few entries
    sample_entries = list(data.items())[:5]  # Check first 5 entries
    required_fields = ['finished_timestamp', 'new_command']
    
    for entry_id, entry_data in sample_entries:
        if not isinstance(entry_data, dict):
            raise DataValidationError(f"Entry {entry_id} is not a dictionary")
        
        missing_fields = [field for field in required_fields if field not in entry_data]
        if missing_fields:
            raise DataValidationError(
                f"Entry {entry_id} missing required fields: {missing_fields}",
                details={"entry_id": entry_id, "missing_fields": missing_fields}
            )
        
        # Validate timestamp format
        timestamp = entry_data.get('finished_timestamp')
        if timestamp:
            try:
                pd.to_datetime(timestamp)
            except Exception:
                raise DataValidationError(
                    f"Entry {entry_id} has invalid timestamp format: {timestamp}",
                    details={"entry_id": entry_id, "timestamp": timestamp}
                )


def interactive_prompts() -> Tuple[bool, bool, bool]:
    """Get user preferences for interactive options"""
    print(f"\n🔧 INTERACTIVE CONFIGURATION")
    
    # Debug mode
    debug_default = get_config_value('debug.enable_debug_mode', False)
    debug_prompt = f"Enable debug mode? ({'Y/n' if debug_default else 'y/N'}): "
    debug_input = input(debug_prompt).strip().lower()
    if debug_input == '':
        debug_mode = debug_default
    else:
        debug_mode = debug_input in ['y', 'yes']
    
    # Mask 4 inclusion (recursive tracing)
    recursive_default = get_config_value('analysis.enable_recursive_tracing', True)
    recursive_prompt = f"Include recursive tracing? ({'Y/n' if recursive_default else 'y/N'}): "
    mask4_input = input(recursive_prompt).strip().lower()
    if mask4_input == '':
        include_mask4 = recursive_default
    else:
        include_mask4 = mask4_input in ['y', 'yes']
    
    # Arrow inclusion
    arrows_default = get_config_value('visualization.default_show_arrows', False)
    arrows_prompt = f"Include connection arrows in plots? ({'Y/n' if arrows_default else 'y/N'}): "
    arrows_input = input(arrows_prompt).strip().lower()
    if arrows_input == '':
        show_arrows = arrows_default
    else:
        show_arrows = arrows_input in ['y', 'yes']
    
    return debug_mode, include_mask4, show_arrows


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Event Tracer v2 - Production Implementation for Sysmon Event Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s --caldera_json /path/to/caldera.json --sysmon_csv /path/to/sysmon.csv --output_dir /path/to/output
  %(prog)s --caldera_json ./apt34-02-07-test-1_event-logs_extracted_information.json --sysmon_csv ./sysmon-2025-02-07-000001.csv --output_dir ./5_entry-events-plots
'''
    )
    
    parser.add_argument('--caldera_json', 
                       help='Path to Caldera JSON file with extracted information',
                       required=True)
    parser.add_argument('--sysmon_csv', 
                       help='Path to Sysmon CSV file',
                       required=True)
    parser.add_argument('--output_dir', 
                       help='Output directory for plots and results',
                       required=True)
    parser.add_argument('--config', 
                       help='Path to configuration YAML file (default: config.yaml)',
                       default='config.yaml')
    parser.add_argument('--debug', 
                       action='store_true',
                       help='Enable debug mode')
    
    return parser.parse_args()

def main():
    """Main entry point"""
    global DEBUG_MODE, config, CLI_ARGS
    
    print("🎯 Event Tracer v2 - Production Implementation")
    print("=" * 60)
    
    # Parse command line arguments
    CLI_ARGS = parse_arguments()
    
    try:
        # Load configuration first with comprehensive error handling
        with error_context("application startup"):
            print(f"\n🔧 Loading configuration...")
            config = load_config(CLI_ARGS.config)
            
            # Interactive configuration
            debug_mode, include_mask4, show_arrows = interactive_prompts()
            DEBUG_MODE = debug_mode or get_config_value('debug.enable_debug_mode', False) or CLI_ARGS.debug
            
            # Print configuration summary
            print(f"\n📋 Configuration Summary:")
            print(f"   Caldera JSON: {CLI_ARGS.caldera_json}")
            print(f"   Sysmon CSV: {CLI_ARGS.sysmon_csv}")
            print(f"   Output Directory: {CLI_ARGS.output_dir}")
            print(f"   Debug Mode: {'ENABLED' if DEBUG_MODE else 'DISABLED'}")
        
        # Load data with error handling and retry logic
        with error_context("data loading phase"):
            print(f"\n🔧 Loading data...")
            sysmon_df = safe_operation_with_retry(load_sysmon_data, max_retries=2)
            caldera_entries = safe_operation_with_retry(load_caldera_data, max_retries=2)
            
            # Initialize tracer
            tracer = EventTracer(sysmon_df, caldera_entries, debug=debug_mode)
        
        # Load batch configuration with error handling
        with error_context("batch configuration loading"):
            print(f"\n📋 Loading batch configuration...")
            configs = tracer.load_batch_configuration()
        
        # Process batch with graceful error handling per entry
        with error_context("batch processing", total_entries=len(configs)):
            print(f"\n🚀 Starting batch processing of {len(configs)} entries...")
            results = tracer.process_entry_batch(configs, include_mask4, show_arrows)
        
        # Generate summary with error handling
        with error_context("report generation"):
            tracer.generate_summary_report(results)
        
        print(f"\n🎉 Batch processing completed!")
        if CLI_ARGS and CLI_ARGS.output_dir:
            output_dir = CLI_ARGS.output_dir
        else:
            output_dir = get_config_value('output.directory', '5_entry-events-plots')
        print(f"   Output directory: {output_dir}")
        
    except ConfigurationError as e:
        print(f"\n❌ Configuration Error: {e}")
        print("   Please check your config.yaml file and fix the issues above.")
        sys.exit(1)
        
    except DataValidationError as e:
        print(f"\n❌ Data Validation Error: {e}")
        print("   Please check your data files for corruption or format issues.")
        if DEBUG_MODE and hasattr(e, 'details'):
            print(f"   Details: {e.details}")
        sys.exit(1)
        
    except FileProcessingError as e:
        print(f"\n❌ File Processing Error: {e}")
        print("   Please check that all required files exist and are accessible.")
        sys.exit(1)
        
    except EventTracerException as e:
        print(f"\n❌ Event Tracer Error: {e}")
        if DEBUG_MODE and hasattr(e, 'details'):
            print(f"   Details: {e.details}")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("   This may be a bug in the application. Please report this issue.")
        if DEBUG_MODE:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()