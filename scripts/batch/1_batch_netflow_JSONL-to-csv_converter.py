#!/usr/bin/env python3
"""
Batch Network Traffic Flow JSONL to CSV Converter

This script provides a comprehensive batch processing interface for converting network traffic 
flow JSONL files to CSV format across multiple APT runs using the 3_network_traffic_csv_creator.py script.

Workflow:
1. Locates compressed JSONL files in dataset/dataset-backup/
2. Extracts ds-logs-network_traffic-flow-default-run-X.jsonl.gz to dataset/apt-Y/apt-Y-run-X/
3. Runs 3_network_traffic_csv_creator.py on each APT directory
4. Generates network_traffic_flow-run-X.csv and processing logs

Features:
- Automatic extraction of compressed JSONL files from backup directory
- Process all runs across all APT types (1-6)
- Process specific APT types (e.g., apt-1, apt-2)
- Process specific run ranges (e.g., runs 04-10)
- Process individual runs
- Parallel processing support
- Performance optimizations (fast mode)
- Comprehensive progress reporting with extraction and conversion timing
- Error handling and recovery
- Two-step processing: extraction → conversion

APT Structure:
- APT-1: Runs 04-20, 51
- APT-2: Runs 21-30  
- APT-3: Runs 31-38
- APT-4: Runs 39-44
- APT-5: Runs 45-47
- APT-6: Runs 48-50

Usage Examples:
    # Process all runs
    python3 1_batch_netflow_JSONL-to-csv_converter.py --all
    
    # Process specific APT types
    python3 1_batch_netflow_JSONL-to-csv_converter.py --apt-types apt-1,apt-2
    
    # Process specific run range
    python3 1_batch_netflow_JSONL-to-csv_converter.py --runs 04-10
    
    # Process individual runs
    python3 1_batch_netflow_JSONL-to-csv_converter.py --runs 04,05,51
    
    # High-performance parallel processing
    python3 1_batch_netflow_JSONL-to-csv_converter.py --all --parallel 4 --fast
"""

import argparse
import os
import sys
import subprocess
import time
import gzip
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import json

class BatchNetflowConverter:
    """Batch processor for network traffic flow JSONL to CSV conversion."""
    
    # APT run mappings based on directory structure
    APT_MAPPINGS = {
        'apt-1': list(range(4, 21)) + [51],  # 04-20, 51
        'apt-2': list(range(21, 31)),        # 21-30
        'apt-3': list(range(31, 39)),        # 31-38
        'apt-4': list(range(39, 45)),        # 39-44
        'apt-5': list(range(45, 48)),        # 45-47
        'apt-6': list(range(48, 51)),        # 48-50
    }
    
    def __init__(self, base_dir: str = None, parallel_workers: int = 1, fast_mode: bool = False, 
                 dry_run: bool = False, cleanup: bool = False):
        """
        Initialize batch converter.
        
        Args:
            base_dir: Base directory (auto-detected if None)
            parallel_workers: Number of parallel processes (1 = sequential)
            fast_mode: Enable performance optimizations
            dry_run: Show what would be done without executing
            cleanup: Delete uncompressed JSONL files after successful CSV creation
        """
        # Auto-detect base directory
        if base_dir is None:
            script_dir = Path(__file__).parent.resolve()
            self.base_dir = script_dir.parent.parent.parent  # batch -> scripts -> dataset -> research
        else:
            self.base_dir = Path(base_dir).resolve()
        
        self.dataset_dir = self.base_dir / "dataset"
        self.backup_dir = self.dataset_dir / "dataset-backup"
        self.converter_script = self.dataset_dir / "scripts" / "pipeline" / "3_network_traffic_csv_creator.py"
        
        self.parallel_workers = parallel_workers
        self.fast_mode = fast_mode
        self.dry_run = dry_run
        self.cleanup = cleanup
        
        # Processing statistics
        self.stats = {
            'total_runs': 0,
            'successful_runs': 0,
            'failed_runs': 0,
            'skipped_runs': 0,
            'start_time': None,
            'end_time': None
        }
        
        # Verify directories and scripts exist
        self._verify_setup()
        
        print(f"🔧 Initialized Batch Network Traffic Converter")
        print(f"📁 Base directory: {self.base_dir}")
        print(f"📊 Dataset directory: {self.dataset_dir}")
        print(f"💾 Backup directory: {self.backup_dir}")
        print(f"🔀 Converter script: {self.converter_script.name}")
        print(f"⚡ Parallel workers: {self.parallel_workers}")
        print(f"🚀 Fast mode: {'ON' if self.fast_mode else 'OFF'}")
        print(f"🗑️ Cleanup mode: {'ON' if self.cleanup else 'OFF'}")
        if self.dry_run:
            print("🔍 DRY RUN MODE: No processing will occur")
    
    def _verify_setup(self):
        """Verify required directories and scripts exist."""
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")
        
        if not self.backup_dir.exists():
            raise FileNotFoundError(f"Backup directory not found: {self.backup_dir}")
        
        if not self.converter_script.exists():
            raise FileNotFoundError(f"Converter script not found: {self.converter_script}")
        
        print("✅ Setup verification passed")
    
    def parse_run_specification(self, run_spec: str) -> List[int]:
        """
        Parse run specification into list of run numbers.
        
        Args:
            run_spec: Run specification (e.g., "04,05,06" or "04-10" or "04-10,51")
            
        Returns:
            List of run numbers
        """
        runs = []
        
        for part in run_spec.split(','):
            part = part.strip()
            if '-' in part and not part.startswith('-'):
                # Range specification (e.g., "04-10")
                start_str, end_str = part.split('-', 1)
                start = int(start_str)
                end = int(end_str)
                runs.extend(range(start, end + 1))
            else:
                # Individual run (e.g., "04" or "51")
                runs.append(int(part))
        
        return sorted(list(set(runs)))  # Remove duplicates and sort
    
    def get_runs_to_process(self, apt_types: Optional[List[str]] = None, 
                           run_numbers: Optional[List[int]] = None, 
                           all_runs: bool = False) -> List[Tuple[str, int]]:
        """
        Get list of (apt_type, run_number) tuples to process.
        
        Args:
            apt_types: List of APT types to include
            run_numbers: List of specific run numbers to include
            all_runs: Process all available runs
            
        Returns:
            List of (apt_type, run_number) tuples
        """
        runs_to_process = []
        
        # Determine APT types to process
        if all_runs or not apt_types:
            apt_types_to_check = list(self.APT_MAPPINGS.keys())
        else:
            apt_types_to_check = apt_types
        
        # Build run list
        for apt_type in apt_types_to_check:
            if apt_type not in self.APT_MAPPINGS:
                print(f"⚠️  Warning: Unknown APT type '{apt_type}', skipping")
                continue
            
            available_runs = self.APT_MAPPINGS[apt_type]
            
            if run_numbers:
                # Filter by specified run numbers
                runs_for_this_apt = [run for run in available_runs if run in run_numbers]
            else:
                # Use all available runs for this APT type
                runs_for_this_apt = available_runs
            
            # Add to processing list
            for run_num in runs_for_this_apt:
                runs_to_process.append((apt_type, run_num))
        
        return sorted(runs_to_process)
    
    def extract_compressed_jsonl(self, apt_type: str, run_number: int) -> Tuple[bool, str]:
        """
        Extract compressed JSONL file from dataset-backup to APT run directory.
        
        Args:
            apt_type: APT type (e.g., 'apt-1')
            run_number: Run number
            
        Returns:
            Tuple of (success, error_message)
        """
        run_str = f"{run_number:02d}"
        
        # Source compressed file
        compressed_file = self.backup_dir / f"ds-logs-network_traffic-flow-default-run-{run_str}.jsonl.gz"
        
        # Target directory and file
        target_dir = self.dataset_dir / apt_type / f"{apt_type}-run-{run_str}"
        target_file = target_dir / f"ds-logs-network_traffic-flow-default-run-{run_str}.jsonl"
        
        # Check if source compressed file exists
        if not compressed_file.exists():
            return False, f"Compressed file not found: {compressed_file}"
        
        # Check if target directory exists
        if not target_dir.exists():
            return False, f"Target directory not found: {target_dir}"
        
        # Check if target file already exists
        if target_file.exists():
            file_size_mb = target_file.stat().st_size / 1024**2
            print(f"   ⚠️  Target JSONL already exists ({file_size_mb:.1f} MB) - skipping extraction")
            return True, "Already extracted"
        
        if self.dry_run:
            print(f"   🔍 DRY RUN: Would extract {compressed_file.name} to {target_file.name}")
            return True, "Dry run - would extract"
        
        try:
            print(f"   📦 Extracting {compressed_file.name} ({compressed_file.stat().st_size / 1024**2:.1f} MB)")
            
            # Extract compressed file
            with gzip.open(compressed_file, 'rb') as f_in:
                with open(target_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Verify extraction
            if target_file.exists():
                file_size_mb = target_file.stat().st_size / 1024**2
                print(f"   ✅ Extraction successful ({file_size_mb:.1f} MB)")
                return True, "Extraction successful"
            else:
                return False, "Target file not created after extraction"
                
        except Exception as e:
            return False, f"Extraction error: {str(e)}"
    
    def check_run_prerequisites(self, apt_type: str, run_number: int) -> Tuple[bool, Path, str]:
        """
        Check if APT run has all prerequisites for processing.
        
        Args:
            apt_type: APT type (e.g., 'apt-1')
            run_number: Run number
            
        Returns:
            Tuple of (has_prerequisites, run_directory, error_message)
        """
        run_str = f"{run_number:02d}"
        run_dir = self.dataset_dir / apt_type / f"{apt_type}-run-{run_str}"
        
        # Check if run directory exists
        if not run_dir.exists():
            return False, run_dir, f"Run directory not found: {run_dir}"
        
        # Check for compressed file in backup
        compressed_file = self.backup_dir / f"ds-logs-network_traffic-flow-default-run-{run_str}.jsonl.gz"
        if not compressed_file.exists():
            return False, run_dir, f"Compressed JSONL file not found in backup: {compressed_file.name}"
        
        return True, run_dir, "Prerequisites met"
    
    def cleanup_jsonl_file(self, apt_type: str, run_number: int) -> Tuple[bool, str]:
        """
        Delete uncompressed JSONL file after successful CSV creation.
        
        Args:
            apt_type: APT type (e.g., 'apt-1')
            run_number: Run number
            
        Returns:
            Tuple of (success, message)
        """
        run_str = f"{run_number:02d}"
        run_dir = self.dataset_dir / apt_type / f"{apt_type}-run-{run_str}"
        jsonl_file = run_dir / f"ds-logs-network_traffic-flow-default-run-{run_str}.jsonl"
        
        if self.dry_run:
            if jsonl_file.exists():
                file_size_mb = jsonl_file.stat().st_size / 1024**2
                print(f"   🔍 DRY RUN: Would delete {jsonl_file.name} ({file_size_mb:.1f} MB)")
                return True, "Dry run - would delete"
            else:
                return True, "Dry run - file not found"
        
        try:
            if jsonl_file.exists():
                file_size_mb = jsonl_file.stat().st_size / 1024**2
                print(f"   🗑️ Cleaning up: Deleting {jsonl_file.name} ({file_size_mb:.1f} MB)")
                
                # Safety check: ensure we're only deleting uncompressed JSONL files
                if jsonl_file.suffix == '.jsonl' and not str(jsonl_file).endswith('.gz'):
                    jsonl_file.unlink()
                    print(f"   ✅ Cleanup successful: {file_size_mb:.1f} MB freed")
                    return True, f"Deleted {file_size_mb:.1f} MB"
                else:
                    return False, "Safety check failed: Not a .jsonl file"
            else:
                return True, "File not found (already cleaned or never extracted)"
                
        except Exception as e:
            return False, f"Cleanup error: {str(e)}"
    
    def process_single_run(self, apt_type: str, run_number: int) -> Dict[str, any]:
        """
        Process a single APT run: extract JSONL and convert to CSV.
        
        Args:
            apt_type: APT type (e.g., 'apt-1')
            run_number: Run number
            
        Returns:
            Processing result dictionary
        """
        run_str = f"{run_number:02d}"
        run_id = f"{apt_type}-run-{run_str}"
        
        result = {
            'run_id': run_id,
            'apt_type': apt_type,
            'run_number': run_number,
            'success': False,
            'error': None,
            'processing_time': 0,
            'extraction_time': 0,
            'conversion_time': 0,
            'cleanup_time': 0,
            'output_size_mb': 0,
            'extracted_size_mb': 0,
            'cleaned_size_mb': 0
        }
        
        start_time = time.time()
        
        try:
            # Step 1: Check prerequisites
            has_prereqs, run_dir, prereq_error = self.check_run_prerequisites(apt_type, run_number)
            if not has_prereqs:
                result['error'] = prereq_error
                return result
            
            if self.dry_run:
                result['success'] = True
                result['processing_time'] = 0.1
                return result
            
            # Step 2: Extract compressed JSONL file
            print(f"   📦 Step 1/2: Extracting compressed JSONL")
            extraction_start = time.time()
            
            extraction_success, extraction_error = self.extract_compressed_jsonl(apt_type, run_number)
            if not extraction_success:
                result['error'] = f"Extraction failed: {extraction_error}"
                return result
            
            result['extraction_time'] = time.time() - extraction_start
            
            # Check extracted file size
            jsonl_file = run_dir / f"ds-logs-network_traffic-flow-default-run-{run_str}.jsonl"
            if jsonl_file.exists():
                result['extracted_size_mb'] = jsonl_file.stat().st_size / 1024**2
            
            # Step 3: Run CSV conversion
            print(f"   🔄 Step 2/2: Converting JSONL to CSV")
            conversion_start = time.time()
            
            # Prepare command for 3_network_traffic_csv_creator.py
            cmd = [
                sys.executable,
                str(self.converter_script),
                '--apt-dir', str(run_dir.relative_to(self.base_dir))
            ]
            
            # Add performance flags if fast mode is enabled
            if self.fast_mode:
                cmd.extend(['--no-validate', '--no-analysis'])
            
            # Change to base directory for execution
            original_cwd = os.getcwd()
            os.chdir(self.base_dir)
            
            try:
                # Execute converter script
                process = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=3600  # 1 hour timeout
                )
                
                result['conversion_time'] = time.time() - conversion_start
                
                if process.returncode == 0:
                    result['success'] = True
                    
                    # Check output CSV file size
                    csv_file = run_dir / f"netflow-run-{run_str}.csv"
                    if csv_file.exists():
                        result['output_size_mb'] = csv_file.stat().st_size / 1024**2
                    
                    print(f"   ✅ Conversion successful")
                    
                    # Step 4: Cleanup uncompressed JSONL file if requested
                    if self.cleanup:
                        print(f"   🗑️ Step 3/3: Cleaning up uncompressed JSONL")
                        cleanup_start = time.time()
                        
                        cleanup_success, cleanup_message = self.cleanup_jsonl_file(apt_type, run_number)
                        result['cleanup_time'] = time.time() - cleanup_start
                        
                        if cleanup_success:
                            # Extract cleaned size from message if available
                            if "Deleted" in cleanup_message and "MB" in cleanup_message:
                                import re
                                size_match = re.search(r'(\d+\.?\d*)\s*MB', cleanup_message)
                                if size_match:
                                    result['cleaned_size_mb'] = float(size_match.group(1))
                        else:
                            print(f"   ⚠️ Cleanup warning: {cleanup_message}")
                
                else:
                    result['error'] = f"CSV conversion failed (exit code {process.returncode}): {process.stderr[:500]}"
                    print(f"   ❌ Conversion failed: {result['error'][:100]}...")
            
            finally:
                os.chdir(original_cwd)
        
        except subprocess.TimeoutExpired:
            result['error'] = "Processing timeout (1 hour limit exceeded)"
        except Exception as e:
            result['error'] = str(e)
        
        result['processing_time'] = time.time() - start_time
        return result
    
    def process_runs_sequential(self, runs_to_process: List[Tuple[str, int]]) -> List[Dict]:
        """Process runs sequentially."""
        results = []
        
        for i, (apt_type, run_number) in enumerate(runs_to_process, 1):
            run_str = f"{run_number:02d}"
            print(f"\n📈 Progress: {i}/{len(runs_to_process)} - Processing {apt_type}-run-{run_str}")
            print("-" * 60)
            
            result = self.process_single_run(apt_type, run_number)
            results.append(result)
            
            # Print immediate result
            if result['success']:
                print(f"   ✅ {result['run_id']} completed in {result['processing_time']:.1f}s")
                if result['extraction_time'] > 0:
                    print(f"   📦 Extraction: {result['extraction_time']:.1f}s ({result['extracted_size_mb']:.1f} MB)")
                if result['conversion_time'] > 0:
                    print(f"   🔄 Conversion: {result['conversion_time']:.1f}s")
                if result['cleanup_time'] > 0:
                    print(f"   🗑️ Cleanup: {result['cleanup_time']:.1f}s ({result['cleaned_size_mb']:.1f} MB freed)")
                if result['output_size_mb'] > 0:
                    print(f"   📁 CSV Output: {result['output_size_mb']:.1f} MB")
            else:
                print(f"   ❌ {result['run_id']} failed: {result['error']}")
        
        return results
    
    def process_runs_parallel(self, runs_to_process: List[Tuple[str, int]]) -> List[Dict]:
        """Process runs in parallel."""
        results = []
        
        print(f"🚀 Starting parallel processing with {self.parallel_workers} workers")
        
        with ProcessPoolExecutor(max_workers=self.parallel_workers) as executor:
            # Submit all jobs
            future_to_run = {
                executor.submit(self.process_single_run, apt_type, run_number): (apt_type, run_number)
                for apt_type, run_number in runs_to_process
            }
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_run):
                apt_type, run_number = future_to_run[future]
                completed += 1
                
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Print progress
                    progress_pct = (completed / len(runs_to_process)) * 100
                    print(f"📈 Progress: {completed}/{len(runs_to_process)} ({progress_pct:.1f}%) - {result['run_id']}")
                    
                    if result['success']:
                        print(f"   ✅ Completed in {result['processing_time']:.1f}s")
                        timing_parts = []
                        if result.get('extraction_time', 0) > 0:
                            timing_parts.append(f"📦 Extract: {result['extraction_time']:.1f}s")
                        if result.get('conversion_time', 0) > 0:
                            timing_parts.append(f"🔄 Convert: {result['conversion_time']:.1f}s")
                        if result.get('cleanup_time', 0) > 0:
                            timing_parts.append(f"🗑️ Clean: {result['cleanup_time']:.1f}s")
                        if timing_parts:
                            print(f"   {' | '.join(timing_parts)}")
                        if result.get('cleaned_size_mb', 0) > 0:
                            print(f"   📁 CSV: {result['output_size_mb']:.1f} MB | 🗑️ Freed: {result['cleaned_size_mb']:.1f} MB")
                        elif result['output_size_mb'] > 0:
                            print(f"   📁 CSV: {result['output_size_mb']:.1f} MB")
                    else:
                        print(f"   ❌ Failed: {result['error'][:100]}{'...' if len(result['error']) > 100 else ''}")
                
                except Exception as e:
                    print(f"   ❌ {apt_type}-run-{run_number:02d} exception: {str(e)}")
                    results.append({
                        'run_id': f"{apt_type}-run-{run_number:02d}",
                        'apt_type': apt_type,
                        'run_number': run_number,
                        'success': False,
                        'error': str(e),
                        'processing_time': 0,
                        'output_size_mb': 0
                    })
        
        return results
    
    def generate_summary_report(self, results: List[Dict]):
        """Generate and display summary report."""
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        
        total_time = sum(r['processing_time'] for r in results)
        total_output_mb = sum(r['output_size_mb'] for r in successful)
        
        print("\n" + "=" * 80)
        print("🎉 BATCH CONVERSION COMPLETED")
        print("=" * 80)
        print(f"🕐 Start time: {self.stats['start_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🕐 End time: {self.stats['end_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏱️ Total duration: {self.stats['end_time'] - self.stats['start_time']}")
        print(f"📊 Total runs processed: {len(results)}")
        print(f"✅ Successful conversions: {len(successful)}")
        print(f"❌ Failed conversions: {len(failed)}")
        print(f"📈 Success rate: {(len(successful)/max(len(results), 1))*100:.1f}%")
        
        if successful:
            avg_time = sum(r['processing_time'] for r in successful) / len(successful)
            avg_extract_time = sum(r.get('extraction_time', 0) for r in successful) / len(successful)
            avg_convert_time = sum(r.get('conversion_time', 0) for r in successful) / len(successful)
            avg_cleanup_time = sum(r.get('cleanup_time', 0) for r in successful) / len(successful)
            total_extracted_mb = sum(r.get('extracted_size_mb', 0) for r in successful)
            total_cleaned_mb = sum(r.get('cleaned_size_mb', 0) for r in successful)
            
            print(f"⚡ Average processing time: {avg_time:.1f}s per run")
            print(f"📦 Average extraction time: {avg_extract_time:.1f}s per run")
            print(f"🔄 Average conversion time: {avg_convert_time:.1f}s per run")
            if avg_cleanup_time > 0:
                print(f"🗑️ Average cleanup time: {avg_cleanup_time:.1f}s per run")
            print(f"📁 Total extracted size: {total_extracted_mb:.1f} MB")
            print(f"📁 Total CSV output size: {total_output_mb:.1f} MB")
            if total_cleaned_mb > 0:
                print(f"🗑️ Total space freed: {total_cleaned_mb:.1f} MB ({total_cleaned_mb/1024:.1f} GB)")
            print(f"🎯 Processing throughput: {len(successful)/(total_time/max(self.parallel_workers,1)):.2f} runs/minute")
        
        # Show failed runs
        if failed:
            print(f"\n❌ FAILED RUNS ({len(failed)}):")
            for result in failed[:10]:  # Show first 10 failures
                error_short = result['error'][:100] + '...' if len(result['error']) > 100 else result['error']
                print(f"   • {result['run_id']}: {error_short}")
            if len(failed) > 10:
                print(f"   ... and {len(failed) - 10} more failures")
        
        # APT type breakdown
        apt_stats = {}
        for result in results:
            apt_type = result['apt_type']
            if apt_type not in apt_stats:
                apt_stats[apt_type] = {'total': 0, 'successful': 0}
            apt_stats[apt_type]['total'] += 1
            if result['success']:
                apt_stats[apt_type]['successful'] += 1
        
        print(f"\n📊 APT TYPE BREAKDOWN:")
        for apt_type, stats in sorted(apt_stats.items()):
            success_rate = (stats['successful'] / stats['total']) * 100
            print(f"   • {apt_type.upper()}: {stats['successful']}/{stats['total']} ({success_rate:.1f}%)")
    
    def run_batch_conversion(self, apt_types: Optional[List[str]] = None,
                           run_numbers: Optional[List[int]] = None,
                           all_runs: bool = False) -> bool:
        """
        Run batch conversion across specified runs.
        
        Args:
            apt_types: List of APT types to process
            run_numbers: List of run numbers to process
            all_runs: Process all available runs
            
        Returns:
            True if all conversions successful, False otherwise
        """
        self.stats['start_time'] = datetime.now()
        
        # Get runs to process
        runs_to_process = self.get_runs_to_process(apt_types, run_numbers, all_runs)
        
        if not runs_to_process:
            print("❌ No runs found to process!")
            return False
        
        print(f"🚀 Starting Batch Network Traffic Flow JSONL to CSV Conversion")
        print(f"📅 Start time: {self.stats['start_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 Runs to process: {len(runs_to_process)}")
        
        if apt_types:
            print(f"🎯 APT types: {', '.join(apt_types)}")
        if run_numbers:
            print(f"🎯 Run numbers: {', '.join(map(str, run_numbers))}")
        if all_runs:
            print(f"🎯 Processing: ALL AVAILABLE RUNS")
        
        print(f"⚡ Processing mode: {'PARALLEL' if self.parallel_workers > 1 else 'SEQUENTIAL'}")
        print(f"🚀 Performance mode: {'FAST' if self.fast_mode else 'STANDARD'}")
        print(f"🗑️ Cleanup mode: {'ON - JSONL files will be deleted after CSV creation' if self.cleanup else 'OFF - JSONL files will be preserved'}")
        
        if self.dry_run:
            print(f"🔍 DRY RUN MODE: Simulating processing")
        
        # Show first few runs
        print(f"\n📋 Runs to process (showing first 10):")
        for i, (apt_type, run_number) in enumerate(runs_to_process[:10]):
            print(f"   {i+1:2d}. {apt_type}-run-{run_number:02d}")
        if len(runs_to_process) > 10:
            print(f"   ... and {len(runs_to_process) - 10} more runs")
        
        print("\n" + "=" * 80)
        
        # Process runs
        if self.parallel_workers > 1 and not self.dry_run:
            results = self.process_runs_parallel(runs_to_process)
        else:
            results = self.process_runs_sequential(runs_to_process)
        
        self.stats['end_time'] = datetime.now()
        
        # Generate summary report
        self.generate_summary_report(results)
        
        # Return success status
        return all(r['success'] for r in results)


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Batch Network Traffic Flow JSONL to CSV Converter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all runs across all APT types
  python3 1_batch_netflow_JSONL-to-csv_converter.py --all
  
  # Process specific APT types
  python3 1_batch_netflow_JSONL-to-csv_converter.py --apt-types apt-1,apt-2
  
  # Process specific run range
  python3 1_batch_netflow_JSONL-to-csv_converter.py --runs 04-10
  
  # Process individual runs
  python3 1_batch_netflow_JSONL-to-csv_converter.py --runs 04,05,51
  
  # High-performance parallel processing
  python3 1_batch_netflow_JSONL-to-csv_converter.py --all --parallel 4 --fast
  
  # Production processing with cleanup (saves ~53GB)
  python3 1_batch_netflow_JSONL-to-csv_converter.py --all --fast --parallel 16 --cleanup
  
  # Process APT-1 runs with parallel processing  
  python3 1_batch_netflow_JSONL-to-csv_converter.py --apt-types apt-1 --parallel 2 --cleanup
  
  # Dry run to see what would be processed
  python3 1_batch_netflow_JSONL-to-csv_converter.py --all --dry-run
        """
    )
    
    # Processing scope
    parser.add_argument('--all', action='store_true',
                       help='Process all available runs across all APT types')
    parser.add_argument('--apt-types',
                       help='Comma-separated list of APT types (e.g., apt-1,apt-2)')
    parser.add_argument('--runs',
                       help='Run specification: individual (04,05,51) or ranges (04-10) or mixed (04-10,51)')
    
    # Performance options
    parser.add_argument('--parallel', type=int, default=1,
                       help='Number of parallel processes (default: 1, max: CPU cores)')
    parser.add_argument('--fast', action='store_true',
                       help='Enable fast mode (skip validation and analysis)')
    parser.add_argument('--cleanup', action='store_true',
                       help='Delete uncompressed JSONL files after successful CSV creation (saves ~53GB)')
    
    # Utility options
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be processed without executing')
    parser.add_argument('--base-dir',
                       help='Base directory (default: auto-detect from script location)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not any([args.all, args.apt_types, args.runs]):
        print("❌ Error: Must specify --all, --apt-types, or --runs")
        print("   Use --help for usage examples")
        sys.exit(1)
    
    # Limit parallel workers
    max_workers = min(args.parallel, mp.cpu_count())
    if args.parallel > max_workers:
        print(f"⚠️  Warning: Requested {args.parallel} workers, using {max_workers} (CPU limit)")
    
    try:
        # Parse arguments
        apt_types = None
        if args.apt_types:
            apt_types = [apt.strip() for apt in args.apt_types.split(',')]
        
        run_numbers = None
        if args.runs:
            converter = BatchNetflowConverter()  # Temporary instance for parsing
            run_numbers = converter.parse_run_specification(args.runs)
        
        # Initialize converter
        converter = BatchNetflowConverter(
            base_dir=args.base_dir,
            parallel_workers=max_workers,
            fast_mode=args.fast,
            dry_run=args.dry_run,
            cleanup=args.cleanup
        )
        
        # Run batch conversion
        success = converter.run_batch_conversion(
            apt_types=apt_types,
            run_numbers=run_numbers,
            all_runs=args.all
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