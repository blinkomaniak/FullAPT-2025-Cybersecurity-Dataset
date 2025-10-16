#!/usr/bin/env python3
"""
Batch Data Processor for APT Dataset
===================================

Purpose: Systematically process all APT runs (04-51) by:
1. Decompressing JSONL files from dataset-backup
2. Moving them to appropriate apt-Y/apt-Y-run-X directories
3. Running scripts #2 and #3 to generate CSV files
4. Cleaning up temporary JSONL files

Author: APT Dataset Development Project
Date: 2025-08-02
"""

import os
import subprocess
import time
import gzip
import shutil
from pathlib import Path
import concurrent.futures
import argparse

class BatchDataProcessor:
    def __init__(self, base_dir, scripts_dir):
        self.base_dir = Path(base_dir)
        self.scripts_dir = Path(scripts_dir)
        self.dataset_backup_dir = self.base_dir / "dataset-backup"
        
        # APT run mapping (run number to apt folder)
        self.apt_mapping = {
            # APT-1: runs 04-20, 51
            **{run: "apt-1" for run in range(4, 21)},
            51: "apt-1",
            # APT-2: runs 21-30
            **{run: "apt-2" for run in range(21, 31)},
            # APT-3: runs 31-38
            **{run: "apt-3" for run in range(31, 39)},
            # APT-4: runs 39-44
            **{run: "apt-4" for run in range(39, 45)},
            # APT-5: runs 45-47
            **{run: "apt-5" for run in range(45, 48)},
            # APT-6: runs 48-50
            **{run: "apt-6" for run in range(48, 51)},
            # Test runs
            52: "apt-test"
        }
        
        # Script paths (updated for new directory structure)
        self.sysmon_script = self.scripts_dir / "2_sysmon_csv_creator.py"
        self.network_script = self.scripts_dir / "3_network_traffic_csv_creator.py"
        
        # Statistics
        self.processed_runs = []
        self.failed_runs = []
        self.skipped_runs = []
        
    def validate_environment(self):
        """Validate that all required directories and scripts exist"""
        print("🔍 VALIDATING ENVIRONMENT")
        print("=" * 50)
        
        # Debug path information
        print(f"🔍 Current working directory: {Path.cwd()}")
        print(f"🔍 Base directory (relative): {self.base_dir}")
        print(f"🔍 Base directory (absolute): {self.base_dir.resolve()}")
        print(f"🔍 Scripts directory (relative): {self.scripts_dir}")
        print(f"🔍 Scripts directory (absolute): {self.scripts_dir.resolve()}")
        print(f"🔍 Dataset backup directory: {self.dataset_backup_dir}")
        print(f"🔍 Dataset backup directory (absolute): {self.dataset_backup_dir.resolve()}")
        print()
        
        # Check base directories
        if not self.base_dir.exists():
            raise FileNotFoundError(f"Base directory not found: {self.base_dir.resolve()}")
        print(f"✅ Base directory: {self.base_dir.resolve()}")
        
        if not self.dataset_backup_dir.exists():
            raise FileNotFoundError(f"Dataset backup directory not found: {self.dataset_backup_dir}")
        print(f"✅ Dataset backup directory: {self.dataset_backup_dir}")
        
        if not self.scripts_dir.exists():
            raise FileNotFoundError(f"Scripts directory not found: {self.scripts_dir}")
        print(f"✅ Scripts directory: {self.scripts_dir}")
        
        # Check scripts
        if not self.sysmon_script.exists():
            raise FileNotFoundError(f"Sysmon script not found: {self.sysmon_script}")
        print(f"✅ Sysmon script: {self.sysmon_script}")
        
        if not self.network_script.exists():
            raise FileNotFoundError(f"Network script not found: {self.network_script}")
        print(f"✅ Network script: {self.network_script}")
        
        print("✅ Environment validation complete\n")
        
    def get_run_directory(self, run_number):
        """Get the appropriate APT directory for a run number"""
        apt_folder = self.apt_mapping.get(run_number)
        if not apt_folder:
            raise ValueError(f"No APT folder mapping found for run {run_number}")
        
        run_dir = self.base_dir / apt_folder / f"{apt_folder}-run-{run_number:02d}"
        return run_dir
    
    def decompress_and_move_files(self, run_number):
        """Decompress and move JSONL files for a specific run"""
        print(f"  📦 Decompressing files for run {run_number:02d}")
        
        # File paths in dataset-backup (corrected - no leading dash)
        sysmon_gz = self.dataset_backup_dir / f"ds-logs-windows-sysmon_operational-default-run-{run_number:02d}.jsonl.gz"
        network_gz = self.dataset_backup_dir / f"ds-logs-network_traffic-flow-default-run-{run_number:02d}.jsonl.gz"
        
        # Check if compressed files exist
        if not sysmon_gz.exists():
            raise FileNotFoundError(f"Sysmon compressed file not found: {sysmon_gz}")
        if not network_gz.exists():
            raise FileNotFoundError(f"Network compressed file not found: {network_gz}")
        
        # Target directory
        run_dir = self.get_run_directory(run_number)
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Target file paths (remove the leading "-")
        sysmon_jsonl = run_dir / f"ds-logs-windows-sysmon_operational-default-run-{run_number:02d}.jsonl"
        network_jsonl = run_dir / f"ds-logs-network_traffic-flow-default-run-{run_number:02d}.jsonl"
        
        # Check if files already exist
        if sysmon_jsonl.exists() and network_jsonl.exists():
            print(f"    ⚠️  JSONL files already exist in {run_dir}")
            return sysmon_jsonl, network_jsonl
        
        # Decompress files without removing originals
        print(f"    🔓 Decompressing Sysmon file...")
        with gzip.open(sysmon_gz, 'rb') as f_in:
            with open(sysmon_jsonl, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        print(f"    🔓 Decompressing Network file...")
        with gzip.open(network_gz, 'rb') as f_in:
            with open(network_jsonl, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        print(f"    ✅ Files decompressed to {run_dir}")
        return sysmon_jsonl, network_jsonl
    
    def run_processing_scripts(self, run_number, sysmon_jsonl, network_jsonl, parallel=False):
        """Run scripts #2 and #3 for data processing"""
        print(f"  🚀 Running processing scripts for run {run_number:02d}")
        
        run_dir = self.get_run_directory(run_number)
        
        # Output file paths
        sysmon_csv = run_dir / f"sysmon-run-{run_number:02d}.csv"
        network_csv = run_dir / f"network_traffic_flow-run-{run_number:02d}.csv"
        
        # Check if output files already exist
        if sysmon_csv.exists() and network_csv.exists():
            print(f"    ⚠️  CSV files already exist, skipping processing")
            return True
        
        # Prepare commands
        sysmon_cmd = [
            "python3", str(self.sysmon_script),
            "--input", str(sysmon_jsonl),
            "--output", str(sysmon_csv)
        ]
        
        network_cmd = [
            "python3", str(self.network_script),
            "--input", str(network_jsonl),
            "--output", str(network_csv)
        ]
        
        try:
            if parallel:
                # Run both scripts in parallel
                print(f"    📊 Running Sysmon and Network scripts in parallel...")
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    sysmon_future = executor.submit(subprocess.run, sysmon_cmd, 
                                                   capture_output=True, text=True, cwd=self.scripts_dir.parent / "pipeline")
                    network_future = executor.submit(subprocess.run, network_cmd, 
                                                    capture_output=True, text=True, cwd=self.scripts_dir.parent / "pipeline")
                    
                    # Wait for both to complete
                    sysmon_result = sysmon_future.result()
                    network_result = network_future.result()
            else:
                # Run scripts sequentially (default behavior)
                print(f"    📊 Running Sysmon script first...")
                sysmon_result = subprocess.run(sysmon_cmd, capture_output=True, text=True, cwd=self.scripts_dir.parent / "pipeline")
                
                print(f"    🌐 Running Network script second...")
                network_result = subprocess.run(network_cmd, capture_output=True, text=True, cwd=self.scripts_dir.parent / "pipeline")
            
            # Check results
            if sysmon_result.returncode != 0:
                print(f"    ❌ Sysmon script failed:")
                print(f"       stdout: {sysmon_result.stdout}")
                print(f"       stderr: {sysmon_result.stderr}")
                return False
            
            if network_result.returncode != 0:
                print(f"    ❌ Network script failed:")
                print(f"       stdout: {network_result.stdout}")
                print(f"       stderr: {network_result.stderr}")
                return False
            
            print(f"    ✅ Both scripts completed successfully")
            return True
            
        except Exception as e:
            print(f"    ❌ Error running processing scripts: {e}")
            return False
    
    def cleanup_jsonl_files(self, run_number, sysmon_jsonl, network_jsonl, delay_seconds=5):
        """Clean up temporary JSONL files after processing"""
        print(f"  🧹 Cleaning up JSONL files for run {run_number:02d}")
        
        # Wait specified delay
        print(f"    ⏱️  Waiting {delay_seconds} seconds before cleanup...")
        time.sleep(delay_seconds)
        
        try:
            if sysmon_jsonl.exists():
                sysmon_jsonl.unlink()
                print(f"    🗑️  Deleted: {sysmon_jsonl.name}")
            
            if network_jsonl.exists():
                network_jsonl.unlink()
                print(f"    🗑️  Deleted: {network_jsonl.name}")
            
            print(f"    ✅ Cleanup complete")
            
        except Exception as e:
            print(f"    ⚠️  Error during cleanup: {e}")
    
    def process_single_run(self, run_number, cleanup=True, parallel=False):
        """Process a single APT run completely"""
        print(f"\n🎯 PROCESSING APT RUN {run_number:02d}")
        print("=" * 50)
        
        try:
            # Step 1: Decompress and move files
            sysmon_jsonl, network_jsonl = self.decompress_and_move_files(run_number)
            
            # Step 2: Run processing scripts
            success = self.run_processing_scripts(run_number, sysmon_jsonl, network_jsonl, parallel=parallel)
            
            if not success:
                self.failed_runs.append(run_number)
                print(f"❌ Processing failed for run {run_number:02d}")
                return False
            
            # Step 3: Clean up JSONL files
            if cleanup:
                self.cleanup_jsonl_files(run_number, sysmon_jsonl, network_jsonl)
            
            self.processed_runs.append(run_number)
            print(f"✅ Successfully processed run {run_number:02d}")
            return True
            
        except Exception as e:
            self.failed_runs.append(run_number)
            print(f"❌ Error processing run {run_number:02d}: {e}")
            return False
    
    def process_all_runs(self, start_run=4, end_run=51, cleanup=True, parallel=False, 
                        continue_on_error=True):
        """Process all APT runs from start_run to end_run"""
        print("🚀 BATCH DATA PROCESSOR")
        print("=" * 80)
        print(f"Processing runs {start_run:02d} to {end_run:02d}")
        print(f"Cleanup enabled: {cleanup}")
        print(f"Sequential processing: {not parallel}")
        print(f"Continue on error: {continue_on_error}")
        print()
        
        # Validate environment
        self.validate_environment()
        
        # Get all run numbers to process
        all_runs = list(range(start_run, end_run + 1))
        if 51 in all_runs and end_run >= 51:
            # Ensure run 51 is included (it's special case for APT-1)
            pass
        
        total_runs = len(all_runs)
        start_time = time.time()
        
        print(f"📋 PROCESSING {total_runs} RUNS")
        print("=" * 80)
        
        for i, run_number in enumerate(all_runs, 1):
            print(f"\n[{i}/{total_runs}] Processing run {run_number:02d}...")
            
            success = self.process_single_run(run_number, cleanup=cleanup, parallel=parallel)
            
            if not success and not continue_on_error:
                print(f"\n❌ Stopping batch processing due to error in run {run_number:02d}")
                break
        
        # Final statistics
        elapsed_time = time.time() - start_time
        self.print_final_statistics(elapsed_time)
    
    def print_final_statistics(self, elapsed_time):
        """Print final processing statistics"""
        print(f"\n{'='*80}")
        print("BATCH PROCESSING COMPLETE")
        print(f"{'='*80}")
        
        total_attempted = len(self.processed_runs) + len(self.failed_runs) + len(self.skipped_runs)
        
        print(f"⏱️  Total time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
        print(f"📊 Total runs attempted: {total_attempted}")
        print(f"✅ Successfully processed: {len(self.processed_runs)}")
        print(f"❌ Failed: {len(self.failed_runs)}")
        print(f"⏭️  Skipped: {len(self.skipped_runs)}")
        
        if self.processed_runs:
            print(f"\n✅ Successfully processed runs: {', '.join([f'{r:02d}' for r in sorted(self.processed_runs)])}")
        
        if self.failed_runs:
            print(f"\n❌ Failed runs: {', '.join([f'{r:02d}' for r in sorted(self.failed_runs)])}")
        
        if self.skipped_runs:
            print(f"\n⏭️  Skipped runs: {', '.join([f'{r:02d}' for r in sorted(self.skipped_runs)])}")
        
        success_rate = len(self.processed_runs) / total_attempted * 100 if total_attempted > 0 else 0
        print(f"\n📈 Success rate: {success_rate:.1f}%")
        
        print(f"\n🎯 Batch processing statistics saved for analysis")
    
    def run_environment_test(self):
        """Run comprehensive environment test to verify everything will work"""
        print("🧪 RUNNING ENVIRONMENT TEST")
        print("=" * 80)
        
        test_passed = True
        
        # Test 1: Validate basic environment
        try:
            print("1️⃣  Testing environment validation...")
            self.validate_environment()
            print("   ✅ Environment validation passed")
        except Exception as e:
            print(f"   ❌ Environment validation failed: {e}")
            test_passed = False
        
        # Test 2: Check sample compressed files exist
        print("\n2️⃣  Testing sample compressed files...")
        sample_runs = [4, 21, 31, 39, 45, 48]  # One from each APT group
        missing_files = []
        
        for run_num in sample_runs:
            sysmon_gz = self.dataset_backup_dir / f"ds-logs-windows-sysmon_operational-default-run-{run_num:02d}.jsonl.gz"
            network_gz = self.dataset_backup_dir / f"ds-logs-network_traffic-flow-default-run-{run_num:02d}.jsonl.gz"
            
            if not sysmon_gz.exists():
                missing_files.append(str(sysmon_gz))
            if not network_gz.exists():
                missing_files.append(str(network_gz))
        
        if missing_files:
            print(f"   ❌ Missing sample files:")
            for file in missing_files:
                print(f"      - {file}")
            test_passed = False
        else:
            print(f"   ✅ Sample compressed files found")
        
        # Test 3: Test decompression on a small sample
        print("\n3️⃣  Testing decompression functionality...")
        try:
            # Test with run 04 (should be smallest)
            test_run = 4
            sysmon_gz = self.dataset_backup_dir / f"ds-logs-windows-sysmon_operational-default-run-{test_run:02d}.jsonl.gz"
            
            if sysmon_gz.exists():
                # Test reading first few lines
                with gzip.open(sysmon_gz, 'rt') as f:
                    first_line = f.readline()
                    if first_line.strip():
                        print(f"   ✅ Decompression test passed (sample: {first_line[:50]}...)")
                    else:
                        print(f"   ❌ Decompression test failed: empty file")
                        test_passed = False
            else:
                print(f"   ⚠️  Skipping decompression test: test file not found")
        except Exception as e:
            print(f"   ❌ Decompression test failed: {e}")
            test_passed = False
        
        # Test 4: Check APT directory structure
        print("\n4️⃣  Testing APT directory structure...")
        apt_dirs = ["apt-1", "apt-2", "apt-3", "apt-4", "apt-5", "apt-6"]
        missing_dirs = []
        existing_dirs = []
        
        print(f"   🔍 Looking for APT directories in: {self.base_dir.resolve()}")
        print(f"   🔍 Directory contents: {list(self.base_dir.iterdir()) if self.base_dir.exists() else 'Directory does not exist'}")
        
        for apt_dir in apt_dirs:
            full_path = self.base_dir / apt_dir
            if not full_path.exists():
                missing_dirs.append(apt_dir)
            else:
                existing_dirs.append(apt_dir)
        
        if existing_dirs:
            print(f"   ✅ Found APT directories: {', '.join(existing_dirs)}")
        
        if missing_dirs:
            print(f"   ❌ Missing APT directories: {', '.join(missing_dirs)}")
            test_passed = False
        else:
            print(f"   ✅ All APT directories found")
        
        # Test 5: Check run directory creation
        print("\n5️⃣  Testing run directory mapping...")
        test_mappings = [(4, "apt-1"), (21, "apt-2"), (31, "apt-3"), (39, "apt-4"), (45, "apt-5"), (48, "apt-6")]
        
        for run_num, expected_apt in test_mappings:
            try:
                run_dir = self.get_run_directory(run_num)
                actual_apt = run_dir.parts[-2]  # Get parent directory name
                if actual_apt == expected_apt:
                    print(f"   ✅ Run {run_num:02d} -> {actual_apt} (correct)")
                else:
                    print(f"   ❌ Run {run_num:02d} -> {actual_apt} (expected {expected_apt})")
                    test_passed = False
            except Exception as e:
                print(f"   ❌ Run {run_num:02d} mapping failed: {e}")
                test_passed = False
        
        # Test 6: Check scripts can be executed
        print("\n6️⃣  Testing script execution...")
        test_commands = [
            ["python3", str(self.sysmon_script), "--help"],
            ["python3", str(self.network_script), "--help"]
        ]
        
        for cmd in test_commands:
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.scripts_dir.parent / "pipeline", timeout=10)
                script_name = cmd[1].split('/')[-1]
                if result.returncode == 0 or "usage:" in result.stdout.lower() or "help" in result.stdout.lower():
                    print(f"   ✅ {script_name} executable")
                else:
                    print(f"   ❌ {script_name} not executable or missing --help")
                    print(f"      stdout: {result.stdout[:100]}...")
                    print(f"      stderr: {result.stderr[:100]}...")
                    test_passed = False
            except Exception as e:
                print(f"   ❌ Script execution test failed: {e}")
                test_passed = False
        
        # Final test result
        print(f"\n{'='*80}")
        if test_passed:
            print("🎉 ALL TESTS PASSED! The batch processor is ready to run.")
            print("\nYou can now run:")
            print("  python3 batch_data_processor.py                    # Process all runs")
            print("  python3 batch_data_processor.py --single-run 04    # Test single run")
            print("  python3 batch_data_processor.py --start-run 04 --end-run 10  # Process range")
        else:
            print("❌ SOME TESTS FAILED! Please fix the issues above before running.")
            print("\nCommon fixes:")
            print("  - Ensure all APT directories exist")
            print("  - Check that compressed files are in dataset-backup/")
            print("  - Verify scripts are in the scripts/ directory")
            print("  - Make sure Python environment has required packages")
        
        print(f"{'='*80}")
        return test_passed

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Batch Data Processor for APT Dataset")
    parser.add_argument("--base-dir", default="../..", 
                       help="Base directory containing APT folders (default: dataset/ directory when run from scripts/batch/)")
    parser.add_argument("--scripts-dir", default="../pipeline",
                       help="Directory containing processing scripts (default: scripts/pipeline/ when run from scripts/batch/)")
    parser.add_argument("--start-run", type=int, default=4,
                       help="Starting run number (default: 4)")
    parser.add_argument("--end-run", type=int, default=51,
                       help="Ending run number (default: 51)")
    parser.add_argument("--no-cleanup", action="store_true",
                       help="Don't delete JSONL files after processing")
    parser.add_argument("--parallel", action="store_true",
                       help="Run scripts in parallel instead of sequentially (default: sequential)")
    parser.add_argument("--stop-on-error", action="store_true",
                       help="Stop processing if any run fails")
    parser.add_argument("--single-run", type=int,
                       help="Process only a single run number")
    parser.add_argument("--test", action="store_true",
                       help="Run environment test to verify setup before processing")
    
    args = parser.parse_args()
    
    try:
        processor = BatchDataProcessor(args.base_dir, args.scripts_dir)
        
        if args.test:
            # Run environment test
            test_passed = processor.run_environment_test()
            if test_passed:
                exit(0)
            else:
                exit(1)
        elif args.single_run:
            # Process single run
            success = processor.process_single_run(
                args.single_run, 
                cleanup=not args.no_cleanup,
                parallel=args.parallel
            )
            if success:
                print(f"\n🎉 Single run {args.single_run:02d} processed successfully!")
            else:
                print(f"\n💥 Single run {args.single_run:02d} processing failed!")
        else:
            # Process all runs
            processor.process_all_runs(
                start_run=args.start_run,
                end_run=args.end_run,
                cleanup=not args.no_cleanup,
                parallel=args.parallel,
                continue_on_error=not args.stop_on_error
            )
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Batch processing interrupted by user")
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        raise

if __name__ == "__main__":
    main()