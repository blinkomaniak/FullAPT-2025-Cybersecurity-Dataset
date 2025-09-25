#!/usr/bin/env python3
"""
Process Identity Consistency Validator  
=====================================

DESCRIPTION:
    Validates process identity consistency within network_community_id groups in 
    netflow datasets. Focuses on process identity patterns within each community ID
    to ensure data quality for temporal correlation analysis.

PREREQUISITES:
    - Raw NetFlow datasets in: data-raw/apt-X/apt-X-run-XX/netflow-run-XX.csv
    - Python packages: pandas, numpy, json, argparse, pathlib

USAGE:
    # Run from project root directory (/home/researcher/Downloads/research/)
    cd /home/researcher/Downloads/research/
    
    # Single run analysis
    python3 dataset/scripts/exploratory/2_process_tuple_uniqueness_validator.py --apt-type apt-1 --run-id 04
    
    # All runs batch analysis (multithreaded)
    python3 dataset/scripts/exploratory/2_process_tuple_uniqueness_validator.py --all --threads 4
    
    # Specific APT types
    python3 dataset/scripts/exploratory/2_process_tuple_uniqueness_validator.py --apt-types apt-1,apt-2 --threads 2

COMMAND LINE OPTIONS:
    --apt-type      APT campaign type (apt-1, apt-2, apt-3, apt-4, apt-5, apt-6)
    --run-id        Specific run ID (01, 02, 03, ..., 51)
    --apt-types     Comma-separated list of APT types (e.g., apt-1,apt-2)
    --all           Process all available APT runs across all types
    --threads       Number of worker threads for parallel processing (default: 4)

INPUT REQUIREMENTS:
    - NetFlow CSV files with columns: network_community_id, process_pid, process_executable
    - Directory structure: data-raw/apt-X/apt-X-run-XX/netflow-run-XX.csv

OUTPUT GENERATED:
    analysis/process-identity-consistency/apt-X/apt-X-run-XX/
    └── process_identity_consistency_analysis.json     # Validation results

EXPECTED RUNTIME:
    - Single run: 30-60 seconds (depends on dataset size)
    - All runs batch: 15-45 minutes (depends on thread count)

RESEARCH QUESTIONS ANSWERED:
    Within the same network_community_id, can we find netflow events that have:
    • Same process_executable but different process_pid? (Multiple instances)
    • Same process_pid but different process_executable? (Data quality issue)

KEY VALIDATION POINTS:
    1. Process identity consistency within each network_community_id
    2. Detection of process switching within same network flows
    3. Identification of data quality issues (PID reuse/collisions)
    4. Analysis of process behavior patterns
    5. Foundation validation for temporal correlation

PURPOSE:
    Ensures process attribution reliability before temporal correlation analysis.
    Part of Phase 1 data validation workflow.

EXAMPLE WORKFLOW:
    # Validate single APT run after network community ID analysis
    python3 dataset/scripts/exploratory/2_process_tuple_uniqueness_validator.py --apt-type apt-1 --run-id 10
    
    # Validate all runs for comprehensive quality assessment
    python3 dataset/scripts/exploratory/2_process_tuple_uniqueness_validator.py --all --threads 4

TROUBLESHOOTING:
    - "No netflow files found": Check data-raw/ directory structure
    - "Missing process columns": Verify NetFlow files have process_pid and process_executable
    - "Memory error": Reduce --threads or process runs individually
    - "Inconsistent results": May indicate data quality issues in process attribution
"""

import argparse
import pandas as pd
import numpy as np
import json
import sys
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import multiprocessing as mp

def clean_for_json(obj):
    """Recursively clean data structure to make it JSON serializable"""
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(item) for item in obj]
    elif pd.isna(obj):
        return None
    elif isinstance(obj, (np.integer, np.floating)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj.item()  # Convert numpy types to Python native types
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

class ProcessIdentityConsistencyValidator:
    """Validates process identity consistency within network_community_id groups"""
    
    def __init__(self, max_workers=None):
        self.base_path = Path("/home/researcher/Downloads/research/dataset")
        self.output_base = Path("/home/researcher/Downloads/research/analysis/process-identity-consistency")
        self.max_workers = max_workers or min(mp.cpu_count(), 8)
        self.results_lock = Lock()
        
        # Results aggregation
        self.global_results = {
            'processed_runs': 0,
            'failed_runs': 0,
            'total_events_analyzed': 0,
            'total_unique_tuples': 0,
            'total_violations': 0,
            'processing_times': []
        }
        
    def find_netflow_files(self):
        """Find all netflow-run-*.csv files across APT runs"""
        netflow_files = []
        
        for apt_dir in sorted(self.base_path.glob("apt-*")):
            if apt_dir.is_dir():
                for run_dir in sorted(apt_dir.glob("*run-*")):
                    if run_dir.is_dir():
                        matches = list(run_dir.glob("netflow-run-*.csv"))
                        netflow_files.extend(matches)
        
        return netflow_files
    
    def extract_run_info(self, file_path):
        """Extract APT type and run ID from file path"""
        parts = file_path.parts
        
        # Find apt directory and run directory
        apt_part = None
        run_part = None
        
        for part in parts:
            if part.startswith('apt-') and '-run-' not in part:
                apt_part = part
            elif part.startswith('apt-') and '-run-' in part:
                run_part = part
                break
        
        if apt_part and run_part:
            # Extract run number from run_part (e.g., "apt-1-run-04" -> "04")
            run_id = run_part.split('-run-')[-1]
            return apt_part, run_id
        
        return None, None
    
    def analyze_process_identity_consistency(self, df):
        """Analyze process identity consistency within network_community_id groups
        
        Research Question: Within same network_community_id, can we find events with:
        - Same process_executable but different process_pid? (Multiple instances)
        - Same process_pid but different process_executable? (Data quality issue)
        """
        
        results = {
            'total_events': len(df),
            'events_with_process_info': 0,
            'community_ids_analyzed': 0,
            'community_ids_with_inconsistencies': 0,
            'same_executable_different_pid_cases': 0,
            'same_pid_different_executable_cases': 0,
            'inconsistency_details': [],
            'process_patterns': {},
            'process_field_coverage': {}
        }
        
        # Check which process fields are available
        process_fields = ['process_pid', 'process_executable']
        required_fields = ['network_community_id'] + process_fields
        
        # Check field availability
        available_fields = [field for field in required_fields if field in df.columns]
        missing_fields = [field for field in required_fields if field not in df.columns]
        
        results['process_field_coverage'] = {
            'available_fields': available_fields,
            'missing_fields': missing_fields,
            'has_all_required': len(missing_fields) == 0
        }
        
        if len(missing_fields) > 0:
            print(f"   ⚠️ Missing required fields: {missing_fields}")
            return results
        
        # Filter to events with process information (non-null values)
        process_filter = df[process_fields].notna().all(axis=1)
        df_with_process = df[process_filter].copy()
        
        results['events_with_process_info'] = len(df_with_process)
        
        if len(df_with_process) == 0:
            print(f"   ⚠️ No events found with complete process information")
            return results
        
        print(f"   📊 Analyzing {len(df_with_process):,} events with process info ({len(df_with_process)/len(df)*100:.1f}% of total)")
        
        # Group by network_community_id to analyze process patterns within each group
        community_groups = df_with_process.groupby('network_community_id')
        results['community_ids_analyzed'] = len(community_groups)
        
        print(f"   🔍 Found {len(community_groups):,} unique network_community_id groups")
        
        same_executable_different_pid_cases = 0
        same_pid_different_executable_cases = 0
        inconsistency_details = []
        process_patterns = {
            'single_process_communities': 0,
            'multiple_process_communities': 0,
            'process_switching_communities': 0,
            'executable_distribution': {},
            'pid_reuse_patterns': []
        }
        
        # Analyze each community ID group
        for community_id, group in community_groups:
            if len(group) < 2:
                # Single event - no opportunity for inconsistencies
                process_patterns['single_process_communities'] += 1
                continue
            
            # Extract unique process patterns within this community ID
            process_info = group[['process_pid', 'process_executable']].copy()
            process_info['process_executable_clean'] = process_info['process_executable'].str.lower()
            
            # Get unique PIDs and executables in this community
            unique_pids = process_info['process_pid'].unique()
            unique_executables = process_info['process_executable_clean'].unique()
            
            # Track executable distribution
            for exe in unique_executables:
                if exe not in process_patterns['executable_distribution']:
                    process_patterns['executable_distribution'][exe] = 0
                process_patterns['executable_distribution'][exe] += 1
            
            # Check for inconsistencies
            has_same_exe_diff_pid = False
            has_same_pid_diff_exe = False
            inconsistency_found = False
            
            # Case 1: Same executable, different PIDs
            for exe in unique_executables:
                exe_events = process_info[process_info['process_executable_clean'] == exe]
                exe_pids = exe_events['process_pid'].unique()
                
                if len(exe_pids) > 1:
                    has_same_exe_diff_pid = True
                    inconsistency_found = True
                    same_executable_different_pid_cases += 1
                    
                    # Store details with samples for each PID (limit to avoid memory issues)
                    if len(inconsistency_details) < 10:
                        # Get sample events for each PID
                        pid_samples = {}
                        for pid in exe_pids:
                            pid_events = group[group['process_pid'] == pid]
                            pid_samples[int(pid)] = pid_events.head(2).to_dict('records')
                        
                        inconsistency_details.append({
                            'community_id': str(community_id),
                            'inconsistency_type': 'same_executable_different_pid',
                            'executable': exe,
                            'pids': [int(pid) for pid in exe_pids],
                            'event_count': len(group),
                            'samples_by_pid': pid_samples
                        })
            
            # Case 2: Same PID, different executables (data quality issue)
            for pid in unique_pids:
                pid_events = process_info[process_info['process_pid'] == pid]
                pid_executables = pid_events['process_executable_clean'].unique()
                
                if len(pid_executables) > 1:
                    has_same_pid_diff_exe = True
                    inconsistency_found = True
                    same_pid_different_executable_cases += 1
                    
                    # Store details with samples for each executable (limit to avoid memory issues)
                    if len(inconsistency_details) < 10:
                        # Get sample events for each executable
                        exe_samples = {}
                        for exe in pid_executables:
                            exe_events = group[group['process_executable'].str.lower() == exe]
                            exe_samples[exe] = exe_events.head(2).to_dict('records')
                        
                        inconsistency_details.append({
                            'community_id': str(community_id),
                            'inconsistency_type': 'same_pid_different_executable',
                            'pid': int(pid),
                            'executables': list(pid_executables),
                            'event_count': len(group),
                            'samples_by_executable': exe_samples
                        })
                    
                    # Track PID reuse pattern
                    process_patterns['pid_reuse_patterns'].append({
                        'pid': int(pid),
                        'executables': list(pid_executables),
                        'community_id': str(community_id)
                    })
            
            # Categorize community patterns
            if len(unique_pids) == 1 and len(unique_executables) == 1:
                # Single process throughout
                process_patterns['single_process_communities'] += 1
            elif inconsistency_found:
                # Process switching or data quality issues
                process_patterns['process_switching_communities'] += 1
            else:
                # Multiple processes but consistent (different processes)
                process_patterns['multiple_process_communities'] += 1
        
        # Update results
        results['same_executable_different_pid_cases'] = same_executable_different_pid_cases
        results['same_pid_different_executable_cases'] = same_pid_different_executable_cases
        results['community_ids_with_inconsistencies'] = len([d for d in inconsistency_details])
        results['inconsistency_details'] = inconsistency_details
        results['process_patterns'] = process_patterns
        
        # Generate summary statistics
        results['summary_statistics'] = {
            'process_coverage_percentage': (results['events_with_process_info'] / len(df) * 100) if len(df) > 0 else 0,
            'communities_with_inconsistencies_percentage': (results['community_ids_with_inconsistencies'] / results['community_ids_analyzed'] * 100) if results['community_ids_analyzed'] > 0 else 0,
            'same_exe_diff_pid_rate': (same_executable_different_pid_cases / results['community_ids_analyzed'] * 100) if results['community_ids_analyzed'] > 0 else 0,
            'same_pid_diff_exe_rate': (same_pid_different_executable_cases / results['community_ids_analyzed'] * 100) if results['community_ids_analyzed'] > 0 else 0,
            'data_quality_score': 100 - (same_pid_different_executable_cases / results['community_ids_analyzed'] * 100) if results['community_ids_analyzed'] > 0 else 100
        }
        
        return results
    
    def analyze_single_run(self, netflow_file):
        """Analyze process tuple uniqueness for a single APT run"""
        
        start_time = datetime.now()
        
        try:
            # Extract run information
            apt_type, run_id = self.extract_run_info(netflow_file)
            if not apt_type or not run_id:
                print(f"❌ Could not extract run info from: {netflow_file}")
                return False
            
            print(f"🔬 Validating {apt_type}-run-{run_id}: {netflow_file.name}")
            
            # Create output directory
            output_dir = self.output_base / apt_type / f"{apt_type}-run-{run_id}"
            print(f"   📁 Creating output directory: {output_dir}")
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
                print(f"   ✅ Output directory created/verified: {output_dir}")
            except Exception as dir_error:
                print(f"   ❌ Failed to create output directory: {dir_error}")
                raise
            
            # Load netflow data
            df = pd.read_csv(netflow_file)
            print(f"   📊 Loaded {len(df):,} netflow events")
            
            # Perform process identity consistency analysis
            print("   🔍 Analyzing process identity consistency...")
            validation_results = self.analyze_process_identity_consistency(df)
            
            # Compile comprehensive results
            comprehensive_results = {
                'analysis_metadata': {
                    'apt_type': apt_type,
                    'run_id': run_id,
                    'input_file': netflow_file.name,
                    'analysis_timestamp': datetime.now().isoformat(),
                    'total_events': len(df),
                    'processing_time_seconds': (datetime.now() - start_time).total_seconds()
                },
                'consistency_analysis': validation_results,
                'summary_statistics': validation_results['summary_statistics']
            }
            
            # Save main analysis results (clean data first to handle NaN values)
            json_file = output_dir / 'process_identity_consistency_analysis.json'
            cleaned_results = clean_for_json(comprehensive_results)
            with open(json_file, 'w') as f:
                json.dump(cleaned_results, f, indent=2, default=str)
            
            # Report inconsistencies if found
            if validation_results['community_ids_with_inconsistencies'] > 0:
                print(f"   ⚠️ {validation_results['community_ids_with_inconsistencies']} inconsistent community IDs found")
                print(f"   📊 Same exe/diff PID: {validation_results['same_executable_different_pid_cases']}, Same PID/diff exe: {validation_results['same_pid_different_executable_cases']}")
            
            # All results are now contained in the main analysis file
            
            processing_time = (datetime.now() - start_time).total_seconds()
            print(f"   ✅ {apt_type}-run-{run_id} completed in {processing_time:.1f}s")
            print(f"   📁 Results saved: {output_dir}")
            
            # Print key findings
            summary = comprehensive_results['summary_statistics']
            print(f"   📊 Key Results:")
            print(f"      • Process Coverage: {summary['process_coverage_percentage']:.1f}%")
            print(f"      • Community IDs Analyzed: {validation_results['community_ids_analyzed']:,}")
            print(f"      • Same Exe, Different PID: {validation_results['same_executable_different_pid_cases']:,} ({summary['same_exe_diff_pid_rate']:.1f}%)")
            print(f"      • Same PID, Different Exe: {validation_results['same_pid_different_executable_cases']:,} ({summary['same_pid_diff_exe_rate']:.1f}%)")
            print(f"      • Data Quality Score: {summary['data_quality_score']:.1f}%")
            
            # Update global results
            with self.results_lock:
                self.global_results['processed_runs'] += 1
                self.global_results['total_events_analyzed'] += len(df)
                self.global_results['total_unique_tuples'] += validation_results['community_ids_analyzed']
                self.global_results['total_violations'] += validation_results['community_ids_with_inconsistencies']
                self.global_results['processing_times'].append(processing_time)
            
            return True
            
        except Exception as e:
            print(f"❌ Error analyzing {netflow_file}: {e}")
            with self.results_lock:
                self.global_results['failed_runs'] += 1
            return False
    
    def analyze_single_run_cli(self, apt_type, run_id):
        """Analyze a specific run specified via CLI"""
        
        # Find the specific netflow file
        netflow_file = None
        for apt_dir in self.base_path.glob(apt_type):
            if apt_dir.is_dir():
                for run_dir in apt_dir.glob(f"*run-{run_id}"):
                    if run_dir.is_dir():
                        matches = list(run_dir.glob("netflow-run-*.csv"))
                        if matches:
                            netflow_file = matches[0]
                            break
        
        if not netflow_file:
            print(f"❌ Netflow file not found for {apt_type}-run-{run_id}")
            return False
        
        print(f"🎯 Single Run Validation: {apt_type}-run-{run_id}")
        print("=" * 60)
        
        return self.analyze_single_run(netflow_file)
    
    def analyze_all_runs(self):
        """Analyze all runs using multithreading"""
        
        print("🚀 Comprehensive Process Tuple Uniqueness Validation - ALL RUNS")
        print("=" * 80)
        
        # Find all netflow files
        netflow_files = self.find_netflow_files()
        
        if not netflow_files:
            print("❌ No netflow files found!")
            return False
        
        print(f"📁 Found {len(netflow_files)} netflow files across all APT runs")
        print(f"⚙️ Using {self.max_workers} worker threads")
        print()
        
        # Process files using multithreading
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_file = {executor.submit(self.analyze_single_run, file_path): file_path 
                             for file_path in netflow_files}
            
            # Process completed tasks
            completed = 0
            for future in as_completed(future_to_file):
                completed += 1
                progress = (completed / len(netflow_files)) * 100
                print(f"📈 Progress: {completed}/{len(netflow_files)} ({progress:.1f}%) - {future_to_file[future].name}")
        
        # Final summary
        print()
        print("🎯 VALIDATION SUMMARY")
        print("=" * 60)
        print(f"✅ Successfully processed: {self.global_results['processed_runs']} runs")
        print(f"❌ Failed runs: {self.global_results['failed_runs']} runs")
        print(f"📊 Total events analyzed: {self.global_results['total_events_analyzed']:,}")
        print(f"🔄 Total community IDs analyzed: {self.global_results['total_unique_tuples']:,}")
        print(f"⚠️ Total inconsistent communities: {self.global_results['total_violations']:,}")
        
        if self.global_results['processing_times']:
            avg_time = np.mean(self.global_results['processing_times'])
            total_time = sum(self.global_results['processing_times'])
            print(f"⏱️ Average processing time per run: {avg_time:.1f}s")
            print(f"⏱️ Total processing time: {total_time/60:.1f} minutes")
        
        # Overall statistics
        if self.global_results['total_unique_tuples'] > 0:
            overall_inconsistency_rate = (self.global_results['total_violations'] / self.global_results['total_unique_tuples']) * 100
            overall_quality_score = 100 - overall_inconsistency_rate
            print(f"📈 Overall inconsistency rate: {overall_inconsistency_rate:.1f}%")
            print(f"🎯 Overall data quality score: {overall_quality_score:.1f}%")
        
        print(f"\n📁 Results saved in: {self.output_base}")
        
        return self.global_results['processed_runs'] > 0

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="Process Identity Consistency Validator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze specific run
  python process_identity_consistency_validator.py --apt-type apt-1 --run-id 04
  
  # Analyze all runs
  python process_identity_consistency_validator.py --all
  
  # Use custom thread count
  python process_identity_consistency_validator.py --all --threads 4
  
  # Analyze specific APT types
  python process_identity_consistency_validator.py --apt-types apt-1,apt-2 --threads 2
        """
    )
    
    # Execution mode
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--all', action='store_true',
                      help='Analyze all APT runs')
    group.add_argument('--apt-type', 
                      help='APT type (e.g., apt-1, apt-2)')
    group.add_argument('--apt-types',
                      help='Comma-separated APT types (e.g., apt-1,apt-2)')
    
    # Single run parameters
    parser.add_argument('--run-id',
                       help='Run ID (e.g., 04, 05) - required with --apt-type')
    
    # Performance parameters
    parser.add_argument('--threads', type=int, default=None,
                       help='Number of worker threads (default: auto)')
    
    args = parser.parse_args()
    
    # Validation
    if args.apt_type and not args.run_id:
        parser.error("--run-id is required when using --apt-type")
    
    try:
        # Initialize validator
        validator = ProcessIdentityConsistencyValidator(max_workers=args.threads)
        
        # Execute based on mode
        if args.all:
            success = validator.analyze_all_runs()
        elif args.apt_types:
            # Process multiple APT types
            apt_types = [apt.strip() for apt in args.apt_types.split(',')]
            print(f"🎯 Multi-APT Analysis: {', '.join(apt_types)}")
            print("=" * 60)
            
            # Find files for specified APT types
            all_files = validator.find_netflow_files()
            filtered_files = []
            for file_path in all_files:
                apt_type, _ = validator.extract_run_info(file_path)
                if apt_type in apt_types:
                    filtered_files.append(file_path)
            
            if not filtered_files:
                print(f"❌ No netflow files found for APT types: {apt_types}")
                sys.exit(1)
            
            print(f"📁 Found {len(filtered_files)} netflow files for specified APT types")
            print(f"⚙️ Using {validator.max_workers} worker threads")
            print()
            
            # Process filtered files
            with ThreadPoolExecutor(max_workers=validator.max_workers) as executor:
                future_to_file = {executor.submit(validator.analyze_single_run, file_path): file_path 
                                 for file_path in filtered_files}
                
                completed = 0
                for future in as_completed(future_to_file):
                    completed += 1
                    progress = (completed / len(filtered_files)) * 100
                    print(f"📈 Progress: {completed}/{len(filtered_files)} ({progress:.1f}%) - {future_to_file[future].name}")
            
            success = validator.global_results['processed_runs'] > 0
        else:
            success = validator.analyze_single_run_cli(args.apt_type, args.run_id)
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n⏹️ Analysis cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()