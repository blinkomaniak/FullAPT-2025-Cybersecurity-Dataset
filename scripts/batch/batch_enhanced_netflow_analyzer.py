#!/usr/bin/env python3
"""
Batch Enhanced Network Traffic Flow Analyzer

This script runs the enhanced netflow traffic splitter with 4-category classification 
and community ID verification across multiple APT runs.

Features:
- Process all APT runs (apt-1 through apt-6) with a single command
- Process specific APT types or run ranges
- Parallel processing support for faster execution
- Comprehensive progress reporting and error handling
- Centralized results aggregation and summary

APT Structure:
- APT-1: Runs 04-20, 51
- APT-2: Runs 21-30  
- APT-3: Runs 31-38
- APT-4: Runs 39-44
- APT-5: Runs 45-47
- APT-6: Runs 48-50

Usage Examples:
    # Process all available APT runs
    python3 batch_enhanced_netflow_analyzer.py --all
    
    # Process specific APT types
    python3 batch_enhanced_netflow_analyzer.py --apt-types apt-1,apt-2
    
    # Process specific run range
    python3 batch_enhanced_netflow_analyzer.py --runs 04-10
    
    # Process with parallel execution
    python3 batch_enhanced_netflow_analyzer.py --all --parallel 4
"""

import argparse
import os
import sys
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import json

class BatchEnhancedNetflowAnalyzer:
    """Batch processor for enhanced network traffic flow analysis across APT runs."""
    
    # APT run mappings based on directory structure
    APT_MAPPINGS = {
        'apt-1': list(range(4, 21)) + [51],  # 04-20, 51
        'apt-2': list(range(21, 31)),        # 21-30
        'apt-3': list(range(31, 39)),        # 31-38
        'apt-4': list(range(39, 45)),        # 39-44
        'apt-5': list(range(45, 48)),        # 45-47
        'apt-6': list(range(48, 51)),        # 48-50
    }
    
    def __init__(self, base_dir: str = None, parallel_workers: int = 1, dry_run: bool = False):
        """
        Initialize batch analyzer.
        
        Args:
            base_dir: Base directory (auto-detected if None)
            parallel_workers: Number of parallel processes (1 = sequential)
            dry_run: Show what would be processed without executing
        """
        # Auto-detect base directory
        if base_dir is None:
            script_dir = Path(__file__).parent.resolve()
            self.base_dir = script_dir.parent.parent.parent  # batch -> scripts -> dataset -> research
        else:
            self.base_dir = Path(base_dir).resolve()
        
        self.dataset_dir = self.base_dir / "dataset"
        self.analysis_dir = self.base_dir / "analysis" / "netflow-analysis"
        self.analyzer_script = self.dataset_dir / "scripts" / "exploratory" / "netflow_traffic_splitter_enhanced.py"
        
        self.parallel_workers = parallel_workers
        self.dry_run = dry_run
        
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
        
        print(f"🔧 Initialized Batch Enhanced Netflow Analyzer")
        print(f"📁 Base directory: {self.base_dir}")
        print(f"📊 Dataset directory: {self.dataset_dir}")
        print(f"📈 Analysis directory: {self.analysis_dir}")
        print(f"🔀 Analyzer script: {self.analyzer_script.name}")
        print(f"⚡ Parallel workers: {self.parallel_workers}")
        if self.dry_run:
            print("🔍 DRY RUN MODE: No processing will occur")
    
    def _verify_setup(self):
        """Verify required directories and scripts exist."""
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")
        
        if not self.analyzer_script.exists():
            raise FileNotFoundError(f"Enhanced analyzer script not found: {self.analyzer_script}")
        
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
    
    def check_run_prerequisites(self, apt_type: str, run_number: int) -> Tuple[bool, str]:
        """
        Check if APT run has prerequisites for processing.
        
        Args:
            apt_type: APT type (e.g., 'apt-1')
            run_number: Run number
            
        Returns:
            Tuple of (has_prerequisites, error_message)
        """
        run_str = f"{run_number:02d}"
        run_dir = self.dataset_dir / apt_type / f"{apt_type}-run-{run_str}"
        
        # Check if run directory exists
        if not run_dir.exists():
            return False, f"Run directory not found: {run_dir}"
        
        # Check for netflow CSV file
        possible_names = [
            f"netflow-run-{run_str}.csv",
            f"network_traffic_flow-run-{run_str}.csv",
            f"network-traffic-run-{run_str}.csv"
        ]
        
        netflow_file_found = False
        for name in possible_names:
            if (run_dir / name).exists():
                netflow_file_found = True
                break
        
        if not netflow_file_found:
            return False, f"No netflow CSV file found. Tried: {possible_names}"
        
        return True, "Prerequisites met"
    
    def process_single_run(self, apt_type: str, run_number: int) -> Dict[str, any]:
        """
        Process a single APT run with enhanced analysis.
        
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
            'total_events': 0,
            'unique_flows': 0,
            'integrity_status': 'UNKNOWN',
            'categories': {
                'non_ip_events': 0,
                'ip_only_events': 0,
                'full_ip_events': 0,
                'partial_events': 0
            }
        }
        
        start_time = time.time()
        
        try:
            # Step 1: Check prerequisites
            has_prereqs, prereq_error = self.check_run_prerequisites(apt_type, run_number)
            if not has_prereqs:
                result['error'] = prereq_error
                return result
            
            if self.dry_run:
                result['success'] = True
                result['processing_time'] = 0.1
                return result
            
            # Step 2: Run enhanced analyzer
            cmd = [
                sys.executable,
                str(self.analyzer_script),
                '--apt-type', apt_type,
                '--run-id', run_str
            ]
            
            # Execute analyzer script
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute timeout
            )
            
            result['processing_time'] = time.time() - start_time
            
            if process.returncode == 0:
                result['success'] = True
                
                # Try to extract results from JSON output
                results_file = self.analysis_dir / apt_type / f"{apt_type}-run-{run_str}" / f"enhanced_netflow_analysis_results-run-{run_str}.json"
                
                if results_file.exists():
                    try:
                        with open(results_file, 'r') as f:
                            analysis_results = json.load(f)
                        
                        # Extract key metrics
                        if 'summary' in analysis_results:
                            summary = analysis_results['summary']
                            result['total_events'] = summary.get('total_events', 0)
                            
                            if 'enhanced_categories' in summary:
                                categories = summary['enhanced_categories']
                                result['categories'] = {
                                    'non_ip_events': categories.get('non_ip_traffic_events', 0),
                                    'ip_only_events': categories.get('ip_only_traffic_events', 0),
                                    'full_ip_events': categories.get('full_ip_traffic_events', 0),
                                    'partial_events': categories.get('partial_traffic_events', 0)
                                }
                        
                        if 'community_id_verification' in analysis_results:
                            verification = analysis_results['community_id_verification']
                            result['unique_flows'] = verification.get('total_unique_flows', 0)
                            result['integrity_status'] = verification.get('integrity_status', 'UNKNOWN')
                    
                    except (json.JSONDecodeError, KeyError) as e:
                        print(f"   ⚠️  Warning: Could not parse results JSON for {run_id}: {e}")
                
                print(f"   ✅ Analysis completed successfully")
                
            else:
                result['error'] = f"Enhanced analysis failed (exit code {process.returncode}): {process.stderr[:500]}"
                print(f"   ❌ Analysis failed: {result['error'][:100]}...")
        
        except subprocess.TimeoutExpired:
            result['error'] = "Processing timeout (30 minute limit exceeded)"
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
            print("-" * 70)
            
            result = self.process_single_run(apt_type, run_number)
            results.append(result)
            
            # Print immediate result
            if result['success']:
                print(f"   ✅ {result['run_id']} completed in {result['processing_time']:.1f}s")
                print(f"   📊 Events: {result['total_events']:,} | Flows: {result['unique_flows']:,} | Status: {result['integrity_status']}")
                categories = result['categories']
                print(f"   📈 Categories: Non-IP:{categories['non_ip_events']:,} IP-Only:{categories['ip_only_events']:,} Full-IP:{categories['full_ip_events']:,} Partial:{categories['partial_events']:,}")
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
                        print(f"   ✅ Completed in {result['processing_time']:.1f}s | Events: {result['total_events']:,} | Flows: {result['unique_flows']:,}")
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
                        'total_events': 0,
                        'unique_flows': 0,
                        'integrity_status': 'ERROR'
                    })
        
        return results
    
    def save_batch_results(self, results: List[Dict]) -> None:
        """Save comprehensive batch analysis results to JSON file."""
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        
        total_events = sum(r['total_events'] for r in successful)
        total_flows = sum(r['unique_flows'] for r in successful)
        
        # Aggregate categories
        total_categories = {
            'non_ip_events': sum(r['categories']['non_ip_events'] for r in successful),
            'ip_only_events': sum(r['categories']['ip_only_events'] for r in successful),
            'full_ip_events': sum(r['categories']['full_ip_events'] for r in successful),
            'partial_events': sum(r['categories']['partial_events'] for r in successful)
        }
        
        # Integrity status summary
        integrity_counts = {}
        for result in successful:
            status = result['integrity_status']
            integrity_counts[status] = integrity_counts.get(status, 0) + 1
        
        # APT type breakdown
        apt_stats = {}
        for result in results:
            apt_type = result['apt_type']
            if apt_type not in apt_stats:
                apt_stats[apt_type] = {'total': 0, 'successful': 0, 'failed': 0}
            apt_stats[apt_type]['total'] += 1
            if result['success']:
                apt_stats[apt_type]['successful'] += 1
            else:
                apt_stats[apt_type]['failed'] += 1
        
        # Prepare comprehensive batch results
        batch_summary = {
            'batch_metadata': {
                'processing_date': self.stats['start_time'].isoformat(),
                'completion_date': self.stats['end_time'].isoformat(),
                'total_duration_seconds': (self.stats['end_time'] - self.stats['start_time']).total_seconds(),
                'script_version': 'batch_enhanced_netflow_analyzer_v1.0',
                'analyzer_script': 'netflow_traffic_splitter_enhanced.py',
                'classification_system': '4-category_enhanced_with_community_id_verification',
                'parallel_workers': self.parallel_workers
            },
            'processing_summary': {
                'total_runs_processed': len(results),
                'successful_analyses': len(successful),
                'failed_analyses': len(failed),
                'success_rate_percentage': (len(successful) / max(len(results), 1)) * 100,
                'average_processing_time_seconds': sum(r['processing_time'] for r in successful) / max(len(successful), 1)
            },
            'aggregated_statistics': {
                'total_network_events': total_events,
                'total_unique_flows': total_flows,
                'enhanced_traffic_categories': {
                    'non_ip_traffic': {
                        'count': total_categories['non_ip_events'],
                        'percentage': (total_categories['non_ip_events'] / max(total_events, 1)) * 100,
                        'description': 'Layer 2 broadcast/local traffic - ALL four fields empty'
                    },
                    'ip_only_traffic': {
                        'count': total_categories['ip_only_events'],
                        'percentage': (total_categories['ip_only_events'] / max(total_events, 1)) * 100,
                        'description': 'Non-port-based IP protocols - IPs present, ports empty (ICMP, IPv6 ND)'
                    },
                    'full_ip_traffic': {
                        'count': total_categories['full_ip_events'],
                        'percentage': (total_categories['full_ip_events'] / max(total_events, 1)) * 100,
                        'description': 'Complete TCP/UDP flows - ALL four fields present'
                    },
                    'partial_traffic': {
                        'count': total_categories['partial_events'],
                        'percentage': (total_categories['partial_events'] / max(total_events, 1)) * 100,
                        'description': 'Incomplete flows - mixed field patterns'
                    }
                }
            },
            'classification_integrity': {
                'integrity_status_summary': integrity_counts,
                'overall_integrity_assessment': 'EXCELLENT' if all(s == 'EXCELLENT' for s in integrity_counts.keys()) else 'MIXED',
                'total_unique_flows_verified': total_flows,
                'cross_category_overlaps_detected': 0 if all(s == 'EXCELLENT' for s in integrity_counts.keys()) else 'UNKNOWN'
            },
            'apt_type_breakdown': {
                apt_type: {
                    'total_runs': stats['total'],
                    'successful_runs': stats['successful'],
                    'failed_runs': stats['failed'],
                    'success_rate_percentage': (stats['successful'] / stats['total']) * 100
                }
                for apt_type, stats in apt_stats.items()
            },
            'failed_runs': [
                {
                    'run_id': result['run_id'],
                    'apt_type': result['apt_type'],
                    'run_number': result['run_number'],
                    'error': result['error'],
                    'processing_time': result['processing_time']
                }
                for result in failed
            ],
            'successful_runs_summary': [
                {
                    'run_id': result['run_id'],
                    'apt_type': result['apt_type'],
                    'run_number': result['run_number'],
                    'total_events': result['total_events'],
                    'unique_flows': result['unique_flows'],
                    'integrity_status': result['integrity_status'],
                    'processing_time': result['processing_time'],
                    'categories': result['categories']
                }
                for result in successful
            ]
        }
        
        # Save batch results file
        output_file = self.analysis_dir / "batch_enhanced_analysis_summary.json"
        
        # Ensure analysis directory exists
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(batch_summary, f, indent=2, default=str)
        
        print(f"\n💾 Saved batch analysis summary: {output_file}")
        print(f"📊 File contains: Aggregated statistics, integrity verification, APT breakdowns, and individual run summaries")
    
    def generate_summary_report(self, results: List[Dict]):
        """Generate and display comprehensive summary report."""
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        
        total_events = sum(r['total_events'] for r in successful)
        total_flows = sum(r['unique_flows'] for r in successful)
        
        # Aggregate categories
        total_categories = {
            'non_ip_events': sum(r['categories']['non_ip_events'] for r in successful),
            'ip_only_events': sum(r['categories']['ip_only_events'] for r in successful),
            'full_ip_events': sum(r['categories']['full_ip_events'] for r in successful),
            'partial_events': sum(r['categories']['partial_events'] for r in successful)
        }
        
        print("\n" + "=" * 90)
        print("🎉 BATCH ENHANCED NETFLOW ANALYSIS COMPLETED")
        print("=" * 90)
        print(f"🕐 Start time: {self.stats['start_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🕐 End time: {self.stats['end_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏱️ Total duration: {self.stats['end_time'] - self.stats['start_time']}")
        print(f"📊 Total runs processed: {len(results)}")
        print(f"✅ Successful analyses: {len(successful)}")
        print(f"❌ Failed analyses: {len(failed)}")
        print(f"📈 Success rate: {(len(successful)/max(len(results), 1))*100:.1f}%")
        
        if successful:
            avg_time = sum(r['processing_time'] for r in successful) / len(successful)
            
            print(f"\n📊 AGGREGATED RESULTS:")
            print(f"⚡ Average processing time: {avg_time:.1f}s per run")
            print(f"📈 Total network events: {total_events:,}")
            print(f"🔗 Total unique flows: {total_flows:,}")
            
            print(f"\n📈 ENHANCED TRAFFIC CATEGORIES (AGGREGATED):")
            print(f"   🔶 Non-IP Traffic (Layer 2): {total_categories['non_ip_events']:,} events ({(total_categories['non_ip_events']/max(total_events,1))*100:.2f}%)")
            print(f"   🔷 IP-Only Traffic (ICMP/IPv6): {total_categories['ip_only_events']:,} events ({(total_categories['ip_only_events']/max(total_events,1))*100:.2f}%)")
            print(f"   🔵 Full-IP Traffic (TCP/UDP): {total_categories['full_ip_events']:,} events ({(total_categories['full_ip_events']/max(total_events,1))*100:.2f}%)")
            print(f"   🔸 Partial Traffic (Incomplete): {total_categories['partial_events']:,} events ({(total_categories['partial_events']/max(total_events,1))*100:.2f}%)")
            
            # Integrity status summary
            integrity_counts = {}
            for result in successful:
                status = result['integrity_status']
                integrity_counts[status] = integrity_counts.get(status, 0) + 1
            
            print(f"\n🔍 CLASSIFICATION INTEGRITY SUMMARY:")
            for status, count in sorted(integrity_counts.items()):
                print(f"   • {status}: {count} runs")
        
        # Show failed runs
        if failed:
            print(f"\n❌ FAILED RUNS ({len(failed)}):")
            for result in failed[:10]:  # Show first 10 failures
                error_short = result['error'][:80] + '...' if len(result['error']) > 80 else result['error']
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
        
        print("=" * 90)
    
    def run_batch_analysis(self, apt_types: Optional[List[str]] = None,
                          run_numbers: Optional[List[int]] = None,
                          all_runs: bool = False) -> bool:
        """
        Run batch enhanced analysis across specified runs.
        
        Args:
            apt_types: List of APT types to process
            run_numbers: List of run numbers to process
            all_runs: Process all available runs
            
        Returns:
            True if all analyses successful, False otherwise
        """
        self.stats['start_time'] = datetime.now()
        
        # Get runs to process
        runs_to_process = self.get_runs_to_process(apt_types, run_numbers, all_runs)
        
        if not runs_to_process:
            print("❌ No runs found to process!")
            return False
        
        print(f"🚀 Starting Batch Enhanced Netflow Analysis")
        print(f"📅 Start time: {self.stats['start_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 Runs to process: {len(runs_to_process)}")
        
        if apt_types:
            print(f"🎯 APT types: {', '.join(apt_types)}")
        if run_numbers:
            print(f"🎯 Run numbers: {', '.join(map(str, run_numbers))}")
        if all_runs:
            print(f"🎯 Processing: ALL AVAILABLE RUNS")
        
        print(f"⚡ Processing mode: {'PARALLEL' if self.parallel_workers > 1 else 'SEQUENTIAL'}")
        
        if self.dry_run:
            print(f"🔍 DRY RUN MODE: Simulating processing")
        
        # Show first few runs
        print(f"\n📋 Runs to process (showing first 10):")
        for i, (apt_type, run_number) in enumerate(runs_to_process[:10]):
            print(f"   {i+1:2d}. {apt_type}-run-{run_number:02d}")
        if len(runs_to_process) > 10:
            print(f"   ... and {len(runs_to_process) - 10} more runs")
        
        print("\n" + "=" * 90)
        
        # Process runs
        if self.parallel_workers > 1 and not self.dry_run:
            results = self.process_runs_parallel(runs_to_process)
        else:
            results = self.process_runs_sequential(runs_to_process)
        
        self.stats['end_time'] = datetime.now()
        
        # Save batch results to JSON file
        self.save_batch_results(results)
        
        # Generate summary report
        self.generate_summary_report(results)
        
        # Return success status
        return all(r['success'] for r in results)


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Batch Enhanced Network Traffic Flow Analyzer (4-Category + Community ID Verification)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all available APT runs with enhanced analysis
  python3 batch_enhanced_netflow_analyzer.py --all
  
  # Process specific APT types
  python3 batch_enhanced_netflow_analyzer.py --apt-types apt-1,apt-2
  
  # Process specific run range
  python3 batch_enhanced_netflow_analyzer.py --runs 04-10
  
  # Process individual runs
  python3 batch_enhanced_netflow_analyzer.py --runs 04,05,51
  
  # High-performance parallel processing
  python3 batch_enhanced_netflow_analyzer.py --all --parallel 4
  
  # Process APT-1 runs with parallel execution
  python3 batch_enhanced_netflow_analyzer.py --apt-types apt-1 --parallel 2
  
  # Dry run to see what would be processed
  python3 batch_enhanced_netflow_analyzer.py --all --dry-run
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
            analyzer = BatchEnhancedNetflowAnalyzer()  # Temporary instance for parsing
            run_numbers = analyzer.parse_run_specification(args.runs)
        
        # Initialize analyzer
        analyzer = BatchEnhancedNetflowAnalyzer(
            base_dir=args.base_dir,
            parallel_workers=max_workers,
            dry_run=args.dry_run
        )
        
        # Run batch analysis
        success = analyzer.run_batch_analysis(
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