#!/usr/bin/env python3
"""
Sysmon Statistics Collector for APT Correlation Analysis
========================================================

Extracts missing Sysmon event statistics from original datasets that were not
captured in the correlation analysis results.

This script fills the data gap by analyzing:
1. Total Sysmon events per APT run
2. Sysmon EventID distribution per APT type
3. Sysmon events that generated network flows (cross-domain correlation)
4. Host-level activity patterns across APT types

Key Missing Statistics:
- How many Sysmon events per APT family?
- How many Sysmon events correlated with network flows?
- EventID distribution across different APT attack patterns
- Host activity volume comparison (Sysmon vs Network events)

Usage:
    cd dataset/scripts/batch/
    python3 sysmon_statistics_collector.py
    python3 sysmon_statistics_collector.py --apt-type apt-1
    python3 sysmon_statistics_collector.py --output-dir ~/custom_output/

Output Location:
    Default: ~/Downloads/research/analysis/dataset_sysmon_statistics/

Output:
    - sysmon_statistics_summary.json: Complete Sysmon statistics per APT
    - sysmon_apt_comparison.csv: APT-level Sysmon event summaries
    - Enhanced correlation analysis with host-level data
"""

import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path
from datetime import datetime
import warnings
from collections import defaultdict
import glob

warnings.filterwarnings('ignore')

class SysmonStatisticsCollector:
    def __init__(self, data_root=None, correlation_results_dir=None, output_dir=None):
        """Initialize Sysmon statistics collector"""
        # Set up directories
        script_dir = Path(__file__).parent
        # Go up to research/ root
        research_root = script_dir.parent.parent.parent
        dataset_root = research_root / "dataset"
        analysis_root = research_root / "analysis" / "correlation-analysis"
        
        self.data_root = Path(data_root) if data_root else dataset_root
        self.correlation_results_dir = Path(correlation_results_dir) if correlation_results_dir else analysis_root / "correlation_analysis_results"
        # Default output to ~/Downloads/research/analysis/dataset_sysmon_statistics/
        default_output = research_root / "analysis" / "dataset_sysmon_statistics"
        self.output_dir = Path(output_dir) if output_dir else default_output
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📊 Sysmon Statistics Collector")
        print(f"Data root: {self.data_root}")
        print(f"Correlation results: {self.correlation_results_dir}")
        print(f"Output directory: {self.output_dir}")
        print("=" * 60)
        
        # APT type metadata
        self.apt_metadata = {
            'apt-1': {'runs': list(range(1, 21)) + [51], 'threat_actor': 'APT-34 (OilRig)'},
            'apt-2': {'runs': list(range(21, 31)), 'threat_actor': 'APT-34 Variant'},
            'apt-3': {'runs': list(range(31, 39)), 'threat_actor': 'APT-34 Variant'},
            'apt-4': {'runs': list(range(39, 45)), 'threat_actor': 'APT-29'},
            'apt-5': {'runs': list(range(45, 48)), 'threat_actor': 'APT-29 Variant'},
            'apt-6': {'runs': list(range(48, 51)), 'threat_actor': 'Wizard Spider'}
        }
        
        self.sysmon_statistics = {}
        self.correlation_data = {}
    
    def load_correlation_summary(self):
        """Load existing correlation analysis results"""
        summary_file = self.correlation_results_dir / "batch_summary_results_multithreaded.csv"
        
        if not summary_file.exists():
            print(f"⚠️  Correlation summary not found: {summary_file}")
            return None
        
        print(f"📖 Loading correlation summary: {summary_file}")
        correlation_df = pd.read_csv(summary_file)
        
        # Convert to dictionary for easy lookup
        for _, row in correlation_df.iterrows():
            apt_type = row['apt_type']
            run_id = str(row['run_id']).zfill(2)
            
            if apt_type not in self.correlation_data:
                self.correlation_data[apt_type] = {}
            
            self.correlation_data[apt_type][run_id] = {
                'total_network_events': row['total_events'],
                'analyzable_flows': row['analyzable_flows'],
                'attributed_count': row['attributed_count'],
                'attributed_pct': row['attributed_pct']
            }
        
        print(f"✅ Loaded correlation data for {len(correlation_df)} runs")
        return correlation_df
    
    def find_sysmon_files(self, apt_type, run_id):
        """Find Sysmon CSV files for given APT type and run"""
        apt_dir = self.data_root / apt_type / f"{apt_type}-run-{run_id:02d}"
        
        # Try multiple possible locations and naming patterns
        possible_paths = [
            apt_dir / "02_data_processing" / "processed_data" / f"sysmon-run-{run_id:02d}.csv",
            apt_dir / f"sysmon-run-{run_id:02d}.csv",
            apt_dir / f"sysmon-run-{run_id:02d}-OLD.csv",
        ]
        
        # Also try glob pattern
        glob_pattern = str(apt_dir / f"sysmon*{run_id:02d}*.csv")
        glob_matches = glob.glob(glob_pattern)
        
        # Return first existing file
        for path in possible_paths:
            if path.exists():
                return path
        
        if glob_matches:
            return Path(glob_matches[0])
        
        return None
    
    def analyze_sysmon_file(self, sysmon_file):
        """Analyze individual Sysmon CSV file"""
        try:
            print(f"📖 Analyzing: {sysmon_file}")
            sysmon_df = pd.read_csv(sysmon_file, low_memory=False)
            
            # Basic statistics
            stats = {
                'total_events': len(sysmon_df),
                'file_path': str(sysmon_file),
                'file_size_mb': sysmon_file.stat().st_size / (1024*1024) if sysmon_file.exists() else 0
            }
            
            # EventID distribution
            if 'EventID' in sysmon_df.columns:
                eventid_counts = sysmon_df['EventID'].value_counts()  # Don't sort by index!
                eventid_sorted = eventid_counts.sort_index()  # Keep sorted version for distribution
                stats['eventid_distribution'] = eventid_sorted.to_dict()
                stats['unique_eventids'] = list(eventid_sorted.index)
                # Most common = highest count (first in value_counts)
                stats['most_common_eventid'] = int(eventid_counts.index[0])  # Most frequent EventID
            
            # Computer distribution
            if 'Computer' in sysmon_df.columns:
                computer_counts = sysmon_df['Computer'].value_counts()
                stats['computer_distribution'] = computer_counts.to_dict()
                stats['unique_computers'] = list(computer_counts.index)
            
            # Process analysis
            if 'ProcessId' in sysmon_df.columns:
                stats['unique_processes'] = sysmon_df['ProcessId'].nunique()
                
                # Valid processes for correlation (similar to correlation script logic)
                valid_processes = sysmon_df[
                    sysmon_df[['ProcessGuid', 'ProcessId', 'Image', 'Computer']].notna().all(axis=1)
                ]
                stats['valid_processes_for_correlation'] = len(valid_processes)
                stats['correlation_eligible_pct'] = (len(valid_processes) / len(sysmon_df)) * 100
            
            # Time span analysis
            if 'UtcTime' in sysmon_df.columns:
                try:
                    # Try to parse timestamps
                    sysmon_df['UtcTime_parsed'] = pd.to_datetime(sysmon_df['UtcTime'], errors='coerce')
                    valid_times = sysmon_df['UtcTime_parsed'].dropna()
                    
                    if len(valid_times) > 0:
                        stats['time_span'] = {
                            'start': valid_times.min().isoformat(),
                            'end': valid_times.max().isoformat(),
                            'duration_minutes': (valid_times.max() - valid_times.min()).total_seconds() / 60
                        }
                except:
                    stats['time_span'] = 'parse_error'
            
            return stats
            
        except Exception as e:
            print(f"❌ Error analyzing {sysmon_file}: {e}")
            return {
                'total_events': 0,
                'error': str(e),
                'file_path': str(sysmon_file)
            }
    
    def collect_apt_statistics(self, apt_type=None):
        """Collect Sysmon statistics for specified APT or all APTs"""
        apt_types = [apt_type] if apt_type else list(self.apt_metadata.keys())
        
        print(f"🔍 Collecting Sysmon statistics for: {apt_types}")
        
        for apt in apt_types:
            print(f"\n📊 Processing {apt.upper()} ({self.apt_metadata[apt]['threat_actor']})...")
            
            apt_stats = {
                'apt_type': apt,
                'threat_actor': self.apt_metadata[apt]['threat_actor'],
                'runs_analyzed': {},
                'apt_summary': {}
            }
            
            runs = self.apt_metadata[apt]['runs']
            successful_runs = 0
            total_sysmon_events = 0
            total_correlation_eligible = 0
            
            for run_id in runs:
                run_id_str = str(run_id).zfill(2)
                print(f"  🔍 Run {run_id_str}...", end=' ')
                
                # Find Sysmon file
                sysmon_file = self.find_sysmon_files(apt, run_id)
                
                if not sysmon_file:
                    print(f"❌ Sysmon file not found")
                    continue
                
                # Analyze Sysmon file
                run_stats = self.analyze_sysmon_file(sysmon_file)
                
                # Add correlation data if available
                if apt in self.correlation_data and run_id_str in self.correlation_data[apt]:
                    run_stats['correlation_data'] = self.correlation_data[apt][run_id_str]
                    
                    # Calculate cross-domain metrics
                    if run_stats['total_events'] > 0:
                        network_events = run_stats['correlation_data']['total_network_events']
                        attributed_flows = run_stats['correlation_data']['attributed_count']
                        
                        run_stats['cross_domain_metrics'] = {
                            'sysmon_to_network_ratio': network_events / run_stats['total_events'],
                            'sysmon_correlation_effectiveness': attributed_flows / run_stats['total_events'] * 100,
                            'network_attribution_rate': run_stats['correlation_data']['attributed_pct']
                        }
                
                apt_stats['runs_analyzed'][run_id_str] = run_stats
                
                if 'error' not in run_stats:
                    successful_runs += 1
                    total_sysmon_events += run_stats['total_events']
                    total_correlation_eligible += run_stats.get('valid_processes_for_correlation', 0)
                    print(f"✅ {run_stats['total_events']:,} events")
                else:
                    print(f"❌ Error")
            
            # APT-level summary
            apt_stats['apt_summary'] = {
                'runs_found': successful_runs,
                'runs_expected': len(runs),
                'total_sysmon_events': total_sysmon_events,
                'average_sysmon_per_run': total_sysmon_events / successful_runs if successful_runs > 0 else 0,
                'total_correlation_eligible': total_correlation_eligible,
                'correlation_eligible_pct': (total_correlation_eligible / total_sysmon_events * 100) if total_sysmon_events > 0 else 0
            }
            
            self.sysmon_statistics[apt] = apt_stats
            
            print(f"  📈 Summary: {successful_runs}/{len(runs)} runs, {total_sysmon_events:,} total Sysmon events")
    
    def generate_cross_domain_comparison(self):
        """Generate cross-domain comparison analysis"""
        print(f"\n📊 Generating cross-domain comparison...")
        
        comparison_data = []
        
        # Count available vs missing correlation data
        total_runs = 0
        runs_with_correlation = 0
        
        for apt, apt_data in self.sysmon_statistics.items():
            for run_id, run_data in apt_data['runs_analyzed'].items():
                total_runs += 1
                
                if 'correlation_data' in run_data and 'error' not in run_data:
                    runs_with_correlation += 1
                    comparison_row = {
                        'apt_type': apt,
                        'run_id': run_id,
                        'threat_actor': apt_data['threat_actor'],
                        # Sysmon data
                        'sysmon_events': run_data['total_events'],
                        'sysmon_valid_for_correlation': run_data.get('valid_processes_for_correlation', 0),
                        # Network data
                        'network_events': run_data['correlation_data']['total_network_events'],
                        'network_analyzable': run_data['correlation_data']['analyzable_flows'],
                        'network_attributed': run_data['correlation_data']['attributed_count'],
                        # Cross-domain metrics
                        'attribution_success_rate': run_data['correlation_data']['attributed_pct'],
                        'sysmon_to_network_ratio': run_data.get('cross_domain_metrics', {}).get('sysmon_to_network_ratio', 0),
                        'sysmon_correlation_effectiveness': run_data.get('cross_domain_metrics', {}).get('sysmon_correlation_effectiveness', 0)
                    }
                    comparison_data.append(comparison_row)
                else:
                    # Add placeholder row with Sysmon data only
                    comparison_row = {
                        'apt_type': apt,
                        'run_id': run_id,
                        'threat_actor': apt_data['threat_actor'],
                        # Sysmon data
                        'sysmon_events': run_data.get('total_events', 0),
                        'sysmon_valid_for_correlation': run_data.get('valid_processes_for_correlation', 0),
                        # Missing network data
                        'network_events': 'N/A - Run correlation analysis first',
                        'network_analyzable': 'N/A',
                        'network_attributed': 'N/A',
                        'attribution_success_rate': 'N/A',
                        'sysmon_to_network_ratio': 'N/A',
                        'sysmon_correlation_effectiveness': 'N/A',
                        'note': 'Correlation data missing - run batch correlation analyzer first'
                    }
                    comparison_data.append(comparison_row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Save comparison data with informative header
        comparison_file = self.output_dir / "sysmon_network_cross_domain_comparison.csv"
        
        # Add metadata to CSV
        with open(comparison_file, 'w') as f:
            f.write(f"# Cross-Domain Sysmon-Network Comparison\n")
            f.write(f"# Generated: {datetime.now().isoformat()}\n")
            f.write(f"# Total runs analyzed: {total_runs}\n")
            f.write(f"# Runs with correlation data: {runs_with_correlation}\n")
            f.write(f"# Note: Run correlation analysis first for complete network data\n")
        
        # Append DataFrame
        comparison_df.to_csv(comparison_file, mode='a', index=False)
        
        print(f"✅ Cross-domain comparison saved: {comparison_file}")
        print(f"   📊 {total_runs} total runs, {runs_with_correlation} with correlation data")
        
        if runs_with_correlation == 0:
            print(f"   ⚠️  No correlation data found - run batch correlation analyzer first for complete analysis")
        
        return comparison_df
    
    def save_statistics(self):
        """Save collected statistics to files"""
        print(f"\n💾 Saving Sysmon statistics...")
        
        # Save detailed statistics
        stats_file = self.output_dir / "sysmon_detailed_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(self.sysmon_statistics, f, indent=2, default=str)
        
        # Create APT summary CSV
        apt_summary_data = []
        for apt, apt_data in self.sysmon_statistics.items():
            summary = apt_data['apt_summary'].copy()
            summary['apt_type'] = apt
            summary['threat_actor'] = apt_data['threat_actor']
            apt_summary_data.append(summary)
        
        summary_df = pd.DataFrame(apt_summary_data)
        summary_file = self.output_dir / "sysmon_apt_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        
        print(f"✅ Detailed statistics: {stats_file}")
        print(f"✅ APT summary: {summary_file}")
        
        # Print file contents summary
        self.print_output_summary(stats_file, summary_file)
        
        return stats_file, summary_file
    
    def print_output_summary(self, stats_file, summary_file):
        """Print detailed summary of what each output file contains"""
        print(f"\n📋 OUTPUT FILES SUMMARY:")
        print("=" * 50)
        
        # Analyze APT summary CSV
        if summary_file.exists():
            summary_df = pd.read_csv(summary_file)
            print(f"\n1️⃣ {summary_file.name}:")
            print(f"   📊 APT-level aggregated statistics")
            print(f"   📈 {len(summary_df)} APT types analyzed")
            print(f"   🏷️  Columns: {', '.join(summary_df.columns.tolist())}")
            
            # Show key metrics
            total_events = summary_df['total_sysmon_events'].sum()
            avg_events = summary_df['total_sysmon_events'].mean()
            print(f"   📊 Total Sysmon events: {total_events:,}")
            print(f"   📊 Average per APT: {avg_events:,.0f}")
        
        # Analyze detailed JSON
        if stats_file.exists():
            print(f"\n2️⃣ {stats_file.name}:")
            print(f"   🔍 Run-level detailed statistics for each APT")
            print(f"   📋 Contains: EventID distributions, computer mappings, time spans")
            print(f"   🎯 Use for: Deep-dive analysis, EventID patterns, host activity")
            
            # Count runs in detailed stats
            total_runs = 0
            for apt_data in self.sysmon_statistics.values():
                total_runs += len(apt_data['runs_analyzed'])
            print(f"   📊 Total runs analyzed: {total_runs}")
        
        # Analyze cross-domain comparison CSV
        cross_domain_file = self.output_dir / "sysmon_network_cross_domain_comparison.csv"
        if cross_domain_file.exists():
            print(f"\n3️⃣ {cross_domain_file.name}:")
            print(f"   🔗 Cross-domain correlation between Sysmon and Network events")
            
            file_size = cross_domain_file.stat().st_size
            if file_size > 100:  # More than just headers
                cross_df = pd.read_csv(cross_domain_file, comment='#')
                print(f"   📊 {len(cross_df)} runs with data")
                
                # Check for N/A values (missing correlation data)
                na_count = cross_df['network_events'].astype(str).str.contains('N/A').sum()
                if na_count > 0:
                    print(f"   ⚠️  {na_count} runs missing correlation data")
                    print(f"   💡 Run correlation analyzer first for complete cross-domain analysis")
                else:
                    print(f"   ✅ Complete correlation data available")
            else:
                print(f"   ⚠️  File is empty ({file_size} bytes) - no correlation data found")
                print(f"   💡 Run batch correlation analyzer first to populate this file")
    
    def run_complete_analysis(self, apt_type=None):
        """Run complete Sysmon statistics collection"""
        print(f"🚀 Running Complete Sysmon Statistics Collection")
        print(f"Target APT: {apt_type.upper() if apt_type else 'ALL APT TYPES'}")
        print("=" * 60)
        
        try:
            # Load existing correlation data
            self.load_correlation_summary()
            
            # Collect Sysmon statistics
            self.collect_apt_statistics(apt_type)
            
            # Generate cross-domain comparison
            comparison_df = self.generate_cross_domain_comparison()
            
            # Save all statistics
            stats_file, summary_file = self.save_statistics()
            
            print(f"\n🎯 SYSMON STATISTICS COLLECTION COMPLETE")
            print(f"✅ Output directory: {self.output_dir}")
            print(f"✅ APT types analyzed: {len(self.sysmon_statistics)}")
            print(f"✅ Cross-domain comparison: {len(comparison_df)} runs")
            
            # Print key findings
            total_sysmon = sum(apt['apt_summary']['total_sysmon_events'] for apt in self.sysmon_statistics.values())
            print(f"✅ Total Sysmon events across all APTs: {total_sysmon:,}")
            
            # Print most common EventIDs across all APTs
            self.print_eventid_analysis()
            
            return True
            
        except Exception as e:
            print(f"❌ Error in Sysmon statistics collection: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def print_eventid_analysis(self):
        """Print analysis of EventID patterns across APTs"""
        print(f"\n📊 EVENTID ANALYSIS ACROSS APTS:")
        print("-" * 40)
        
        all_eventids = defaultdict(int)
        apt_eventids = {}
        
        for apt, apt_data in self.sysmon_statistics.items():
            apt_eventids[apt] = defaultdict(int)
            
            for run_id, run_data in apt_data['runs_analyzed'].items():
                if 'eventid_distribution' in run_data:
                    for eventid, count in run_data['eventid_distribution'].items():
                        all_eventids[eventid] += count
                        apt_eventids[apt][eventid] += count
        
        # Top 5 most common EventIDs overall
        if all_eventids:
            sorted_eventids = sorted(all_eventids.items(), key=lambda x: x[1], reverse=True)
            print(f"\n🔝 Top 5 EventIDs across all APTs:")
            for i, (eventid, count) in enumerate(sorted_eventids[:5]):
                print(f"   {i+1}. EventID {eventid}: {count:,} events")
            
            # Most common EventID per APT
            print(f"\n🎯 Most common EventID per APT:")
            for apt in sorted(apt_eventids.keys()):
                if apt_eventids[apt]:
                    most_common = max(apt_eventids[apt].items(), key=lambda x: x[1])
                    print(f"   {apt}: EventID {most_common[0]} ({most_common[1]:,} events)")
        else:
            print(f"   ⚠️  No EventID data available")


def main():
    parser = argparse.ArgumentParser(description='Sysmon Statistics Collector for APT Correlation Analysis')
    parser.add_argument('--apt-type', help='Specific APT type to analyze (e.g., apt-1)')
    parser.add_argument('--data-root', help='Data root directory (default: ../../)')
    parser.add_argument('--correlation-results', help='Correlation results directory')
    parser.add_argument('--output-dir', help='Output directory for statistics')
    
    args = parser.parse_args()
    
    collector = SysmonStatisticsCollector(args.data_root, args.correlation_results, args.output_dir)
    success = collector.run_complete_analysis(args.apt_type)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())