#!/usr/bin/env python3
"""
Multithreaded Batch Cross-Domain Correlation Analyzer (Scripts Folder Version)
===============================================================================

High-performance multithreaded version optimized for 32-CPU servers.
Supports --run-id flag for individual runs and batch processing.

Key Features:
- Multithreaded processing with configurable worker count
- IPv6-enhanced correlation analysis
- Thread-safe chunk processing
- Optimized for large datasets (1M+ network events)
- Memory-efficient processing with progress tracking

Usage:
    cd dataset/scripts/batch/
    python3 scripts_batch_correlation_analyzer_multithreaded.py --run-id 04 --workers 32
    python3 scripts_batch_correlation_analyzer_multithreaded.py --run-id 51 --apt-type apt-2 --workers 16

Architecture:
- Main thread: Data loading, coordination, results aggregation
- Worker threads: Independent chunk processing (thread-safe)
- No shared state between chunks (network flows are independent)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import ast
import warnings
import argparse
import os
from pathlib import Path
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import time
warnings.filterwarnings('ignore')

# Set style for academic plots
plt.style.use('default')
sns.set_palette("husl")

class MultithreadedCorrelationAnalyzer:
    def __init__(self, run_id, apt_type="apt-1", max_workers=None):
        self.run_id = run_id
        self.apt_type = apt_type
        self.max_workers = max_workers or min(32, multiprocessing.cpu_count())
        
        # Set up paths relative to scripts directory
        self.scripts_dir = Path.cwd()
        # Go up from batch/ to scripts/ to dataset/ 
        self.data_root = self.scripts_dir.parent.parent
        self.apt_dir = self.data_root / self.apt_type / f"{self.apt_type}-run-{self.run_id}"
        # Create outputs in analysis/correlation-analysis/correlation_analysis_results/
        # Go up to research/ root, then to analysis/
        research_root = self.scripts_dir.parent.parent.parent
        analysis_root = research_root / "analysis" / "correlation-analysis"
        self.output_dir = analysis_root / "correlation_analysis_results" / self.apt_type / f"run-{self.run_id}"
        
        self.ipv6_mapping = {}
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Computer-IP mapping (shared read-only data - thread safe)
        self.COMPUTER_IP_MAPPING = {
        'theblock.boombox.local': '10.1.0.5',
        'waterfalls.boombox.local': '10.1.0.6',
        'endofroad.boombox.local': '10.1.0.7',
        'diskjockey.boombox.local': '10.1.0.4',
        'toto.boombox.local': '10.1.0.8'
        }
        # self.COMPUTER_IP_MAPPING = {
        #     'theblock.boombox.local': '10.1.0.5',
        #     'THEBLOCK.boombox.local': '10.1.0.5',
        #     'Theblock.boombox.local': '10.1.0.5',
        #     'waterfalls.boombox.local': '10.1.0.6', 
        #     'WATERFALLS.boombox.local': '10.1.0.6',
        #     'Waterfalls.boombox.local': '10.1.0.6',
        #     'endofroad.boombox.local': '10.1.0.7',
        #     'ENDOFROAD.boombox.local': '10.1.0.7',
        #     'Endofroad.boombox.local': '10.1.0.7',
        #     'diskjockey.boombox.local': '10.1.0.4',
        #     'DISKJOCKEY.boombox.local': '10.1.0.4',
        #     'Diskjockey.boombox.local': '10.1.0.4',
        #     'toto.boombox.local': '10.1.0.8',
        #     'TOTO.boombox.local': '10.1.0.8',
        #     'Toto.boombox.local': '10.1.0.8'
        # }
        
        self.IP_COMPUTER_MAPPING = {v: k for k, v in self.COMPUTER_IP_MAPPING.items()}
        
        # Computer name mapping (without boombox.local)
        self.COMPUTER_SHORT_NAMES = {
            'theblock.boombox.local': 'theblock',
            'waterfalls.boombox.local': 'waterfalls', 
            'endofroad.boombox.local': 'endofroad',
            'diskjockey.boombox.local': 'diskjockey',
            'toto.boombox.local': 'toto',
        }
        # self.COMPUTER_SHORT_NAMES = {
        #     'theblock.boombox.local': 'theblock',
        #     'THEBLOCK.boombox.local': 'theblock',
        #     'Theblock.boombox.local': 'theblock',
        #     'waterfalls.boombox.local': 'waterfalls', 
        #     'WATERFALLS.boombox.local': 'waterfalls',
        #     'Waterfalls.boombox.local': 'waterfalls',
        #     'endofroad.boombox.local': 'endofroad',
        #     'ENDOFROAD.boombox.local': 'endofroad',
        #     'Endofroad.boombox.local': 'endofroad',
        #     'diskjockey.boombox.local': 'diskjockey',
        #     'DISKJOCKEY.boombox.local': 'diskjockey',
        #     'Diskjockey.boombox.local': 'diskjockey',
        #     'toto.boombox.local': 'toto',
        #     'TOTO.boombox.local': 'toto',
        #     'Toto.boombox.local': 'toto'
        # }
        
        # Thread-safe progress tracking
        self.progress_lock = threading.Lock()
        self.processed_chunks = 0
        self.total_chunks = 0
    
    def find_dataset_files(self):
        """Find dataset files in APT directory structure - handles multiple naming patterns"""
        print(f"🔍 Looking for datasets in: {self.apt_dir}")
        
        # Try organized structure first, then various naming patterns
        sysmon_paths = [
            self.apt_dir / "02_data_processing" / "processed_data" / f"sysmon-run-{self.run_id}.csv",
            self.apt_dir / f"sysmon-run-{self.run_id}.csv",
            self.apt_dir / f"sysmon-run-{self.run_id}-OLD.csv"
        ]
        
        network_paths = [
            self.apt_dir / "02_data_processing" / "processed_data" / f"netflow-run-{self.run_id}.csv",
            self.apt_dir / f"netflow-run-{self.run_id}.csv"
        ]
        
        sysmon_file = None
        network_file = None
        
        # Try predefined paths first
        for path in sysmon_paths:
            if path.exists():
                sysmon_file = path
                break
        
        # If not found, try glob pattern for sysmon files
        if not sysmon_file:
            import glob
            pattern = str(self.apt_dir / f"sysmon*{self.run_id}*.csv")
            candidates = glob.glob(pattern)
            if candidates:
                sysmon_file = Path(candidates[0])  # Take the first match
        
        for path in network_paths:
            if path.exists():
                network_file = path
                break
        
        if not sysmon_file or not network_file:
            # Show what files we actually found for debugging
            print(f"❌ Could not find required files for {self.apt_type}-run-{self.run_id}")
            print(f"   Directory contents:")
            for item in self.apt_dir.iterdir():
                if item.is_file() and ('.csv' in item.name):
                    print(f"     {item.name}")
            raise FileNotFoundError(f"Could not find dataset files for {self.apt_type}-run-{self.run_id}")
        
        return sysmon_file, network_file
    
    def load_datasets(self):
        """Load datasets for analysis"""
        print(f"🔍 Loading Datasets for {self.apt_type.upper()}-Run-{self.run_id}")
        print("=" * 60)
        
        sysmon_file, network_file = self.find_dataset_files()
        
        print(f"Loading Sysmon events: {sysmon_file}")
        self.sysmon_df = pd.read_csv(sysmon_file)
        
        print(f"Loading Network events: {network_file}")
        self.network_df = pd.read_csv(network_file)
        
        print(f"✅ Loaded {len(self.sysmon_df):,} Sysmon events")
        print(f"✅ Loaded {len(self.network_df):,} Network flow events")
        
        # Extract IPv6 mappings
        self.extract_ipv6_mapping()
        
        return self.sysmon_df, self.network_df
    
    def extract_ipv6_mapping(self):
        """Extract IPv6 to IPv4 mapping dynamically from host_ip column"""
        print("Extracting IPv6 mappings...")
        
        if 'host_ip' not in self.network_df.columns:
            print("⚠️  No host_ip column found - skipping IPv6 mapping")
            return
        
        unique_host_ips = self.network_df['host_ip'].dropna().unique()
        
        for host_ip_str in unique_host_ips:
            try:
                ip_list = ast.literal_eval(host_ip_str)
                if len(ip_list) == 2:
                    ipv6, ipv4 = None, None
                    for ip in ip_list:
                        if ':' in ip and 'fe80:' in ip:
                            ipv6 = ip
                        elif '.' in ip and ip.startswith('10.1.0'):
                            ipv4 = ip
                    if ipv6 and ipv4:
                        self.ipv6_mapping[ipv6] = ipv4
            except:
                continue
        
        print(f"✅ Extracted {len(self.ipv6_mapping)} IPv6 mappings")
    
    def enhanced_ip_lookup(self, ip_address):
        """Enhanced IP lookup that handles IPv6 addresses - THREAD SAFE"""
        if pd.isna(ip_address):
            return None
        if ip_address in self.IP_COMPUTER_MAPPING:
            return ip_address
        if ip_address in self.ipv6_mapping:
            return self.ipv6_mapping[ip_address]
        return ip_address
    
    def process_chunk(self, chunk_data):
        """
        Process a chunk of network flows - THREAD SAFE FUNCTION
        
        This function is completely independent - no shared state between chunks.
        Each network flow is analyzed independently against the Sysmon dataset.
        """
        chunk_id, chunk_df, valid_sysmon = chunk_data
        
        # Local results for this chunk
        local_attribution_results = {
            'attributed': 0,
            'unattributed_pid_missing': 0,
            'unattributed_ip_mismatch': 0,
            'unattributed_exe_mismatch': 0
        }
        
        local_detailed_results = []
        
        # Process each flow in the chunk
        for idx, flow in chunk_df.iterrows():
            process_id = flow['process_pid']
            process_exe = str(flow['process_executable']).lower()
            
            # Check if PID exists in Sysmon
            sysmon_with_pid = valid_sysmon[valid_sysmon['ProcessId'] == process_id]
            
            if len(sysmon_with_pid) == 0:
                local_attribution_results['unattributed_pid_missing'] += 1
                local_detailed_results.append({
                    'process_executable': flow['process_executable'],
                    'attribution_result': 'PID Missing',
                    'computer': 'Unknown'
                })
                continue
            
            # PID exists, check IP matching
            found_ip_match = False
            matching_computer = None
            matching_computer_sysmon = None
            
            for ip_field in ['source_ip', 'destination_ip']:
                flow_ip = flow[ip_field]
                mapped_ip = self.enhanced_ip_lookup(flow_ip)
                
                if mapped_ip in self.IP_COMPUTER_MAPPING:
                    computer = self.IP_COMPUTER_MAPPING[mapped_ip]
                    computer_sysmon = sysmon_with_pid[sysmon_with_pid['Computer'] == computer]
                    
                    if len(computer_sysmon) > 0:
                        found_ip_match = True
                        matching_computer = computer
                        matching_computer_sysmon = computer_sysmon
                        break
            
            if not found_ip_match:
                local_attribution_results['unattributed_ip_mismatch'] += 1
                local_detailed_results.append({
                    'process_executable': flow['process_executable'],
                    'attribution_result': 'IP Mismatch',
                    'computer': 'Unknown'
                })
                continue
            
            # Check executable matching
            exe_matches = matching_computer_sysmon[
                matching_computer_sysmon['Image'].str.lower() == process_exe
            ]
            
            if len(exe_matches) == 0:
                local_attribution_results['unattributed_exe_mismatch'] += 1
                local_detailed_results.append({
                    'process_executable': flow['process_executable'],
                    'attribution_result': 'Executable Mismatch',
                    'computer': self.COMPUTER_SHORT_NAMES.get(matching_computer, matching_computer)
                })
                continue
            
            # Success!
            local_attribution_results['attributed'] += 1
            local_detailed_results.append({
                'process_executable': flow['process_executable'],
                'attribution_result': 'Successfully Attributed',
                'computer': self.COMPUTER_SHORT_NAMES.get(matching_computer, matching_computer)
            })
        
        # Update progress (thread-safe)
        with self.progress_lock:
            self.processed_chunks += 1
            progress = (self.processed_chunks / self.total_chunks) * 100
            print(f"  Progress: {self.processed_chunks}/{self.total_chunks} chunks ({progress:.1f}%)")
        
        return {
            'chunk_id': chunk_id,
            'attribution_results': local_attribution_results,
            'detailed_results': local_detailed_results,
            'chunk_size': len(chunk_df)
        }
    
    def perform_correlation_analysis(self):
        """Perform multithreaded cross-domain correlation analysis"""
        print(f"\n🔍 Performing Multithreaded Cross-Domain Correlation Analysis")
        print(f"💻 Using {self.max_workers} worker threads")
        print("-" * 60)
        
        # Filter analyzable flows
        analyzable_flows = self.network_df[
            self.network_df['source_ip'].notna() & 
            self.network_df['destination_ip'].notna() &
            self.network_df['source_port'].notna() & 
            self.network_df['destination_port'].notna() &
            self.network_df['process_pid'].notna() & 
            self.network_df['process_executable'].notna()
        ].copy()
        
        incomplete_flows = len(self.network_df) - len(analyzable_flows)
        
        print(f"✅ Total network events: {len(self.network_df):,}")
        print(f"✅ Analyzable TCP/UDP flows: {len(analyzable_flows):,} ({len(analyzable_flows)/len(self.network_df)*100:.1f}%)")
        print(f"✅ Incomplete/ARP flows: {incomplete_flows:,} ({incomplete_flows/len(self.network_df)*100:.1f}%)")
        
        # Prepare Sysmon lookup (shared read-only data)
        valid_sysmon = self.sysmon_df[
            self.sysmon_df[['ProcessGuid', 'ProcessId', 'Image', 'Computer']].notna().all(axis=1) &
            self.sysmon_df['Computer'].isin(self.COMPUTER_IP_MAPPING.keys())
        ].copy()
        
        print(f"✅ Valid Sysmon events for correlation: {len(valid_sysmon):,}")
        
        # Calculate optimal chunk size based on worker count and dataset size
        base_chunk_size = max(1000, len(analyzable_flows) // (self.max_workers * 4))  # 4 chunks per worker
        chunk_size = min(base_chunk_size, 50000)  # Cap at 50K for memory efficiency
        
        print(f"✅ Chunk size: {chunk_size:,} flows per chunk")
        
        # Create chunks for multithreading
        chunks = []
        chunk_id = 0
        for start_idx in range(0, len(analyzable_flows), chunk_size):
            end_idx = min(start_idx + chunk_size, len(analyzable_flows))
            chunk_df = analyzable_flows.iloc[start_idx:end_idx].copy()
            chunks.append((chunk_id, chunk_df, valid_sysmon))
            chunk_id += 1
        
        self.total_chunks = len(chunks)
        print(f"✅ Created {self.total_chunks} chunks for processing")
        
        # Process chunks in parallel
        print(f"\n🚀 Starting multithreaded processing...")
        start_time = time.time()
        
        # Aggregate results
        global_attribution_results = {
            'attributed': 0,
            'unattributed_pid_missing': 0,
            'unattributed_ip_mismatch': 0,
            'unattributed_exe_mismatch': 0
        }
        
        global_detailed_results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all chunks
            future_to_chunk = {executor.submit(self.process_chunk, chunk): chunk[0] for chunk in chunks}
            
            # Collect results as they complete
            for future in as_completed(future_to_chunk):
                try:
                    result = future.result()
                    
                    # Aggregate attribution results
                    for key, value in result['attribution_results'].items():
                        global_attribution_results[key] += value
                    
                    # Aggregate detailed results
                    global_detailed_results.extend(result['detailed_results'])
                    
                except Exception as e:
                    chunk_id = future_to_chunk[future]
                    print(f"❌ Chunk {chunk_id} failed: {e}")
        
        processing_time = time.time() - start_time
        flows_per_second = len(analyzable_flows) / processing_time if processing_time > 0 else 0
        
        print(f"\n⚡ Processing completed in {processing_time:.1f} seconds")
        print(f"⚡ Throughput: {flows_per_second:,.0f} flows/second")
        
        # Calculate results
        total_analyzable = len(analyzable_flows)
        results_pct = {key: (value / total_analyzable * 100) for key, value in global_attribution_results.items()}
        
        # Store results
        self.correlation_results = {
            'metadata': {
                'apt_type': self.apt_type,
                'run_id': self.run_id,
                'analysis_timestamp': datetime.now().isoformat(),
                'total_network_events': len(self.network_df),
                'analyzable_flows': total_analyzable,
                'incomplete_flows': incomplete_flows,
                'data_directory': str(self.apt_dir),
                'multithreading': {
                    'max_workers': self.max_workers,
                    'chunks_processed': self.total_chunks,
                    'chunk_size': chunk_size,
                    'processing_time_seconds': processing_time,
                    'throughput_flows_per_second': flows_per_second
                }
            },
            'attribution_counts': global_attribution_results,
            'attribution_percentages': results_pct,
            'detailed_results': global_detailed_results
        }
        
        print(f"\n📊 CORRELATION ANALYSIS RESULTS:")
        print(f"✅ Successfully Attributed: {global_attribution_results['attributed']:,} ({results_pct['attributed']:.1f}%)")
        print(f"❌ PID Missing: {global_attribution_results['unattributed_pid_missing']:,} ({results_pct['unattributed_pid_missing']:.1f}%)")
        print(f"❌ IP Mismatch: {global_attribution_results['unattributed_ip_mismatch']:,} ({results_pct['unattributed_ip_mismatch']:.1f}%)")
        print(f"❌ Executable Mismatch: {global_attribution_results['unattributed_exe_mismatch']:,} ({results_pct['unattributed_exe_mismatch']:.1f}%)")
        
        return self.correlation_results
    
    def create_focused_plots(self):
        """Create the 2 focused plots requested - identical to original"""
        print(f"\n🎨 Creating Focused Visualizations")
        print("-" * 40)
        
        results = self.correlation_results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Attribution Results by Count (with ARP traffic)
        categories = ['Attributed', 'PID Missing', 'IP Mismatch', 'Exe Mismatch', 'ARP Traffic']
        counts = [
            results['attribution_counts']['attributed'],
            results['attribution_counts']['unattributed_pid_missing'],
            results['attribution_counts']['unattributed_ip_mismatch'],
            results['attribution_counts']['unattributed_exe_mismatch'],
            results['metadata']['incomplete_flows']
        ]
        
        total_events = results['metadata']['total_network_events']
        percentages = [(count / total_events * 100) for count in counts]
        
        colors = ['#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#95a5a6']
        
        bars = ax1.bar(categories, counts, color=colors)
        ax1.set_title(f'{self.apt_type.upper()}-Run-{self.run_id}: Attribution Results (MT)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Number of Network Events')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add labels with counts and percentages
        for bar, count, pct in zip(bars, counts, percentages):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                     f'{count:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Plot 2: Top 10 Unattributed Executables (with computer info)
        detailed_df = pd.DataFrame(results['detailed_results'])
        unattributed_df = detailed_df[detailed_df['attribution_result'] != 'Successfully Attributed']
        
        if len(unattributed_df) > 0:
            # Extract executable names and add computer info
            unattributed_df = unattributed_df.copy()
            unattributed_df['exe_name'] = unattributed_df['process_executable'].apply(
                lambda x: x.split('\\\\')[-1] if '\\\\' in str(x) else str(x)
            )
            
            # Group by executable and computer
            exe_computer_counts = unattributed_df.groupby(['exe_name', 'computer']).size().reset_index(name='count')
            exe_computer_counts['label'] = exe_computer_counts['exe_name'] + ' (' + exe_computer_counts['computer'] + ')'
            
            # Get top 10
            top_10 = exe_computer_counts.nlargest(10, 'count')
            
            if len(top_10) > 0:
                bars = ax2.barh(range(len(top_10)), top_10['count'], color='#e74c3c')
                ax2.set_yticks(range(len(top_10)))
                ax2.set_yticklabels(top_10['label'], fontsize=9)
                ax2.set_xlabel('Number of Unattributed Flows')
                ax2.set_title(f'Top 10 Unattributed Executables (MT)', fontsize=14, fontweight='bold')
                ax2.invert_yaxis()
                
                # Add count labels
                for i, (bar, count) in enumerate(zip(bars, top_10['count'])):
                    width = bar.get_width()
                    ax2.text(width + width*0.01, bar.get_y() + bar.get_height()/2.,
                             f'{count:,}', ha='left', va='center', fontsize=9)
            else:
                ax2.text(0.5, 0.5, 'No unattributed\\nexecutables', ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Top Unattributed Executables (None Found)', fontsize=14)
        else:
            ax2.text(0.5, 0.5, 'No unattributed\\flows found', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Top Unattributed Executables (All Attributed)', fontsize=14)
        
        plt.tight_layout()
        
        # Save plots
        plot_filename = f'{self.output_dir}/correlation_analysis_{self.apt_type}_run_{self.run_id}_multithreaded'
        plt.savefig(f'{plot_filename}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{plot_filename}.pdf', bbox_inches='tight')
        
        # Save NPZ for individual manipulation
        plot_data = {
            'plot1_attribution_results': {
                'categories': categories,
                'counts': counts,
                'percentages': percentages,
                'colors': colors
            },
            'plot2_top_unattributed': {
                'labels': top_10['label'].tolist() if len(unattributed_df) > 0 and len(top_10) > 0 else [],
                'counts': top_10['count'].tolist() if len(unattributed_df) > 0 and len(top_10) > 0 else []
            }
        }
        np.savez(f'{plot_filename}.npz', **plot_data)
        
        print(f"✅ Plots saved: {plot_filename}.[png|pdf|npz]")
        plt.close()
        
        return plot_data
    
    def export_batch_results(self):
        """Export results in formats suitable for batch analysis - identical to original"""
        print(f"\n💾 Exporting Batch Analysis Results")
        print("-" * 40)
        
        # Summary results for table building
        summary = {
            'apt_type': self.apt_type,
            'run_id': self.run_id,
            'total_events': self.correlation_results['metadata']['total_network_events'],
            'analyzable_flows': self.correlation_results['metadata']['analyzable_flows'],
            'attributed_count': self.correlation_results['attribution_counts']['attributed'],
            'attributed_pct': self.correlation_results['attribution_percentages']['attributed'],
            'pid_missing_count': self.correlation_results['attribution_counts']['unattributed_pid_missing'],
            'pid_missing_pct': self.correlation_results['attribution_percentages']['unattributed_pid_missing'],
            'ip_mismatch_count': self.correlation_results['attribution_counts']['unattributed_ip_mismatch'],
            'ip_mismatch_pct': self.correlation_results['attribution_percentages']['unattributed_ip_mismatch'],
            'exe_mismatch_count': self.correlation_results['attribution_counts']['unattributed_exe_mismatch'],
            'exe_mismatch_pct': self.correlation_results['attribution_percentages']['unattributed_exe_mismatch'],
            'arp_traffic_count': self.correlation_results['metadata']['incomplete_flows'],
            'arp_traffic_pct': self.correlation_results['metadata']['incomplete_flows'] / self.correlation_results['metadata']['total_network_events'] * 100,
            # Multithreading metadata
            'processing_time_seconds': self.correlation_results['metadata']['multithreading']['processing_time_seconds'],
            'throughput_flows_per_second': self.correlation_results['metadata']['multithreading']['throughput_flows_per_second'],
            'workers_used': self.correlation_results['metadata']['multithreading']['max_workers']
        }
        
        # Export summary CSV (append mode for batch processing)
        analysis_root = self.scripts_dir.parent.parent.parent / "analysis" / "correlation-analysis"
        summary_file = analysis_root / 'correlation_analysis_results' / 'batch_summary_results_multithreaded.csv'
        summary_df = pd.DataFrame([summary])
        
        if summary_file.exists():
            summary_df.to_csv(summary_file, mode='a', header=False, index=False)
        else:
            summary_file.parent.mkdir(parents=True, exist_ok=True)
            summary_df.to_csv(summary_file, index=False)
        
        # Export detailed results
        results_file = self.output_dir / 'detailed_correlation_results_multithreaded.json'
        with open(results_file, 'w') as f:
            json.dump(self.correlation_results, f, indent=2, default=str)
        
        # Export unattributed executables for this run
        if self.correlation_results['detailed_results']:
            detailed_df = pd.DataFrame(self.correlation_results['detailed_results'])
            unattrib_file = self.output_dir / 'unattributed_executables_multithreaded.csv'
            detailed_df.to_csv(unattrib_file, index=False)
            
            print(f"✅ Detailed results: {results_file}")
            print(f"✅ Unattributed executables: {unattrib_file}")
        
        print(f"✅ Summary added to: {summary_file}")
        
        return summary
    
    def run_complete_analysis(self):
        """Run complete correlation analysis"""
        print(f"🚀 Multithreaded Cross-Domain Correlation Analysis")
        print(f"APT Type: {self.apt_type.upper()}, Run ID: {self.run_id}")
        print(f"Working from: {self.scripts_dir}")
        print(f"Data directory: {self.apt_dir}")
        print(f"💻 Max workers: {self.max_workers}")
        print("=" * 80)
        
        try:
            # Verify we're in scripts directory
            if not self.scripts_dir.name == 'scripts':
                print(f"⚠️  Warning: Not running from scripts directory (current: {self.scripts_dir.name})")
            
            # Load datasets
            self.load_datasets()
            
            # Perform correlation analysis
            self.perform_correlation_analysis()
            
            # Create focused plots
            self.create_focused_plots()
            
            # Export results
            summary = self.export_batch_results()
            
            print(f"\n🎯 MULTITHREADED ANALYSIS COMPLETE FOR {self.apt_type.upper()}-Run-{self.run_id}")
            print(f"✅ Attribution Success Rate: {summary['attributed_pct']:.1f}%")
            print(f"⚡ Processing Speed: {summary['throughput_flows_per_second']:,.0f} flows/second")
            print(f"💻 Workers Used: {summary['workers_used']}")
            print(f"✅ Results exported to: {self.output_dir}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error in analysis for {self.apt_type}-run-{self.run_id}: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    parser = argparse.ArgumentParser(description='Multithreaded Cross-Domain Correlation Analyzer')
    parser.add_argument('--run-id', required=True, help='Run ID (e.g., 04, 51)')
    parser.add_argument('--apt-type', default='apt-1', help='APT type (default: apt-1)')
    parser.add_argument('--workers', type=int, help='Number of worker threads (default: min(32, CPU count))')
    
    args = parser.parse_args()
    
    analyzer = MultithreadedCorrelationAnalyzer(args.run_id, args.apt_type, args.workers)
    success = analyzer.run_complete_analysis()
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())