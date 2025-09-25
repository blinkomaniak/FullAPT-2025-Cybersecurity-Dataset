#!/usr/bin/env python3
"""
Comprehensive Network Community ID Analyzer
==========================================

DESCRIPTION:
    Provides comprehensive analysis of network community IDs across APT runs,
    including traffic categorization, bidirectionality analysis, and detailed 
    reporting of inconsistent flows.

PREREQUISITES:
    - Raw NetFlow datasets in: data-raw/apt-X/apt-X-run-XX/netflow-run-XX.csv
    - Python packages: pandas, numpy, matplotlib, argparse, multiprocessing

USAGE:
    # Run from project root directory (/home/researcher/Downloads/research/)
    cd /home/researcher/Downloads/research/
    
    # Single run analysis
    python3 dataset/scripts/exploratory/1_comprehensive_network_community_id_analyzer.py --apt-type apt-1 --run-id 04
    
    # All runs batch analysis (multithreaded)
    python3 dataset/scripts/exploratory/1_comprehensive_network_community_id_analyzer.py --all --threads 8
    
    # Custom thread count
    python3 dataset/scripts/exploratory/1_comprehensive_network_community_id_analyzer.py --all --threads 4

COMMAND LINE OPTIONS:
    --apt-type      APT campaign type (apt-1, apt-2, apt-3, apt-4, apt-5, apt-6)
    --run-id        Specific run ID (01, 02, 03, ..., 51)
    --all           Process all available APT runs across all types
    --threads       Number of worker threads for parallel processing (default: auto)

INPUT REQUIREMENTS:
    - NetFlow CSV files with network_community_id column
    - Directory structure: data-raw/apt-X/apt-X-run-XX/netflow-run-XX.csv

OUTPUT GENERATED:
    analysis/network-community-id-analysis/apt-X/apt-X-run-XX/
    ├── network_community_id_analysis.json          # Comprehensive analysis results
    ├── traffic_categories_distribution.png         # Traffic type breakdown
    ├── traffic_categories_distribution.npz         # Data for batch processing
    ├── protocol_distribution_stacked.png           # Protocol analysis
    └── protocol_distribution_stacked.npz           # Data for batch processing

EXPECTED RUNTIME:
    - Single run: 30-90 seconds (depends on dataset size)
    - All runs batch: 10-30 minutes (depends on thread count)

KEY FEATURES:
    - Traffic categorization: Non-IP, ICMP/ICMPv6, IP-Traffic
    - Bidirectionality analysis for IP-Traffic flows  
    - Detailed inconsistent flow reporting
    - Protocol distribution analysis
    - Multithreaded processing for performance
    - Dual visualizations (PNG + NPZ format)

PURPOSE:
    Validates network flow data quality and consistency before temporal correlation.
    Part of Phase 1 data validation workflow.

EXAMPLE WORKFLOW:
    # Validate single APT run before correlation
    python3 dataset/scripts/exploratory/1_comprehensive_network_community_id_analyzer.py --apt-type apt-1 --run-id 10
    
    # Validate all runs for comprehensive analysis
    python3 dataset/scripts/exploratory/1_comprehensive_network_community_id_analyzer.py --all --threads 8

TROUBLESHOOTING:
    - "No netflow files found": Check data-raw/ directory structure
    - "Memory error": Reduce --threads or process runs individually
    - "Permission error": Ensure write access to analysis/ directory
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for threading compatibility
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

class NetworkCommunityIdAnalyzer:
    """Comprehensive Network Community ID Analysis"""
    
    def __init__(self, max_workers=None):
        self.base_path = Path("/home/researcher/Downloads/research/dataset")
        self.output_base = Path("/home/researcher/Downloads/research/analysis/network-community-id-analysis")
        self.max_workers = max_workers or min(mp.cpu_count(), 8)
        self.results_lock = Lock()
        
        # Results aggregation
        self.global_results = {
            'processed_runs': 0,
            'failed_runs': 0,
            'total_events_analyzed': 0,
            'total_community_ids_analyzed': 0,
            'processing_times': []
        }
        
        # Setup matplotlib for better plots
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
    
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
    
    def categorize_traffic(self, df):
        """Categorize netflow events into Non-IP, ICMP/ICMPv6, and IP-Traffic"""
        
        # Required columns
        ip_cols = ['source_ip', 'destination_ip']
        port_cols = ['source_port', 'destination_port']
        transport_col = 'network_transport'
        
        results = {
            'non_ip': {'count': 0, 'events': []},
            'icmp_icmpv6': {'count': 0, 'protocols': {}, 'events': []},
            'ip_traffic': {'count': 0, 'protocols': {}, 'events': []},
            'total_events': len(df)
        }
        
        for idx, row in df.iterrows():
            # Check if required columns exist
            has_ips = all(col in df.columns for col in ip_cols)
            has_ports = all(col in df.columns for col in port_cols)
            has_transport = transport_col in df.columns
            
            if not (has_ips and has_ports and has_transport):
                continue
            
            # Extract values
            src_ip = row.get('source_ip')
            dst_ip = row.get('destination_ip')
            src_port = row.get('source_port')
            dst_port = row.get('destination_port')
            transport = row.get('network_transport')
            
            # Check for nulls (pd.isna handles various null types)
            ip_valid = not (pd.isna(src_ip) or pd.isna(dst_ip))
            ports_valid = not (pd.isna(src_port) or pd.isna(dst_port))
            transport_valid = not pd.isna(transport)
            
            # Categorize
            if not ip_valid and not ports_valid and not transport_valid:
                # Non-IP: all null
                results['non_ip']['count'] += 1
                results['non_ip']['events'].append({
                    'index': idx,
                    'source_ip': src_ip,
                    'destination_ip': dst_ip,
                    'source_port': src_port,
                    'destination_port': dst_port,
                    'network_transport': transport
                })
                
            elif ip_valid and not ports_valid and transport_valid:
                # ICMP/ICMPv6: IPs valid, ports null
                results['icmp_icmpv6']['count'] += 1
                protocol = str(transport).lower()
                results['icmp_icmpv6']['protocols'][protocol] = results['icmp_icmpv6']['protocols'].get(protocol, 0) + 1
                
                # Store sample events (limit to avoid memory issues)
                if len(results['icmp_icmpv6']['events']) < 100:
                    results['icmp_icmpv6']['events'].append({
                        'index': idx,
                        'source_ip': src_ip,
                        'destination_ip': dst_ip,
                        'network_transport': transport
                    })
                
            elif ip_valid and ports_valid and transport_valid:
                # IP-Traffic: all valid
                results['ip_traffic']['count'] += 1
                protocol = str(transport).lower()
                results['ip_traffic']['protocols'][protocol] = results['ip_traffic']['protocols'].get(protocol, 0) + 1
                
                # Store sample events (limit to avoid memory issues)
                if len(results['ip_traffic']['events']) < 100:
                    results['ip_traffic']['events'].append({
                        'index': idx,
                        'source_ip': src_ip,
                        'destination_ip': dst_ip,
                        'source_port': src_port,
                        'destination_port': dst_port,
                        'network_transport': transport
                    })
        
        return results
    
    def analyze_bidirectionality(self, df):
        """Analyze bidirectionality patterns for IP-Traffic flows only"""
        
        # Filter to IP-Traffic flows only (all fields valid)
        required_cols = ['network_community_id', 'source_ip', 'destination_ip', 
                        'source_port', 'destination_port', 'network_transport']
        
        if not all(col in df.columns for col in required_cols):
            return {'error': 'Missing required columns', 'analyzed_flows': 0}
        
        # Filter to rows with all valid values
        ip_traffic_df = df.dropna(subset=required_cols)
        
        if len(ip_traffic_df) == 0:
            return {'analyzed_flows': 0, 'bidirectional_flows': 0, 'unidirectional_flows': 0, 'inconsistent_flows': 0}
        
        results = {
            'analyzed_flows': 0,
            'bidirectional_flows': 0,
            'unidirectional_flows': 0,
            'inconsistent_flows': 0,
            'bidirectional_examples': [],
            'inconsistent_details': []
        }
        
        # Group by network_community_id
        community_groups = ip_traffic_df.groupby('network_community_id')
        results['analyzed_flows'] = len(community_groups)
        
        for community_id, group in community_groups:
            if len(group) < 2:
                results['unidirectional_flows'] += 1
                continue
            
            # Extract unique 5-tuples
            tuples_df = group[['source_ip', 'destination_ip', 'source_port', 'destination_port', 'network_transport']].drop_duplicates()
            
            if len(tuples_df) == 1:
                # All events have identical 5-tuple
                results['unidirectional_flows'] += 1
                
            elif len(tuples_df) == 2:
                # Check if bidirectional (swapped source/dest)
                tuple1 = tuples_df.iloc[0]
                tuple2 = tuples_df.iloc[1]
                
                is_bidirectional = (
                    tuple1['source_ip'] == tuple2['destination_ip'] and
                    tuple1['destination_ip'] == tuple2['source_ip'] and
                    tuple1['source_port'] == tuple2['destination_port'] and
                    tuple1['destination_port'] == tuple2['source_port'] and
                    tuple1['network_transport'] == tuple2['network_transport']
                )
                
                if is_bidirectional:
                    results['bidirectional_flows'] += 1
                    
                    # Collect example (limit to avoid memory issues)
                    if len(results['bidirectional_examples']) < 10:
                        example = {
                            'community_id': community_id,
                            'event_count': len(group),
                            'direction_1': {
                                'source_ip': tuple1['source_ip'],
                                'destination_ip': tuple1['destination_ip'],
                                'source_port': int(tuple1['source_port']),
                                'destination_port': int(tuple1['destination_port']),
                                'transport': tuple1['network_transport']
                            },
                            'direction_2': {
                                'source_ip': tuple2['source_ip'],
                                'destination_ip': tuple2['destination_ip'],
                                'source_port': int(tuple2['source_port']),
                                'destination_port': int(tuple2['destination_port']),
                                'transport': tuple2['network_transport']
                            }
                        }
                        results['bidirectional_examples'].append(example)
                else:
                    results['inconsistent_flows'] += 1
                    
                    # Collect inconsistent details
                    if len(results['inconsistent_details']) < 20:
                        inconsistent_detail = {
                            'community_id': community_id,
                            'event_count': len(group),
                            'unique_tuples_count': len(tuples_df),
                            'tuples': []
                        }
                        
                        for _, tuple_row in tuples_df.iterrows():
                            inconsistent_detail['tuples'].append({
                                'source_ip': tuple_row['source_ip'],
                                'destination_ip': tuple_row['destination_ip'],
                                'source_port': int(tuple_row['source_port']),
                                'destination_port': int(tuple_row['destination_port']),
                                'transport': tuple_row['network_transport']
                            })
                        
                        # Add sample events from this inconsistent community ID
                        sample_events = group.head(5).to_dict('records')
                        inconsistent_detail['sample_events'] = sample_events
                        
                        results['inconsistent_details'].append(inconsistent_detail)
            else:
                # More than 2 unique tuples - highly inconsistent
                results['inconsistent_flows'] += 1
                
                # Collect details for complex inconsistencies
                if len(results['inconsistent_details']) < 20:
                    inconsistent_detail = {
                        'community_id': community_id,
                        'event_count': len(group),
                        'unique_tuples_count': len(tuples_df),
                        'complexity': 'high',
                        'tuples': []
                    }
                    
                    # Limit tuples to avoid memory issues
                    for _, tuple_row in tuples_df.head(10).iterrows():
                        inconsistent_detail['tuples'].append({
                            'source_ip': tuple_row['source_ip'],
                            'destination_ip': tuple_row['destination_ip'],
                            'source_port': int(tuple_row['source_port']),
                            'destination_port': int(tuple_row['destination_port']),
                            'transport': tuple_row['network_transport']
                        })
                    
                    results['inconsistent_details'].append(inconsistent_detail)
        
        return results
    
    def create_visualizations(self, traffic_results, bidirectionality_results, output_dir):
        """Create PNG and NPZ visualizations"""
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # 1. Traffic Categories Bar Chart
            self.create_traffic_categories_chart(traffic_results, output_dir)
            
            # 2. Protocol Distribution Stacked Chart
            self.create_protocol_distribution_chart(traffic_results, output_dir)
        finally:
            # Clean up any remaining matplotlib resources
            plt.close('all')
    
    def create_traffic_categories_chart(self, traffic_results, output_dir):
        """Create traffic categories distribution chart"""
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Data preparation
        categories = ['Non-IP', 'ICMP/ICMPv6', 'IP-Traffic']
        counts = [
            traffic_results['non_ip']['count'],
            traffic_results['icmp_icmpv6']['count'],
            traffic_results['ip_traffic']['count']
        ]
        total_events = traffic_results['total_events']
        percentages = [(count / total_events * 100) if total_events > 0 else 0 for count in counts]
        
        # Color scheme
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
        
        # Create bar chart
        bars = ax.bar(categories, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add value labels on bars
        for bar, count, pct in zip(bars, counts, percentages):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                   f'{count:,}\n({pct:.1f}%)', 
                   ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # Formatting
        ax.set_title('NetFlow Events Distribution by Traffic Category', fontsize=16, fontweight='bold', pad=20)
        ax.set_ylabel('Number of Events', fontsize=12, fontweight='bold')
        ax.set_xlabel('Traffic Category', fontsize=12, fontweight='bold')
        
        # Add grid
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_axisbelow(True)
        
        # Format y-axis with comma separator
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
        
        # Add summary text
        summary_text = f'Total Events: {total_events:,}'
        ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, 
               fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save PNG
        png_file = output_dir / 'traffic_categories_distribution.png'
        plt.savefig(png_file, dpi=300, bbox_inches='tight')
        
        # Save NPZ (matplotlib figure object)
        npz_file = output_dir / 'traffic_categories_distribution.npz'
        np.savez_compressed(npz_file, 
                           categories=categories, 
                           counts=counts, 
                           percentages=percentages,
                           total_events=total_events,
                           colors=colors)
        
        plt.close(fig)  # Explicitly close the figure to prevent memory leaks
    
    def create_protocol_distribution_chart(self, traffic_results, output_dir):
        """Create stacked bar chart for protocol distribution within categories"""
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Data preparation
        categories = []
        protocol_data = {}
        
        # ICMP/ICMPv6 protocols
        if traffic_results['icmp_icmpv6']['count'] > 0:
            categories.append('ICMP/ICMPv6')
            protocol_data['ICMP/ICMPv6'] = traffic_results['icmp_icmpv6']['protocols']
        
        # IP-Traffic protocols
        if traffic_results['ip_traffic']['count'] > 0:
            categories.append('IP-Traffic')
            protocol_data['IP-Traffic'] = traffic_results['ip_traffic']['protocols']
        
        if not categories:
            # No protocols to display
            ax.text(0.5, 0.5, 'No Protocol Data Available', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=16, fontweight='bold')
            ax.set_title('Protocol Distribution by Traffic Category', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # Save empty chart
            png_file = output_dir / 'protocol_distribution_stacked.png'
            plt.savefig(png_file, dpi=300, bbox_inches='tight')
            
            npz_file = output_dir / 'protocol_distribution_stacked.npz'
            np.savez_compressed(npz_file, categories=[], protocols=[], data=[])
            
            plt.close(fig)  # Explicitly close the figure to prevent memory leaks
            return
        
        # Collect all unique protocols across categories
        all_protocols = set()
        for protocols in protocol_data.values():
            all_protocols.update(protocols.keys())
        all_protocols = sorted(all_protocols)
        
        # Color palette for protocols
        protocol_colors = plt.cm.Set3(np.linspace(0, 1, len(all_protocols)))
        protocol_color_map = dict(zip(all_protocols, protocol_colors))
        
        # Prepare data for stacking
        bottoms = np.zeros(len(categories))
        legend_elements = []
        
        for protocol in all_protocols:
            values = []
            for category in categories:
                count = protocol_data[category].get(protocol, 0)
                values.append(count)
            
            if sum(values) > 0:  # Only plot if there's data
                bars = ax.bar(categories, values, bottom=bottoms, 
                             color=protocol_color_map[protocol], 
                             label=protocol.upper(),
                             alpha=0.8, edgecolor='white', linewidth=0.5)
                
                # Add protocol count labels on bars
                for i, (bar, value) in enumerate(zip(bars, values)):
                    if value > 0:
                        # Calculate percentage within category
                        total_for_category = sum(protocol_data[categories[i]].values())
                        pct = (value / total_for_category * 100) if total_for_category > 0 else 0
                        
                        # Position label in middle of bar segment
                        label_y = bottoms[i] + value / 2
                        
                        # Only show label if segment is large enough
                        if value > max(sum(protocol_data[cat].values()) for cat in categories) * 0.02:
                            ax.text(bar.get_x() + bar.get_width()/2., label_y,
                                   f'{value:,}\n({pct:.1f}%)', 
                                   ha='center', va='center', 
                                   fontweight='bold', fontsize=9,
                                   color='black' if pct > 15 else 'white')
                
                bottoms += values
                
                # Create legend element with absolute count
                total_protocol_count = sum(values)
                legend_elements.append(
                    mpatches.Patch(color=protocol_color_map[protocol], 
                                  label=f'{protocol.upper()}: {total_protocol_count:,}')
                )
        
        # Formatting
        ax.set_title('Protocol Distribution within Traffic Categories', fontsize=16, fontweight='bold', pad=20)
        ax.set_ylabel('Number of Events', fontsize=12, fontweight='bold')
        ax.set_xlabel('Traffic Category', fontsize=12, fontweight='bold')
        
        # Format y-axis
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
        
        # Add legend
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper right', 
                     bbox_to_anchor=(1.15, 1), fontsize=10)
        
        # Add grid
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        
        # Save PNG
        png_file = output_dir / 'protocol_distribution_stacked.png'
        plt.savefig(png_file, dpi=300, bbox_inches='tight')
        
        # Save NPZ
        npz_file = output_dir / 'protocol_distribution_stacked.npz'
        np.savez_compressed(npz_file,
                           categories=categories,
                           protocols=all_protocols,
                           protocol_data=protocol_data,
                           colors=protocol_colors)
        
        plt.close(fig)  # Explicitly close the figure to prevent memory leaks
    
    def analyze_single_run(self, netflow_file):
        """Analyze a single APT run"""
        
        start_time = datetime.now()
        
        try:
            # Extract run information
            apt_type, run_id = self.extract_run_info(netflow_file)
            if not apt_type or not run_id:
                print(f"❌ Could not extract run info from: {netflow_file}")
                return False
            
            print(f"🔬 Analyzing {apt_type}-run-{run_id}: {netflow_file.name}")
            
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
            
            # Perform traffic categorization
            print("   🔍 Categorizing traffic...")
            traffic_results = self.categorize_traffic(df)
            
            # Perform bidirectionality analysis
            print("   🔄 Analyzing bidirectionality...")
            bidirectionality_results = self.analyze_bidirectionality(df)
            
            # Create visualizations
            print("   📊 Creating visualizations...")
            self.create_visualizations(traffic_results, bidirectionality_results, output_dir)
            
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
                'traffic_categorization': traffic_results,
                'bidirectionality_analysis': bidirectionality_results,
                'summary_statistics': {
                    'non_ip_percentage': (traffic_results['non_ip']['count'] / len(df) * 100) if len(df) > 0 else 0,
                    'icmp_percentage': (traffic_results['icmp_icmpv6']['count'] / len(df) * 100) if len(df) > 0 else 0,
                    'ip_traffic_percentage': (traffic_results['ip_traffic']['count'] / len(df) * 100) if len(df) > 0 else 0,
                    'bidirectional_ratio': (bidirectionality_results['bidirectional_flows'] / bidirectionality_results['analyzed_flows'] * 100) if bidirectionality_results['analyzed_flows'] > 0 else 0
                }
            }
            
            # Save JSON results (clean data first to handle NaN values)
            json_file = output_dir / 'network_community_id_analysis.json'
            cleaned_results = clean_for_json(comprehensive_results)
            with open(json_file, 'w') as f:
                json.dump(cleaned_results, f, indent=2, default=str)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            print(f"   ✅ {apt_type}-run-{run_id} completed in {processing_time:.1f}s")
            print(f"   📁 Results saved: {output_dir}")
            
            # Update global results
            with self.results_lock:
                self.global_results['processed_runs'] += 1
                self.global_results['total_events_analyzed'] += len(df)
                self.global_results['total_community_ids_analyzed'] += len(df['network_community_id'].unique()) if 'network_community_id' in df.columns else 0
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
        
        print(f"🎯 Single Run Analysis: {apt_type}-run-{run_id}")
        print("=" * 60)
        
        return self.analyze_single_run(netflow_file)
    
    def analyze_all_runs(self):
        """Analyze all runs using multithreading"""
        
        print("🚀 Comprehensive Network Community ID Analysis - ALL RUNS")
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
        print("🎯 ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"✅ Successfully processed: {self.global_results['processed_runs']} runs")
        print(f"❌ Failed runs: {self.global_results['failed_runs']} runs")
        print(f"📊 Total events analyzed: {self.global_results['total_events_analyzed']:,}")
        print(f"🔄 Total community IDs: {self.global_results['total_community_ids_analyzed']:,}")
        
        if self.global_results['processing_times']:
            avg_time = np.mean(self.global_results['processing_times'])
            total_time = sum(self.global_results['processing_times'])
            print(f"⏱️ Average processing time per run: {avg_time:.1f}s")
            print(f"⏱️ Total processing time: {total_time/60:.1f} minutes")
        
        print(f"\n📁 Results saved in: {self.output_base}")
        
        return self.global_results['processed_runs'] > 0

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="Comprehensive Network Community ID Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze specific run
  python comprehensive_network_community_id_analyzer.py --apt-type apt-1 --run-id 04
  
  # Analyze all runs
  python comprehensive_network_community_id_analyzer.py --all
  
  # Use custom thread count
  python comprehensive_network_community_id_analyzer.py --all --threads 4
        """
    )
    
    # Execution mode
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--all', action='store_true',
                      help='Analyze all APT runs')
    group.add_argument('--apt-type', 
                      help='APT type (e.g., apt-1, apt-2)')
    
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
        # Initialize analyzer
        analyzer = NetworkCommunityIdAnalyzer(max_workers=args.threads)
        
        # Execute based on mode
        if args.all:
            success = analyzer.analyze_all_runs()
        else:
            success = analyzer.analyze_single_run_cli(args.apt_type, args.run_id)
        
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