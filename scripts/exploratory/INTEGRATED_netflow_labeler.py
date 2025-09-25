#!/usr/bin/env python3
"""
INTEGRATED NetFlow Labeling System - Single Script Approach
=========================================================

Combines functionality from TESTING_8, TESTING_9, and TESTING_10 into a single
interactive workflow with manual checkpoints.

COMPLETE WORKFLOW:
1. Load seed events and NetFlow data
2. Perform temporal correlation analysis  
3. ⏸️  MANUAL: Mark Direct-attribution and Subnetflow-attribution in verification_matrix
4. Generate sub-NetFlow timeline analysis (selective processing)
5. ⏸️  MANUAL: Fill seed_event assignments in subnetflow_assignment_template
6. Apply two-tier labeling system
7. Generate timeline visualizations
8. Create final labeled dataset

INPUT FILES:
- all_target_events_run-XX.csv (manually marked seed events)
- netflow-run-XX.csv (original NetFlow dataset)

OUTPUT FILES:
- verification_matrix_v2_run-XX.csv (correlation analysis + manual marks)
- subnetflow_assignment_template_run-XX.csv (sub-NetFlow assignments template)
- netflow-run-XX-labeled.csv (final labeled dataset)
- Timeline visualizations

USAGE:
    python3 INTEGRATED_netflow_labeler.py --apt-type apt-1 --run-id 04
    python3 INTEGRATED_netflow_labeler.py --resume --apt-type apt-1 --run-id 04
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Set
import warnings
warnings.filterwarnings('ignore')


class IntegratedNetFlowLabeler:
    """Integrated NetFlow labeling system with interactive workflow."""
    
    def __init__(self, debug: bool = False, use_automated_assignment: bool = False):
        """Initialize integrated NetFlow labeler."""
        # Set up logging
        log_level = logging.DEBUG if debug else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Base paths  
        self.scripts_dir = Path(__file__).parent  # Current scripts/exploratory/ directory
        project_root = self.scripts_dir.parent.parent  # Go to research/
        self.base_path = project_root / "dataset"  # Point to dataset folder
        
        # C2 server configuration
        self.c2_server_ip = "192.168.0.4"
        self.internal_ips = ['10.1.0.5', '10.1.0.6', '10.1.0.7', '10.1.0.8']

        # Feature flags
        self.use_automated_assignment = use_automated_assignment
        
        # Color schemes (consistent across all visualizations)
        self.colors = {
            'sysmon_eventid1': '#FF0000',    # Red - Process Creation
            'sysmon_eventid11': '#0000FF',   # Blue - File Create
            'sysmon_eventid23': '#00AA00',   # Green - File Delete
            'netflow_outbound': '#0066CC',   # Blue - Victim → C2
            'netflow_inbound': '#00AA44',    # Green - C2 → Victim
            'netflow_bidirectional': '#8800CC' # Purple - Bidirectional
        }
        
        # MITRE Tactic color palette (matching Sysmon dataset)
        self.tactic_colors = {
            'initial-access': '#000000',      # Black
            'execution': '#4169E1',           # Royal Blue
            'persistence': '#228B22',         # Forest Green
            'privilege-escalation': '#B22222', # Fire Brick Red
            'defense-evasion': '#FF8C00',     # Dark Orange
            'credential-access': '#FFD700',   # Gold
            'discovery': '#8B4513',           # Saddle Brown
            'lateral-movement': '#FF1493',    # Deep Pink
            'collection': '#9932CC',          # Dark Orchid
            'command-and-control': '#00CED1', # Dark Turquoise
            'exfiltration': '#32CD32',        # Lime Green
            'impact': '#DC143C'               # Crimson
        }
        self.benign_color = '#808080'  # Gray for benign events
        
        # Attribution tracking
        self.direct_attributions = 0
        self.subnetflow_attributions = 0
        self.conflicts = 0
        
        # Data holders
        self.seed_events = None
        self.netflow_data = None
        
        # Workflow state
        self.workflow_state = {'step': 'initial', 'files_created': [], 'manual_steps_completed': []}
        self.unlabeled = 0
        self.conflict_records = []
    
    def _get_machine_name(self, ip: str) -> str:
        """Get friendly machine name from IP address."""
        machine_names = {
            '10.1.0.5': 'WS1',      # Workstation 1
            '10.1.0.6': 'WebSrv',   # Email & Web Server  
            '10.1.0.7': 'DBSrv',    # Database Server
            '192.168.0.4': 'C2'     # C2 Server
        }
        return machine_names.get(ip, ip[-3:])  # Return last 3 chars if not found
    
    def _get_flow_type_color(self, flow_type_key: str) -> str:
        """Get unique color for each flow type combination."""
        # Predefined color palette for different flow types
        flow_colors = {
            'C2-WS1': '#FF6B6B',        # Red/Pink - C2 ↔ Workstation 1
            'C2-WebSrv': '#4ECDC4',     # Teal - C2 ↔ Web Server
            'C2-DBSrv': '#45B7D1',      # Blue - C2 ↔ Database Server
            'DBSrv-WebSrv': '#96CEB4',  # Green - Database ↔ Web Server
            'WebSrv-WS1': '#FECA57',    # Yellow - Web Server ↔ Workstation 1
            'DBSrv-WS1': '#FF9FF3',     # Purple - Database ↔ Workstation 1
        }
        
        # Return assigned color or generate a default
        return flow_colors.get(flow_type_key, '#808080')  # Gray as fallback
        self.unlabeled = 0
        self.conflict_records = []
        
        # Data caching (avoid reloading)
        self.seed_events = None
        self.netflow_data = None
        self.tactic_lookup = None
        
        # File paths (will be set based on arguments)
        self.file_paths = {}
        
        # Workflow state tracking
        self.workflow_state = {
            'step': 'initial',
            'files_created': [],
            'manual_steps_completed': [],
            'timestamp': None
        }
        
    def setup_file_paths(self, apt_type: str, run_id: str):
        """Set up all file paths based on apt_type and run_id."""
        apt_dir = self.base_path / apt_type / f"{apt_type}-run-{run_id}"
        results_dir = apt_dir / "netflow_event_tracing_analysis_results"
        
        self.file_paths = {
            'apt_dir': apt_dir,
            'results_dir': results_dir,
            'timeline_dir': results_dir,  # Use same directory for timeline outputs
            'seed_events': apt_dir / f"all_target_events_run-{run_id}.csv",
            'netflow_data': apt_dir / f"netflow-run-{run_id}.csv",
            'verification_matrix_create': results_dir / f"verification_matrix_run-{run_id}.csv",
            'verification_matrix': results_dir / f"verification_matrix_v2_run-{run_id}.csv",
            'subnetflow_template_create': results_dir / f"subnetflow_assignment_template_run-{run_id}.csv",
            'subnetflow_template': results_dir / f"subnetflow_assignment_template_v2_run-{run_id}.csv",
            'labeled_netflow': apt_dir / f"netflow-run-{run_id}-labeled.csv",
            'workflow_state': results_dir / f"workflow_state_run-{run_id}.json"
        }
        
        # Ensure directories exist
        results_dir.mkdir(exist_ok=True)
        
    def load_workflow_state(self) -> dict:
        """Load workflow state from disk if available."""
        if self.file_paths['workflow_state'].exists():
            try:
                with open(self.file_paths['workflow_state'], 'r') as f:
                    state = json.load(f)
                self.logger.info(f"📄 Loaded workflow state: {state['step']}")
                return state
            except Exception as e:
                self.logger.warning(f"⚠️ Could not load workflow state: {e}")
        
        return {
            'step': 'initial',
            'files_created': [],
            'manual_steps_completed': [],
            'timestamp': None
        }
        
    def save_workflow_state(self, step: str, files_created: List[str] = None):
        """Save current workflow state to disk."""
        self.workflow_state.update({
            'step': step,
            'files_created': files_created or self.workflow_state['files_created'],
            'timestamp': datetime.now().isoformat()
        })
        
        try:
            with open(self.file_paths['workflow_state'], 'w') as f:
                json.dump(self.workflow_state, f, indent=2)
        except Exception as e:
            self.logger.warning(f"⚠️ Could not save workflow state: {e}")
    
    def load_seed_events(self, apt_type: str, run_id: str) -> pd.DataFrame:
        """Load Sysmon seed events from all_target_events_run-X.csv."""
        if self.seed_events is not None:
            return self.seed_events
            
        seed_file = self.file_paths['seed_events']
        
        if not seed_file.exists():
            raise FileNotFoundError(f"Seed events file not found: {seed_file}")
        
        self.logger.info(f"📥 Loading seed events: {seed_file}")
        try:
            seed_df = pd.read_csv(seed_file)
            self.logger.info(f"✅ Loaded {len(seed_df):,} potential seed events")
            
            # Filter for manually selected seed events (marked with 'x' in Local column)
            selected_mask = seed_df['Local'].astype(str).str.strip().str.lower() == 'x'
            selected_seed_df = seed_df[selected_mask].copy()
            
            self.logger.info(f"🎯 Filtered to {len(selected_seed_df):,} manually selected seed events")
            
            if len(selected_seed_df) == 0:
                self.logger.warning("⚠️ No seed events marked with 'x' in Local column found!")
                return pd.DataFrame()
            
            # Create tactic lookup dictionary
            self.tactic_lookup = {}
            for _, row in selected_seed_df.iterrows():
                if pd.notna(row['Tactic']) and pd.notna(row['Technique']):
                    self.tactic_lookup[row['OriginalRowNumber']] = {
                        'Tactic': row['Tactic'].lower(),
                        'Technique': row['Technique']
                    }
            
            self.logger.info(f"📊 Built tactic lookup for {len(self.tactic_lookup)} seed events")
            self.seed_events = selected_seed_df
            return selected_seed_df
            
        except Exception as e:
            self.logger.error(f"❌ Error loading seed events: {e}")
            raise

    def load_netflow_data(self, apt_type: str, run_id: str) -> pd.DataFrame:
        """Load and cache NetFlow dataset."""
        if self.netflow_data is not None:
            return self.netflow_data
            
        netflow_file = self.file_paths['netflow_data']
        
        if not netflow_file.exists():
            raise FileNotFoundError(f"NetFlow file not found: {netflow_file}")
        
        self.logger.info(f"📥 Loading NetFlow dataset: {netflow_file}")
        try:
            netflow_df = pd.read_csv(netflow_file, low_memory=False)
            self.logger.info(f"✅ Loaded {len(netflow_df):,} NetFlow events")
            
            self.netflow_data = netflow_df
            return netflow_df
            
        except Exception as e:
            self.logger.error(f"❌ Error loading NetFlow dataset: {e}")
            raise

    def run_correlation_analysis(self, apt_type: str, run_id: str):
        """Run temporal correlation analysis between seed events and NetFlow."""
        self.logger.info("🔍 Running temporal correlation analysis...")
        
        # Load data
        seed_df = self.load_seed_events(apt_type, run_id)
        netflow_df = self.load_netflow_data(apt_type, run_id)
        
        if len(seed_df) == 0:
            self.logger.error("❌ No seed events available for correlation")
            return False
        
        # Filter for C2 and internal communications
        c2_mask = (
            (netflow_df['source_ip'] == self.c2_server_ip) |
            (netflow_df['destination_ip'] == self.c2_server_ip)
        )
        
        internal_mask = (
            (netflow_df['source_ip'].isin(self.internal_ips)) & 
            (netflow_df['destination_ip'].isin(self.internal_ips))
        )
        
        relevant_flows = netflow_df[c2_mask | internal_mask].copy()
        self.logger.info(f"🎯 Found {len(relevant_flows):,} C2/internal NetFlow events for correlation")
        
        # Parse timestamps
        seed_df['timestamp_parsed'] = pd.to_datetime(seed_df['timestamp'], unit='ms')
        relevant_flows['event_start_parsed'] = pd.to_datetime(relevant_flows['event_start'], unit='ms')
        relevant_flows['event_end_parsed'] = pd.to_datetime(relevant_flows['event_end'], unit='ms')
        
        # Group flows by network_community_id
        correlation_results = []
        
        for community_id, flow_group in relevant_flows.groupby('network_community_id'):
            if pd.isna(community_id):
                continue
                
            flow_start = flow_group['event_start_parsed'].min()
            flow_end = flow_group['event_end_parsed'].max()
            
            # Find temporally correlated seed events (within 30-second window)
            time_window = timedelta(seconds=30)
            
            for _, seed_row in seed_df.iterrows():
                seed_time = seed_row['timestamp_parsed']
                
                # Enhanced criteria: Check multiple temporal relationships
                include_flow = False
                time_diff_seconds = 0.0
                
                # Criterion 1: Seed event falls INSIDE the netflow timespan
                if flow_start <= seed_time <= flow_end:
                    include_flow = True
                    time_diff_seconds = 0.0  # Inside the flow
                
                # Criterion 2: NetFlow is on the LEFT of seed event (flow ends before seed)
                elif seed_time > flow_end and (seed_time - flow_end) <= time_window:
                    include_flow = True
                    time_diff_seconds = abs((seed_time - flow_end).total_seconds())
                
                # Criterion 3: NetFlow is on the RIGHT of seed event (flow starts after seed) 
                elif seed_time < flow_start and (flow_start - seed_time) <= time_window:
                    include_flow = True
                    time_diff_seconds = abs((flow_start - seed_time).total_seconds())
                
                if include_flow:
                    correlation_results.append({
                        'network_community_id': community_id,
                        'seed_event': seed_row['OriginalRowNumber'],
                        'seed_timestamp': seed_time,
                        'flow_start': flow_start,
                        'flow_end': flow_end,
                        'time_diff_seconds': time_diff_seconds,
                        'flow_type': self._determine_flow_type(flow_group),
                        'event_count': len(flow_group),
                        'Direct-attribution': '',  # Manual column
                        'Subnetflow-attribution': ''  # Manual column
                    })
        
        # Create verification matrix (preserve existing manual markings)
        if correlation_results:
            verification_df = pd.DataFrame(correlation_results)
            
            # Ensure correct column order as specified by user
            column_order = [
                'network_community_id', 'seed_event', 'seed_timestamp', 'flow_start', 'flow_end',
                'time_diff_seconds', 'flow_type', 'event_count', 'Direct-attribution', 'Subnetflow-attribution'
            ]
            verification_df = verification_df[column_order]
            verification_df = verification_df.sort_values(['seed_timestamp', 'time_diff_seconds'])
            
            verification_file_create = self.file_paths['verification_matrix_create']
            verification_file_manual = self.file_paths['verification_matrix']
            
            # Check if manual verification matrix (v2) already exists with markings
            if verification_file_manual.exists():
                try:
                    existing_df = pd.read_csv(verification_file_manual)
                    
                    # Check if there are any manual markings to preserve
                    has_direct_markings = (existing_df['Direct-attribution'].astype(str).str.strip().str.lower() == 'x').any()
                    has_subnetflow_markings = (existing_df['Subnetflow-attribution'].astype(str).str.strip().str.lower() == 'x').any()
                    
                    if has_direct_markings or has_subnetflow_markings:
                        self.logger.info("🔄 Preserving existing manual markings in verification matrix")
                        
                        # Preserve manual markings by merging with existing data
                        # Create a mapping of (network_community_id, seed_event) -> markings
                        marking_map = {}
                        for _, row in existing_df.iterrows():
                            key = (row['network_community_id'], row['seed_event'])
                            marking_map[key] = {
                                'Direct-attribution': row['Direct-attribution'],
                                'Subnetflow-attribution': row['Subnetflow-attribution']
                            }
                        
                        # Apply preserved markings to new dataframe
                        for idx, row in verification_df.iterrows():
                            key = (row['network_community_id'], row['seed_event'])
                            if key in marking_map:
                                verification_df.at[idx, 'Direct-attribution'] = marking_map[key]['Direct-attribution']
                                verification_df.at[idx, 'Subnetflow-attribution'] = marking_map[key]['Subnetflow-attribution']
                    else:
                        self.logger.info("📄 No existing manual markings found - creating fresh verification matrix")
                        
                except Exception as e:
                    self.logger.warning(f"⚠️ Could not read existing verification matrix: {e}")
                    self.logger.info("📄 Creating fresh verification matrix")
            
            # Save verification matrix (original naming for manual editing)
            verification_df.to_csv(verification_file_create, index=False)
            
            self.logger.info(f"✅ Created verification matrix with {len(verification_df)} correlations: {verification_file_create}")
            
            # Generate correlation visualization plots
            self.logger.info("📊 Creating correlation timeline visualizations...")
            self._create_correlation_plots(seed_df, relevant_flows, apt_type, run_id)
            
            # NEW: Export direct attribution metadata CSV files
            self.logger.info("📊 Exporting direct attribution metadata...")
            self._export_direct_attribution_metadata(correlation_results, relevant_flows, apt_type, run_id)
            
            return True
        else:
            self.logger.warning("⚠️ No temporal correlations found")
            return False
            
    def _determine_flow_type(self, flow_group: pd.DataFrame) -> str:
        """Determine flow type (C2, internal, etc.)."""
        has_c2 = ((flow_group['source_ip'] == self.c2_server_ip) | 
                 (flow_group['destination_ip'] == self.c2_server_ip)).any()
        
        if has_c2:
            return 'c2'
        else:
            return 'internal'
    
    def validate_verification_matrix(self) -> bool:
        """Validate manual edits in verification matrix."""
        verification_file = self.file_paths['verification_matrix']
        
        if not verification_file.exists():
            self.logger.error(f"❌ Verification matrix not found: {verification_file}")
            return False
            
        try:
            df = pd.read_csv(verification_file)
            
            # Check required columns exist
            required_cols = ['Direct-attribution', 'Subnetflow-attribution']
            if not all(col in df.columns for col in required_cols):
                self.logger.error(f"❌ Missing columns: {required_cols}")
                return False
                
            # Check for 'x' markings (handle NaN values safely)
            direct_marked = (df['Direct-attribution'].astype(str).str.strip().str.lower() == 'x').sum()
            sub_marked = (df['Subnetflow-attribution'].astype(str).str.strip().str.lower() == 'x').sum()
            
            self.logger.info(f"✅ Found {direct_marked} Direct-attribution markings")
            self.logger.info(f"✅ Found {sub_marked} Subnetflow-attribution markings")
            
            if direct_marked == 0 and sub_marked == 0:
                self.logger.warning("⚠️ No manual markings found - workflow will continue with empty attributions")
            
            return True
        except Exception as e:
            self.logger.error(f"❌ File validation error: {e}")
            return False
    
    def run_subnetflow_analysis(self, apt_type: str, run_id: str):
        """Generate sub-NetFlow analysis for communities marked for sub-NetFlow attribution."""
        self.logger.info("🔍 Running sub-NetFlow analysis...")
        
        # Load verification matrix to find communities needing sub-NetFlow analysis
        verification_file = self.file_paths['verification_matrix']
        verification_df = pd.read_csv(verification_file)
        
        # Find communities marked for sub-NetFlow analysis
        subnetflow_mask = verification_df['Subnetflow-attribution'].astype(str).str.strip().str.lower() == 'x'
        subnetflow_communities = verification_df[subnetflow_mask]['network_community_id'].unique()
        
        if len(subnetflow_communities) == 0:
            self.logger.info("✅ No communities marked for sub-NetFlow analysis - skipping this step")
            return True
        
        self.logger.info(f"🎯 Processing {len(subnetflow_communities)} communities for sub-NetFlow analysis")
        
        # Load NetFlow data
        netflow_df = self.load_netflow_data(apt_type, run_id)
        
        # Parse timestamps
        netflow_df['event_start_parsed'] = pd.to_datetime(netflow_df['event_start'], unit='ms')
        netflow_df['event_end_parsed'] = pd.to_datetime(netflow_df['event_end'], unit='ms')
        
        # Create sub-NetFlow assignment template
        template_entries = []
        
        # Process each community needing sub-NetFlow analysis
        for community_id in subnetflow_communities:
            community_data = netflow_df[netflow_df['network_community_id'] == community_id]
            
            if len(community_data) == 0:
                continue
            
            # Group by (event_start, event_end) to identify sub-NetFlows
            grouped = community_data.groupby(['event_start_parsed', 'event_end_parsed'])
            
            # Create timeline plot for this community
            self._create_subnetflow_timeline(community_data, community_id, apt_type, run_id)
            
            # Create assignment-colored timeline plot (if assignments are available)
            self._create_assignment_colored_subnetflow_timeline(community_data, community_id, apt_type, run_id)
            
            # NEW: Export metadata CSV for this community
            self._export_subnetflow_metadata(community_data, community_id, apt_type, run_id, grouped)
            
            # Add entries to template
            for i, ((start_time, end_time), group) in enumerate(grouped):
                template_entries.append({
                    'seed_event': '',  # Manual column - user fills this
                    'network_community_id': community_id,
                    'subnetflow_id': i+1  # Numeric for proper sorting
                })
        
        # Create template file
        if template_entries:
            template_df = pd.DataFrame(template_entries)
            
            # Ensure correct column order as specified by user
            column_order = ['seed_event', 'network_community_id', 'subnetflow_id']
            template_df = template_df[column_order]
            template_df = template_df.sort_values(['network_community_id', 'subnetflow_id'])
            
            # Create original template file (for reference)
            template_file_create = self.file_paths['subnetflow_template_create']
            template_df.to_csv(template_file_create, index=False)
            
            # Create v2 template file (for manual editing)  
            template_file_manual = self.file_paths['subnetflow_template']
            template_df.to_csv(template_file_manual, index=False)
            
            self.logger.info(f"✅ Created sub-NetFlow assignment template with {len(template_df)} entries:")
            self.logger.info(f"   📋 Original: {template_file_create}")
            self.logger.info(f"   📋 Manual (v2): {template_file_manual}")
            return True
        else:
            self.logger.warning("⚠️ No sub-NetFlow segments found")
            return False
    
    def _create_subnetflow_timeline(self, community_data: pd.DataFrame, community_id: str, 
                                  apt_type: str, run_id: str):
        """Create timeline plot for sub-NetFlow analysis with seed events overlay."""
        # Load seed events for this run to overlay on the plot
        seed_df = self.load_seed_events(apt_type, run_id)
        
        # Parse seed event timestamps if not already parsed
        if 'timestamp_parsed' not in seed_df.columns:
            seed_df['timestamp_parsed'] = pd.to_datetime(seed_df['timestamp'], unit='ms')
        
        # Group NetFlow by (start, end) times to create sub-NetFlows
        grouped = community_data.groupby(['event_start_parsed', 'event_end_parsed'])
        num_groups = len(grouped)
        
        # Adjust figure height based on number of subnetflows for better visibility
        if num_groups <= 20:
            fig_height = 10
        elif num_groups <= 100:
            fig_height = 15
        else:
            fig_height = 20  # Taller plot for many subnetflows
            
        fig, ax = plt.subplots(figsize=(16, fig_height))
        
        # Get time range for the plot
        all_times = []
        for (start_time, end_time), group in grouped:
            all_times.extend([start_time, end_time])
        
        # Add seed event times for complete time range
        if len(seed_df) > 0:
            all_times.extend(seed_df['timestamp_parsed'].tolist())
        
        time_min = min(all_times)
        time_max = max(all_times)
        time_range = time_max - time_min
        
        # Plot each sub-NetFlow as a horizontal bar (SAME COLOR for all)
        bar_color = '#4472C4'  # Single blue color for all bars
        y_labels = []
        
        # Calculate bar height based on number of subnetflows for visibility
        if num_groups <= 20:
            bar_height = 0.8
        elif num_groups <= 50:
            bar_height = 0.9
        else:
            bar_height = 0.95  # Use almost full height for many subnetflows
        
        for i, ((start_time, end_time), group) in enumerate(grouped):
            duration_mpl = mdates.date2num(end_time) - mdates.date2num(start_time)
            
            # Plot horizontal bar with consistent color and increased visibility
            ax.barh(i, duration_mpl,
                   left=mdates.date2num(start_time),
                   height=bar_height,
                   color=bar_color,
                   alpha=0.8,
                   edgecolor='navy',
                   linewidth=0.5)  # Thinner edge for dense plots
            
            # Create simplified y-axis label (just subnetflow number)
            y_labels.append(f'{i+1}')
        
        # Set y-axis with dynamic tick spacing to avoid overcrowding
        if num_groups <= 20:
            # Show all labels for small numbers
            ax.set_yticks(range(len(y_labels)))
            ax.set_yticklabels(y_labels, fontsize=9)
        elif num_groups <= 100:
            # Show every 5th label for medium numbers
            tick_positions = range(0, len(y_labels), 5)
            tick_labels = [y_labels[i] for i in tick_positions]
            ax.set_yticks(tick_positions)
            ax.set_yticklabels(tick_labels, fontsize=8)
        else:
            # Show every 10th label for large numbers
            tick_positions = range(0, len(y_labels), 10)
            tick_labels = [y_labels[i] for i in tick_positions]
            ax.set_yticks(tick_positions)
            ax.set_yticklabels(tick_labels, fontsize=7)
        
        # Plot seed events as dashed vertical lines with OriginalRowNumber labels
        seed_positions = []
        seed_labels = []
        seed_colors = []
        
        # Filter seed events to those within or near the NetFlow time range (±5min buffer)
        time_buffer = timedelta(minutes=5)
        relevant_seeds = seed_df[
            (seed_df['timestamp_parsed'] >= (time_min - time_buffer)) &
            (seed_df['timestamp_parsed'] <= (time_max + time_buffer))
        ]
        
        for _, seed_event in relevant_seeds.iterrows():
            timestamp = seed_event['timestamp_parsed']
            event_id = seed_event['EventID']
            orig_row_num = str(seed_event['OriginalRowNumber'])
            
            # Color based on EventID (consistent with other plots)
            if event_id == 1:
                color = self.colors['sysmon_eventid1']  # Red
            elif event_id == 11:
                color = self.colors['sysmon_eventid11']  # Blue
            elif event_id == 23:
                color = self.colors['sysmon_eventid23']  # Green
            else:
                color = '#666666'  # Gray for other EventIDs
            
            # Plot dashed vertical line with millisecond precision
            ax.axvline(x=timestamp, color=color, linestyle='--', linewidth=1.5, alpha=0.8, zorder=10)
            
            # Store for top axis labels
            seed_positions.append(timestamp)
            seed_labels.append(orig_row_num)
            seed_colors.append(color)
        
        # Set up top axis for OriginalRowNumber labels with 45° rotation
        if len(seed_positions) > 0:
            ax_top = ax.twiny()
            ax_top.set_xlim(ax.get_xlim())
            
            # Convert timestamps to matplotlib date numbers for precise positioning
            mpl_positions = [mdates.date2num(pos) for pos in seed_positions]
            
            ax_top.set_xticks(mpl_positions)
            ax_top.set_xticklabels(seed_labels, rotation=45, fontsize=8, ha='left')
            ax_top.set_xlabel('OriginalRowNumber (Seed Events)', fontsize=10, fontweight='bold')
            
            # Add some padding at the top for rotated labels
            ax_top.tick_params(axis='x', pad=8)
        
        # Customize main plot with descriptive labels
        ax.set_xlabel('Timeline', fontsize=11, fontweight='bold')
        ax.set_ylabel(f'Sub-NetFlow ID (Total: {num_groups})', fontsize=11, fontweight='bold')
        
        # Set title with extra padding to avoid overlap with top axis labels
        title_text = (f'Sub-NetFlow Timeline Analysis\n'
                     f'Community ID: {community_id[:30]}...\n'
                     f'{apt_type.upper()}-Run-{run_id}')
        ax.set_title(title_text, fontsize=12, fontweight='bold', pad=60)
        
        # Dynamic time axis formatting based on time range
        total_seconds = time_range.total_seconds()
        
        if total_seconds <= 10:  # Less than 10 seconds - show milliseconds
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S.%f'))
            ax.xaxis.set_major_locator(mdates.SecondLocator(interval=1))
            formatter_str = "millisecond"
        elif total_seconds <= 60:  # Less than 1 minute - show seconds
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            ax.xaxis.set_major_locator(mdates.SecondLocator(interval=max(1, int(total_seconds / 8))))
            formatter_str = "second"
        elif total_seconds <= 3600:  # Less than 1 hour - show minutes:seconds
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            interval = max(1, int(total_seconds / 480))
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=interval))
            formatter_str = "minute"
        elif total_seconds <= 14400:  # Less than 4 hours - show minutes with larger intervals
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            interval = max(5, int(total_seconds / 720))
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=interval))
            formatter_str = "hour"
        else:  # Very long ranges - use AutoDateLocator to prevent tick overload
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=20))
            formatter_str = "auto"
            self.logger.warning(f"⚠️  Using AutoDateLocator for long time range: {total_seconds/3600:.1f} hours")
        
        # Rotate labels with dynamic font size based on time range
        fontsize = 9 if total_seconds <= 60 else 8
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, fontsize=fontsize)
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Create simple legend for seed events (no sub-NetFlow legend to avoid clutter)
        if len(seed_positions) > 0:
            legend_elements = []
            if any(c == self.colors['sysmon_eventid1'] for c in seed_colors):
                legend_elements.append(plt.Line2D([0], [0], color=self.colors['sysmon_eventid1'], 
                                                linestyle='--', linewidth=2, label='EventID 1 (Process Creation)'))
            if any(c == self.colors['sysmon_eventid11'] for c in seed_colors):
                legend_elements.append(plt.Line2D([0], [0], color=self.colors['sysmon_eventid11'], 
                                                linestyle='--', linewidth=2, label='EventID 11 (File Create)'))
            if any(c == self.colors['sysmon_eventid23'] for c in seed_colors):
                legend_elements.append(plt.Line2D([0], [0], color=self.colors['sysmon_eventid23'], 
                                                linestyle='--', linewidth=2, label='EventID 23 (File Delete)'))
            
            if legend_elements:
                ax.legend(handles=legend_elements, loc='upper right', fontsize=9, 
                         frameon=True, fancybox=True, shadow=True)
        
        # Tight layout with extra padding for top labels
        plt.tight_layout(rect=[0.05, 0.05, 0.98, 0.85])
        
        # Save plot
        output_dir = self.file_paths['results_dir'] / "subnetflow_analysis"
        output_dir.mkdir(exist_ok=True)
        
        # Clean community ID for filename (match reference format)
        clean_id = community_id.replace(':', '_').replace('/', '_').replace('=', '')
        clean_id = clean_id[:20] if len(clean_id) > 20 else clean_id
        output_file = output_dir / f"subnetflow_timeline_1_{clean_id}_analysis.png"
        
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.logger.info(f"📊 Created sub-NetFlow timeline: {output_file}")
        
        # Also return info about seed events found for logging
        if len(relevant_seeds) > 0:
            self.logger.info(f"🎯 Overlaid {len(relevant_seeds)} seed events on sub-NetFlow timeline")
    
    def _create_assignment_colored_subnetflow_timeline(self, community_data: pd.DataFrame, 
                                                     community_id: str, apt_type: str, run_id: str):
        """Create enhanced subnetflow timeline plot with assignment-based color coding."""
        # Check if assignment file exists
        assignment_file = self.file_paths['subnetflow_template']
        if not assignment_file.exists():
            self.logger.info("⚠️ No assignment file found - skipping assignment-colored plot")
            return
        
        try:
            # Load assignment information
            assignments_df = pd.read_csv(assignment_file)
            
            # Create assignment lookup: subnetflow_id -> seed_event (or None if unassigned)
            assignment_lookup = {}
            for _, row in assignments_df.iterrows():
                subnetflow_id = row['subnetflow_id']
                seed_event = row['seed_event']
                assignment_lookup[subnetflow_id] = seed_event if pd.notna(seed_event) and seed_event != '' else None
                
            assigned_count = sum(1 for v in assignment_lookup.values() if v is not None)
            unassigned_count = len(assignment_lookup) - assigned_count
            
            self.logger.info(f"🎨 Creating assignment-colored plot: {assigned_count} assigned, {unassigned_count} unassigned")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Could not load assignments: {e} - skipping assignment-colored plot")
            return
        
        # Load seed events for overlay
        seed_df = self.load_seed_events(apt_type, run_id)
        if 'timestamp_parsed' not in seed_df.columns:
            seed_df['timestamp_parsed'] = pd.to_datetime(seed_df['timestamp'], unit='ms')
        
        # Group NetFlow by (start, end) times to create sub-NetFlows
        grouped = community_data.groupby(['event_start_parsed', 'event_end_parsed'])
        num_groups = len(grouped)
        
        # Adjust figure height based on number of subnetflows
        if num_groups <= 20:
            fig_height = 10
        elif num_groups <= 100:
            fig_height = 15
        else:
            fig_height = 20
            
        fig, ax = plt.subplots(figsize=(16, fig_height))
        
        # Get time range for the plot
        all_times = []
        for (start_time, end_time), group in grouped:
            all_times.extend([start_time, end_time])
        
        # Add seed event times for complete time range
        if len(seed_df) > 0:
            all_times.extend(seed_df['timestamp_parsed'].tolist())
        
        time_min = min(all_times)
        time_max = max(all_times)
        
        # Define colors for assignment status
        colors = {
            'assigned': '#4472C4',      # Blue for assigned subnetflows
            'unassigned': '#D3D3D3'     # Light gray for unassigned subnetflows  
        }
        
        # Calculate bar height for visibility
        if num_groups <= 20:
            bar_height = 0.8
        elif num_groups <= 50:
            bar_height = 0.9
        else:
            bar_height = 0.95
        
        y_labels = []
        assigned_bars = 0
        unassigned_bars = 0
        
        # Plot each sub-NetFlow with assignment-based coloring
        for i, ((start_time, end_time), group) in enumerate(grouped):
            duration_mpl = mdates.date2num(end_time) - mdates.date2num(start_time)
            
            # Determine color based on assignment status
            subnetflow_id = i + 1  # Subnetflow IDs start from 1
            is_assigned = assignment_lookup.get(subnetflow_id) is not None
            
            if is_assigned:
                bar_color = colors['assigned']
                edge_color = 'navy'
                assigned_bars += 1
            else:
                bar_color = colors['unassigned']
                edge_color = 'gray'
                unassigned_bars += 1
            
            # Plot horizontal bar with assignment-based color
            ax.barh(i, duration_mpl,
                   left=mdates.date2num(start_time),
                   height=bar_height,
                   color=bar_color,
                   alpha=0.8,
                   edgecolor=edge_color,
                   linewidth=0.5)
            
            # Create y-axis label
            y_labels.append(f'{subnetflow_id}')
        
        # Set y-axis with dynamic tick spacing
        if num_groups <= 20:
            ax.set_yticks(range(len(y_labels)))
            ax.set_yticklabels(y_labels, fontsize=9)
        elif num_groups <= 100:
            tick_positions = range(0, len(y_labels), 5)
            tick_labels = [y_labels[i] for i in tick_positions]
            ax.set_yticks(tick_positions)
            ax.set_yticklabels(tick_labels, fontsize=8)
        else:
            tick_positions = range(0, len(y_labels), 10)
            tick_labels = [y_labels[i] for i in tick_positions]
            ax.set_yticks(tick_positions)
            ax.set_yticklabels(tick_labels, fontsize=7)
        
        # Enhanced title and labels
        ax.set_xlabel('Timeline (HH:MM:SS)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Sub-NetFlow ID', fontsize=12, fontweight='bold')
        
        # Create comprehensive title
        community_short = community_id.split(':')[1][:10] if ':' in community_id else community_id[:10]
        ax.set_title(f'Sub-NetFlow Timeline Analysis (Assignment-Based Coloring)\n' + 
                    f'Community: {community_short}... | APT-{apt_type.upper()} Run-{run_id}\n' +
                    f'Assigned: {assigned_bars} | Unassigned: {unassigned_bars} | Total: {num_groups}',
                    fontsize=14, fontweight='bold', pad=20)
        
        # Format x-axis for timeline with smart tick limiting
        # Get time range from the actual timestamp columns in the DataFrame
        all_times = []
        all_times.extend(community_data['event_start_parsed'].tolist())
        all_times.extend(community_data['event_end_parsed'].tolist())
        time_range = max(all_times) - min(all_times)
        total_seconds = time_range.total_seconds()

        if total_seconds <= 3600:  # Less than 1 hour
            interval = max(1, int(total_seconds / 480))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=interval))
        elif total_seconds <= 14400:  # Less than 4 hours
            interval = max(5, int(total_seconds / 720))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=interval))
        else:  # Very long ranges
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=20))
            self.logger.warning(f"⚠️  Using AutoDateLocator for assignment timeline: {total_seconds/3600:.1f} hours")

        plt.xticks(rotation=45)
        
        # Add grid for better readability
        ax.grid(True, axis='x', alpha=0.3, linestyle='--')
        
        # Add legend for color coding
        legend_elements = [
            plt.Rectangle((0,0),1,1, facecolor=colors['assigned'], alpha=0.8, 
                         edgecolor='navy', label=f'Assigned to Seed Event ({assigned_bars})'),
            plt.Rectangle((0,0),1,1, facecolor=colors['unassigned'], alpha=0.8, 
                         edgecolor='gray', label=f'Unassigned - Background ({unassigned_bars})')
        ]
        
        # Add seed events overlay
        relevant_seeds = seed_df[
            (seed_df['timestamp_parsed'] >= time_min) & 
            (seed_df['timestamp_parsed'] <= time_max)
        ]
        
        if len(relevant_seeds) > 0:
            for _, seed in relevant_seeds.iterrows():
                ax.axvline(x=seed['timestamp_parsed'], color='red', alpha=0.7, 
                         linewidth=1.5, linestyle='--', zorder=10)
            
            # Add seed events to legend
            legend_elements.append(
                plt.Line2D([0], [0], color='red', alpha=0.7, linestyle='--', 
                         linewidth=1.5, label=f'Seed Events ({len(relevant_seeds)})')
            )
        
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10, 
                 frameon=True, fancybox=True, shadow=True)
        
        # Tight layout
        plt.tight_layout(rect=[0.05, 0.05, 0.98, 0.90])
        
        # Generate output filename with assignment indicator
        output_dir = self.file_paths['results_dir'] / "subnetflow_analysis"
        output_dir.mkdir(exist_ok=True)
        
        # Clean community ID for filename (match reference format)
        clean_id = community_id.replace(':', '_').replace('/', '_').replace('=', '')
        clean_id = clean_id[:20] if len(clean_id) > 20 else clean_id
        
        # NEW filename with assignment indicator
        output_file = output_dir / f"subnetflow_timeline_ASSIGNED_1_{clean_id}_analysis.png"
        
        # Save plot
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.logger.info(f"🎨 Created assignment-colored timeline: {output_file}")
        self.logger.info(f"   Assigned: {assigned_bars} | Unassigned: {unassigned_bars}")
    
    def validate_subnetflow_assignments(self) -> bool:
        """Validate manual sub-NetFlow assignments."""
        template_file = self.file_paths['subnetflow_template']
        
        if not template_file.exists():
            self.logger.info("✅ No sub-NetFlow template found - skipping validation")
            return True
            
        try:
            df = pd.read_csv(template_file)
            
            # Check if seed_event column has been filled
            filled_assignments = df[df['seed_event'].notna() & (df['seed_event'] != '')].shape[0]
            total_assignments = len(df)
            
            self.logger.info(f"✅ Found {filled_assignments}/{total_assignments} manual assignments")
            
            if filled_assignments == 0:
                self.logger.warning("⚠️ No manual assignments found - sub-NetFlow attribution will be skipped")
            
            return True
        except Exception as e:
            self.logger.error(f"❌ Template validation error: {e}")
            return False
    
    def apply_two_tier_labeling(self, apt_type: str, run_id: str) -> pd.DataFrame:
        """Apply two-tier labeling system to NetFlow dataset."""
        self.logger.info("🏷️ Applying two-tier labeling system...")
        
        # Load NetFlow data
        netflow_df = self.load_netflow_data(apt_type, run_id)
        
        # Load attribution mappings
        direct_mapping = self._load_direct_mapping()
        subnetflow_mapping = self._load_subnetflow_mapping()
        
        if not self.tactic_lookup:
            self.load_seed_events(apt_type, run_id)  # This builds tactic_lookup
        
        # Initialize label columns
        result_df = netflow_df.copy()
        result_df['Tactic'] = ''
        result_df['Technique'] = ''
        result_df['Label'] = 'benign'  # Default to benign
        result_df['attribution_source'] = 'none'  # Track attribution source
        
        # Add subnetflow_id column for communities that need it
        result_df['subnetflow_id'] = ''
        
        # Apply selective sub-NetFlow segmentation
        if subnetflow_mapping:
            result_df = self._apply_selective_subnetflow_segmentation(result_df, subnetflow_mapping.keys())
        
        total_records = len(result_df)
        
        # Apply Direct NetFlow Attribution (Tier 1)
        self.logger.info("🎯 Applying Tier 1: Direct NetFlow Attribution...")
        for community_id, seed_event in direct_mapping.items():
            if seed_event in self.tactic_lookup:
                mask = result_df['network_community_id'] == community_id
                affected_records = mask.sum()
                
                result_df.loc[mask, 'Tactic'] = self.tactic_lookup[seed_event]['Tactic']
                result_df.loc[mask, 'Technique'] = self.tactic_lookup[seed_event]['Technique']
                result_df.loc[mask, 'Label'] = 'malicious'
                result_df.loc[mask, 'attribution_source'] = 'direct'
                
                self.direct_attributions += affected_records
                self.logger.info(f"📊 Labeled {affected_records} events for community {community_id[:20]}...")
        
        # Apply Sub-NetFlow Attribution (Tier 2)
        self.logger.info("🎯 Applying Tier 2: Sub-NetFlow Attribution...")
        for (community_id, subnetflow_id), seed_event in subnetflow_mapping.items():
            if seed_event in self.tactic_lookup:
                mask = ((result_df['network_community_id'] == community_id) & 
                       (result_df['subnetflow_id'] == subnetflow_id))
                affected_records = mask.sum()
                
                if affected_records > 0:
                    # Check for conflicts
                    conflict_mask = mask & (result_df['attribution_source'] == 'direct')
                    conflict_count = conflict_mask.sum()
                    
                    if conflict_count > 0:
                        self.conflicts += conflict_count
                        self.conflict_records.append({
                            'community_id': community_id,
                            'subnetflow_id': subnetflow_id,
                            'subnetflow_seed_event': seed_event,
                            'affected_records': conflict_count
                        })
                    
                    # Sub-NetFlow takes precedence
                    result_df.loc[mask, 'Tactic'] = self.tactic_lookup[seed_event]['Tactic']
                    result_df.loc[mask, 'Technique'] = self.tactic_lookup[seed_event]['Technique']
                    result_df.loc[mask, 'Label'] = 'malicious'
                    result_df.loc[mask, 'attribution_source'] = 'subnetflow'
                    
                    self.subnetflow_attributions += affected_records
        
        # Count unlabeled records
        self.unlabeled = (result_df['attribution_source'] == 'none').sum()
        
        # Remove temporary column but keep subnetflow_id for analysis
        result_df = result_df.drop(['attribution_source'], axis=1)
        
        # Log summary
        self._log_attribution_summary()
        
        return result_df
    
    def _apply_selective_subnetflow_segmentation(self, netflow_df: pd.DataFrame, 
                                               subnetflow_keys: Set[Tuple[str, str]]) -> pd.DataFrame:
        """Apply sub-NetFlow segmentation only to communities that need it."""
        communities_needing_segmentation = set(key[0] for key in subnetflow_keys)
        
        if not communities_needing_segmentation:
            return netflow_df
            
        self.logger.info(f"🔧 Applying selective sub-NetFlow segmentation to {len(communities_needing_segmentation)} communities")
        
        # Parse timestamps
        netflow_df['event_start_parsed'] = pd.to_datetime(netflow_df['event_start'], unit='ms')
        netflow_df['event_end_parsed'] = pd.to_datetime(netflow_df['event_end'], unit='ms')
        
        for community_id in communities_needing_segmentation:
            community_mask = netflow_df['network_community_id'] == community_id
            community_data = netflow_df[community_mask]
            
            if len(community_data) == 0:
                continue
            
            # Group by (start, end) times
            grouped = community_data.groupby(['event_start_parsed', 'event_end_parsed'])
            
            # Assign sub-NetFlow IDs  
            for i, ((start_time, end_time), group) in enumerate(grouped):
                subnetflow_id = i+1  # Numeric for proper sorting
                netflow_df.loc[group.index, 'subnetflow_id'] = subnetflow_id
        
        return netflow_df
    
    def _load_direct_mapping(self) -> Dict[str, int]:
        """Load direct NetFlow attribution mapping from verification matrix."""
        verification_file = self.file_paths['verification_matrix']
        
        if not verification_file.exists():
            self.logger.warning("⚠️ Verification matrix not found - no direct attributions")
            return {}
        
        try:
            df = pd.read_csv(verification_file)
            direct_mask = df['Direct-attribution'].astype(str).str.strip().str.lower() == 'x'
            direct_df = df[direct_mask]
            
            mapping = {}
            for _, row in direct_df.iterrows():
                mapping[row['network_community_id']] = row['seed_event']
            
            self.logger.info(f"📊 Loaded {len(mapping)} direct attributions")
            return mapping
        except Exception as e:
            self.logger.error(f"❌ Error loading direct mapping: {e}")
            return {}
    
    def _load_subnetflow_mapping(self) -> Dict[Tuple[str, str], int]:
        """Load sub-NetFlow attribution mapping from template."""
        template_file = self.file_paths['subnetflow_template']
        
        if not template_file.exists():
            self.logger.info("✅ No sub-NetFlow template - no sub-NetFlow attributions")
            return {}
        
        try:
            df = pd.read_csv(template_file)
            filled_mask = df['seed_event'].notna() & (df['seed_event'] != '')
            filled_df = df[filled_mask]
            
            mapping = {}
            for _, row in filled_df.iterrows():
                key = (row['network_community_id'], row['subnetflow_id'])
                mapping[key] = int(row['seed_event'])
            
            self.logger.info(f"📊 Loaded {len(mapping)} sub-NetFlow attributions")
            return mapping
        except Exception as e:
            self.logger.error(f"❌ Error loading sub-NetFlow mapping: {e}")
            return {}
    
    def _log_attribution_summary(self):
        """Log detailed attribution statistics."""
        total = self.direct_attributions + self.subnetflow_attributions + self.unlabeled
        
        self.logger.info("📊 TWO-TIER ATTRIBUTION SUMMARY:")
        self.logger.info(f"   🎯 Direct NetFlow Attribution: {self.direct_attributions:,} records")
        self.logger.info(f"   🔍 Sub-NetFlow Attribution: {self.subnetflow_attributions:,} records")
        self.logger.info(f"   ⚠️ Conflicts (Sub-NetFlow override): {self.conflicts:,} records")
        self.logger.info(f"   ❌ Unlabeled (Benign): {self.unlabeled:,} records")
        self.logger.info(f"   📊 Total Records: {total:,}")
        
        if self.conflicts > 0:
            self.logger.info("⚠️ CONFLICT DETAILS:")
            for conflict in self.conflict_records:
                self.logger.info(f"   Community: {conflict['community_id'][:20]}...")
                self.logger.info(f"   Sub-NetFlow: {conflict['subnetflow_id']}")
                self.logger.info(f"   Affected: {conflict['affected_records']}")
    
    def _export_subnetflow_metadata(self, community_data: pd.DataFrame, community_id: str, 
                                  apt_type: str, run_id: str, grouped):
        """Export detailed metadata CSV files for subnetflow analysis."""
        # Create metadata export directory in dataset structure  
        metadata_dir = self.file_paths['results_dir'] / "subnetflow_analysis"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # Clean community ID for filename (remove problematic characters)
        clean_id = community_id.replace('/', '_').replace(':', '_').replace('=', '').replace('+', 'plus')
        
        # File 1: Community-level metadata overview
        community_metadata = self._create_community_metadata(community_data, community_id, grouped, 
                                                            apt_type, run_id, 'subnetflow')
        community_file = metadata_dir / f"community_{clean_id}_metadata.csv"
        community_metadata.to_csv(community_file, index=False)
        
        # File 2: Individual subnetflow details
        subnetflows_data = self._create_subnetflows_detail(community_data, community_id, grouped, 
                                                          apt_type, run_id)
        subnetflows_file = metadata_dir / f"community_{clean_id}_subnetflows.csv"  
        subnetflows_data.to_csv(subnetflows_file, index=False)
        
        self.logger.info(f"📊 Exported subnetflow metadata: {len(grouped)} subnetflows for {clean_id[:20]}...")
    
    def _create_community_metadata(self, community_data: pd.DataFrame, community_id: str, 
                                 grouped, apt_type: str, run_id: str, analysis_type: str) -> pd.DataFrame:
        """Create community-level metadata overview."""
        # Load seed events and manual assignments if available
        seed_df = self.load_seed_events(apt_type, run_id)
        
        # Calculate community statistics
        total_events = len(community_data)
        total_subnetflows = len(grouped) if analysis_type == 'subnetflow' else 1
        
        # Time span calculation
        start_times = community_data['event_start_parsed']
        end_times = community_data['event_end_parsed']
        first_event = start_times.min()
        last_event = end_times.max()
        time_span_minutes = (last_event - first_event).total_seconds() / 60
        
        # Network statistics
        source_ips = community_data['source_ip'].unique()
        dest_ips = community_data['destination_ip'].unique()
        protocols = community_data['network_transport'].unique()
        ports = pd.concat([community_data['source_port'], community_data['destination_port']]).unique()
        processes = community_data['process_executable'].dropna().unique()
        
        # Volume statistics
        total_bytes = community_data['network_bytes'].fillna(0).sum()
        total_packets = community_data['network_packets'].fillna(0).sum()
        
        # Seed events in time range
        if len(seed_df) > 0 and 'timestamp_parsed' in seed_df.columns:
            time_buffer = timedelta(minutes=5)
            relevant_seeds = seed_df[
                (seed_df['timestamp_parsed'] >= (first_event - time_buffer)) &
                (seed_df['timestamp_parsed'] <= (last_event + time_buffer))
            ]
            seed_events_in_range = len(relevant_seeds)
        else:
            seed_events_in_range = 0
        
        # Manual assignment status (try to load from template)
        assignment_status = self._get_assignment_status(community_id, run_id, analysis_type)
        
        metadata = {
            'community_id': community_id,
            'total_netflow_events': total_events,
            'total_subnetflows': total_subnetflows,
            'time_span_minutes': round(time_span_minutes, 2),
            'seed_events_in_range': seed_events_in_range,
            'assigned_subnetflows': assignment_status.get('assigned', 0),
            'unassigned_subnetflows': assignment_status.get('unassigned', 0),
            'source_ips': '|'.join(map(str, source_ips)),
            'destination_ips': '|'.join(map(str, dest_ips)),
            'protocols': '|'.join(map(str, protocols)),
            'ports': '|'.join(map(str, ports[pd.notna(ports)])),
            'process_executables': '|'.join(map(str, processes)),
            'bytes_total': int(total_bytes),
            'packets_total': int(total_packets),
            'first_event_timestamp': first_event.isoformat(),
            'last_event_timestamp': last_event.isoformat(),
            'png_filename': f"subnetflow_timeline_{community_id[:20]}_run-{run_id}.png",
            'analysis_type': analysis_type
        }
        
        return pd.DataFrame([metadata])
    
    def _create_subnetflows_detail(self, community_data: pd.DataFrame, community_id: str,
                                 grouped, apt_type: str, run_id: str) -> pd.DataFrame:
        """Create detailed subnetflow-level metadata."""
        subnetflow_details = []
        
        # Try to load manual assignments
        manual_assignments = self._load_manual_assignments(run_id)
        
        for i, ((start_time, end_time), group) in enumerate(grouped):
            subnetflow_id = i + 1
            duration_ms = (end_time - start_time).total_seconds() * 1000
            
            # Get assignment info for this subnetflow
            assignment_key = (community_id, subnetflow_id)
            manual_seed = manual_assignments.get(assignment_key, '')
            
            # Get tactic/technique if assigned
            assigned_tactic = ''
            assigned_technique = ''
            if manual_seed and manual_seed in self.tactic_lookup:
                assigned_tactic = self.tactic_lookup[manual_seed]['Tactic']
                assigned_technique = self.tactic_lookup[manual_seed]['Technique']
            
            # Network statistics for this subnetflow
            source_ips = group['source_ip'].unique()
            dest_ips = group['destination_ip'].unique()
            source_ports = group['source_port'].unique()
            dest_ports = group['destination_port'].unique()
            transports = group['network_transport'].unique()
            
            # Process information
            processes = group['process_executable'].dropna().unique()
            pids = group['process_pid'].dropna().unique()
            process_names = group['process_name'].dropna().unique()
            
            # Volume statistics
            total_bytes = group['network_bytes'].fillna(0).sum()
            total_packets = group['network_packets'].fillna(0).sum()
            
            detail = {
                'subnetflow_id': subnetflow_id,
                'start_timestamp': start_time.isoformat(),
                'end_timestamp': end_time.isoformat(),
                'duration_ms': round(duration_ms, 2),
                'event_count': len(group),
                'source_ips': '|'.join(map(str, source_ips)),
                'destination_ips': '|'.join(map(str, dest_ips)),
                'source_ports': '|'.join(map(str, source_ports[pd.notna(source_ports)])),
                'destination_ports': '|'.join(map(str, dest_ports[pd.notna(dest_ports)])),
                'network_transport': '|'.join(map(str, transports)),
                'network_bytes': int(total_bytes),
                'network_packets': int(total_packets),
                'process_executables': '|'.join(map(str, processes)),
                'process_pids': '|'.join(map(str, pids[pd.notna(pids)])),
                'process_names': '|'.join(map(str, process_names)),
                'manual_seed_assignment': manual_seed,
                'assigned_tactic': assigned_tactic,
                'assigned_technique': assigned_technique,
                'attribution_confidence': 1.0 if manual_seed else 0.0,
                'timeline_position_seconds': round((start_time - community_data['event_start_parsed'].min()).total_seconds(), 2),
                'community_id': community_id
            }
            
            subnetflow_details.append(detail)
        
        return pd.DataFrame(subnetflow_details)
    
    def _get_assignment_status(self, community_id: str, run_id: str, analysis_type: str) -> dict:
        """Get assignment status for community or subnetflows."""
        if analysis_type == 'direct':
            return {'assigned': 1, 'unassigned': 0}  # Direct attribution is binary
        
        # For subnetflow attribution, check template
        template_file = self.file_paths.get('subnetflow_template')
        if not template_file or not template_file.exists():
            return {'assigned': 0, 'unassigned': 0}
        
        try:
            df = pd.read_csv(template_file)
            community_assignments = df[df['network_community_id'] == community_id]
            assigned = community_assignments[
                community_assignments['seed_event'].notna() & 
                (community_assignments['seed_event'] != '')
            ].shape[0]
            total = len(community_assignments)
            return {'assigned': assigned, 'unassigned': total - assigned}
        except:
            return {'assigned': 0, 'unassigned': 0}
    
    def _load_manual_assignments(self, run_id: str) -> dict:
        """Load manual assignments from template file."""
        template_file = self.file_paths.get('subnetflow_template')
        if not template_file or not template_file.exists():
            return {}
        
        try:
            df = pd.read_csv(template_file)
            assignments = {}
            for _, row in df.iterrows():
                if pd.notna(row['seed_event']) and row['seed_event'] != '':
                    key = (row['network_community_id'], row['subnetflow_id'])
                    assignments[key] = int(row['seed_event'])
            return assignments
        except:
            return {}
    
    def _export_direct_attribution_metadata(self, correlation_results: list, relevant_flows: pd.DataFrame,
                                          apt_type: str, run_id: str):
        """Export detailed metadata CSV files for direct attribution analysis."""
        # Create metadata export directory in dataset structure
        metadata_dir = self.file_paths['results_dir']  # This is apt-Y/apt-Y-run-XX/netflow_event_tracing_analysis_results/
        metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each correlated community
        for result in correlation_results:
            community_id = result['network_community_id']
            
            # Get community data
            community_data = relevant_flows[relevant_flows['network_community_id'] == community_id]
            
            if len(community_data) == 0:
                continue
                
            # Clean community ID for filename
            clean_id = community_id.replace('/', '_').replace(':', '_').replace('=', '').replace('+', 'plus')
            
            # File 1: Community-level metadata overview
            community_metadata = self._create_community_metadata(community_data, community_id, None,
                                                               apt_type, run_id, 'direct')
            
            # Add direct attribution specific fields
            community_metadata['assigned_seed_event'] = result.get('closest_seed_event', '')
            community_metadata['correlation_distance_seconds'] = result.get('correlation_distance_ms', 0) / 1000
            community_metadata['flow_type'] = result.get('flow_type', 'unknown')
            
            community_file = metadata_dir / f"community_{clean_id}_metadata.csv"
            community_metadata.to_csv(community_file, index=False)
            
            # File 2: Individual NetFlow event details
            events_data = self._create_netflow_events_detail(community_data, community_id, result, 
                                                           apt_type, run_id)
            events_file = metadata_dir / f"community_{clean_id}_events.csv"
            events_data.to_csv(events_file, index=False)
            
            self.logger.info(f"📊 Exported direct attribution metadata: {len(community_data)} events for {clean_id[:20]}...")
    
    def _create_netflow_events_detail(self, community_data: pd.DataFrame, community_id: str,
                                    correlation_result: dict, apt_type: str, run_id: str) -> pd.DataFrame:
        """Create detailed individual NetFlow event metadata for direct attribution."""
        events_detail = []
        
        # Get assignment information
        assigned_seed = correlation_result.get('closest_seed_event', '')
        assigned_tactic = ''
        assigned_technique = ''
        
        if assigned_seed and assigned_seed in self.tactic_lookup:
            assigned_tactic = self.tactic_lookup[assigned_seed]['Tactic']
            assigned_technique = self.tactic_lookup[assigned_seed]['Technique']
        
        # Calculate timeline baseline
        timeline_start = community_data['event_start_parsed'].min()
        
        for idx, (_, event) in enumerate(community_data.iterrows()):
            # Calculate timeline position
            timeline_position = (event['event_start_parsed'] - timeline_start).total_seconds()
            
            detail = {
                'event_index': idx + 1,
                'timestamp': event['timestamp'],
                'event_start': event['event_start'],
                'event_end': event['event_end'],
                'duration_ms': event.get('event_duration', 0),
                'source_ip': event['source_ip'],
                'destination_ip': event['destination_ip'], 
                'source_port': event.get('source_port', ''),
                'destination_port': event.get('destination_port', ''),
                'network_transport': event.get('network_transport', ''),
                'network_bytes': event.get('network_bytes', 0),
                'network_packets': event.get('network_packets', 0),
                'process_executable': event.get('process_executable', ''),
                'process_pid': event.get('process_pid', ''),
                'process_name': event.get('process_name', ''),
                'process_args': str(event.get('process_args', '')),
                'destination_process_executable': event.get('destination_process_executable', ''),
                'destination_process_pid': event.get('destination_process_pid', ''),
                'host_hostname': event.get('host_hostname', ''),
                'host_ip': str(event.get('host_ip', '')),
                'network_community_id': community_id,
                'assigned_seed_event': assigned_seed,
                'assigned_tactic': assigned_tactic,
                'assigned_technique': assigned_technique,
                'timeline_position_seconds': round(timeline_position, 2),
                'correlation_type': 'direct',
                'correlation_distance_ms': correlation_result.get('correlation_distance_ms', 0),
                'flow_type': correlation_result.get('flow_type', 'unknown')
            }
            
            events_detail.append(detail)
        
        return pd.DataFrame(events_detail)
    
    def create_timeline_visualizations(self, labeled_df: pd.DataFrame, apt_type: str, run_id: str):
        """Create all timeline visualizations."""
        self.logger.info("📊 Creating timeline visualizations...")
        
        # Create only multi-track timeline (user's preferred visualization)
        self._create_multi_track_timeline(labeled_df, apt_type, run_id)
        
        # Create dual-domain combined visualization
        self._create_dual_domain_timeline(labeled_df, apt_type, run_id)
        
        # Optional: Create other visualizations (user questioned their value)
        # Uncomment if needed:
        # self._create_flow_community_timeline(labeled_df, apt_type, run_id)
        # self._create_interactive_gantt_plot(labeled_df, apt_type, run_id)
    
    def _create_multi_track_timeline(self, labeled_df: pd.DataFrame, apt_type: str, run_id: str):
        """Create Multi-Track Timeline visualization."""
        self.logger.info("📊 Creating Multi-Track Timeline...")
        
        # Parse timestamps - use 'timestamp' for proper chronological progression
        labeled_df['timestamp_parsed'] = pd.to_datetime(labeled_df['timestamp'], unit='ms')
        
        # Create figure
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Separate malicious and benign events
        malicious_df = labeled_df[labeled_df['Label'] == 'malicious']
        benign_df = labeled_df[labeled_df['Label'] == 'benign']
        
        # Y-level assignments: Benign at level 1, tactics at levels 2+
        y_levels = {'benign': 1}
        if len(malicious_df) > 0:
            unique_tactics = sorted(malicious_df['Tactic'].unique())
            for i, tactic in enumerate(unique_tactics):
                y_levels[tactic] = i + 2
        
        # Plot benign events (sample if too many)
        if len(benign_df) > 0:
            if len(benign_df) > 10000:
                benign_sample = benign_df.sample(n=10000, random_state=42)
                self.logger.info(f"📊 Sampling {len(benign_sample):,} benign events (out of {len(benign_df):,})")
            else:
                benign_sample = benign_df
            
            ax.scatter(benign_sample['timestamp_parsed'], 
                      [y_levels['benign']] * len(benign_sample),
                      c=self.benign_color, 
                      alpha=0.4, s=15, 
                      label=f'Benign Events ({len(benign_df):,})')
        
        # Plot malicious events by tactic
        if len(malicious_df) > 0:
            for tactic in sorted(malicious_df['Tactic'].unique()):
                tactic_df = malicious_df[malicious_df['Tactic'] == tactic]
                color = self.tactic_colors.get(tactic, '#696969')
                y_level = y_levels[tactic]
                
                ax.scatter(tactic_df['timestamp_parsed'], 
                          [y_level] * len(tactic_df),
                          c=color, alpha=0.7, s=30, 
                          label=f'{tactic.title()} ({len(tactic_df)} events)')
        
        # Customize plot
        ax.set_ylabel('Tactic Levels')
        ax.set_xlabel('Timeline')
        ax.set_title(f'Multi-Track Timeline Analysis\n{apt_type.upper()}-Run-{run_id} - NetFlow Events by MITRE Tactic Level')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Set y-axis
        y_ticks = list(y_levels.values())
        y_labels = [key.title() if key != 'benign' else 'Benign' for key in y_levels.keys()]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)
        ax.set_ylim(0.5, max(y_ticks) + 0.5)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
        plt.xticks(rotation=45)
        
        # Add legend
        ax.legend(bbox_to_anchor=(1.02, 1.0), loc='upper left')
        
        plt.tight_layout()
        
        # Save plot
        output_file = self.file_paths['timeline_dir'] / "multi_track_timeline.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"✅ Multi-Track Timeline saved: {output_file}")
    
    def _create_dual_domain_timeline(self, netflow_labeled_df: pd.DataFrame, apt_type: str, run_id: str):
        """Create dual-domain combined visualization: Sysmon + NetFlow timelines."""
        self.logger.info("📊 Creating Dual-Domain Combined Timeline...")
        
        # Load Sysmon labeled dataset
        sysmon_labeled_df = self._load_sysmon_labeled_dataset(apt_type, run_id)
        if sysmon_labeled_df is None:
            self.logger.warning("⚠️ Sysmon labeled dataset not found - creating NetFlow-only timeline")
            self._create_multi_track_timeline(netflow_labeled_df, apt_type, run_id)
            return
        
        # Parse timestamps for both datasets
        netflow_labeled_df['timestamp_parsed'] = pd.to_datetime(netflow_labeled_df['timestamp'], unit='ms')
        sysmon_labeled_df['timestamp_parsed'] = pd.to_datetime(sysmon_labeled_df['timestamp'], unit='ms')
        
        # Determine unified time range
        all_times = pd.concat([
            netflow_labeled_df['timestamp_parsed'], 
            sysmon_labeled_df['timestamp_parsed']
        ]).dropna()
        time_min, time_max = all_times.min(), all_times.max()
        
        # Create 2-panel figure with shared time axis
        fig, (ax_sysmon, ax_netflow) = plt.subplots(2, 1, figsize=(20, 12), 
                                                    sharex=True, 
                                                    height_ratios=[1.2, 1])
        
        # Top panel: Sysmon events by computer
        self._plot_sysmon_by_computer(ax_sysmon, sysmon_labeled_df, apt_type, run_id, time_min, time_max)
        
        # Bottom panel: NetFlow events by tactic
        self._plot_netflow_by_tactic(ax_netflow, netflow_labeled_df, apt_type, run_id, time_min, time_max)
        
        # Synchronize and style the combined plot
        self._finalize_dual_domain_plot(fig, ax_sysmon, ax_netflow, apt_type, run_id, time_min, time_max)
        
        # Save combined visualization
        output_file = self.file_paths['timeline_dir'] / "dual_domain_attack_timeline.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"✅ Dual-Domain Timeline saved: {output_file}")
    
    def _load_sysmon_labeled_dataset(self, apt_type: str, run_id: str) -> pd.DataFrame:
        """Load Sysmon labeled dataset if available."""
        apt_dir = self.base_path / apt_type / f"{apt_type}-run-{run_id}"
        sysmon_labeled_file = apt_dir / f"sysmon-run-{run_id}-labeled.csv"
        
        if not sysmon_labeled_file.exists():
            self.logger.warning(f"⚠️ Sysmon labeled dataset not found: {sysmon_labeled_file}")
            return None
        
        try:
            sysmon_df = pd.read_csv(sysmon_labeled_file)
            # Filter for malicious events only (handle both case variations)
            malicious_sysmon = sysmon_df[
                (sysmon_df['Label'].str.lower() == 'malicious') |
                (sysmon_df['Label'] == 'Malicious')
            ]
            self.logger.info(f"📊 Loaded {len(malicious_sysmon):,} malicious Sysmon events")
            return malicious_sysmon
        except Exception as e:
            self.logger.error(f"❌ Error loading Sysmon dataset: {e}")
            return None
    
    def _plot_sysmon_by_computer(self, ax, sysmon_df: pd.DataFrame, apt_type: str, run_id: str, 
                                time_min, time_max):
        """Plot Sysmon malicious events grouped by computer."""
        if len(sysmon_df) == 0:
            ax.text(0.5, 0.5, 'No Sysmon malicious events found', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=12)
            ax.set_ylabel('Host Events')
            return
        
        # Group by computer (use Computer column or extract from other fields)
        if 'Computer' in sysmon_df.columns:
            computers = sysmon_df.groupby('Computer')
        else:
            # Fallback: group by unique event origins
            sysmon_df['computer_id'] = 'Host'  # Generic grouping
            computers = sysmon_df.groupby('computer_id')
        
        # Create y-levels for each computer
        computer_names = sorted(computers.groups.keys())
        y_positions = {comp: i for i, comp in enumerate(computer_names)}
        
        # Plot events for each computer
        for computer_name, computer_events in computers:
            y_pos = y_positions[computer_name]
            
            # Group by tactic and plot with different colors
            if 'Tactic' in computer_events.columns:
                for tactic, tactic_events in computer_events.groupby('Tactic'):
                    if pd.notna(tactic) and tactic != '':
                        color = self.tactic_colors.get(tactic, '#696969')
                        ax.scatter(tactic_events['timestamp_parsed'], 
                                  [y_pos] * len(tactic_events),
                                  c=color, alpha=0.7, s=25, 
                                  label=f'{tactic.title()}' if tactic not in [h.get_label() for h in ax.get_children() if hasattr(h, 'get_label')] else "")
            else:
                # No tactic info - plot all as generic malicious
                ax.scatter(computer_events['timestamp_parsed'], 
                          [y_pos] * len(computer_events),
                          c='#FF0000', alpha=0.7, s=25, 
                          label='Malicious Events')
        
        # Customize top panel (reduced title padding to avoid overlap)
        ax.set_ylabel('Host Systems', fontsize=12, fontweight='bold')
        ax.set_title('🖥️  HOST-LEVEL MALICIOUS EVENTS (Sysmon)', 
                    fontsize=12, fontweight='bold', pad=10)
        ax.set_yticks(list(y_positions.values()))
        ax.set_yticklabels([f'{comp}' for comp in computer_names])
        ax.set_xlim(time_min, time_max)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add legend for top panel (remove duplicate tactics)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            # Remove duplicate tactics from legend
            unique_legend = {}
            for handle, label in zip(handles, labels):
                if label not in unique_legend:
                    unique_legend[label] = handle
            
            ax.legend(list(unique_legend.values()), list(unique_legend.keys()), 
                     loc='upper right', fontsize=10)
    
    def _plot_netflow_by_tactic(self, ax, netflow_df: pd.DataFrame, apt_type: str, run_id: str, 
                               time_min, time_max):
        """Plot NetFlow malicious events grouped by MITRE tactic."""
        # Separate malicious and benign events
        malicious_df = netflow_df[netflow_df['Label'] == 'malicious']
        benign_df = netflow_df[netflow_df['Label'] == 'benign']
        
        # Y-level assignments: Benign at level 0, tactics at levels 1+
        y_levels = {}
        if len(benign_df) > 0:
            y_levels['benign'] = 0
        
        if len(malicious_df) > 0:
            unique_tactics = sorted(malicious_df['Tactic'].unique())
            for i, tactic in enumerate(unique_tactics):
                y_levels[tactic] = i + 1
        
        # Plot benign events (sample if too many)
        if len(benign_df) > 0:
            if len(benign_df) > 5000:
                benign_sample = benign_df.sample(n=5000, random_state=42)
                self.logger.info(f"📊 Sampling {len(benign_sample):,} benign NetFlow events")
            else:
                benign_sample = benign_df
            
            ax.scatter(benign_sample['timestamp_parsed'], 
                      [y_levels['benign']] * len(benign_sample),
                      c=self.benign_color, 
                      alpha=0.3, s=10, 
                      label='Benign Events')
        
        # Plot malicious events by tactic
        if len(malicious_df) > 0:
            for tactic in sorted(malicious_df['Tactic'].unique()):
                tactic_df = malicious_df[malicious_df['Tactic'] == tactic]
                color = self.tactic_colors.get(tactic, '#696969')
                y_level = y_levels[tactic]
                
                ax.scatter(tactic_df['timestamp_parsed'], 
                          [y_level] * len(tactic_df),
                          c=color, alpha=0.8, s=20, 
                          label=f'{tactic.title()}')
        
        # Customize bottom panel
        ax.set_ylabel('Network Tactics', fontsize=12, fontweight='bold')
        ax.set_xlabel('Timeline', fontsize=12, fontweight='bold')
        ax.set_title('🌐 NETWORK-LEVEL MALICIOUS EVENTS (NetFlow)', 
                    fontsize=12, fontweight='bold', pad=15)
        
        # Set y-axis
        if y_levels:
            y_ticks = list(y_levels.values())
            y_labels = [key.title() if key != 'benign' else 'Benign' for key in y_levels.keys()]
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_labels)
            ax.set_ylim(-0.5, max(y_ticks) + 0.5)
        
        ax.set_xlim(time_min, time_max)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add legend for bottom panel
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc='upper right', fontsize=10)
    
    def _finalize_dual_domain_plot(self, fig, ax_sysmon, ax_netflow, apt_type: str, run_id: str, 
                                  time_min, time_max):
        """Apply final styling and synchronization to dual-domain plot."""
        # Synchronize x-axis formatting
        ax_netflow.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
        plt.setp(ax_netflow.xaxis.get_majorticklabels(), rotation=45)
        
        # Add overall title with better spacing
        fig.suptitle(f'DUAL-DOMAIN ATTACK PROGRESSION ANALYSIS\n{apt_type.upper()}-Run-{run_id} - Host Events vs Network Flow Timeline', 
                     fontsize=16, fontweight='bold', y=0.96)
        
        # Add time range info
        duration_hours = (time_max - time_min).total_seconds() / 3600
        fig.text(0.02, 0.02, f'Time Range: {time_min.strftime("%H:%M:%S")} - {time_max.strftime("%H:%M:%S")} (Duration: {duration_hours:.1f}h)', 
                 fontsize=10, alpha=0.7)
        
        # Improved layout with better spacing
        plt.tight_layout()
        plt.subplots_adjust(top=0.90, bottom=0.08, hspace=0.3)
    
    def _create_correlation_plots(self, seed_df: pd.DataFrame, netflow_df: pd.DataFrame, 
                                 apt_type: str, run_id: str):
        """Create correlation timeline visualizations to assist manual marking."""
        # Group NetFlow by community ID
        grouped_flows = self._group_netflow_by_community_id(netflow_df)
        
        # Create both correlation plots
        self._create_complete_timeline_plot(seed_df, grouped_flows, apt_type, run_id)
        self._create_correlation_hotspots_plot(seed_df, grouped_flows, apt_type, run_id)
    
    def _group_netflow_by_community_id(self, flows: pd.DataFrame) -> dict:
        """Group flows by network_community_id for timeline plotting."""
        self.logger.info("🔗 Grouping NetFlow by community ID...")
        
        grouped_flows = {}
        c2_count = 0
        internal_count = 0
        
        for community_id, group in flows.groupby('network_community_id'):
            if pd.isna(community_id):
                continue
            
            # Determine flow type (C2 or Internal)
            has_c2 = ((group['source_ip'] == self.c2_server_ip) | 
                     (group['destination_ip'] == self.c2_server_ip)).any()
            
            if has_c2:
                flow_type = 'c2'
                c2_count += 1
            else:
                flow_type = 'internal'
                internal_count += 1
            
            # Determine flow direction for C2 flows
            direction_info = 'bidirectional'  # Default
            if has_c2:
                outbound = (group['source_ip'].isin(['10.1.0.5', '10.1.0.6', '10.1.0.7'])).any()
                inbound = (group['destination_ip'].isin(['10.1.0.5', '10.1.0.6', '10.1.0.7'])).any()
                
                if outbound and inbound:
                    direction_info = 'bidirectional'
                elif outbound:
                    direction_info = 'outbound'
                elif inbound:
                    direction_info = 'inbound'
            
            grouped_flows[community_id] = {
                'data': group,
                'flow_type': flow_type,
                'direction_info': direction_info,
                'start_time': group['event_start_parsed'].min(),
                'end_time': group['event_end_parsed'].max(),
                'event_count': len(group)
            }
        
        self.logger.info(f"📊 Grouped {c2_count} C2 flows and {internal_count} internal flows")
        return grouped_flows
    
    def _create_complete_timeline_plot(self, seed_df: pd.DataFrame, grouped_flows: dict, 
                                      apt_type: str, run_id: str):
        """Create complete timeline visualization using TESTING_8 approach with proper community ID labels."""
        self.logger.info("📊 Creating complete timeline visualization...")
        
        # Check if we need to split into multiple plots (more than 50 flows)
        total_flows = len(grouped_flows)
        flows_per_plot = 50
        
        if total_flows <= flows_per_plot:
            # Single plot - use existing logic with dynamic height
            plot_height = max(8, min(20, total_flows * 0.3))
            fig, ax = plt.subplots(1, 1, figsize=(18, plot_height))
            self._create_single_timeline_plot(fig, ax, seed_df, grouped_flows, apt_type, run_id, 0, part_suffix="", all_flows=grouped_flows)
        else:
            # Multi-plot division
            self.logger.info(f"🔀 Splitting {total_flows} flows into multiple plots ({flows_per_plot} flows per plot)")
            
            # Convert grouped_flows dict to list for easy splitting
            flow_items = list(grouped_flows.items())
            num_plots = (total_flows + flows_per_plot - 1) // flows_per_plot  # Ceiling division
            
            for plot_idx in range(num_plots):
                start_idx = plot_idx * flows_per_plot
                end_idx = min(start_idx + flows_per_plot, total_flows)
                flows_for_this_plot = dict(flow_items[start_idx:end_idx])
                
                # Dynamic height based on flows in this plot
                flows_in_plot = len(flows_for_this_plot)
                plot_height = max(8, min(20, flows_in_plot * 0.3))
                
                fig, ax = plt.subplots(1, 1, figsize=(18, plot_height))
                part_suffix = f"_part{plot_idx + 1}"
                
                # Pass original grouped_flows for universal time range calculation
                self._create_single_timeline_plot(fig, ax, seed_df, flows_for_this_plot, apt_type, run_id, start_idx, part_suffix, all_flows=grouped_flows)
    
    def _create_single_timeline_plot(self, fig, ax, seed_df: pd.DataFrame, grouped_flows: dict, 
                                   apt_type: str, run_id: str, y_offset: int, part_suffix: str, all_flows: dict = None):
        """Create a single timeline plot (used by both single and multi-plot modes)."""
        
        # Calculate time range from ALL original data (not just this plot's subset)
        # This ensures universal time axis span across all subplots
        all_times = []
        # Get time range from all seed events to ensure universal span
        all_times.extend(seed_df['timestamp_parsed'].tolist())
        
        # Use all_flows for universal time range calculation when available (multi-plot mode)
        flows_for_time_calc = all_flows if all_flows is not None else grouped_flows
        for flow_info in flows_for_time_calc.values():
            all_times.extend([flow_info['start_time'], flow_info['end_time']])
        
        time_range = {
            'min_time': min(all_times),
            'max_time': max(all_times),
            'duration_hours': (max(all_times) - min(all_times)).total_seconds() / 3600
        }
        
        # ========================
        # NETFLOW COMMUNICATIONS (Base Layer) - TESTING_8 approach
        # ========================
        self.logger.info("🌐 Plotting NetFlow communications (C2 + Internal)...")
        
        # Plot NetFlow communications as horizontal bars
        y_positions = []
        flow_labels = []
        netflow_legend_elements = []
        flow_type_counts = {}  # Track detailed flow types
        
        for i, (community_id, flow_info) in enumerate(grouped_flows.items()):
            y_pos = i
            y_positions.append(y_pos)
            
            # Extract source and destination IPs from flow_info
            flow_data = flow_info['data']
            if not flow_data.empty:
                src_ip = flow_data.iloc[0]['source_ip']
                dst_ip = flow_data.iloc[0]['destination_ip']
            else:
                src_ip = '0.0.0.0'
                dst_ip = '0.0.0.0'
            
            flow_type = flow_info['flow_type']
            
            # Always use machine names for consistent labeling
            src_name = self._get_machine_name(src_ip)
            dst_name = self._get_machine_name(dst_ip)
            
            # Create detailed flow type key for color assignment
            if flow_type == 'c2':
                # C2 flows - create bidirectional flow type key
                internal_machine = dst_name if src_name == 'C2' else src_name
                flow_type_key = f"C2-{internal_machine}"
                label = f"C2↔{internal_machine} ({community_id[:8]}...)"
            else:
                # Internal flows - create alphabetically sorted flow type key  
                machines = sorted([src_name, dst_name])
                flow_type_key = f"{machines[0]}-{machines[1]}"
                label = f"{machines[0]}↔{machines[1]} ({community_id[:8]}...)"
            
            # Assign unique color based on flow type key
            color = self._get_flow_type_color(flow_type_key)
            
            flow_labels.append(label)
            # Track detailed flow type counts
            if flow_type_key not in flow_type_counts:
                flow_type_counts[flow_type_key] = {'count': 0, 'color': color, 'label': flow_type_key.replace('-', '↔')}
            flow_type_counts[flow_type_key]['count'] += 1
            
            start_time = flow_info['start_time']
            end_time = flow_info['end_time']
            
            # Draw horizontal bar for flow duration (TESTING_8 approach)
            if start_time and end_time and (end_time - start_time).total_seconds() > 0:
                duration_mpl = mdates.date2num(end_time) - mdates.date2num(start_time)
                ax.barh(y_pos, duration_mpl, left=mdates.date2num(start_time), 
                       height=0.6, color=color, alpha=0.7, edgecolor='navy', linewidth=0.5)
        
        # ========================
        # SEED EVENTS (Overlay Layer) - TESTING_8 approach
        # ========================
        self.logger.info("🎯 Overlaying seed events as vertical lines...")
        
        # Group seed events by EventID for coloring (TESTING_8 approach)
        seed_eventids = seed_df.groupby('EventID')
        seed_legend_elements = []
        
        # Track MITRE tactics if available
        if 'Tactic' in seed_df.columns:
            tactic_counts = seed_df['Tactic'].value_counts().to_dict()
            self.logger.info(f"🎯 MITRE Tactics in selected seeds: {tactic_counts}")
        
        for event_id, group in seed_eventids:
            # Determine color based on EventID (TESTING_8 approach)
            if event_id == 1:
                color = self.colors['sysmon_eventid1']
                label = f'EventID {event_id} (Process Creation)'
            elif event_id == 11:
                color = self.colors['sysmon_eventid11'] 
                label = f'EventID {event_id} (File Create)'
            elif event_id == 23:
                color = self.colors['sysmon_eventid23']
                label = f'EventID {event_id} (File Delete)'
            else:
                color = '#666666'
                label = f'EventID {event_id}'
            
            # Plot seed events as thin vertical lines spanning the entire plot height
            for _, event in group.iterrows():
                timestamp = event['timestamp_parsed']
                ax.axvline(x=timestamp, color=color, alpha=0.8, linewidth=0.8, zorder=10)
            
            # Add to legend
            seed_legend_elements.append(
                plt.Line2D([0], [0], color=color, linewidth=3, label=f'{label} ({len(group)} events)')
            )
            
            self.logger.info(f"  EventID {event_id}: {len(group)} events")
        
        # ========================
        # FORMATTING (Set time range FIRST) - TESTING_8 approach
        # ========================
        
        # Format main plot
        ax.set_yticks(y_positions)
        ax.set_yticklabels(flow_labels, fontsize=9)
        ax.set_ylabel('C2 NetFlow Community IDs', fontsize=12, fontweight='bold')
        ax.set_xlabel('Timeline', fontsize=12, fontweight='bold')
        # Update title for multi-part plots
        if part_suffix:
            title = f'Complete Timeline: Seed Events vs C2 NetFlow - {apt_type.upper()}-Run-{run_id} {part_suffix.replace("_", " ").title()}'
        else:
            title = f'Complete Timeline: Seed Events vs C2 NetFlow - {apt_type.upper()}-Run-{run_id}'
        ax.set_title(title, fontsize=14, fontweight='bold', pad=50)
        ax.grid(True, alpha=0.3)
        
        # Format bottom x-axis (time) with smart tick limiting
        time_range_seconds = (time_range['max_time'] - time_range['min_time']).total_seconds()

        if time_range_seconds <= 3600:  # Less than 1 hour
            interval = max(1, int(time_range_seconds / 480))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=interval))
        elif time_range_seconds <= 14400:  # Less than 4 hours
            interval = max(5, int(time_range_seconds / 720))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=interval))
        else:  # Very long ranges
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=20))

        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # CRITICAL: Set time range BEFORE creating top axis
        time_buffer = timedelta(minutes=2)
        ax.set_xlim(time_range['min_time'] - time_buffer, time_range['max_time'] + time_buffer)
        
        # ========================
        # TOP X-AXIS WITH ORIGINAL ROW NUMBERS (TESTING_8 approach)
        # ========================
        self.logger.info("🏷️ Adding OriginalRowNumber labels on top axis...")
        
        # Create secondary x-axis on top with SYNCHRONIZED positioning
        ax_top = ax.twiny()
        # CRITICAL FIX: Set identical xlim AFTER main axis is configured
        ax_top.set_xlim(ax.get_xlim())
        
        # PRECISION FIX: Ensure exact timestamp-label pairing
        seed_positions = []
        seed_labels = []
        
        for _, event in seed_df.iterrows():
            timestamp = event['timestamp_parsed']
            label = str(event['OriginalRowNumber'])
            seed_positions.append(timestamp)
            seed_labels.append(label)
        
        if len(seed_positions) > 0:
            # Convert timestamps to matplotlib date numbers with high precision
            mpl_positions = [mdates.date2num(pos) for pos in seed_positions]
            
            # Set ticks with exact positions
            ax_top.set_xticks(mpl_positions)
            ax_top.set_xticklabels(seed_labels, rotation=45, fontsize=8, ha='left')
            ax_top.set_xlabel('Seed Event OriginalRowNumber', fontsize=10, fontweight='bold')
        
        # ========================
        # EXTERNAL LEGENDS (Right Side) - TESTING_8 approach
        # ========================
        
        # Create detailed NetFlow legend with unique colors for each flow type
        for flow_type_key, flow_info in flow_type_counts.items():
            count = flow_info['count']
            color = flow_info['color']
            label = flow_info['label']
            
            netflow_legend_elements.append(
                plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.7, 
                             label=f'{label} ({count} flows)')
            )
        
        # Position legends outside plot area
        seed_legend = ax.legend(handles=seed_legend_elements, title='Seed Events', 
                               bbox_to_anchor=(1.02, 1), loc='upper left')
        seed_legend.get_title().set_fontweight('bold')
        
        netflow_legend = ax.legend(handles=netflow_legend_elements, title='NetFlow Communications',
                                  bbox_to_anchor=(1.02, 0.6), loc='upper left')
        netflow_legend.get_title().set_fontweight('bold')
        
        # Add both legends to the plot
        ax.add_artist(seed_legend)  # Keep both legends visible
        
        # Add statistics box in plot area
        total_seed_events = len(seed_df)
        total_netflow_events = sum(len(flow_info['data']) for flow_info in grouped_flows.values())
        
        # Count C2 vs Internal flows
        c2_flows = sum(1 for key in flow_type_counts.keys() if 'C2' in key)
        internal_flows = len(flow_type_counts) - c2_flows
        
        stats_text = (f"Statistics:\n"
                     f"• Selected Seeds: {total_seed_events}\n"
                     f"• NetFlow Events: {total_netflow_events:,}\n"
                     f"• Flow Types: {len(flow_type_counts)}\n"
                     f"• C2 Flows: {c2_flows}\n"
                     f"• Internal Flows: {internal_flows}\n"
                     f"• Time Span: {time_range['duration_hours']:.1f}h")
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Adjust layout to accommodate external legends
        plt.subplots_adjust(right=0.75)
        
        # Save plot with appropriate filename
        output_filename = f"complete_timeline_seed_events_vs_c2_netflow_run-{run_id}{part_suffix}.png"
        output_file = self.file_paths['results_dir'] / output_filename
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"✅ Complete timeline plot saved: {output_file}")
    
    def _create_correlation_hotspots_plot(self, seed_df: pd.DataFrame, grouped_flows: dict, 
                                         apt_type: str, run_id: str):
        """Create correlation hotspots with one subplot per seed event (30-second window)."""
        self.logger.info("📊 Creating individual seed event correlation hotspots (30-second window)...")
        
        # Initialize flow type tracking for consistent color mapping
        flow_type_counts = {}  # Track detailed flow types
        
        # ========================
        # STEP 1: CREATE ONE SUBPLOT PER SEED EVENT
        # ========================
        seed_events_list = seed_df.to_dict('records')
        n_seeds = len(seed_events_list)
        
        if n_seeds == 0:
            self.logger.warning("⚠️ No seed events found - skipping correlation hotspots plot")
            return None
        
        self.logger.info(f"🎯 Creating {n_seeds} subplots (one per seed event)")
        
        # ========================
        # STEP 2: CREATE SUBPLOT GRID
        # ========================
        # Determine optimal grid layout with reasonable figure size limits
        if n_seeds <= 2:
            rows, cols = 1, n_seeds
            figsize = (8 * cols, 6)
        elif n_seeds <= 4:
            rows, cols = 2, 2
            figsize = (16, 12)
        elif n_seeds <= 6:
            rows, cols = 2, 3
            figsize = (18, 12)
        elif n_seeds <= 9:
            rows, cols = 3, 3
            figsize = (18, 18)
        elif n_seeds <= 16:
            rows, cols = 4, 4
            figsize = (20, 20)
        elif n_seeds <= 25:
            rows, cols = 5, 5
            figsize = (25, 25)
        elif n_seeds <= 36:
            rows, cols = 6, 6
            figsize = (30, 30)
        elif n_seeds <= 49:
            rows, cols = 7, 7
            figsize = (35, 35)
        else:
            # For very large numbers (>49), split into multiple figures
            max_subplots_per_figure = 36  # 6x6 grid maximum
            if n_seeds > max_subplots_per_figure:
                return self._create_multiple_correlation_hotspots_plots(seed_df, grouped_flows, apt_type, run_id, max_subplots_per_figure)
            else:
                # Use 8x8 grid as fallback for 50-64 seeds
                rows, cols = 8, 8
                figsize = (32, 32)  # Reduced from 40x40
        
        # Increase horizontal spacing between subplots
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        plt.subplots_adjust(wspace=0.4, hspace=0.3)  # Increase horizontal and vertical spacing
        
        if n_seeds == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        # ========================
        # STEP 3: PLOT EACH SEED EVENT WITH ITS CORRELATED FLOWS
        # ========================
        correlation_threshold = timedelta(seconds=30)  # ±30s window for correlation detection
        
        for i, seed_event in enumerate(seed_events_list):
            if i >= len(axes):
                break
                
            ax = axes[i]
            seed_timestamp = seed_event['timestamp_parsed']
            
            # Find correlated NetFlow events for this seed event using enhanced criteria
            correlated_flows = {}
            
            for community_id, flow_info in grouped_flows.items():
                flow_start = flow_info['start_time']
                flow_end = flow_info['end_time']
                
                # Apply enhanced correlation criteria (matching run_correlation_analysis logic)
                include_flow = False
                time_diff_seconds = 0.0
                
                # Criterion 1: Seed event falls INSIDE the netflow timespan
                if flow_start <= seed_timestamp <= flow_end:
                    include_flow = True
                    time_diff_seconds = 0.0  # Inside the flow
                
                # Criterion 2: NetFlow is on the LEFT of seed event (flow ends before seed)
                elif seed_timestamp > flow_end and (seed_timestamp - flow_end) <= correlation_threshold:
                    include_flow = True
                    time_diff_seconds = abs((seed_timestamp - flow_end).total_seconds())
                
                # Criterion 3: NetFlow is on the RIGHT of seed event (flow starts after seed) 
                elif seed_timestamp < flow_start and (flow_start - seed_timestamp) <= correlation_threshold:
                    include_flow = True
                    time_diff_seconds = abs((flow_start - seed_timestamp).total_seconds())
                
                if include_flow:
                    correlated_flows[community_id] = {
                        'flow_data': flow_info,
                        'time_diff_seconds': time_diff_seconds
                    }
            
            self.logger.info(f"🔍 Seed event {i+1} (Row {seed_event['OriginalRowNumber']}): "
                           f"{len(correlated_flows)} correlated flows")
            
            # Plot NetFlow bars
            y_positions = []
            flow_labels = []
            
            for j, (community_id, corr_info) in enumerate(correlated_flows.items()):
                flow_data = corr_info['flow_data']
                y_pos = j
                y_positions.append(y_pos)
                flow_labels.append(f"{community_id[:8]}...")
                
                start_time = flow_data['start_time']
                end_time = flow_data['end_time']
                
                # Use INTEGRATED flow structure - get color based on flow type
                # Extract IP addresses from the DataFrame (similar to complete timeline)
                if not flow_data['data'].empty:
                    src_ip = flow_data['data'].iloc[0]['source_ip']
                    dst_ip = flow_data['data'].iloc[0]['destination_ip']
                else:
                    src_ip = '0.0.0.0'
                    dst_ip = '0.0.0.0'
                
                src_name = self._get_machine_name(src_ip)
                dst_name = self._get_machine_name(dst_ip)
                
                if flow_data['flow_type'] == 'c2':
                    internal_machine = dst_name if src_name == 'C2' else src_name
                    flow_type_key = f"C2-{internal_machine}"
                else:
                    machines = sorted([src_name, dst_name])
                    flow_type_key = f"{machines[0]}-{machines[1]}"
                
                color = self._get_flow_type_color(flow_type_key)
                
                # Track flow types for consistent legend
                if flow_type_key not in flow_type_counts:
                    flow_type_counts[flow_type_key] = {'count': 0, 'color': color, 'label': flow_type_key.replace('-', '↔')}
                flow_type_counts[flow_type_key]['count'] += 1
                
                # Draw horizontal bar
                if start_time and end_time and (end_time - start_time).total_seconds() > 0:
                    duration_mpl = mdates.date2num(end_time) - mdates.date2num(start_time)
                    ax.barh(y_pos, duration_mpl, left=mdates.date2num(start_time), 
                           height=0.6, color=color, alpha=0.7, edgecolor='navy', linewidth=0.5)
            
            # Plot the seed event as a centered vertical line
            event_id = seed_event['EventID']
            
            # Create descriptive title based on EventID
            if event_id == 1:
                # Process Creation - use CommandLine
                command_line = seed_event.get('CommandLine', 'N/A')
                # Truncate long command lines for readability
                title = command_line[:50] + '...' if len(command_line) > 50 else command_line
                color = self.colors['sysmon_eventid1']
            elif event_id == 11 or event_id == 23:
                # File Create/Delete - use TargetFilename
                target_filename = seed_event.get('TargetFilename', 'N/A')
                # Extract just filename from full path for readability
                import os
                title = os.path.basename(target_filename) if target_filename != 'N/A' else 'N/A'
                color = self.colors['sysmon_eventid11'] if event_id == 11 else self.colors['sysmon_eventid23']
            else:
                # Other EventIDs - fallback to OriginalRowNumber
                title = f"EventID {event_id}"
                color = '#666666'
            
            # Plot vertical line for seed event (centered)
            ax.axvline(x=seed_timestamp, color=color, alpha=0.8, linewidth=2.0, zorder=10, 
                      linestyle='-', label=f"Seed Event {seed_event['OriginalRowNumber']}")
            
            # Format subplot FIRST before setting up top axis
            if len(y_positions) > 0:
                ax.set_yticks(y_positions)
                ax.set_yticklabels(flow_labels, fontsize=8)
            ax.set_ylabel('NetFlow IDs', fontsize=9)
            ax.set_xlabel('Time (±30s window)', fontsize=9)
            
            # CRITICAL: Set time limits to ±30s display window (seed event centered)
            display_padding = timedelta(seconds=30)
            display_start = seed_timestamp - display_padding
            display_end = seed_timestamp + display_padding
            ax.set_xlim(display_start, display_end)
            
            # Add OriginalRowNumber label on top axis (centered on seed event)
            ax_top = ax.twiny()
            # CRITICAL FIX: Set identical xlim AFTER main axis is configured
            ax_top.set_xlim(ax.get_xlim())
            
            # Single tick at the seed event position
            mpl_position = mdates.date2num(seed_timestamp)
            ax_top.set_xticks([mpl_position])
            ax_top.set_xticklabels([str(seed_event['OriginalRowNumber'])], 
                                 rotation=45, fontsize=10, ha='center', fontweight='bold')
            ax_top.set_xlabel('OriginalRowNumber', fontsize=8)
            
            # Format time axis with finer granularity for 30-second window
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            ax.xaxis.set_major_locator(mdates.SecondLocator(interval=10))  # 10-second intervals
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, fontsize=7)
            
            # Add subplot title with descriptive seed event information
            ax.set_title(f'Seed Event {seed_event["OriginalRowNumber"]} (EventID {event_id})\n'
                        f'{title[:40]}{"..." if len(title) > 40 else ""}\n'
                        f'{len(correlated_flows)} correlated flows',
                        fontsize=9, fontweight='bold', pad=15)
            
            ax.grid(True, alpha=0.3)
        
        # Intelligently use empty subplots for legends instead of hiding them
        empty_subplots = list(range(n_seeds, len(axes)))
        
        # Hide remaining empty subplots (keep at least 1 for combined legend if available)
        legend_subplots_used = 0
        for i in empty_subplots:
            if legend_subplots_used < 1 and len(empty_subplots) >= 1:
                # Keep this subplot for combined legend
                legend_subplots_used += 1
            else:
                # Hide unused subplot
                axes[i].set_visible(False)
        
        # ========================
        # STEP 5: ADD OVERALL LEGEND
        # ========================
        # Create global legend elements
        seed_legend_elements = [
            # plt.Line2D([0], [0], color=self.colors['sysmon_eventid1'], linewidth=2, label='EventID 1 (Process Creation)'),
            plt.Line2D([0], [0], color=self.colors['sysmon_eventid1'], linewidth=2, label='EventID 1'),
            # plt.Line2D([0], [0], color=self.colors['sysmon_eventid11'], linewidth=2, label='EventID 11 (File Create)'),
            plt.Line2D([0], [0], color=self.colors['sysmon_eventid11'], linewidth=2, label='EventID 11'),
            # plt.Line2D([0], [0], color=self.colors['sysmon_eventid23'], linewidth=2, label='EventID 23 (File Delete)')
            plt.Line2D([0], [0], color=self.colors['sysmon_eventid23'], linewidth=2, label='EventID 23')
        ]
        
        # Create detailed NetFlow legend with actual flow types (consistent with complete_timeline)
        netflow_legend_elements = []
        for flow_type_key, flow_info in flow_type_counts.items():
            label = f"{flow_info['label']} ({flow_info['count']})"
            color = flow_info['color']
            netflow_legend_elements.append(
                plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.7, 
                             edgecolor='navy', linewidth=0.5, label=label)
            )
        
        # Flow marker legend removed per user request
        
        # ========================
        # STEP 5.1: INTELLIGENT COMBINED LEGEND IN EMPTY SUBPLOT
        # ========================
        if len(empty_subplots) >= 1:
            # Use first empty subplot for combined legend
            legend_ax = axes[empty_subplots[0]]
            legend_ax.axis('off')  # Hide axes
            
            # Combine both legend elements into one comprehensive legend
            combined_legend_elements = []
            
            # Add section separator for Seed Events
            combined_legend_elements.append(plt.Line2D([0], [0], color='none', label='── Seed Events ──'))
            combined_legend_elements.extend(seed_legend_elements)
            
            # Add section separator for NetFlow Communications  
            combined_legend_elements.append(plt.Line2D([0], [0], color='none', label='── NetFlow Communications ──'))
            combined_legend_elements.extend(netflow_legend_elements)
            
            # Create single comprehensive legend
            legend_ax.legend(handles=combined_legend_elements, 
                            title='Legend: Seed Events & NetFlow Communications', 
                            loc='center', 
                            fontsize=9,
                            title_fontsize=11,
                            frameon=True,
                            fancybox=True,
                            shadow=True,
                            ncol=1)  # Single column for better organization
        else:
            # Fallback to figure legends if no empty subplots available
            fig.legend(seed_legend_elements, [elem.get_label() for elem in seed_legend_elements],
                      title='Seed Events', bbox_to_anchor=(0.02, 0.88), loc='upper left', fontsize=9)
            fig.legend(netflow_legend_elements, [elem.get_label() for elem in netflow_legend_elements],
                      title='NetFlow Communications', bbox_to_anchor=(0.02, 0.12), loc='lower left', fontsize=9)
        
        # ========================
        # STEP 6: SAVE AND FINALIZE
        # ========================
        plt.suptitle(f'Correlation Hotspots: Seed Events vs C2 NetFlow - {apt_type.upper()}-Run-{run_id}', 
                    fontsize=16, fontweight='bold', y=0.96)
        
        # IMPROVED LAYOUT: More space for subplots, no side legend interference
        if len(empty_subplots) >= 1:
            # Legend is in subplot, use full width
            plt.tight_layout(rect=[0.03, 0.03, 0.98, 0.92])
        else:
            # Fallback: Reserve space for side legends
            plt.tight_layout(rect=[0.18, 0.03, 0.98, 0.92])
        
        # Save the correlation hotspots plot
        output_file = self.file_paths['results_dir'] / f"correlation_hotspots_seed_events_vs_c2_netflow_run-{run_id}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        self.logger.info(f"💾 Correlation hotspots visualization saved: {output_file}")
        
        plt.close()
        
        return output_file
    
    def _create_multiple_correlation_hotspots_plots(self, seed_df: pd.DataFrame, grouped_flows: dict, 
                                                   apt_type: str, run_id: str, max_subplots_per_figure: int):
        """Create multiple correlation hotspots plots when there are too many seed events."""
        self.logger.info(f"📊 Creating multiple correlation hotspots plots ({max_subplots_per_figure} subplots per figure)...")
        
        seed_events_list = seed_df.to_dict('records')
        n_seeds = len(seed_events_list)
        n_figures = (n_seeds + max_subplots_per_figure - 1) // max_subplots_per_figure  # Ceiling division
        
        self.logger.info(f"🎯 Splitting {n_seeds} seed events into {n_figures} figures")
        
        output_files = []
        
        for fig_idx in range(n_figures):
            start_idx = fig_idx * max_subplots_per_figure
            end_idx = min(start_idx + max_subplots_per_figure, n_seeds)
            seeds_for_figure = seed_events_list[start_idx:end_idx]
            seeds_in_figure = len(seeds_for_figure)
            
            self.logger.info(f"📈 Creating figure {fig_idx + 1}/{n_figures} with {seeds_in_figure} seed events")
            
            # Create subset DataFrame for this figure
            figure_seed_df = pd.DataFrame(seeds_for_figure)
            
            # Create single figure with manageable size (6x6 grid)
            rows, cols = 6, 6
            figsize = (30, 30)
            
            # Initialize flow type tracking for consistent color mapping
            flow_type_counts = {}
            
            # Create subplot grid
            fig, axes = plt.subplots(rows, cols, figsize=figsize)
            plt.subplots_adjust(wspace=0.4, hspace=0.3)
            
            if seeds_in_figure == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
            
            # Plot each seed event in this figure
            correlation_threshold = timedelta(seconds=30)
            
            for i, seed_event in enumerate(seeds_for_figure):
                if i >= len(axes):
                    break
                    
                ax = axes[i]
                seed_timestamp = seed_event['timestamp_parsed']
                
                # Find correlated NetFlow events (same logic as main method)
                correlated_flows = {}
                
                for community_id, flow_info in grouped_flows.items():
                    flow_start = flow_info['start_time']
                    flow_end = flow_info['end_time']
                    
                    include_flow = False
                    time_diff_seconds = 0.0
                    
                    # Apply enhanced correlation criteria
                    if flow_start <= seed_timestamp <= flow_end:
                        include_flow = True
                        time_diff_seconds = 0.0
                    elif seed_timestamp > flow_end and (seed_timestamp - flow_end) <= correlation_threshold:
                        include_flow = True
                        time_diff_seconds = abs((seed_timestamp - flow_end).total_seconds())
                    elif seed_timestamp < flow_start and (flow_start - seed_timestamp) <= correlation_threshold:
                        include_flow = True
                        time_diff_seconds = abs((flow_start - seed_timestamp).total_seconds())
                    
                    if include_flow:
                        correlated_flows[community_id] = {
                            'flow_data': flow_info,
                            'time_diff_seconds': time_diff_seconds
                        }
                
                # Plot NetFlow bars (same logic as main method)
                y_positions = []
                flow_labels = []
                
                for j, (community_id, corr_info) in enumerate(correlated_flows.items()):
                    flow_data = corr_info['flow_data']
                    y_pos = j
                    y_positions.append(y_pos)
                    flow_labels.append(f"{community_id[:8]}...")
                    
                    start_time = flow_data['start_time']
                    end_time = flow_data['end_time']
                    
                    if not flow_data['data'].empty:
                        src_ip = flow_data['data'].iloc[0]['source_ip']
                        dst_ip = flow_data['data'].iloc[0]['destination_ip']
                    else:
                        src_ip = '0.0.0.0'
                        dst_ip = '0.0.0.0'
                    
                    src_name = self._get_machine_name(src_ip)
                    dst_name = self._get_machine_name(dst_ip)
                    
                    if flow_data['flow_type'] == 'c2':
                        internal_machine = dst_name if src_name == 'C2' else src_name
                        flow_type_key = f"C2-{internal_machine}"
                    else:
                        machines = sorted([src_name, dst_name])
                        flow_type_key = f"{machines[0]}-{machines[1]}"
                    
                    color = self._get_flow_type_color(flow_type_key)
                    
                    if flow_type_key not in flow_type_counts:
                        flow_type_counts[flow_type_key] = {'count': 0, 'color': color, 'label': flow_type_key.replace('-', '↔')}
                    flow_type_counts[flow_type_key]['count'] += 1
                    
                    if start_time and end_time and (end_time - start_time).total_seconds() > 0:
                        duration_mpl = mdates.date2num(end_time) - mdates.date2num(start_time)
                        ax.barh(y_pos, duration_mpl, left=mdates.date2num(start_time), 
                               height=0.6, color=color, alpha=0.7, edgecolor='navy', linewidth=0.5)
                
                # Plot seed event vertical line (same logic as main method)
                event_id = seed_event['EventID']
                
                if event_id == 1:
                    command_line = seed_event.get('CommandLine', 'N/A')
                    title = command_line[:50] + '...' if len(command_line) > 50 else command_line
                    color = self.colors['sysmon_eventid1']
                elif event_id == 11 or event_id == 23:
                    target_filename = seed_event.get('TargetFilename', 'N/A')
                    import os
                    title = os.path.basename(target_filename) if target_filename != 'N/A' else 'N/A'
                    color = self.colors['sysmon_eventid11'] if event_id == 11 else self.colors['sysmon_eventid23']
                else:
                    title = f"EventID {event_id}"
                    color = '#666666'
                
                ax.axvline(x=seed_timestamp, color=color, alpha=0.8, linewidth=2.0, zorder=10, 
                          linestyle='-', label=f"Seed Event {seed_event['OriginalRowNumber']}")
                
                # Format subplot
                if len(y_positions) > 0:
                    ax.set_yticks(y_positions)
                    ax.set_yticklabels(flow_labels, fontsize=8)
                ax.set_ylabel('NetFlow IDs', fontsize=9)
                ax.set_xlabel('Time (±30s window)', fontsize=9)
                
                display_padding = timedelta(seconds=30)
                display_start = seed_timestamp - display_padding
                display_end = seed_timestamp + display_padding
                ax.set_xlim(display_start, display_end)
                
                # Top axis with OriginalRowNumber
                ax_top = ax.twiny()
                ax_top.set_xlim(ax.get_xlim())
                
                mpl_position = mdates.date2num(seed_timestamp)
                ax_top.set_xticks([mpl_position])
                ax_top.set_xticklabels([str(seed_event['OriginalRowNumber'])], 
                                     rotation=45, fontsize=10, ha='center', fontweight='bold')
                ax_top.set_xlabel('OriginalRowNumber', fontsize=8)
                
                # Format time axis
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                ax.xaxis.set_major_locator(mdates.SecondLocator(interval=10))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, fontsize=7)
                
                # Add subplot title
                ax.set_title(f'Seed Event {seed_event["OriginalRowNumber"]} (EventID {event_id})\n'
                            f'{title[:40]}{"..." if len(title) > 40 else ""}\n'
                            f'{len(correlated_flows)} correlated flows',
                            fontsize=9, fontweight='bold', pad=15)
                
                ax.grid(True, alpha=0.3)
            
            # Hide empty subplots
            for i in range(seeds_in_figure, len(axes)):
                axes[i].set_visible(False)
            
            # Add figure title
            plt.suptitle(f'Correlation Hotspots: Seed Events vs C2 NetFlow - {apt_type.upper()}-Run-{run_id} (Part {fig_idx + 1}/{n_figures})', 
                        fontsize=16, fontweight='bold', y=0.96)
            
            plt.tight_layout(rect=[0.03, 0.03, 0.98, 0.92])
            
            # Save this figure
            part_suffix = f"_part{fig_idx + 1}" if n_figures > 1 else ""
            output_file = self.file_paths['results_dir'] / f"correlation_hotspots_seed_events_vs_c2_netflow_run-{run_id}{part_suffix}.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            self.logger.info(f"💾 Correlation hotspots part {fig_idx + 1} saved: {output_file}")
            
            plt.close()
            output_files.append(output_file)
        
        return output_files

    def run_automated_subnetflow_assignment(self, apt_type: str, run_id: str) -> bool:
        """
        Automated replacement for Manual Checkpoint 2
        Implements end_time_proximity_assignment logic with multi-assignment capability
        """
        self.logger.info("🤖 AUTOMATED SUBNETFLOW ASSIGNMENT")
        self.logger.info("   📊 Processing communities marked for Subnetflow-attribution...")

        try:
            # 1. Pre-flight checks
            if not self._pre_flight_checks(apt_type, run_id):
                return False

            # 2. Load verification matrix to find communities needing automation
            verification_file = self.file_paths['verification_matrix']
            verification_df = pd.read_csv(verification_file)

            # Find communities marked for sub-NetFlow analysis
            subnetflow_mask = verification_df['Subnetflow-attribution'].astype(str).str.strip().str.lower() == 'x'
            subnetflow_communities = verification_df[subnetflow_mask]['network_community_id'].unique()

            if len(subnetflow_communities) == 0:
                self.logger.info("✅ No communities marked for automated sub-NetFlow assignment")
                return True

            self.logger.info(f"🎯 Processing {len(subnetflow_communities)} communities for automated assignment")

            # 3. Load required data
            seed_events_df = self.load_seed_events(apt_type, run_id)
            self.seed_events_df = seed_events_df  # Store for community filtering
            netflow_df = self.load_netflow_data(apt_type, run_id)

            # 4. Create pattern_analysis directory
            pattern_analysis_dir = self.file_paths['results_dir'] / "pattern_analysis"
            pattern_analysis_dir.mkdir(exist_ok=True)

            # 5. Process all communities systematically
            all_assignments = {}
            all_assignment_confidence = {}
            community_stats = []
            successful_communities = 0
            failed_communities = 0

            for community_id in subnetflow_communities:
                self.logger.info(f"🔄 Processing community: {community_id[:20]}...")

                try:
                    community_assignments = self._safe_community_processing(
                        community_id, netflow_df, seed_events_df, apt_type, run_id
                    )

                    if community_assignments is None:  # Error occurred
                        failed_communities += 1
                        self.logger.warning(f"⚠️  Skipping failed community: {community_id[:20]}...")
                        continue
                    elif len(community_assignments[0]) == 0:  # Skip empty
                        self.logger.info(f"✅ Community processed (no assignments): {community_id[:20]}...")
                        successful_communities += 1
                        continue

                    assignments, confidence = community_assignments
                    all_assignments.update(assignments)
                    all_assignment_confidence.update(confidence)

                    # Collect statistics
                    community_stats.append({
                        'community_id': community_id,
                        'total_subnetflows': len(assignments),
                        'assignment_rate': 100.0 if len(assignments) > 0 else 0.0
                    })

                    successful_communities += 1
                    self.logger.info(f"✅ Community processed successfully: {community_id[:20]}... ({len(assignments)} assignments)")

                except Exception as e:
                    failed_communities += 1
                    self.logger.error(f"❌ Error processing community {community_id[:20]}...: {e}")
                    continue  # Continue with next community

            # Summary of processing results
            self.logger.info(f"📊 Community Processing Summary:")
            self.logger.info(f"   ✅ Successful: {successful_communities}")
            self.logger.info(f"   ❌ Failed: {failed_communities}")
            self.logger.info(f"   📦 Total assignments: {len(all_assignments)}")

            # Only fail if ALL communities failed
            if successful_communities == 0 and failed_communities > 0:
                self.logger.error("❌ All communities failed - automated assignment cannot proceed")
                return False

            # 6. Generate automated assignment CSV (replaces manual template)
            self._generate_automated_assignment_csv(
                all_assignments, all_assignment_confidence, apt_type, run_id
            )

            # 7. Export detailed analysis
            self._export_automated_assignment_analysis(
                all_assignments, all_assignment_confidence, community_stats, pattern_analysis_dir
            )

            self.logger.info("✅ Automated subnetflow assignment completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"❌ Automated assignment failed: {e}")
            return False

    def _fallback_to_manual_assignment(self, apt_type: str, run_id: str) -> bool:
        """Fallback to original manual workflow if automation fails."""
        print("\n" + "="*70)
        print("📝 FALLING BACK TO MANUAL ASSIGNMENT WORKFLOW")
        print(f"   📝 Reference file: {self.file_paths['subnetflow_template_create']}")
        print(f"   📝 Please edit: {self.file_paths['subnetflow_template']}")
        print("   📋 Fill 'seed_event' column for each sub-NetFlow assignment")
        print("   💡 Use OriginalRowNumber values from your seed events")
        print("   💡 Leave blank for sub-NetFlows you want to remain unlabeled")
        print("="*70)
        input("   ⏵  Press ENTER when manual editing is complete...")

        # Validate manual assignments
        return self.validate_subnetflow_assignments()

    def _pre_flight_checks(self, apt_type: str, run_id: str) -> bool:
        """Comprehensive pre-flight validation."""
        checks = {
            'seed_events_exist': self.file_paths['seed_events'].exists(),
            'netflow_data_exist': self.file_paths['netflow_data'].exists(),
            'verification_matrix_exist': self.file_paths['verification_matrix'].exists(),
        }

        failed_checks = [check for check, result in checks.items() if not result]
        if failed_checks:
            self.logger.error(f"❌ Pre-flight checks failed: {failed_checks}")
            return False

        self.logger.info("✅ All pre-flight checks passed")
        return True

    def _safe_community_processing(self, community_id: str, netflow_df: pd.DataFrame,
                                 seed_events_df: pd.DataFrame, apt_type: str, run_id: str) -> Optional[Tuple]:
        """Process individual community with error handling."""
        try:
            # Get community data and create subnetflows
            community_data = netflow_df[netflow_df['network_community_id'] == community_id]

            if len(community_data) == 0:
                self.logger.warning(f"⚠️  No data for community: {community_id[:20]}...")
                return ({}, {})

            # Parse timestamps
            community_data = community_data.copy()
            community_data['event_start_parsed'] = pd.to_datetime(community_data['event_start'], unit='ms')
            community_data['event_end_parsed'] = pd.to_datetime(community_data['event_end'], unit='ms')

            # Group by (event_start, event_end) to identify sub-NetFlows
            grouped = community_data.groupby(['event_start_parsed', 'event_end_parsed'])

            # Create subnetflows DataFrame
            subnetflows_data = []
            for i, ((start_time, end_time), group) in enumerate(grouped):
                subnetflows_data.append({
                    'subnetflow_id': f"1_{community_id}_{i+1}",
                    'start_timestamp': start_time.strftime('%Y-%m-%d %H:%M:%S.%f'),
                    'end_timestamp': end_time.strftime('%Y-%m-%d %H:%M:%S.%f'),
                    'event_count': len(group),
                    'community_id': community_id
                })

            subnetflows_df = pd.DataFrame(subnetflows_data)

            if len(subnetflows_df) > 100:  # Large community
                self.logger.warning(f"⚠️  Large community ({len(subnetflows_df)} subnetflows): {community_id[:20]}...")

            # Filter seed events to only those correlated with this community (CRITICAL FIX)
            community_seed_events_df = self._get_community_seed_events(community_id, apt_type, run_id)

            if len(community_seed_events_df) == 0:
                self.logger.warning(f"⚠️  No seed events found for community {community_id[:20]}...")
                return ({}, {})

            # Apply end-time proximity assignment (match working script: ±5s = 10s window)
            assignments, confidence = self._apply_end_time_proximity_assignment(
                subnetflows_df, community_seed_events_df, time_window_seconds=10
            )

            # Generate timeline visualization with community-specific seed events
            self._create_assignment_timeline_plot(
                community_id, subnetflows_df, community_seed_events_df, assignments, apt_type, run_id, time_window_seconds=10
            )

            return (assignments, confidence)

        except Exception as e:
            self.logger.error(f"❌ Error processing community {community_id[:20]}...: {e}")
            import traceback
            self.logger.debug(f"Full traceback: {traceback.format_exc()}")
            return None

    def _get_community_seed_events(self, community_id: str, apt_type: str, run_id: str) -> pd.DataFrame:
        """Filter seed events to only those correlated with the specific community - EXACT COPY FROM WORKING SCRIPT."""
        try:
            # Load verification matrix to get community-specific seed events
            verification_file = self.file_paths['results_dir'] / f'verification_matrix_v2_run-{run_id}.csv'
            if not verification_file.exists():
                verification_file = self.file_paths['results_dir'] / f'verification_matrix_run-{run_id}.csv'

            if not verification_file.exists():
                self.logger.warning(f"⚠️  Verification matrix not found for community filtering")
                return pd.DataFrame()

            verification_df = pd.read_csv(verification_file)

            # Filter to this community's seed events (EXACT logic from working script)
            community_seeds = verification_df[
                (verification_df['network_community_id'] == community_id) &
                (verification_df['Subnetflow-attribution'].astype(str).str.strip().str.lower() == 'x')
            ]

            self.logger.info(f"   📋 Found {len(community_seeds)} seed events marked for Subnetflow-attribution")

            if len(community_seeds) == 0:
                return pd.DataFrame()

            # Get details for each seed event FROM VERIFICATION MATRIX (EXACT COPY from working script)
            seed_details = []
            for _, row in community_seeds.iterrows():
                seed_id = row['seed_event']
                seed_timestamp = row['seed_timestamp']

                # Find the seed event in all_target_events to get tactic/technique details
                seed_info = self.seed_events_df[self.seed_events_df['OriginalRowNumber'] == seed_id]

                if len(seed_info) > 0:
                    seed_row = seed_info.iloc[0]
                    seed_details.append({
                        'OriginalRowNumber': seed_id,
                        'seed_timestamp': pd.to_datetime(seed_timestamp),  # KEY FIX: Use verification matrix timestamp
                        'Tactic': seed_row['Tactic'],
                        'Technique': seed_row['Technique'],
                        'correlation_distance_ms': row['time_diff_seconds'] * 1000
                    })

            if len(seed_details) == 0:
                return pd.DataFrame()

            # Create DataFrame exactly like the working script
            seed_events_df = pd.DataFrame(seed_details).sort_values('seed_timestamp')

            self.logger.info(f"   🎯 Loaded {len(seed_events_df)} community-specific seed events with correlation timestamps")

            return seed_events_df

        except Exception as e:
            self.logger.error(f"❌ Error filtering community seed events: {e}")
            return pd.DataFrame()

    def _apply_end_time_proximity_assignment(self, subnetflows_df: pd.DataFrame,
                                           seed_events_df: pd.DataFrame,
                                           time_window_seconds: int = 10) -> Tuple[Dict, Dict]:
        """Core integration matching end_time_proximity_assignment.py exactly."""
        assignments = {}
        assignment_confidence = {}
        assigned_seeds = set()

        self.logger.info(f"🎯 PERFORMING END-TIME PROXIMITY ASSIGNMENT (±{time_window_seconds//2}s window)...")

        # EXACT LOGIC from working script: For each seed event, find ALL matching subnetflows
        for _, seed in seed_events_df.iterrows():
            # Use the already-parsed seed_timestamp from verification matrix (KEY FIX)
            seed_time = seed['seed_timestamp']
            seed_id = seed['OriginalRowNumber']

            if seed_id in assigned_seeds:
                continue

            # Find all subnetflows within time window
            matching_subnetflows = []

            # Check all subnetflows for end-time proximity (EXACT from working script)
            for _, subnetflow in subnetflows_df.iterrows():
                try:
                    end_time = pd.to_datetime(subnetflow['end_timestamp'], format='mixed')
                    distance = abs((seed_time - end_time).total_seconds())

                    # Include if within time window and not already assigned
                    if (distance <= time_window_seconds and
                        subnetflow['subnetflow_id'] not in assignments):

                        matching_subnetflows.append({
                            'subnetflow': subnetflow,
                            'distance': distance
                        })
                except:
                    # Skip timestamp parsing errors
                    continue

            # Assign ALL matching subnetflows (multi-assignment approach)
            for match in matching_subnetflows:
                sub_id = match['subnetflow']['subnetflow_id']
                distance = match['distance']
                assignments[sub_id] = seed_id
                assignment_confidence[sub_id] = f'END_TIME_PROXIMITY_{distance:.1f}s'

        return assignments, assignment_confidence

    def _create_assignment_timeline_plot(self, community_id: str, subnetflows_df: pd.DataFrame,
                                       seed_events_df: pd.DataFrame, assignments: Dict,
                                       apt_type: str, run_id: str, time_window_seconds: int = 10) -> str:
        """Create timeline visualization matching end_time_proximity_assignment.py style."""
        try:
            pattern_analysis_dir = self.file_paths['results_dir'] / "pattern_analysis"
            clean_id = community_id.replace('/', '_').replace(':', '_').replace('=', '').replace('+', 'plus')[:20]
            output_path = pattern_analysis_dir / f'end_time_proximity_timeline_{clean_id}.png'

            # Set up the plot (matching end_time_proximity_assignment.py)
            fig, ax = plt.subplots(figsize=(20, 12))

            # Define colors for different tactics (EXACT COPY from working script)
            tactic_colors = {
                'initial-access': '#000000',      # Black
                'execution': '#4169E1',           # Royal Blue
                'persistence': '#228B22',         # Forest Green
                'privilege-escalation': '#B22222', # Fire Brick Red
                'defense-evasion': '#FF8C00',     # Dark Orange
                'credential-access': '#FFD700',   # Gold
                'discovery': '#8B4513',           # Saddle Brown
                'lateral-movement': '#FF1493',    # Deep Pink
                'collection': '#9932CC',          # Dark Orchid
                'command-and-control': '#00CED1', # Dark Turquoise
                'exfiltration': '#32CD32',        # Lime Green
                'impact': '#DC143C',              # Crimson
                'unassigned': '#CCCCCC'           # Light Gray
            }

            # Plot subnetflows as horizontal bars (EXACT LOGIC from working script)
            bar_height = 0.8
            y_labels = []

            for i, (_, subnetflow) in enumerate(subnetflows_df.iterrows()):
                start_time = subnetflow['start_timestamp']
                end_time = subnetflow['end_timestamp']
                sub_id = subnetflow['subnetflow_id']

                # Use simple ordinal number for y-axis labels (1, 2, 3, ...)
                ordinal_number = i + 1

                # Get assignment and color BY TACTIC (key difference!)
                assigned_seed = assignments.get(sub_id, None)
                if assigned_seed:
                    # Find seed info to get tactic
                    seed_info = seed_events_df[seed_events_df['OriginalRowNumber'] == assigned_seed]
                    if len(seed_info) > 0:
                        tactic = seed_info.iloc[0]['Tactic']
                        color = tactic_colors.get(tactic, '#666666')
                        alpha = 0.8
                    else:
                        tactic = 'unassigned'
                        color = tactic_colors['unassigned']
                        alpha = 0.4
                else:
                    tactic = 'unassigned'
                    color = tactic_colors['unassigned']
                    alpha = 0.4

                # Calculate duration for bar width (matplotlib date format)
                start_time_parsed = pd.to_datetime(start_time)
                end_time_parsed = pd.to_datetime(end_time, format='mixed')
                duration_mpl = mdates.date2num(end_time_parsed) - mdates.date2num(start_time_parsed)

                # CRITICAL FIX: Set minimum bar width for visibility (absolute time-based)
                min_bar_width_seconds = 1.0  # 1 second minimum width for visibility
                min_bar_width = min_bar_width_seconds / 86400.0  # Convert to matplotlib date units (1 day = 1.0)
                if duration_mpl <= 0 or duration_mpl < min_bar_width:
                    duration_mpl = min_bar_width

                # Plot horizontal bar (CLEAN: no border, just tactic color)
                ax.barh(i, duration_mpl,
                       left=mdates.date2num(start_time_parsed),
                       height=bar_height,
                       color=color,
                       alpha=alpha,
                       edgecolor='none',  # Remove border color
                       linewidth=0)

                # Create CLEAN y-axis label: just ordinal → seed_event (no network_community_id)
                if assigned_seed:
                    y_labels.append(f'{ordinal_number} → {assigned_seed}')
                else:
                    y_labels.append(f'{ordinal_number}')

            # Set y-axis (SAME as working script)
            ax.set_yticks(range(len(y_labels)))
            ax.set_yticklabels(y_labels, fontsize=8)
            ax.set_ylabel('Subnetflow ID → Assigned Seed Event', fontsize=12)

            # Plot seed events as vertical lines (SAME as working script)
            seed_positions = []
            seed_labels = []
            seed_colors = []

            for _, seed in seed_events_df.iterrows():
                timestamp = seed['seed_timestamp']  # KEY FIX: Use verification matrix timestamp (EXACT COPY from working script)
                seed_id = seed['OriginalRowNumber']
                tactic = seed['Tactic']

                color = tactic_colors.get(tactic, '#666666')

                # Plot vertical line for seed event (EXACT SAME as working script)
                ax.axvline(x=timestamp, color=color, linestyle='--', linewidth=2, alpha=0.9, zorder=10)

                # Store for top axis labels (EXACT SAME as working script)
                seed_positions.append(timestamp)
                seed_labels.append(str(seed_id))
                seed_colors.append(color)

            # Format x-axis with enhanced tick limiting and EXPLICIT range setting (MOVED UP)
            if len(subnetflows_df) > 0:
                start_times = [pd.to_datetime(sf['start_timestamp']) for _, sf in subnetflows_df.iterrows()]
                end_times = [pd.to_datetime(sf['end_timestamp'], format='mixed') for _, sf in subnetflows_df.iterrows()]
                time_range_minutes = (max(end_times) - min(start_times)).total_seconds() / 60

                # CRITICAL: Set explicit x-axis limits BEFORE applying locator
                min_time = min(start_times)
                max_time = max(end_times)
                ax.set_xlim(min_time, max_time)

                self.logger.info(f"📊 Time range for visualization: {time_range_minutes:.1f} minutes")
                self.logger.info(f"📊 X-axis range: {min_time} to {max_time}")

                # Enhanced interval calculation to prevent tick overload
                if time_range_minutes <= 5:
                    interval = 1  # 1 minute intervals for short ranges
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=interval))
                elif time_range_minutes <= 30:
                    interval = 5  # 5 minute intervals for medium ranges
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=interval))
                elif time_range_minutes <= 180:  # 3 hours
                    interval = 15  # 15 minute intervals for long ranges
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=interval))
                else:
                    # For very long ranges, use AutoDateLocator to prevent overload
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=15))
                    self.logger.warning(f"⚠️  Using AutoDateLocator for long time range: {time_range_minutes:.1f} minutes")

                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, fontsize=10)
                self.logger.info(f"📊 Applied time axis formatting for {time_range_minutes:.1f} minute range")

            ax.set_xlabel('Timeline (HH:MM:SS)', fontsize=12)

            # NOW add top axis for seed event labels (AFTER main axis limits are set)
            ax_top = ax.twiny()
            ax_top.set_xlim(ax.get_xlim())  # Now this captures the CORRECT limits

            # Limit number of seed event labels to prevent matplotlib overload
            max_labels = 20
            if len(seed_positions) <= max_labels:
                # Show all labels if not too many (EXACT SAME as working script)
                ax_top.set_xticks([mdates.date2num(pos) for pos in seed_positions])
                ax_top.set_xticklabels(seed_labels, rotation=90, fontsize=8)
            else:
                # Show only every Nth label to avoid overload
                step = len(seed_positions) // max_labels + 1
                selected_positions = seed_positions[::step]
                selected_labels = seed_labels[::step]
                ax_top.set_xticks([mdates.date2num(pos) for pos in selected_positions])
                ax_top.set_xticklabels(selected_labels, rotation=90, fontsize=8)
                self.logger.info(f"⚠️  Showing {len(selected_labels)} of {len(seed_labels)} seed event labels on plot")

            ax_top.set_xlabel('Seed Event OriginalRowNumber', fontsize=12)

            # Add grid (SAME as working script)
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

            # Create legend for tactics (SAME logic as working script)
            unique_tactics = set()
            for _, seed in seed_events_df.iterrows():
                unique_tactics.add(seed['Tactic'])
            if len(subnetflows_df) - len(assignments) > 0:
                unique_tactics.add('unassigned')

            legend_elements = []
            for tactic in sorted(unique_tactics):
                color = tactic_colors.get(tactic, '#666666')
                legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.8, label=tactic))

            ax.legend(handles=legend_elements, loc='upper right', fontsize=10,
                     title='Attack Tactics', frameon=True, fancybox=True, shadow=True)

            # Title and layout (SAME as working script)
            plt.suptitle(f'End-Time Proximity Assignments - Community {clean_id}', fontsize=16, y=0.95)

            assignment_count = len(assignments)
            unassigned_count = len(subnetflows_df) - assignment_count
            # Use actual time window (±half of time_window_seconds)
            half_window = time_window_seconds // 2
            plt.title(f'{assignment_count} Assigned Subnetflows | {unassigned_count} Unassigned | '
                     f'±{half_window}s End-Time Proximity Algorithm', fontsize=12, pad=20)

            plt.tight_layout(rect=[0.05, 0.05, 0.98, 0.90])

            # Save plot (SAME as working script)
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()

            self.logger.info(f"✅ Timeline visualization saved to: {output_path}")
            return str(output_path)

        except Exception as e:
            self.logger.warning(f"⚠️  Timeline plot creation failed: {e}")
            import traceback
            self.logger.debug(f"Plot creation traceback: {traceback.format_exc()}")
            return ""

    def _generate_automated_assignment_csv(self, assignments: Dict, confidence: Dict,
                                         apt_type: str, run_id: str):
        """Generate the automated assignment CSV that replaces manual template."""
        try:
            template_path = self.file_paths['subnetflow_template']

            # Load the original template structure
            if template_path.exists():
                template_df = pd.read_csv(template_path)

                # Fill assignments automatically
                for idx, row in template_df.iterrows():
                    subnetflow_id = row['subnetflow_id']
                    if subnetflow_id in assignments:
                        template_df.loc[idx, 'seed_event'] = assignments[subnetflow_id]

                # Save back to same file (replaces manual requirement)
                template_df.to_csv(template_path, index=False)
                self.logger.info(f"✅ Automated assignment CSV saved: {template_path}")

        except Exception as e:
            self.logger.error(f"❌ Error generating automated assignment CSV: {e}")

    def _export_automated_assignment_analysis(self, assignments: Dict, confidence: Dict,
                                            community_stats: List, pattern_analysis_dir: Path):
        """Export detailed analysis of automated assignments."""
        try:
            # Summary statistics
            total_assignments = len(assignments)
            avg_confidence = sum(float(conf.split('_')[-1][:-1]) for conf in confidence.values()) / len(confidence) if confidence else 0

            analysis_data = {
                'total_assignments': total_assignments,
                'avg_confidence_seconds': avg_confidence,
                'community_stats': community_stats
            }

            analysis_file = pattern_analysis_dir / "automated_assignment_summary.json"
            with open(analysis_file, 'w') as f:
                json.dump(analysis_data, f, indent=2)

            self.logger.info(f"📊 Assignment analysis exported: {analysis_file}")

        except Exception as e:
            self.logger.warning(f"⚠️  Analysis export failed: {e}")

    def run_complete_pipeline(self, apt_type: str, run_id: str, resume: bool = False):
        """Run the complete integrated NetFlow labeling pipeline."""
        self.logger.info("🚀 Starting Integrated NetFlow Labeling Pipeline...")
        
        # Setup file paths
        self.setup_file_paths(apt_type, run_id)
        
        # Load workflow state
        if resume:
            self.workflow_state = self.load_workflow_state()
            self.logger.info(f"🔄 Resuming from step: {self.workflow_state['step']}")
        
        try:
            # STEP 1: Correlation Analysis
            if not resume or self.workflow_state['step'] in ['initial', 'correlation']:
                self.logger.info("📊 STEP 1: Running correlation analysis...")
                if not self.run_correlation_analysis(apt_type, run_id):
                    self.logger.error("❌ Correlation analysis failed")
                    return False
                
                self.save_workflow_state('awaiting_verification_manual')
            
            # MANUAL CHECKPOINT 1: Verification Matrix
            if self.workflow_state['step'] == 'awaiting_verification_manual':
                print("\n" + "="*70)
                print("⏸️  MANUAL STEP REQUIRED:")
                print(f"   📝 Copy and rename: {self.file_paths['verification_matrix_create']}")
                print(f"   📝 Create manually: {self.file_paths['verification_matrix']}")
                print("   📋 Mark 'x' in Direct-attribution and Subnetflow-attribution columns")
                print("   💡 Direct-attribution: Label entire NetFlow with one tactic")
                print("   💡 Subnetflow-attribution: Granular labeling of specific segments")
                print("")
                print("   📊 Use these correlation plots to guide your manual marking:")
                print(f"   📈 Complete timeline: complete_timeline_seed_events_vs_c2_netflow_run-{run_id}.png")
                print(f"   🎯 Correlation hotspots: correlation_hotspots_seed_events_vs_c2_netflow_run-{run_id}.png")
                print("="*70)
                input("   ⏵  Press ENTER when manual editing is complete...")
                
                # Validate manual file
                if not self.validate_verification_matrix():
                    self.logger.error("❌ Manual file validation failed")
                    return False
                
                self.workflow_state['manual_steps_completed'].append('verification_matrix')
                self.save_workflow_state('verification_complete')
            
            # STEP 2: Sub-NetFlow Analysis (conditional)
            if self.workflow_state['step'] in ['verification_complete', 'subnetflow_analysis']:
                self.logger.info("📊 STEP 2: Running sub-NetFlow analysis...")
                if not self.run_subnetflow_analysis(apt_type, run_id):
                    self.logger.warning("⚠️ Sub-NetFlow analysis failed or skipped")
                
                self.save_workflow_state('awaiting_subnetflow_manual')
            
            # STEP 2.5: Sub-NetFlow Assignments (CONDITIONAL - Automated vs Manual)
            if (self.workflow_state['step'] == 'awaiting_subnetflow_manual' and
                self.file_paths['subnetflow_template'].exists()):

                if self.use_automated_assignment:
                    print("\n" + "="*70)
                    print("🤖 AUTOMATED SUBNETFLOW ASSIGNMENT MODE")
                    print("   🔄 Running automated end-time proximity assignment...")
                    print("="*70)

                    success = self.run_automated_subnetflow_assignment(apt_type, run_id)
                    if not success:
                        self.logger.warning("⚠️  Automated assignment failed - falling back to manual mode")
                        success = self._fallback_to_manual_assignment(apt_type, run_id)
                        if not success:
                            self.logger.error("❌ Manual fallback also failed")
                            return False

                    self.workflow_state['manual_steps_completed'].append('automated_subnetflow_assignments')
                    self.save_workflow_state('assignments_complete')

                else:
                    print("\n" + "="*70)
                    print("⏸️  MANUAL SUBNETFLOW ASSIGNMENT MODE:")
                    print(f"   📝 Reference file: {self.file_paths['subnetflow_template_create']}")
                    print(f"   📝 Please edit: {self.file_paths['subnetflow_template']}")
                    print("   📋 Fill 'seed_event' column for each sub-NetFlow assignment")
                    print("   💡 Use OriginalRowNumber values from your seed events")
                    print("   💡 Leave blank for sub-NetFlows you want to remain unlabeled")
                    print("   💡 subnetflow_id is now numeric (1,2,3...) for easy sorting")
                    print("="*70)
                    input("   ⏵  Press ENTER when manual editing is complete...")

                    # Validate manual assignments
                    if not self.validate_subnetflow_assignments():
                        self.logger.error("❌ Manual assignment validation failed")
                        return False

                    self.workflow_state['manual_steps_completed'].append('manual_subnetflow_assignments')
                    self.save_workflow_state('assignments_complete')
            else:
                self.save_workflow_state('assignments_complete')
            
            # STEP 3: Two-Tier Labeling
            if self.workflow_state['step'] in ['assignments_complete', 'labeling']:
                self.logger.info("🏷️  STEP 3: Applying two-tier labeling...")
                labeled_df = self.apply_two_tier_labeling(apt_type, run_id)
                
                # Save labeled dataset
                labeled_file = self.file_paths['labeled_netflow']
                labeled_df.to_csv(labeled_file, index=False)
                self.logger.info(f"✅ Saved labeled dataset: {labeled_file}")
                
                self.save_workflow_state('labeling_complete')
            else:
                # Load existing labeled dataset
                labeled_df = pd.read_csv(self.file_paths['labeled_netflow'])
            
            # STEP 4: Generate Visualizations
            if self.workflow_state['step'] in ['labeling_complete', 'visualization']:
                self.logger.info("📊 STEP 4: Creating timeline visualizations...")
                self.create_timeline_visualizations(labeled_df, apt_type, run_id)
                
                self.save_workflow_state('complete')
            
            # Pipeline completed successfully
            self.logger.info("✅ INTEGRATED PIPELINE COMPLETED SUCCESSFULLY!")
            self.logger.info("📁 Output files created:")
            self.logger.info(f"   📋 Verification matrix: {self.file_paths['verification_matrix']}")
            if self.file_paths['subnetflow_template'].exists():
                self.logger.info(f"   📋 Sub-NetFlow template (original): {self.file_paths['subnetflow_template_create']}")
                self.logger.info(f"   📋 Sub-NetFlow template (manual): {self.file_paths['subnetflow_template']}")
            self.logger.info(f"   📊 Labeled dataset: {self.file_paths['labeled_netflow']}")
            self.logger.info(f"   📊 Timeline plots: {self.file_paths['results_dir']}")
            
            return True
            
        except KeyboardInterrupt:
            self.logger.info("🔴 Pipeline interrupted by user")
            self.save_workflow_state('interrupted')
            return False
        except Exception as e:
            self.logger.error(f"❌ Pipeline failed: {e}")
            self.save_workflow_state('error')
            return False


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Integrated NetFlow Labeling System - Single Script Approach',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python3 INTEGRATED_netflow_labeler.py --apt-type apt-1 --run-id 04
    python3 INTEGRATED_netflow_labeler.py --resume --apt-type apt-1 --run-id 04
    python3 INTEGRATED_netflow_labeler.py --apt-type apt-2 --run-id 25 --debug
        """
    )
    
    parser.add_argument('--apt-type', required=True,
                       choices=['apt-1', 'apt-2', 'apt-3', 'apt-4', 'apt-5', 'apt-6'],
                       help='APT dataset type')
    
    parser.add_argument('--run-id', required=True,
                       help='Run ID (e.g., 04, 25)')
    
    parser.add_argument('--resume', action='store_true',
                       help='Resume from previous workflow state')
    
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')

    parser.add_argument('--automated-assignment', action='store_true',
                       help='Use automated subnetflow assignment (default: manual)')

    parser.add_argument('--no-automated-assignment', action='store_true',
                       help='Force manual subnetflow assignment')

    args = parser.parse_args()

    # Determine automated assignment mode
    use_automated = args.automated_assignment and not args.no_automated_assignment

    # Create and run integrated labeler
    labeler = IntegratedNetFlowLabeler(debug=args.debug, use_automated_assignment=use_automated)
    success = labeler.run_complete_pipeline(args.apt_type, args.run_id, args.resume)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()