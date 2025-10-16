#!/usr/bin/env python3
"""
Create labeled Sysmon dataset with Tactic/Technique columns and generate v2 timeline plots.

This script:
1. Reads the original sysmon-run-XX.csv file
2. Reads the manually-corrected traced_sysmon_events_with_tactics_v2.csv file 
3. Creates sysmon-run-XX-labeled.csv with Tactic/Technique labels
4. Generates v2 timeline plots based on the labeled dataset:
   - timeline_all_malicious_events_v2.png
   - timeline_all_malicious_events_with_tactics_v2.png

Note: This script uses traced_sysmon_events_with_tactics_v2.csv to preserve manual corrections
made to the Correct_SeedRowNumber column, avoiding overwriting when script 6 is re-run.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta
import argparse
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import shared utilities (with fallback for compatibility)
try:
    from utils.apt_config import TacticColors, PlottingConfig, FeatureFlags, FilePaths
    from utils.apt_plotting_utils import plot_simple_timeline_v2, plot_tactics_timeline_v2
    from utils.apt_path_utils import PathManager
    SHARED_UTILS_AVAILABLE = True
except ImportError:
    print("⚠️ Warning: Shared utilities not available - using original implementation")
    SHARED_UTILS_AVAILABLE = False

def setup_logging() -> logging.Logger:
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

class LabeledSysmonDatasetCreator:
    """Creates labeled Sysmon dataset and generates timeline visualizations."""
    
    def __init__(self, apt_type: str, run_id: str):
        self.apt_type = apt_type
        self.run_id = run_id
        self.logger = setup_logging()
        
        # Define base paths
        scripts_dir = Path(__file__).parent.parent  # Go up to scripts/
        project_root = scripts_dir.parent  # Go to research/
        dataset_dir = project_root / "dataset"  # Point to dataset folder
        self.base_path = dataset_dir / apt_type / f"{apt_type}-run-{run_id}"
        
        # Input files
        self.sysmon_file = self.base_path / f"sysmon-run-{run_id}.csv"
        self.traced_events_file = self.base_path / "sysmon_event_tracing_analysis_results" / f"traced_sysmon_events_with_tactics_v2.csv"
        
        # Output files  
        # self.labeled_dataset_file = self.base_path / f"sysmon-run-{run_id}-labeled-v2.csv"
        self.labeled_dataset_file = self.base_path / f"sysmon-run-{run_id}-labeled.csv"
        self.results_dir = self.base_path / "sysmon_event_tracing_analysis_results"
        
        # MITRE ATT&CK tactic colors (same as used in other scripts)
        self.tactic_colors = {
            'initial-access': '#000000',      # Black (STANDARDIZED)
            'execution': '#4169E1',           # Royal Blue
            'persistence': '#228B22',         # Forest Green
            'privilege-escalation': '#8A2BE2', # Blue Violet
            'defense-evasion': '#FF4500',     # Orange Red
            'credential-access': '#FFD700',   # Gold/Strong Yellow
            'discovery': '#8B4513',           # Saddle Brown
            'lateral-movement': '#FF1493',    # Deep Pink
            'collection': '#2F4F4F',          # Dark Slate Gray
            'command-and-control': '#00CED1', # Dark Turquoise
            'exfiltration': '#FF8C00',        # Dark Orange
            'impact': '#32CD32',              # Lime Green
            'Defense-evasion': '#B22222',     # Fire Brick (for capitalized version)
            'Initial-access': '#000000',      # Black (same as initial-access)
            'Unknown': '#696969',             # Dim Gray
            'no_attack_tactic': '#D3D3D3',    # Light Gray
        }
        
        # Create results directory
        self.results_dir.mkdir(exist_ok=True)
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load the original Sysmon dataset and traced events with tactics."""
        self.logger.info(f"📖 Reading original Sysmon dataset: {self.sysmon_file}")
        
        if not self.sysmon_file.exists():
            raise FileNotFoundError(f"Sysmon file not found: {self.sysmon_file}")
        
        sysmon_df = pd.read_csv(self.sysmon_file)
        self.logger.info(f"Original dataset shape: {sysmon_df.shape}")
        
        self.logger.info(f"📖 Reading manually-corrected traced events with tactics: {self.traced_events_file}")
        
        if not self.traced_events_file.exists():
            raise FileNotFoundError(f"Traced events file not found: {self.traced_events_file}")
        
        traced_df = pd.read_csv(self.traced_events_file)
        self.logger.info(f"Traced events shape: {traced_df.shape}")
        
        return sysmon_df, traced_df
    
    def load_master_tactics(self) -> pd.DataFrame:
        """Load the master tactics file to get correct tactics for seed events."""
        master_file = self.base_path / f"all_target_events_run-{self.run_id}.csv"
        
        if not master_file.exists():
            raise FileNotFoundError(f"Master tactics file not found: {master_file}")
        
        self.logger.info(f"📖 Reading master tactics file: {master_file}")
        master_df = pd.read_csv(master_file)
        self.logger.info(f"Master tactics shape: {master_df.shape}")
        
        return master_df
    
    def create_labeled_dataset(self, sysmon_df: pd.DataFrame, traced_df: pd.DataFrame) -> pd.DataFrame:
        """Create labeled dataset with Tactic/Technique columns."""
        self.logger.info("➕ Adding default Tactic/Technique/Label columns")
        
        # Add default labels
        sysmon_df = sysmon_df.copy()
        sysmon_df['Tactic'] = 'Benign'
        sysmon_df['Technique'] = 'Benign'
        sysmon_df['Label'] = 'Benign'
        
        # Load master tactics file
        master_df = self.load_master_tactics()
        
        self.logger.info("🔍 Processing traced events for labeling")
        
        # Create mapping key function (same logic as Script #6 for consistency)
        def create_mapping_key(row):
            """Create robust unique key matching Script #6's deduplication logic."""
            def safe_str(value):
                return str(value) if pd.notna(value) else 'NULL'

            event_id = safe_str(row.get('EventID', ''))
            computer = safe_str(row.get('Computer', ''))
            timestamp = safe_str(row.get('timestamp', ''))
            process_guid = safe_str(row.get('ProcessGuid', ''))
            process_id = safe_str(row.get('ProcessId', ''))
            image = safe_str(row.get('Image', ''))

            # Event-specific unique identifiers matching Script #6's logic
            if event_id == '1':  # Process Creation
                return f"{event_id}_{computer}_{timestamp}_{process_guid}_{process_id}_{image}"
            elif event_id == '7':  # Image Load
                image_loaded = safe_str(row.get('ImageLoaded', ''))
                return f"{event_id}_{computer}_{timestamp}_{process_guid}_{process_id}_{image}_{image_loaded}"
            elif event_id == '10':  # Process Access
                source_guid = safe_str(row.get('SourceProcessGUID', ''))
                target_guid = safe_str(row.get('TargetProcessGUID', ''))
                return f"{event_id}_{computer}_{timestamp}_{source_guid}_{target_guid}"
            elif event_id in ['11', '23']:  # File Create/Delete
                target_filename = safe_str(row.get('TargetFilename', ''))
                return f"{event_id}_{computer}_{timestamp}_{process_guid}_{process_id}_{target_filename}"
            elif event_id == '13':  # Registry Event
                target_object = safe_str(row.get('TargetObject', ''))
                return f"{event_id}_{computer}_{timestamp}_{process_guid}_{process_id}_{target_object}"
            else:  # Generic fallback
                return f"{event_id}_{computer}_{timestamp}_{process_guid}_{process_id}_{image}"

        # Create mapping keys for sysmon dataset (using apply for robust key generation)
        self.logger.info("🗝️ Creating mapping keys for sysmon dataset (using Script #6's matching logic)")
        sysmon_df['mapping_key'] = sysmon_df.apply(create_mapping_key, axis=1)
        
        # Create mapping dictionary for traced events with corrected tactics
        self.logger.info("🏷️ Building tactic mapping dictionary")
        
        mapping_dict = {}
        processed_count = 0
        
        for _, traced_row in traced_df.iterrows():
            # Determine the final seed row number (manual correction takes priority)
            seed_row_number = traced_row.get('Correct_SeedRowNumber')
            if pd.isna(seed_row_number):
                seed_row_number = traced_row.get('OriginatorRow')
            
            if pd.isna(seed_row_number):
                continue
            
            try:
                seed_row_number = int(seed_row_number)
                
                # Look up the correct tactic/technique from master file
                master_row = master_df[master_df['RawDatasetRowNumber'] == seed_row_number]
                if len(master_row) == 0:
                    self.logger.warning(f"⚠️ Seed row {seed_row_number} not found in master tactics file")
                    continue
                
                # Get the correct tactic/technique
                correct_tactic = master_row.iloc[0]['Tactic']
                correct_technique = master_row.iloc[0]['Technique']
                
                # Create mapping key for this traced event
                traced_mapping_key = create_mapping_key(traced_row)
                
                # Store in mapping dictionary
                mapping_dict[traced_mapping_key] = {
                    'Tactic': correct_tactic,
                    'Technique': correct_technique,
                    'Label': 'Malicious'
                }
                
                processed_count += 1
                    
            except (ValueError, TypeError) as e:
                self.logger.warning(f"⚠️ Error processing traced event: {e}")
                continue
        
        self.logger.info(f"📋 Created mapping for {processed_count} traced events")
        
        # Apply labels using vectorized operations
        self.logger.info("🏷️ Applying malicious event labels")
        labeled_events_count = 0
        
        for mapping_key, labels in mapping_dict.items():
            matching_mask = sysmon_df['mapping_key'] == mapping_key
            matching_count = matching_mask.sum()
            
            if matching_count > 0:
                sysmon_df.loc[matching_mask, 'Tactic'] = labels['Tactic']
                sysmon_df.loc[matching_mask, 'Technique'] = labels['Technique'] 
                sysmon_df.loc[matching_mask, 'Label'] = labels['Label']
                
                labeled_events_count += matching_count
                
                if matching_count > 1:
                    self.logger.debug(f"🔄 Labeled {matching_count} duplicate events with key: {mapping_key}")
            else:
                self.logger.warning(f"⚠️ No matching events found for traced event: {mapping_key}")
        
        self.logger.info(f"✅ Labeled {labeled_events_count} malicious events (including duplicates)")
        
        # Remove temporary mapping key column
        sysmon_df.drop('mapping_key', axis=1, inplace=True)
        
        # Reorder columns to put Tactic, Technique, and Label at the beginning
        cols = sysmon_df.columns.tolist()
        cols.remove('Tactic')
        cols.remove('Technique') 
        cols.remove('Label')
        new_cols = ['Tactic', 'Technique', 'Label'] + cols
        sysmon_df = sysmon_df[new_cols]
        
        return sysmon_df
    
    def save_labeled_dataset(self, labeled_df: pd.DataFrame):
        """Save the labeled dataset to CSV."""
        self.logger.info(f"💾 Saving labeled dataset: {self.labeled_dataset_file}")
        labeled_df.to_csv(self.labeled_dataset_file, index=False)
        
        # Print summary statistics
        total_events = len(labeled_df)
        malicious_events = len(labeled_df[labeled_df['Label'] == 'Malicious'])
        benign_events = len(labeled_df[labeled_df['Label'] == 'Benign'])
        
        self.logger.info("📊 Summary Statistics:")
        self.logger.info(f"Total events: {total_events:,}")
        self.logger.info(f"Malicious events: {malicious_events:,}")
        self.logger.info(f"Benign events: {benign_events:,}")
        self.logger.info(f"Malicious percentage: {(malicious_events / total_events * 100):.2f}%")
        
        # Show tactic distribution (excluding Benign for brevity)
        tactic_counts = labeled_df['Tactic'].value_counts()
        self.logger.info("\n📋 Tactic Distribution:")
        for tactic, count in tactic_counts.items():
            if tactic != 'Benign':
                self.logger.info(f"  {tactic}: {count:,} events")
        
        # Show benign count separately
        benign_count = tactic_counts.get('Benign', 0)
        if benign_count > 0:
            self.logger.info(f"  Benign: {benign_count:,} events")
    
    def prepare_plotting_data(self, labeled_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for plotting by converting timestamps and filtering malicious events."""
        self.logger.info("🎨 Preparing data for timeline plotting")
        
        # Convert timestamp to datetime (milliseconds format)
        labeled_df['datetime'] = pd.to_datetime(labeled_df['timestamp'], unit='ms', errors='coerce')
        
        # Filter malicious events only
        malicious_df = labeled_df[labeled_df['Label'] == 'Malicious'].copy()
        self.logger.info(f"Malicious events for plotting: {len(malicious_df):,}")
        
        # Sort by timestamp
        malicious_df = malicious_df.sort_values('datetime')
        
        return malicious_df
    
    def create_simple_timeline_plot(self, malicious_df: pd.DataFrame):
        """Create group timeline plot (v2 version) - copied from script 6."""
        self.logger.info("📈 Creating group timeline plot (v2)")
        
        # Organize events by computer and sort by event count (descending)
        computers_with_counts = []
        for computer in malicious_df['Computer'].unique():
            computer_events = malicious_df[malicious_df['Computer'] == computer]
            event_count = len(computer_events)
            computers_with_counts.append((computer, event_count))
        
        # Sort by event count (descending) for top-to-bottom arrangement
        computers_with_counts.sort(key=lambda x: x[1], reverse=True)
        computers = [computer for computer, count in computers_with_counts]
        
        if not computers:
            self.logger.warning("⚠️ No computers found for group timeline")
            return
        
        # Create subplots for each computer
        fig, axes = plt.subplots(len(computers), 1, figsize=(16, 8 * len(computers)), sharex=True)
        if len(computers) == 1:
            axes = [axes]
        
        # Tactic colors (consistent with tactics timeline plot)
        tactic_colors = {
            'initial-access': '#000000',      # Black (as requested)
            'execution': '#4169E1',           # Royal Blue (distinct from others)
            'persistence': '#228B22',         # Forest Green (strong)
            'privilege-escalation': '#8A2BE2', # Blue Violet (distinct purple)
            'defense-evasion': '#FF4500',     # Orange Red (vibrant)
            'credential-access': '#FFD700',   # Gold/Strong Yellow (high contrast!)
            'discovery': '#8B4513',           # Saddle Brown (earthy)
            'lateral-movement': '#FF1493',    # Deep Pink (vibrant)
            'collection': '#2F4F4F',          # Dark Slate Gray (distinct from others)
            'command-and-control': '#00CED1', # Dark Turquoise (cyan family)
            'exfiltration': '#FF8C00',        # Dark Orange (different from orange red)
            'impact': '#32CD32',              # Lime Green (bright, distinct)
            'Defense-evasion': '#B22222',     # Fire Brick (for capitalized version - different red)
            'Initial-access': '#000000',      # Black (same as initial-access)  
            'Unknown': '#696969'              # Dim Gray (neutral)
        }
        
        # Plot each computer's events
        for i, (computer, ax) in enumerate(zip(computers, axes)):
            computer_events = malicious_df[malicious_df['Computer'] == computer].copy()
            
            if len(computer_events) == 0:
                continue
            
            # Group by originator row (we need to simulate this from the labeled data)
            # For v2, we'll use a simpler approach - group by Tactic for visual distinction
            unique_tactics = computer_events['Tactic'].unique()
            
            for j, tactic in enumerate(sorted(unique_tactics)):
                tactic_events = computer_events[computer_events['Tactic'] == tactic]
                
                # Use consistent tactic color across all computers
                color = tactic_colors.get(tactic, '#000000')  # Default to black if tactic not found
                
                ax.scatter(tactic_events['datetime'], tactic_events['EventID'],
                         color=color, 
                         s=60, 
                         alpha=0.8, 
                         label=f'{tactic} ({len(tactic_events)})',
                         edgecolors='black', 
                         linewidths=0.5)
            
            # Customize subplot
            computer_short = computer.replace('.boombox.local', '').replace('.local', '')
            ax.set_ylabel('EventID')
            ax.set_title(f"Attack Events Timeline: {computer_short}")
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Set Y-axis to show only EventIDs that exist
            if len(computer_events) > 0:
                unique_event_ids = sorted(computer_events['EventID'].unique())
                ax.set_yticks(unique_event_ids)
                ax.set_yticklabels(unique_event_ids)
                ax.set_ylim(min(unique_event_ids) - 0.5, max(unique_event_ids) + 0.5)
        
        # Format time axis
        if len(computers) > 0:
            axes[-1].set_xlabel('Time')
            axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
            axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.setp(axes[-1].xaxis.get_ticklabels(), rotation=45, ha='right')
        
        plt.suptitle('Multi-EventID Attack Progression - Group Timeline\n\n', 
                    fontsize=16, y=0.98)
        plt.tight_layout()
        
        # Save plot
        output_file = self.results_dir / "timeline_all_malicious_events_v2.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"✅ Group timeline plot saved: {output_file}")
    
    def create_tactics_timeline_plot(self, labeled_df: pd.DataFrame):
        """Create tactics timeline plot (v2 version) - copied from script 6."""
        self.logger.info("📈 Creating tactics timeline plot (v2)")
        
        # Create single plot for all events
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        
        # Convert timestamps to datetime
        labeled_df = labeled_df.copy()
        labeled_df['datetime'] = pd.to_datetime(labeled_df['timestamp'], unit='ms', errors='coerce')
        # Remove any invalid timestamps
        invalid_timestamps = labeled_df['datetime'].isna().sum()
        if invalid_timestamps > 0:
            self.logger.warning(f"⚠️ Removed {invalid_timestamps} events with invalid timestamps")
            labeled_df = labeled_df.dropna(subset=['datetime'])
        
        # First, plot ALL Sysmon events as pale gray background
        self.logger.info("🎨 Plotting all Sysmon events as background...")
        benign_events = labeled_df[labeled_df['Label'] == 'Benign']
        if len(benign_events) > 0:
            ax.scatter(benign_events['datetime'], benign_events['EventID'], 
                      c='#d0d0d0', alpha=0.4, s=20, 
                      label=f'Benign Events ({len(benign_events):,})', zorder=1)
            self.logger.info(f"📊 Plotted {len(benign_events):,} benign events as background")
        
        # Now organize and plot malicious events by tactic
        malicious_events = labeled_df[labeled_df['Label'] == 'Malicious']
        tactics_events = {}
        for tactic in malicious_events['Tactic'].unique():
            tactics_events[tactic] = malicious_events[malicious_events['Tactic'] == tactic]
        
        if not tactics_events:
            self.logger.warning("⚠️ No malicious tactics found for timeline")
            return
        
        # Define tactic colors (exact copy from script 6)
        tactic_colors = {
            'initial-access': '#000000',      # Black (as requested)
            'execution': '#4169E1',           # Royal Blue (distinct from others)
            'persistence': '#228B22',         # Forest Green (strong)
            'privilege-escalation': '#8A2BE2', # Blue Violet (distinct purple)
            'defense-evasion': '#FF4500',     # Orange Red (vibrant)
            'credential-access': '#FFD700',   # Gold/Strong Yellow (high contrast!)
            'discovery': '#8B4513',           # Saddle Brown (earthy)
            'lateral-movement': '#FF1493',    # Deep Pink (vibrant)
            'collection': '#2F4F4F',          # Dark Slate Gray (distinct from others)
            'command-and-control': '#00CED1', # Dark Turquoise (cyan family)
            'exfiltration': '#FF8C00',        # Dark Orange (different from orange red)
            'impact': '#32CD32',              # Lime Green (bright, distinct)
            'Defense-evasion': '#B22222',     # Fire Brick (for capitalized version - different red)
            'Initial-access': '#000000',      # Black (same as initial-access)  
            'Unknown': '#696969'              # Dim Gray (neutral)
        }
        
        # Plot malicious events by tactic
        total_malicious_events = 0
        
        for tactic in sorted(tactics_events.keys()):
            tactic_df = tactics_events[tactic]
            
            if len(tactic_df) == 0:
                continue
                
            # Get color for this tactic
            tactic_color = tactic_colors.get(tactic, '#000000')
            
            # Create scatter plot for this tactic
            ax.scatter(tactic_df['datetime'], tactic_df['EventID'], 
                       c=tactic_color, alpha=0.8, s=60, 
                       label=f'{tactic.title()} ({len(tactic_df)} events)', zorder=2)
            
            total_malicious_events += len(tactic_df)
            self.logger.info(f"📊 Plotted {len(tactic_df)} events for tactic: {tactic}")
        
        self.logger.info(f"📊 Total malicious events plotted: {total_malicious_events}")
        
        # Customize plot
        ax.set_ylabel('EventID')
        ax.set_xlabel('Timeline')
        ax.grid(True, alpha=0.3)
        
        # Set y-axis ticks based on actual EventIDs present in the data
        all_eventids = set(labeled_df['EventID'].unique())
        
        if all_eventids:
            sorted_eventids = sorted(all_eventids)
            ax.set_yticks(sorted_eventids)
            ax.set_yticklabels(sorted_eventids)
            ax.set_ylim(min(sorted_eventids) - 0.5, max(sorted_eventids) + 0.5)
        else:
            # Fallback to common EventIDs if no data
            ax.set_yticks([1, 3, 5, 7, 8, 9, 10, 11, 12, 13, 23])
            ax.set_ylim(0, 25)
        
        # Format time axis
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.xaxis.get_ticklabels(), rotation=45, ha='right')
        
        # Set main title with event counts
        total_events = len(labeled_df)
        ax.set_title(f'Complete Sysmon Timeline with MITRE Tactics Highlighting\n'
                    f'Total Events: {total_events:,} | Malicious: {total_malicious_events:,} | Benign: {len(benign_events):,}', 
                    fontsize=14, fontweight='bold')
        
        # Add legend
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0),
                 title='MITRE Tactics', title_fontsize=12, fontsize=10)
        
        plt.tight_layout()
        
        # Save plot
        output_file = self.results_dir / "timeline_all_malicious_events_with_tactics_v2.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"✅ Tactics timeline plot saved: {output_file}")
    
    # ==================== V2 METHODS USING SHARED UTILITIES ====================
    
    def create_simple_timeline_plot_v2(self, malicious_df: pd.DataFrame) -> bool:
        """Create simple timeline plot using shared plotting utilities (v2)."""
        if not SHARED_UTILS_AVAILABLE or not FeatureFlags.USE_SHARED_PLOTTING:
            self.create_simple_timeline_plot(malicious_df)  # Fallback to original
            return True
        
        self.logger.info("📈 Creating group timeline plot (v2 - shared utilities)")
        
        # Use shared plotting utility
        return plot_simple_timeline_v2(malicious_df, self.results_dir, self.logger)
    
    def create_tactics_timeline_plot_v2(self, labeled_df: pd.DataFrame, traced_df: pd.DataFrame) -> bool:
        """Create tactics timeline plot using TRACED EVENTS directly (not labeled dataset)."""
        if not SHARED_UTILS_AVAILABLE or not FeatureFlags.USE_SHARED_PLOTTING:
            self.create_tactics_timeline_plot(labeled_df)  # Fallback to original
            return True

        self.logger.info("📈 Creating tactics timeline plot (v2 - plotting traced events directly)")

        # Plot directly from traced CSV to preserve multi-originator associations
        # Read sysmon dataset for gray background
        sysmon_df = pd.read_csv(self.sysmon_file, low_memory=False)

        # Use custom plotting logic (same as Script #6)
        return self._plot_tactics_from_traced_events(traced_df, sysmon_df)
    
    def _plot_tactics_from_traced_events(self, traced_df: pd.DataFrame, sysmon_df: pd.DataFrame) -> bool:
        """Plot tactics timeline directly from traced events (same logic as Script #6)."""
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        # Create single plot
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))

        # Convert timestamps to datetime
        self.logger.info("🎨 Preparing traced events for plotting...")
        try:
            traced_df['datetime'] = pd.to_datetime(traced_df['timestamp'], unit='ms', errors='coerce')
            traced_df = traced_df.dropna(subset=['datetime'])
        except Exception as e:
            self.logger.error(f"❌ Error converting timestamps: {e}")
            return False

        # Plot benign events as background
        self.logger.info("🎨 Plotting all Sysmon events as background...")
        try:
            sysmon_df['datetime'] = pd.to_datetime(sysmon_df['timestamp'], unit='ms', errors='coerce')
            sysmon_df = sysmon_df.dropna(subset=['datetime'])
        except Exception as e:
            self.logger.error(f"❌ Error converting timestamps: {e}")
            return False

        ax.scatter(sysmon_df['datetime'], sysmon_df['EventID'],
                  c='#d0d0d0', alpha=0.4, s=20,
                  label=f'Benign Events ({len(sysmon_df):,})', zorder=1)

        # Organize traced events by tactic
        tactics_events = {}
        for tactic in traced_df['Tactic'].unique():
            if pd.notna(tactic):
                tactics_events[tactic] = traced_df[traced_df['Tactic'] == tactic]

        if not tactics_events:
            self.logger.warning("⚠️ No malicious tactics found")
            return False

        # Sort by count (largest first) so smaller groups appear on top
        tactics_sorted = sorted(tactics_events.keys(), key=lambda t: len(tactics_events[t]), reverse=True)

        total_malicious = 0
        for idx, tactic in enumerate(tactics_sorted):
            tactic_df = tactics_events[tactic]
            if len(tactic_df) == 0:
                continue

            tactic_color = self.tactic_colors.get(tactic, '#000000')
            zorder_value = 2 + idx  # Larger groups get lower z-order

            ax.scatter(tactic_df['datetime'], tactic_df['EventID'],
                      c=tactic_color, alpha=0.8, s=60,
                      label=f'{tactic.title()} ({len(tactic_df)} events)',
                      zorder=zorder_value)

            total_malicious += len(tactic_df)
            self.logger.info(f"📊 Plotted {len(tactic_df)} events for tactic: {tactic} (z-order: {zorder_value})")

        # Customize plot
        ax.set_ylabel('EventID')
        ax.set_xlabel('Timeline')
        ax.grid(True, alpha=0.3)

        # Set y-axis ticks
        all_eventids = set(sysmon_df['EventID'].unique())
        if len(traced_df) > 0:
            all_eventids.update(traced_df['EventID'].unique())

        if all_eventids:
            sorted_eventids = sorted(all_eventids)
            ax.set_yticks(sorted_eventids)
            ax.set_yticklabels(sorted_eventids)
            ax.set_ylim(min(sorted_eventids) - 0.5, max(sorted_eventids) + 0.5)

        # Format time axis
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.xaxis.get_ticklabels(), rotation=45, ha='right')

        # Set title
        total_events = len(sysmon_df)
        ax.set_title(f'Complete Sysmon Timeline with MITRE Tactics Highlighting (v2)\n'
                    f'Total Events: {total_events:,} | Malicious: {total_malicious:,} | Benign: ~{total_events:,}',
                    fontsize=14, fontweight='bold')

        # Add legend
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0),
                 title='MITRE Tactics', title_fontsize=12, fontsize=10)

        plt.tight_layout()

        # Save plot
        output_file = self.results_dir / "timeline_all_malicious_events_with_tactics_v2.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"📊 Tactics timeline saved: {output_file}")
        return True

    # ==================== END V2 METHODS ====================

    def run(self):
        """Main execution method."""
        try:
            self.logger.info(f"🚀 Starting labeled dataset creation for {self.apt_type.upper()}-Run-{self.run_id}")
            
            # Load data
            sysmon_df, traced_df = self.load_data()
            
            # Create labeled dataset
            labeled_df = self.create_labeled_dataset(sysmon_df, traced_df)
            
            # Save labeled dataset
            self.save_labeled_dataset(labeled_df)
            
            # Store for plotting
            self.current_labeled_df = labeled_df
            
            # Prepare plotting data
            malicious_df = self.prepare_plotting_data(labeled_df)
            
            if len(malicious_df) == 0:
                self.logger.warning("⚠️ No malicious events found for plotting")
                return
            
            # Create timeline plots (use v2 methods with shared utilities if available)
            if SHARED_UTILS_AVAILABLE and FeatureFlags.USE_SHARED_PLOTTING:
                self.logger.info("🔧 Using shared plotting utilities (v2)")
                self.create_simple_timeline_plot_v2(malicious_df)
                self.create_tactics_timeline_plot_v2(labeled_df, traced_df)  # Pass traced events directly
            else:
                self.logger.info("🔧 Using original plotting implementation")
                self.create_simple_timeline_plot(malicious_df)
                self.create_tactics_timeline_plot(labeled_df)  # Pass full dataset for gray background
            
            self.logger.info("✅ Labeled dataset creation and plotting completed successfully!")
            
        except Exception as e:
            self.logger.error(f"❌ Error during execution: {e}")
            import traceback
            traceback.print_exc()
            raise

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Create labeled Sysmon dataset and generate v2 timeline plots')
    parser.add_argument('--apt-type', type=str, required=True,
                       help='APT type (e.g., apt-1)')
    parser.add_argument('--run-id', type=str, required=True,
                       help='Run ID (e.g., 04)')
    
    args = parser.parse_args()
    
    try:
        creator = LabeledSysmonDatasetCreator(args.apt_type, args.run_id)
        creator.run()
        
        print(f"\n✅ Labeled Sysmon dataset and v2 plots created successfully!")
        print(f"📁 Output directory: {creator.results_dir}")
        print(f"📄 Labeled dataset: {creator.labeled_dataset_file}")
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())