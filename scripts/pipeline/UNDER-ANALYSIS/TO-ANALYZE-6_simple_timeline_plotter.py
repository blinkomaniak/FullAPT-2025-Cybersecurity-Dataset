#!/usr/bin/env python3
"""
Simple Timeline Plotter
Follows sysmon_event_analysis.py exactly, just organizes results by computer in timeline plots
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime

# Import from sysmon_event_analysis.py (one directory up)
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sysmon_event_analysis import (
    load_config,
    load_sysmon_data,
    load_caldera_data,
    get_config_value,
    EventTracer,
    EntryConfig
)

class EventTracerWithCapture(EventTracer):
    """EventTracer subclass that captures processed events"""
    
    def __init__(self, sysmon_df, caldera_entries, debug=False):
        super().__init__(sysmon_df, caldera_entries, debug)
        self.last_combined_events = None
        self.last_spawned_count = 0
    
    def plot_zero_level_events(self, entry_id, zero_level_event, combined_events, show_arrows, method_type):
        """Override to capture combined_events before plotting"""
        # Store the combined events for access after analysis
        self.last_combined_events = combined_events.copy()
        self.last_spawned_count = len(combined_events)  # This is the spawned events count
        
        # Call the original plot method
        return super().plot_zero_level_events(entry_id, zero_level_event, combined_events, show_arrows, method_type)


class SimpleTimelinePlotter:
    """Simple timeline plotter that follows sysmon_event_analysis.py exactly"""
    
    def __init__(self, plot_config=None):
        print("🔧 Initializing Simple Timeline Plotter...")
        
        # Default plot configuration
        self.plot_config = {
            'figure_width': 15,
            'subplot_height': 4,
            'dpi': 300,
            'marker_size': 60,
            'marker_alpha': 0.8,
            'output_format': 'png'
        }
        
        # Override with user configuration if provided
        if plot_config:
            self.plot_config.update(plot_config)
        
        # Load everything exactly like sysmon_event_analysis.py
        self.config = load_config('config.yaml')
        self.sysmon_df = load_sysmon_data()
        self.caldera_entries = load_caldera_data()
        self.entry_config_df = self._load_entry_config()
        
        # Data quality validation
        self._validate_data_quality()
        
        # Initialize custom tracer that captures processed events
        self.tracer = EventTracerWithCapture(self.sysmon_df, self.caldera_entries, debug=False)
        
        # Store all events for plotting (organized by computer)
        self.computer_events = {}  # computer -> list of events
        
        print("✅ Simple Timeline Plotter initialized")
    
    def _load_entry_config(self):
        """Load entry_config.csv"""
        entry_config_file = get_config_value('data_sources.entry_config_file', 'entry_config.csv')
        df = pd.read_csv(entry_config_file, header=None, 
                        names=['entry_id', 'computer_short', 'totem_eventid'])
        return df
    
    def _validate_data_quality(self):
        """Validate data quality and consistency"""
        print(f"\n🔍 Data Quality Validation:")
        
        issues = []
        
        # Check Sysmon data
        if len(self.sysmon_df) == 0:
            issues.append("❌ No Sysmon events loaded")
        else:
            print(f"   ✅ Sysmon events: {len(self.sysmon_df):,}")
        
        # Check Caldera data
        if len(self.caldera_entries) == 0:
            issues.append("❌ No Caldera entries loaded")
        else:
            print(f"   ✅ Caldera entries: {len(self.caldera_entries)}")
        
        # Check entry config
        if len(self.entry_config_df) == 0:
            issues.append("❌ No entry config loaded")
        else:
            print(f"   ✅ Entry config: {len(self.entry_config_df)} entries")
        
        # Check for missing entries
        caldera_ids = set(self.caldera_entries.keys())
        config_ids = set(str(x) for x in self.entry_config_df['entry_id'].values)
        missing_in_config = caldera_ids - config_ids
        missing_in_caldera = config_ids - caldera_ids
        
        if missing_in_config:
            issues.append(f"⚠️ {len(missing_in_config)} Caldera entries not in config: {sorted(list(missing_in_config))[:5]}...")
        
        if missing_in_caldera:
            issues.append(f"⚠️ {len(missing_in_caldera)} config entries not in Caldera: {sorted(list(missing_in_caldera))[:5]}...")
        
        if not issues:
            print(f"   ✅ All data quality checks passed")
        else:
            print(f"   Data quality issues found:")
            for issue in issues:
                print(f"      {issue}")
        
        return len(issues) == 0
    
    def process_all_entries(self):
        """Process each entry exactly like sysmon_event_analysis.py does"""
        print(f"\n🚀 Processing entries from entry_config.csv...")
        
        # Only process entries in the config file
        valid_entry_ids = set(str(eid) for eid in self.entry_config_df['entry_id'].values)
        entries_to_process = [eid for eid in self.caldera_entries.keys() if eid in valid_entry_ids]
        
        print(f"📋 Processing {len(entries_to_process)} entries: {sorted([int(x) for x in entries_to_process])}")
        
        for entry_id in entries_to_process:
            try:
                print(f"\n{'='*40}")
                print(f"🔍 Processing Entry {entry_id}")
                print(f"{'='*40}")
                
                # Process exactly like sysmon_event_analysis.py
                self._process_single_entry(entry_id)
                
            except Exception as e:
                print(f"❌ Entry {entry_id}: Error - {e}")
                print(f"   Continuing with next entry...")
                continue
        
        print(f"\n📊 Collection Summary:")
        for computer, events in self.computer_events.items():
            print(f"   {computer}: {len(events)} events")
    
    def _process_single_entry(self, entry_id):
        """Process single entry exactly like sysmon_event_analysis.py"""
        
        # Get entry data
        entry_data = self.caldera_entries[entry_id]
        
        # Get configuration
        entry_config_row = self.entry_config_df[
            self.entry_config_df['entry_id'] == int(entry_id)
        ]
        
        if len(entry_config_row) == 0:
            print(f"⚠️ Entry {entry_id} not in config - skipping")
            return
        
        computer_short = entry_config_row.iloc[0]['computer_short']
        totem_eventid = entry_config_row.iloc[0]['totem_eventid']
        
        # Determine analysis type based on EventID from entry_config.csv
        if totem_eventid == 1:
            analysis_type = "Type1 (CommandLine with recursive tracing)"
        elif totem_eventid == 11:
            analysis_type = "Type2a (FileCreate)"
        elif totem_eventid == 23:
            analysis_type = "Type2b (FileDelete)"
        else:
            analysis_type = f"Unsupported EventID {totem_eventid}"
        
        print(f"📋 Entry {entry_id}: {computer_short}, EventID {totem_eventid}")
        print(f"   Analysis Type: {analysis_type}")
        print(f"   Command: {entry_data['new_command'][:60]}...")
        print(f"   Tactic: {entry_data.get('tactic', 'unknown')}")
        
        # Create EntryConfig
        entry_config = EntryConfig(
            entry_id=int(entry_id),
            computer_short=computer_short,
            totem_eventid=totem_eventid
        )
        
        # Step 1: Detect zero-level event (exactly like sysmon_event_analysis.py)
        result = self.tracer.detect_zero_level_event(entry_config)
        
        if not result.success:
            print(f"❌ Detection failed: {result.error_message}")
            return
        
        print(f"✅ Detection successful: {result.matches_found} matches")
        
        # Step 2: Apply analysis based on EventID from entry_config.csv
        spawned_events_count = 0
        
        if totem_eventid == 1:  # Type1 - CommandLine with recursive tracing
            print(f"🔄 Applying Type1 analysis (recursive tracing enabled)...")
            plot_filename = self.tracer.apply_type1_analysis(
                result, include_mask4=False, show_arrows=False
            )
            # Get spawned events count from our custom tracer
            spawned_events_count = self.tracer.last_spawned_count
            
        elif totem_eventid == 11:  # Type2a - FileCreate (no recursive tracing)
            print(f"📁 Applying Type2a analysis (FileCreate, no recursive tracing)...")
            plot_filename = self.tracer.apply_type2_analysis(
                result, show_arrows=False
            )
            # Type2 doesn't use recursive tracing, but check what was captured
            spawned_events_count = self.tracer.last_spawned_count
            
        elif totem_eventid == 23:  # Type2b - FileDelete (no recursive tracing)
            print(f"🗑️ Applying Type2b analysis (FileDelete, no recursive tracing)...")
            plot_filename = self.tracer.apply_type2_analysis(
                result, show_arrows=False
            )
            # Type2 doesn't use recursive tracing, but check what was captured
            spawned_events_count = self.tracer.last_spawned_count
            
        else:
            print(f"⚠️ Unsupported EventID: {totem_eventid}")
            return
        
        # Log the spawned events count clearly
        print(f"📊 SPAWNED EVENTS COUNT: {spawned_events_count}")
        if totem_eventid == 1:
            print(f"   (Type1 entries can have spawned events from recursive tracing)")
        else:
            print(f"   (Type2 entries do not use recursive tracing)")
        
        # Step 3: Collect the events that were just processed
        self._collect_processed_events(result, entry_id, entry_data, spawned_events_count)
    
    def _normalize_tactic(self, tactic):
        """Normalize tactic names to handle cross-dataset inconsistencies"""
        if tactic == 'defensive-evasion':
            return 'defense-evasion'
        return tactic
    
    def _collect_processed_events(self, result, entry_id, entry_data, spawned_events_count):
        """Collect the events that were processed by the tracer"""
        
        # Get events from our custom tracer that captured them during plotting
        if self.tracer.last_combined_events is not None:
            events_df = self.tracer.last_combined_events
        else:
            # Fallback to just the zero-level event
            events_df = pd.DataFrame([result.zero_level_event])
        
        # Add the zero-level event to get the complete set
        zero_level_df = pd.DataFrame([result.zero_level_event])
        complete_events = pd.concat([events_df, zero_level_df], ignore_index=True).drop_duplicates()
        
        total_events = len(complete_events)
        zero_level_events = 1
        actual_spawned = total_events - zero_level_events
        
        print(f"📝 Collected {total_events} total events from tracer")
        print(f"   Zero-level events: {zero_level_events}")
        print(f"   Spawned events: {actual_spawned}")
        
        # Normalize tactic name for consistency across datasets
        raw_tactic = entry_data.get('tactic', 'unknown')
        normalized_tactic = self._normalize_tactic(raw_tactic)
        
        if raw_tactic != normalized_tactic:
            print(f"🔄 Normalized tactic: '{raw_tactic}' → '{normalized_tactic}'")
        
        # Organize events by computer for timeline plotting
        for _, event in complete_events.iterrows():
            computer = event['Computer']
            
            # Create event record for timeline
            event_record = {
                'entry_id': entry_id,
                'tactic': normalized_tactic,
                'event_data': event.copy(),
                'spawned_count': actual_spawned
            }
            
            # Add to computer's event list
            if computer not in self.computer_events:
                self.computer_events[computer] = []
            
            self.computer_events[computer].append(event_record)
    
    def plot_timeline_by_computer(self):
        """Create timeline plots organized by computer"""
        print(f"\n📈 Creating timeline plots by computer...")
        
        if not self.computer_events:
            print("⚠️ No events to plot")
            return
        
        # Order computers by first appearance in entry_config.csv: theblock → waterfalls → endofroad
        all_computers = set(self.computer_events.keys())
        preferred_order = ['theblock.boombox.local', 'waterfalls.boombox.local', 'endofroad.boombox.local']
        
        # Use preferred order for computers that exist, then add any others
        computers = [comp for comp in preferred_order if comp in all_computers]
        for comp in sorted(all_computers):
            if comp not in computers:
                computers.append(comp)
        
        print(f"📊 Computers with events (ordered by first appearance): {[comp.replace('.boombox.local', '') for comp in computers]}")
        
        # Create figure with subplots for each computer
        fig, axes = plt.subplots(nrows=len(computers), ncols=1, 
                                figsize=(self.plot_config['figure_width'], 
                                        self.plot_config['subplot_height'] * len(computers)), 
                                sharex=True)
        
        if len(computers) == 1:
            axes = [axes]
        
        # Get all unique tactics and event IDs for plotting
        all_event_ids = set()
        all_tactics = set()
        for events_list in self.computer_events.values():
            for event_record in events_list:
                all_event_ids.add(event_record['event_data']['EventID'])
                all_tactics.add(event_record['tactic'])
        
        event_ids = sorted(all_event_ids)
        unique_tactics = sorted(all_tactics)
        
        print(f"🏷️ Tactics found: {unique_tactics}")
        
        # Create comprehensive color-and-shape mapping for all possible tactics
        all_possible_tactics = [
            'reconnaissance', 'initial-access', 'persistence', 'privilege-escalation', 
            'defense-evasion', 'command-and-control', 'credential-access', 'discovery',
            'lateral-movement', 'data-collection', 'exfiltration', 'impact', 'execution'
        ]
        
        # Define distinct colors (using qualitative color palettes)
        colors = [
            '#e41a1c',  # Red
            '#377eb8',  # Blue  
            '#4daf4a',  # Green
            '#984ea3',  # Purple
            '#ff7f00',  # Orange
            '#ffff33',  # Yellow
            '#a65628',  # Brown
            '#f781bf',  # Pink
            '#999999',  # Gray
            '#1f78b4',  # Dark Blue
            '#33a02c',  # Dark Green
            '#fb9a99',  # Light Red
            '#000000'   # Black
        ]
        
        # Define distinct marker shapes
        shapes = ['o', 's', '^', 'v', '<', '>', 'D', 'p', '*', 'h', 'H', '+', 'x']
        
        # Create mapping for all possible tactics
        tactic_style_map = {}
        for i, tactic in enumerate(all_possible_tactics):
            tactic_style_map[tactic] = {
                'color': colors[i % len(colors)],
                'shape': shapes[i % len(shapes)]
            }
        
        print(f"🎨 Tactic color-and-shape mapping:")
        for tactic in unique_tactics:
            if tactic in tactic_style_map:
                style = tactic_style_map[tactic]
                print(f"   {tactic}: {style['color']} {style['shape']}")
            else:
                print(f"   {tactic}: UNKNOWN TACTIC - using default")
        
        # Plot events for each computer
        legend_handles = []  # Collect legend handles from all tactics
        legend_labels = []   # Collect legend labels from all tactics
        
        for ax, computer in zip(axes, computers):
            computer_short = computer.replace('.boombox.local', '')
            events_list = self.computer_events[computer]
            
            print(f"📊 Plotting {len(events_list)} events for {computer_short}")
            
            if not events_list:
                ax.text(0.5, 0.5, f'No events on {computer_short}', 
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes, fontsize=12)
                ax.set_title(f"Timeline: {computer_short}")
                continue
            
            # Extract event data for plotting
            plot_data = []
            for event_record in events_list:
                event = event_record['event_data']
                plot_data.append({
                    'UtcTime': pd.to_datetime(event['UtcTime']),
                    'EventID': event['EventID'],
                    'Tactic': event_record['tactic'],
                    'EntryID': event_record['entry_id']
                })
            
            plot_df = pd.DataFrame(plot_data)
            
            # Plot events colored by tactic (y-axis still shows EventID)
            spacing_factor = 2.0
            
            # Group events by tactic for consistent coloring and shapes
            for tactic in unique_tactics:
                tactic_events = plot_df[plot_df['Tactic'] == tactic]
                if len(tactic_events) > 0:
                    # Get style for this tactic (fallback to defaults if unknown)
                    if tactic in tactic_style_map:
                        style = tactic_style_map[tactic]
                        color = style['color']
                        marker = style['shape']
                    else:
                        color = '#000000'  # Black fallback
                        marker = 'o'       # Circle fallback
                    
                    scatter = ax.scatter(tactic_events['UtcTime'], 
                                       tactic_events['EventID'] * spacing_factor,
                                       color=color, 
                                       marker=marker,
                                       s=self.plot_config['marker_size'], 
                                       alpha=self.plot_config['marker_alpha'], 
                                       label=f'{tactic}',
                                       edgecolors='black', linewidths=0.5)
                    
                    # Collect legend entries from all tactics across all computers
                    if tactic not in legend_labels:
                        legend_handles.append(scatter)
                        legend_labels.append(tactic)
            
            # Customize subplot
            ax.set_ylabel('EventID')
            ax.set_title(f"Attack Timeline: {computer_short}")
            ax.grid(True, alpha=0.3)
            
            # Set y-axis ticks
            ytick_values = [1, 3, 5, 7, 10, 11, 12, 13, 17, 18, 23]
            available_ticks = [y for y in ytick_values if y in event_ids]
            ax.set_yticks([y * spacing_factor for y in available_ticks])
            ax.set_yticklabels(available_ticks)
        
        # Add legend with ALL tactics from ALL computers to the first subplot
        if len(axes) > 0 and legend_handles:
            axes[0].legend(legend_handles, legend_labels, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Format time axis
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.setp(axes[-1].xaxis.get_ticklabels(), rotation=45, ha='right')
        
        plt.suptitle('Attack Timeline by Computer - Tactic Progression (Y-axis: EventID, Color+Shape: Tactic)', 
                    fontsize=16, y=0.98)
        plt.tight_layout()
        
        # Save plot
        output_file = f"timeline_by_computer.{self.plot_config['output_format']}"
        plt.savefig(output_file, dpi=self.plot_config['dpi'], bbox_inches='tight')
        print(f"💾 Timeline plot saved: {output_file}")
        
        plt.show()
    
    def print_spawned_events_summary(self):
        """Print summary of spawned events per entry"""
        print(f"\n📊 SPAWNED EVENTS SUMMARY:")
        print(f"{'='*60}")
        
        # Collect spawned events data by entry
        entry_summary = {}
        for computer, events_list in self.computer_events.items():
            for event_record in events_list:
                entry_id = event_record['entry_id']
                spawned_count = event_record['spawned_count']
                
                if entry_id not in entry_summary:
                    entry_summary[entry_id] = {
                        'spawned_count': spawned_count,
                        'computer': computer.replace('.boombox.local', ''),
                        'tactic': event_record['tactic']
                    }
        
        # Print summary sorted by entry ID
        for entry_id in sorted(entry_summary.keys(), key=int):
            data = entry_summary[entry_id]
            print(f"Entry {int(entry_id):2d}: {data['spawned_count']:3d} spawned events | {data['computer']:10s} | {data['tactic']}")
        
        # Print statistics
        spawned_counts = [data['spawned_count'] for data in entry_summary.values()]
        total_spawned = sum(spawned_counts)
        max_spawned = max(spawned_counts) if spawned_counts else 0
        avg_spawned = total_spawned / len(spawned_counts) if spawned_counts else 0
        
        print(f"\n📈 Statistics:")
        print(f"   Total entries: {len(entry_summary)}")
        print(f"   Total spawned events: {total_spawned}")
        print(f"   Average per entry: {avg_spawned:.1f}")
        print(f"   Maximum per entry: {max_spawned}")
        print(f"{'='*60}")
    
    def save_labeled_dataset(self):
        """Create and save labeled dataset"""
        print(f"\n💾 Creating labeled dataset...")
        
        # Start with normal labels
        self.sysmon_df['TacticLabel'] = 'Normal'
        
        # Label events based on collected data
        labeled_count = 0
        for computer, events_list in self.computer_events.items():
            for event_record in events_list:
                event = event_record['event_data']
                tactic = event_record['tactic']
                
                # Find matching events in main dataframe
                mask = (
                    (self.sysmon_df['UtcTime'] == event['UtcTime']) &
                    (self.sysmon_df['EventID'] == event['EventID']) &
                    (self.sysmon_df['Computer'] == event['Computer'])
                )
                
                # Additional matching for precision
                if 'ProcessGuid' in event and pd.notna(event['ProcessGuid']):
                    mask &= (self.sysmon_df['ProcessGuid'] == event['ProcessGuid'])
                
                # Apply label
                matches = self.sysmon_df[mask]
                if len(matches) > 0:
                    self.sysmon_df.loc[mask, 'TacticLabel'] = tactic
                    labeled_count += len(matches)
        
        # Save labeled dataset
        sysmon_file = get_config_value('data_sources.sysmon_file')
        labeled_file = sysmon_file.replace('.csv', '-labeled.csv')
        self.sysmon_df.to_csv(labeled_file, index=False)
        
        print(f"✅ Labeled dataset saved: {labeled_file}")
        print(f"📊 Labeled {labeled_count:,} events")
        
        # Print label distribution
        label_counts = self.sysmon_df['TacticLabel'].value_counts()
        print(f"\n📊 Label Distribution:")
        for label, count in label_counts.items():
            percentage = (count / len(self.sysmon_df)) * 100
            print(f"   {label}: {count:,} events ({percentage:.2f}%)")
        
        return labeled_file
    
    def generate_analysis_report(self):
        """Generate comprehensive analysis report"""
        print(f"\n📋 COMPREHENSIVE TIMELINE ANALYSIS REPORT")
        print(f"{'='*70}")
        
        # Overall statistics
        total_events = sum(len(events) for events in self.computer_events.values())
        total_computers = len(self.computer_events)
        
        print(f"\n📊 OVERALL STATISTICS:")
        print(f"   Total Events Processed: {total_events:,}")
        print(f"   Total Computers: {total_computers}")
        print(f"   Total Entries Analyzed: {len(set(event['entry_id'] for events in self.computer_events.values() for event in events))}")
        
        # Tactic distribution
        tactic_counts = {}
        for events_list in self.computer_events.values():
            for event_record in events_list:
                tactic = event_record['tactic']
                tactic_counts[tactic] = tactic_counts.get(tactic, 0) + 1
        
        print(f"\n🏷️ TACTIC DISTRIBUTION:")
        for tactic, count in sorted(tactic_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_events) * 100
            print(f"   {tactic}: {count:,} events ({percentage:.1f}%)")
        
        # Computer-wise breakdown
        print(f"\n💻 COMPUTER-WISE BREAKDOWN:")
        for computer, events_list in self.computer_events.items():
            computer_short = computer.replace('.boombox.local', '')
            print(f"   {computer_short}: {len(events_list):,} events")
            
            # Tactic breakdown per computer
            computer_tactics = {}
            for event_record in events_list:
                tactic = event_record['tactic']
                computer_tactics[tactic] = computer_tactics.get(tactic, 0) + 1
            
            for tactic, count in sorted(computer_tactics.items(), key=lambda x: x[1], reverse=True):
                print(f"      {tactic}: {count}")
        
        # Timeline summary
        if total_events > 0:
            all_times = []
            for events_list in self.computer_events.values():
                for event_record in events_list:
                    all_times.append(pd.to_datetime(event_record['event_data']['UtcTime']))
            
            if all_times:
                start_time = min(all_times)
                end_time = max(all_times)
                duration = end_time - start_time
                
                print(f"\n⏰ TIMELINE SUMMARY:")
                print(f"   Attack Start: {start_time.strftime('%H:%M:%S')}")
                print(f"   Attack End: {end_time.strftime('%H:%M:%S')}")
                print(f"   Total Duration: {duration}")
        
        print(f"{'='*70}")
    
    def run_complete_analysis(self):
        """Run the complete simple analysis"""
        print("🚀 Starting Simple Timeline Analysis")
        print("=" * 60)
        
        # Process all entries (follows sysmon_event_analysis.py exactly)
        self.process_all_entries()
        
        # Print spawned events summary
        self.print_spawned_events_summary()
        
        # Create timeline plots by computer
        self.plot_timeline_by_computer()
        
        # Save labeled dataset
        labeled_file = self.save_labeled_dataset()
        
        # Generate comprehensive analysis report
        self.generate_analysis_report()
        
        print("\n" + "=" * 60)
        print("🎉 Simple Timeline Analysis Complete!")
        print(f"📊 Generated timeline plots by computer")
        print(f"💾 Labeled dataset: {labeled_file}")
        print("=" * 60)


def main():
    """Main execution"""
    print("🎯 Simple Timeline Plotter")
    print("Follows sysmon_event_analysis.py exactly, organizes results by computer")
    print("=" * 70)
    
    try:
        plotter = SimpleTimelinePlotter()
        plotter.run_complete_analysis()
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Simple timeline plotting completed!")
    else:
        print("\n❌ Simple timeline plotting failed!")