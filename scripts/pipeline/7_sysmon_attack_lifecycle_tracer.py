#!/usr/bin/env python3
"""
Multi-EventID Attack Lifecycle Tracer - Process & File Event Analysis

Specialized script for tracing attack lifecycles across multiple Sysmon EventID types:
- EventID 1 (Process Creation): Full lifecycle tracing with ProcessGuid correlation
- EventID 11 (File Create): Individual event plotting with TargetFilename analysis  
- EventID 23 (File Delete): Individual event plotting with TargetFilename analysis

Features:
- Multi-EventID support (1, 11, 23) with appropriate correlation methods
- ProcessGuid-based correlation for child processes and spawned events (EventID 1)
- Seed_Event/Tactic selection support from all_target_events_run-X.csv
- Individual timeline plots with tactic metadata for all EventID types
- Unified group timeline showing all traced events across EventID types
- Enhanced titles with Command (EventID 1) or TargetFilename (EventID 11/23) information

Usage:
    python3 EVENTID1_attack_lifecycle_tracer.py --apt-type apt-1 --run-id 04
    python3 EVENTID1_attack_lifecycle_tracer.py --sysmon-csv /path/to/sysmon.csv --originators-csv /path/to/all_target_events_run-04.csv
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field

try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.lines import Line2D
except ImportError as e:
    print(f"❌ Error: Required library not installed: {e}")
    print("   Install with: pip install pandas numpy matplotlib")
    sys.exit(1)

# Import shared utilities (with fallback for compatibility)
try:
    from utils.apt_config import TacticColors, PlottingConfig, FeatureFlags
    from utils.apt_plotting_utils import plot_simple_timeline_v1, plot_tactics_timeline_v1
    from utils.apt_path_utils import PathManager
    SHARED_UTILS_AVAILABLE = True
except ImportError:
    print("⚠️ Warning: Shared utilities not available - using original implementation")
    SHARED_UTILS_AVAILABLE = False


class MultiEventIDAttackLifecycleTracer:
    """
    Multi-EventID tracer for attack lifecycle analysis across process and file events.
    
    Supports EventID 1 (Process Creation) with full lifecycle tracing, and EventID 11/23 
    (File Create/Delete) with individual event analysis.
    """
    
    # APT run ranges per dataset type
    APT_RUN_RANGES = {
        'apt-1': list(range(1, 21)) + [51],  # 01-20, 51
        'apt-2': list(range(21, 31)),        # 21-30
        'apt-3': list(range(31, 39)),        # 31-38
        'apt-4': list(range(39, 45)),        # 39-44
        'apt-5': list(range(45, 48)),        # 45-47
        'apt-6': list(range(48, 51))         # 48-50
    }
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.logger = self._setup_logger()
        
        # Data containers
        self.sysmon_df = None
        self.selected_originators = []
        self.traced_events = {}  # originator_row -> traced data
        self.all_traced_events = []  # Combined list for group timeline
        
        # Labeling statistics
        self.labeling_stats = {}
        
        # Computer organization
        self.computer_events = {}  # computer -> list of events
        
        # Statistics and results
        self.stats = {
            'total_originators_selected': 0,
            'successfully_traced': 0,
            'total_traced_events': 0,
            'events_by_computer': {},
            'events_by_eventid': {},
            'processing_start_time': datetime.now().isoformat(),
            'processing_duration_seconds': 0,
            'seed_event_originators_selected': 0,
            'tactic_originators_selected': 0,
            'eventid1_selected': 0,
            'eventid11_selected': 0,
            'eventid23_selected': 0,
            'eventid1_plots_created': 0,
            'eventid11_plots_created': 0,
            'eventid23_plots_created': 0
        }
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logging configuration."""
        level = logging.DEBUG if self.debug else logging.INFO
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        return logging.getLogger(__name__)
    
    def load_selected_originators(self, originators_file: Path) -> bool:
        """
        Load selected EventID 1, 11, and 23 originators from CSV file.
        
        Supports Seed_Event/Tactic/Technique column structure.
        Filters to include EventID 1 (process), 11 (file create), and 23 (file delete) events.
        """
        if not originators_file.exists():
            self.logger.error(f"❌ Originators file not found: {originators_file}")
            return False
        
        self.logger.info(f"📂 Loading attack originators from: {originators_file}")
        
        try:
            df = pd.read_csv(originators_file)
            
            # Filter to EventID 1, 11, and 23 only
            target_events_df = df[df['EventID'].isin([1, 11, 23])].copy()
            if len(target_events_df) == 0:
                self.logger.error("❌ No EventID 1, 11, or 23 events found in originators file")
                return False
            
            # Count by EventID
            eventid_counts = target_events_df['EventID'].value_counts().sort_index()
            count_details = []
            for eid, count in eventid_counts.items():
                event_type = "Process Creation" if eid == 1 else "File Create" if eid == 11 else "File Delete"
                count_details.append(f"EventID {eid} ({event_type}): {count}")
            count_str = ", ".join(count_details)
            self.logger.info(f"🔍 Filtered events ({count_str}) from {len(df)} total events")
            
            # Check for column structure
            has_seed_event = 'Seed_Event' in target_events_df.columns
            has_tactic = 'Tactic' in target_events_df.columns
            has_technique = 'Technique' in target_events_df.columns
            
            if not has_seed_event:
                self.logger.error("❌ CSV file must have Seed_Event column")
                return False
            
            required_columns = ['Seed_Event', 'RawDatasetRowNumber']
            missing_columns = [col for col in required_columns if col not in target_events_df.columns]
            if missing_columns:
                self.logger.error(f"❌ Missing required columns: {missing_columns}")
                return False
            
            # Filter for selected originators (EventID 1, 11, 23) - avoid duplicates
            # Priority: Tactic > Seed_Event (to preserve Tactic/Technique information)

            # Start with all selected entries (Seed_Event marked as 'X')
            seed_event_selected = target_events_df[
                (target_events_df['Seed_Event'].astype(str).str.strip().str.upper() == 'X')
            ].copy()
            
            if len(seed_event_selected) == 0:
                self.logger.warning("⚠️ No attack originators selected in Seed_Event column")
                return False
            
            # Add processing type (simplified to Seed_Event only)
            seed_event_selected['processing_type'] = 'Seed_Event'
            
            # Override processing type to 'Tactic' if Tactic information is available
            if has_tactic:
                tactic_mask = (
                    (seed_event_selected['Tactic'].astype(str).str.strip() != '') &
                    (seed_event_selected['Tactic'].astype(str).str.strip().str.upper() != 'NAN')
                )
                seed_event_selected.loc[tactic_mask, 'processing_type'] = 'Tactic'
            
            # Use only unique RawDatasetRowNumber entries (no duplicates)
            selected_df = seed_event_selected.drop_duplicates(subset=['RawDatasetRowNumber'], keep='first')
            
            if len(selected_df) == 0:
                self.logger.warning("⚠️ No attack originators selected in Seed_Event or Tactic columns")
                return False
            
            # Statistics by EventID type (avoid duplication from Seed_Event+Tactic)
            # Get unique rows by dropping duplicates based on RawDatasetRowNumber
            unique_selected_df = selected_df.drop_duplicates(subset=['RawDatasetRowNumber'], keep='first')
            eventid_stats = unique_selected_df['EventID'].value_counts().sort_index()
            self.stats['eventid1_selected'] = eventid_stats.get(1, 0)
            self.stats['eventid11_selected'] = eventid_stats.get(11, 0)  
            self.stats['eventid23_selected'] = eventid_stats.get(23, 0)
            
            # General statistics - count by processing type
            processing_type_counts = selected_df['processing_type'].value_counts()
            self.stats['seed_event_originators_selected'] = processing_type_counts.get('Seed_Event', 0)
            self.stats['tactic_originators_selected'] = processing_type_counts.get('Tactic', 0)
            
            self.logger.info(f"📊 Attack Event Selection Summary:")
            self.logger.info(f"   🌱 Seed Event originators: {self.stats['seed_event_originators_selected']}")
            if self.stats['tactic_originators_selected'] > 0:
                self.logger.info(f"   🏷️ Tactic-labeled originators: {self.stats['tactic_originators_selected']}")
            
            # EventID breakdown
            for eid, count in eventid_stats.items():
                event_type = "Process Creation" if eid == 1 else "File Create" if eid == 11 else "File Delete"
                self.logger.info(f"   📋 EventID {eid} ({event_type}): {count} events")
            
            self.logger.info(f"   ✅ Total unique attack events selected: {len(selected_df)}")
            
            self.selected_originators = selected_df.to_dict('records')
            self.stats['total_originators_selected'] = len(selected_df)
            
            # Display selected originators
            self.logger.info(f"📋 Selected Attack Event Originators:")
            for i, orig in enumerate(self.selected_originators, 1):
                processing_type = orig.get('processing_type', 'Seed_Event')
                row_num = orig['RawDatasetRowNumber']
                event_id = orig['EventID']
                
                # Get display text based on EventID
                if event_id == 1:
                    display_text = orig.get('CommandLine', 'N/A')
                else:  # EventID 11 or 23
                    display_text = orig.get('TargetFilename', 'N/A')
                
                type_emoji = '🌱' if processing_type == 'Seed_Event' else '🏷️'
                self.logger.info(f"   {i}. {type_emoji} Row {row_num} (EventID {event_id}, {processing_type}): {str(display_text)[:80]}...")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error loading originators: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            return False
    
    def load_sysmon_data(self, sysmon_file: Path) -> bool:
        """Load Sysmon dataset."""
        if not sysmon_file.exists():
            self.logger.error(f"❌ Sysmon file not found: {sysmon_file}")
            return False
        
        self.logger.info(f"📊 Loading Sysmon dataset: {sysmon_file}")
        
        try:
            self.sysmon_df = pd.read_csv(sysmon_file, low_memory=False)
            self.logger.info(f"✅ Loaded {len(self.sysmon_df):,} Sysmon events")
            
            # Validate required columns
            required_columns = ['EventID', 'ProcessGuid', 'ProcessId', 'Computer', 'timestamp']
            missing_columns = [col for col in required_columns if col not in self.sysmon_df.columns]
            if missing_columns:
                self.logger.error(f"❌ Missing required columns in Sysmon data: {missing_columns}")
                return False
                
            # Log EventID distribution
            eventid_counts = self.sysmon_df['EventID'].value_counts().sort_index()
            self.logger.info(f"📊 EventID Distribution:")
            for eventid, count in eventid_counts.items():
                self.logger.info(f"   EventID {eventid}: {count:,} events")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error loading Sysmon data: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            return False
    
    def get_originator_event_from_sysmon(self, originator_info: Dict) -> Optional[pd.Series]:
        """Find the originator event in Sysmon dataset using multiple search strategies."""
        try:
            original_row_number = originator_info['RawDatasetRowNumber']
            event_id = originator_info['EventID']
            
            # Primary search: EventID + row proximity 
            eventid_mask = self.sysmon_df['EventID'] == event_id
            eventid_events = self.sysmon_df[eventid_mask]
            
            if len(eventid_events) == 0:
                self.logger.error(f"❌ No EventID {event_id} events found in Sysmon data")
                return None
            
            # Look for events near the original row number
            target_rows = range(max(1, original_row_number - 10), original_row_number + 10)
            nearby_events = eventid_events[eventid_events.index.isin(target_rows)]
            
            if len(nearby_events) > 0:
                # Find closest match based on EventID type
                if event_id == 1:
                    # EventID 1: Match by CommandLine similarity
                    target_cmd = str(originator_info.get('CommandLine', '')).strip().lower()
                    if target_cmd and target_cmd != 'nan':
                        for idx, event in nearby_events.iterrows():
                            event_cmd = str(event.get('CommandLine', '')).strip().lower()
                            if target_cmd in event_cmd or event_cmd in target_cmd:
                                self.logger.debug(f"✅ Found EventID 1 originator by CommandLine match")
                                return event
                elif event_id in [11, 23]:
                    # EventID 11/23: Match by TargetFilename similarity
                    target_filename = str(originator_info.get('TargetFilename', '')).strip().lower()
                    if target_filename and target_filename != 'nan':
                        for idx, event in nearby_events.iterrows():
                            event_filename = str(event.get('TargetFilename', '')).strip().lower()
                            if target_filename in event_filename or event_filename in target_filename:
                                self.logger.debug(f"✅ Found EventID {event_id} originator by TargetFilename match")
                                return event
                
                # Fallback: return first nearby event of the correct EventID
                event = nearby_events.iloc[0]
                self.logger.debug(f"✅ Found EventID {event_id} originator by proximity")
                return event
            
            # Final fallback: return any event of the correct EventID
            self.logger.warning(f"⚠️ Using fallback EventID {event_id} event for row {original_row_number}")
            return eventid_events.iloc[0]
            
        except Exception as e:
            self.logger.error(f"❌ Error searching for EventID {event_id} originator: {e}")
            return None
    
    # Sysmon EventID Column Reference (for mask design)
    # Note: 'UtcTime' -> 'timestamp', 'SourceProcessGuid' -> 'SourceProcessGUID', 'TargetProcessGuid' -> 'TargetProcessGUID'
    #
    # EventID 1 (Process Create): ProcessGuid, ProcessId, ParentProcessGuid, ParentProcessId
    # EventID 8 (CreateRemoteThread): SourceProcessGUID, SourceProcessId, TargetProcessGUID, TargetProcessId  
    # EventID 10 (ProcessAccess): SourceProcessGUID, SourceProcessId, TargetProcessGUID, TargetProcessId
    # Other EventIDs: ProcessGuid, ProcessId (standard process correlation fields)
    #
    # Mask Logic:
    # - spawned_events: Find events where THIS process is the source (ProcessGuid/SourceProcessGUID match)
    # - child_processcreate: Find EventID 1 where THIS process is the parent (ParentProcessGuid match)
    # - parent_eventid_8_or_10: Find EventID 8/10 where THIS process is the target (TargetProcessGUID match)

    def mask_to_find_spawned_events(self, process_guid: str, process_pid: int, target_computer: Optional[str] = None) -> pd.Series:
        """Create mask to find spawned events using ProcessGuid and PID (excludes EventID 1)."""
        mask = (
            ((self.sysmon_df['ProcessGuid'] == process_guid) & (self.sysmon_df['ProcessId'] == process_pid) | 
            (self.sysmon_df['SourceProcessGUID'] == process_guid) & (self.sysmon_df['SourceProcessId'] == process_pid)) &
            (~self.sysmon_df['EventID'].isin([1]))
        )
        
        if target_computer:
            mask = mask & (self.sysmon_df['Computer'] == target_computer)
        
        return mask
    
    def mask_to_find_child_processcreate(self, process_guid: str, process_pid: int, target_computer: Optional[str] = None) -> pd.Series:
        """Create mask to find child process creation events."""
        mask = (
            (self.sysmon_df['EventID'] == 1) &
            (self.sysmon_df['ParentProcessGuid'] == process_guid) &
            (self.sysmon_df['ParentProcessId'] == process_pid)
        )
        
        if target_computer:
            mask = mask & (self.sysmon_df['Computer'] == target_computer)
        
        return mask
    
    def mask_to_find_parent_eventid_8_or_10(self, process_guid: str, process_pid: int, target_computer: Optional[str] = None) -> pd.Series:
        """Create mask to find EventID 8 and 10 events (CreateRemoteThread, ProcessAccess)."""
        mask = (
            (self.sysmon_df['EventID'].isin([8, 10])) &
            (self.sysmon_df['TargetProcessGUID'] == process_guid) &
            (self.sysmon_df['TargetProcessId'] == process_pid)
        )
        
        if target_computer:
            mask = mask & (self.sysmon_df['Computer'] == target_computer)
        
        return mask
    
    def trace_eventid1_lifecycle(self, originator_event: pd.Series, originator_info: Dict) -> Dict[str, any]:
        """
        Trace EventID 1 attack lifecycle using process-centric correlation.
        
        Focuses on ProcessGuid-based tracing for child processes and spawned events.
        """
        process_guid = originator_event['ProcessGuid']
        process_pid = int(float(originator_event['ProcessId'])) if pd.notna(originator_event['ProcessId']) else None
        computer = originator_event['Computer']
        
        self.logger.info(f"🔍 Tracing EventID 1 lifecycle for ProcessGuid: {process_guid}")
        self.logger.info(f"   Process: PID {process_pid} on {computer}")
        
        # Display command
        command = originator_event.get('CommandLine', 'N/A')
        self.logger.info(f"   Command: {str(command)[:100]}...")
        
        # Check if we have valid process information
        if process_pid is None:
            self.logger.warning(f"⚠️ No valid ProcessId for EventID 1 originator row {originator_info['RawDatasetRowNumber']}")
            return self._create_minimal_trace_result(originator_event, originator_info)
        
        # Convert originator_event to dict and add Sysmon row index for deduplication
        originator_event_dict = dict(originator_event)
        originator_event_dict['_sysmon_row_index'] = originator_event.name  # pandas Series.name is the index
        
        # Initialize traced events collection
        traced_events = {
            'originator_event': originator_event_dict,
            'spawned_events': [],
            'child_process_create_events': [],
            'eight_and_ten_events': [],
            'all_events': [originator_event_dict],
            'analysis_details': {
                'process_guid': process_guid,
                'process_pid': process_pid,
                'computer': computer,
                'originator_row': originator_info['RawDatasetRowNumber'],
                'event_id': originator_event['EventID'],
                'timestamp': originator_event['timestamp']
            }
        }
        
        # Apply process-centric correlation masks
        self._apply_spawned_events_mask(process_guid, process_pid, computer, traced_events)
        self._apply_child_processcreate_mask(process_guid, process_pid, computer, traced_events)
        self._apply_8_and_10_events_mask(process_guid, process_pid, computer, traced_events)
        
        # Recursive child process tracing
        if traced_events['child_process_create_events']:
            self.logger.info(f"   🔄 Applying recursive tracing for {len(traced_events['child_process_create_events'])} child processes")
            self._recursive_child_tracing(traced_events['child_process_create_events'], traced_events)
        
        # Organize events by computer
        self._organize_events_by_computer(traced_events, originator_info)
        
        total_events = len(traced_events['all_events'])
        self.stats['total_traced_events'] += total_events
        
        self.logger.info(f"✅ EventID 1 lifecycle traced: {total_events} total events")
        self.logger.info(f"   Spawned events: {len(traced_events['spawned_events'])}")
        self.logger.info(f"   Child Process Create events: {len(traced_events['child_process_create_events'])}")
        self.logger.info(f"   EventID 8/10 events: {len(traced_events['eight_and_ten_events'])}")
        
        return traced_events
    
    def _create_minimal_trace_result(self, originator_event: pd.Series, originator_info: Dict) -> Dict[str, any]:
        """Create minimal trace result when full tracing cannot be performed."""
        # Convert originator_event to dict and add Sysmon row index for deduplication
        originator_event_dict = dict(originator_event)
        originator_event_dict['_sysmon_row_index'] = originator_event.name  # pandas Series.name is the index
        
        return {
            'originator_event': originator_event_dict,
            'spawned_events': [],
            'child_process_create_events': [],
            'eight_and_ten_events': [],
            'all_events': [originator_event_dict],
            'analysis_details': {
                'process_guid': originator_event['ProcessGuid'],
                'process_pid': None,
                'computer': originator_event['Computer'],
                'originator_row': originator_info['RawDatasetRowNumber'],
                'event_id': originator_event['EventID'],
                'timestamp': originator_event['timestamp']
            }
        }
    
    def _apply_spawned_events_mask(self, process_guid: str, process_pid: int, computer: str, traced_events: Dict) -> None:
        """Apply spawned events mask to find events from the same process."""
        mask = self.mask_to_find_spawned_events(process_guid, process_pid, computer)
        spawned = self.sysmon_df[mask]
        
        if len(spawned) > 0:
            spawned_records = spawned.to_dict('records')
            # Add original Sysmon row index for deduplication
            for i, record in enumerate(spawned_records):
                record['_sysmon_row_index'] = spawned.index[i]
            traced_events['spawned_events'].extend(spawned_records)
            traced_events['all_events'].extend(spawned_records)
            if self.debug:
                self.logger.debug(f"   Found {len(spawned)} spawned events")
    
    def _apply_child_processcreate_mask(self, process_guid: str, process_pid: int, computer: str, traced_events: Dict) -> None:
        """Apply child process creation mask to find new processes."""
        mask = self.mask_to_find_child_processcreate(process_guid, process_pid, computer)
        children = self.sysmon_df[mask]
        
        if len(children) > 0:
            children_records = children.to_dict('records')
            # Add original Sysmon row index for deduplication
            for i, record in enumerate(children_records):
                record['_sysmon_row_index'] = children.index[i]
            traced_events['child_process_create_events'].extend(children_records)
            traced_events['all_events'].extend(children_records)
            if self.debug:
                self.logger.debug(f"   Found {len(children)} child processes")
    
    def _apply_8_and_10_events_mask(self, process_guid: str, process_pid: int, computer: str, traced_events: Dict) -> None:
        """Apply EventID 8 and 10 events mask."""
        mask = self.mask_to_find_parent_eventid_8_or_10(process_guid, process_pid, computer)
        events_8_10 = self.sysmon_df[mask]
        
        if len(events_8_10) > 0:
            events_8_10_records = events_8_10.to_dict('records')
            # Add original Sysmon row index for deduplication
            for i, record in enumerate(events_8_10_records):
                record['_sysmon_row_index'] = events_8_10.index[i]
            traced_events['eight_and_ten_events'].extend(events_8_10_records)
            traced_events['all_events'].extend(events_8_10_records)
            if self.debug:
                self.logger.debug(f"   Found {len(events_8_10)} EventID 8/10 events")
    
    def _recursive_child_tracing(self, child_process_create_events: List[Dict], traced_events: Dict) -> None:
        """Recursively trace ALL child processes."""
        for child_event in child_process_create_events:
            child_guid = child_event['ProcessGuid']
            child_pid = int(float(child_event['ProcessId'])) if pd.notna(child_event['ProcessId']) else None
            child_computer = child_event['Computer']
            
            if child_pid is None:
                continue
            
            if self.debug:
                self.logger.debug(f"   Recursively tracing child: {child_guid}")
            
            # Apply masks to child process
            self._apply_spawned_events_mask(child_guid, child_pid, child_computer, traced_events)
            self._apply_8_and_10_events_mask(child_guid, child_pid, child_computer, traced_events)
            
            # Find grandchildren
            grandchild_mask = self.mask_to_find_child_processcreate(child_guid, child_pid, child_computer)
            grandchildren = self.sysmon_df[grandchild_mask]
            
            if len(grandchildren) > 0:
                new_children = grandchildren.to_dict('records')
                # Add original Sysmon row index for deduplication
                for i, record in enumerate(new_children):
                    record['_sysmon_row_index'] = grandchildren.index[i]
                traced_events['child_process_create_events'].extend(new_children)
                traced_events['all_events'].extend(new_children)
                
                # Recursive call for unlimited depth
                self._recursive_child_tracing(new_children, traced_events)
    
    def _organize_events_by_computer(self, traced_events: Dict, originator_info: Dict) -> None:
        """Organize traced events by computer for timeline visualization."""
        originator_row = originator_info['RawDatasetRowNumber']
        
        if self.debug:
            self.logger.debug(f"📊 Organizing {len(traced_events['all_events'])} events for Row {originator_row}")
        
        for event in traced_events['all_events']:
            try:
                computer = event['Computer']
                
                if computer not in self.computer_events:
                    self.computer_events[computer] = []
                
                # Count events by computer and EventID for statistics
                if computer not in self.stats['events_by_computer']:
                    self.stats['events_by_computer'][computer] = 0
                self.stats['events_by_computer'][computer] += 1
                
                event_id = event['EventID']
                if event_id not in self.stats['events_by_eventid']:
                    self.stats['events_by_eventid'][event_id] = 0
                self.stats['events_by_eventid'][event_id] += 1
                
                # Create event record for timeline
                event_record = {
                    'originator_row': originator_row,
                    'event_data': event,
                    'classification': 'malicious',
                    'timestamp': event['timestamp']  # Preserve original millisecond format
                }
                
                self.computer_events[computer].append(event_record)
                self.all_traced_events.append(event_record)
                
            except Exception as e:
                self.logger.error(f"❌ Error organizing event for Row {originator_row}: {e}")
                if self.debug:
                    self.logger.error(f"Event causing error: {event}")
    
    def _organize_file_events_by_computer(self, traced_data: Dict, originator_info: Dict) -> None:
        """Organize EventID 11/23 file events by computer for timeline visualization."""
        originator_row = originator_info['RawDatasetRowNumber']
        
        if self.debug:
            self.logger.debug(f"📊 Organizing {len(traced_data['traced_events'])} file events for Row {originator_row}")
        
        for event in traced_data['traced_events']:
            try:
                computer = event['Computer']
                
                if computer not in self.computer_events:
                    self.computer_events[computer] = []
                
                # Count events by computer and EventID for statistics
                if computer not in self.stats['events_by_computer']:
                    self.stats['events_by_computer'][computer] = 0
                self.stats['events_by_computer'][computer] += 1
                
                event_id = traced_data['event_id']
                if event_id not in self.stats['events_by_eventid']:
                    self.stats['events_by_eventid'][event_id] = 0
                self.stats['events_by_eventid'][event_id] += 1
                
                # Create event record for timeline - convert file event format to timeline format
                event_record = {
                    'originator_row': originator_row,
                    'event_data': {
                        'EventID': event_id,
                        'Computer': computer,
                        'timestamp': event['timestamp'],
                        'TargetFilename': event.get('target_filename', ''),
                        'Image': event.get('image', '')
                    },
                    'classification': 'malicious',
                    'timestamp': event['timestamp']  # Preserve original millisecond format
                }
                
                self.computer_events[computer].append(event_record)
                self.all_traced_events.append(event_record)
                
            except Exception as e:
                self.logger.error(f"❌ Error organizing file event for Row {originator_row}: {e}")
                if self.debug:
                    self.logger.error(f"File event causing error: {event}")
    
    def _deduplicate_events(self, all_events: List[Dict]) -> List[Dict]:
        """
        Deduplicate events to ensure each unique Sysmon event appears only once.
        
        For overlapping process trees, keep the event attribution with the most specific/closest originator.
        Uses EventID, Computer, timestamp, and ProcessGuid as the unique key.
        """
        if not all_events:
            return all_events
            
        self.logger.info(f"🔄 Deduplicating {len(all_events)} events...")
        
        # Debug: Check for EventID 1 events at specific timestamps that should be deduplicated
        test_cases = [
            (1, 1742361238003, '4a85d404-5296-67da-3f01-000000005500'),
            (1, 1742361274995, '4a85d404-52ba-67da-4101-000000005500')
        ]
        
        for event_id, timestamp, process_guid in test_cases:
            test_events = [e for e in all_events 
                         if e.get('EventID') == event_id 
                         and e.get('timestamp') == timestamp 
                         and e.get('ProcessGuid') == process_guid]
            if test_events:
                self.logger.info(f"   DEBUG: Found {len(test_events)} EventID {event_id} events at timestamp {timestamp}")
                for e in test_events:
                    self.logger.info(f"     - OriginatorRow: {e.get('OriginatorRow')}, _sysmon_row_index: {e.get('_sysmon_row_index', 'MISSING')}")
        
        # Group events by unique identifiers
        event_groups = {}
        
        for event in all_events:
            # Create unique key based on critical event attributes + original Sysmon row index
            event_id = event.get('EventID', '')
            computer = event.get('Computer', '')
            timestamp = event.get('timestamp', '')
            process_guid = event.get('ProcessGuid', '')
            image = event.get('Image', '')
            target_filename = event.get('TargetFilename', '')
            image_loaded = event.get('ImageLoaded', '')
            sysmon_row_index = event.get('_sysmon_row_index', '')  # Original Sysmon row index
            
            # Different key strategies for different event types
            # CRITICAL: Include sysmon_row_index to preserve legitimate duplicate Sysmon events
            if event_id in [1]:  # Process Creation
                unique_key = (event_id, computer, timestamp, process_guid, image, sysmon_row_index)
            elif event_id in [7]:  # Image Load
                unique_key = (event_id, computer, timestamp, process_guid, image, image_loaded, sysmon_row_index)
            elif event_id in [10]:  # Process Access
                unique_key = (event_id, computer, timestamp, 
                            event.get('SourceProcessGUID', ''), 
                            event.get('TargetProcessGUID', ''), sysmon_row_index)
            elif event_id in [11, 23]:  # File Create/Delete
                unique_key = (event_id, computer, timestamp, process_guid, target_filename, sysmon_row_index)
            elif event_id in [13]:  # Registry Event
                unique_key = (event_id, computer, timestamp, process_guid, 
                            event.get('TargetObject', ''), sysmon_row_index)
            else:  # Generic fallback
                unique_key = (event_id, computer, timestamp, process_guid, image, sysmon_row_index)
            
            if unique_key not in event_groups:
                event_groups[unique_key] = []
            
            event_groups[unique_key].append(event)
        
        # Select the best event from each group
        deduplicated_events = []
        duplicates_found = 0
        
        for unique_key, events_in_group in event_groups.items():
            if len(events_in_group) == 1:
                # Single event, keep it
                deduplicated_events.append(events_in_group[0])
            else:
                # Multiple events with same signature - choose the best one
                duplicates_found += len(events_in_group) - 1
                
                # Priority logic: prefer the event from the latest/most specific originator
                # (higher OriginatorRow numbers are processed later and are more specific)
                best_event = max(events_in_group, 
                               key=lambda e: int(e.get('OriginatorRow', 0)))
                
                deduplicated_events.append(best_event)
                
                if self.debug:
                    originator_rows = [e.get('OriginatorRow', 'N/A') for e in events_in_group]
                    self.logger.debug(f"   Duplicate found for {unique_key}: "
                                    f"OriginatorRows {originator_rows}, "
                                    f"kept latest/most specific OriginatorRow {best_event.get('OriginatorRow')}")
        
        self.logger.info(f"✅ Deduplication completed: {len(all_events)} → {len(deduplicated_events)} "
                        f"({duplicates_found} duplicates removed)")
        
        return deduplicated_events
    
    def _lookup_sysmon_row_by_raw_dataset_row_number(self, raw_dataset_row_number: int) -> pd.Series:
        """Look up the actual Sysmon row from CSV using the raw dataset row number."""
        try:
            # The raw dataset row number in all_target_events_run-04.csv corresponds to the actual line in sysmon CSV (1-indexed, including header)
            # Since pandas is 0-indexed and excludes the header row, we need to subtract 2
            actual_row_index = raw_dataset_row_number - 2  # -1 for 0-indexing, -1 for header row
            
            if 0 <= actual_row_index < len(self.sysmon_df):
                return self.sysmon_df.iloc[actual_row_index]
            else:
                self.logger.warning(f"⚠️ Row number {raw_dataset_row_number} is out of bounds (CSV has {len(self.sysmon_df)} rows)")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error looking up Sysmon row {raw_dataset_row_number}: {e}")
            return None
    
    def create_file_event_timeline(self, originator_event: pd.Series, originator_info: Dict) -> Dict[str, any]:
        """
        Create a timeline for EventID 11/23 (file create/delete) events.
        
        Unlike EventID 1, file events are standalone and don't have child processes to trace.
        """
        event_id = originator_info['EventID']
        originator_row = originator_info['RawDatasetRowNumber']
        
        try:
            # File events are standalone - just create single event data
            traced_result = {
                'originator_row': originator_row,
                'event_id': event_id,
                'traced_events': [],
                'total_events': 1,
                'spawned_events': 0,
                'child_processes': 0,
                'eventid_8_10_events': 0,
                'computers': [originator_event['Computer']],
                'processing_duration_seconds': 0,
                'success': True
            }
            
            # Look up the actual Sysmon row from CSV using RawDatasetRowNumber
            actual_sysmon_row = self._lookup_sysmon_row_by_raw_dataset_row_number(originator_info['RawDatasetRowNumber'])
            
            if actual_sysmon_row is not None:
                # Use the actual Sysmon row data instead of synthetic data
                event_dict = actual_sysmon_row.to_dict()
                # Add Sysmon row index for deduplication
                event_dict['_sysmon_row_index'] = actual_sysmon_row.name
                traced_result['traced_events'].append(event_dict)
                self.logger.info(f"✅ EventID {event_id} timeline created: 1 file event (actual Sysmon row)")
                self.logger.info(f"   Target File: {actual_sysmon_row.get('TargetFilename', 'N/A')}")
                self.logger.info(f"   Process: {actual_sysmon_row.get('Image', 'N/A')}")
            else:
                self.logger.warning(f"⚠️ Could not find actual Sysmon row for RawDatasetRowNumber {originator_info['RawDatasetRowNumber']}")
                traced_result['traced_events'].append({})  # Empty placeholder
            
            return traced_result
            
        except Exception as e:
            self.logger.error(f"❌ Error creating file event timeline for row {originator_row}: {e}")
            return {
                'originator_row': originator_row,
                'event_id': event_id,
                'traced_events': [],
                'success': False,
                'error': str(e)
            }
    
    def process_all_originators(self) -> bool:
        """Process all selected attack event originators (EventID 1, 11, 23)."""
        self.logger.info("🚀 STARTING ATTACK LIFECYCLE ANALYSIS")
        self.logger.info("=" * 70)
        
        if not self.selected_originators:
            self.logger.error("❌ No attack originators to process")
            return False
        
        self.logger.info(f"📊 Processing {len(self.selected_originators)} selected attack originators...")
        
        # Process each originator
        for originator_info in self.selected_originators:
            try:
                originator_row = originator_info['RawDatasetRowNumber']
                event_id = originator_info['EventID']
                processing_type = originator_info.get('processing_type', 'Seed_Event')
                
                self.logger.info(f"🎯 Processing EventID {event_id} Originator Row {originator_row}")
                self.logger.info("=" * 50)
                
                # Find the originator event in Sysmon data
                originator_event = self.get_originator_event_from_sysmon(originator_info)
                if originator_event is None:
                    self.logger.warning(f"⚠️ Could not find EventID {event_id} originator event for row {originator_row}")
                    continue
                
                # Route to appropriate processing based on EventID
                if event_id == 1:
                    # Process creation - use full lifecycle tracing
                    self.logger.info(f"🏠 LOCAL PROCESSING: Using process-centric tracing for EventID 1")
                    traced_data = self.trace_eventid1_lifecycle(originator_event, originator_info)
                elif event_id in [11, 23]:
                    # File events - create standalone plots
                    event_type_name = "File Create" if event_id == 11 else "File Delete"
                    self.logger.info(f"📄 FILE PROCESSING: Creating timeline for EventID {event_id} ({event_type_name})")
                    traced_data = self.create_file_event_timeline(originator_event, originator_info)
                else:
                    self.logger.warning(f"⚠️ Unsupported EventID {event_id} for row {originator_row}")
                    continue
                
                traced_data['processing_type'] = processing_type
                
                # Extract tactic and technique metadata if available
                tactic_label = originator_info.get('Tactic', '').strip()
                technique_label = originator_info.get('Technique', '').strip()
                if tactic_label:
                    traced_data['tactic_label'] = tactic_label
                    self.logger.info(f"🏷️ Tactic: {tactic_label}")
                if technique_label:
                    traced_data['technique_label'] = technique_label
                    self.logger.info(f"🔧 Technique: {technique_label}")
                
                # Organize events by computer for group timeline (EventID 11/23 only)
                if event_id in [11, 23]:
                    self._organize_file_events_by_computer(traced_data, originator_info)
                
                # Only increment if this is a new unique originator row (avoid duplicates from Seed_Event+Tactic)
                if originator_row not in self.traced_events:
                    self.stats['successfully_traced'] += 1
                self.traced_events[originator_row] = traced_data
                
            except Exception as e:
                self.logger.error(f"❌ Error processing EventID {event_id} originator row {originator_row}: {e}")
                if self.debug:
                    import traceback
                    self.logger.error(f"Full traceback: {traceback.format_exc()}")
                continue
        
        # Calculate processing duration
        end_time = datetime.now()
        start_time = datetime.fromisoformat(self.stats['processing_start_time'])
        self.stats['processing_duration_seconds'] = (end_time - start_time).total_seconds()
        
        self.logger.info("✅ MULTI-EVENTID LIFECYCLE ANALYSIS COMPLETED")
        self.logger.info("=" * 70)
        
        return True
    
    def create_individual_timeline_plots(self, output_dir: Path) -> List[Path]:
        """Create individual timeline plots for each traced attack event originator."""
        if not self.traced_events:
            self.logger.warning("⚠️ No traced attack events to plot")
            return []
        
        self.logger.info("📊 Creating individual attack timeline plots...")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_files = []
        
        for originator_row, traced_data in self.traced_events.items():
            try:
                event_id = traced_data.get('event_id', 1)
                if event_id in [1, 11, 23]:
                    plot_file = self._create_timeline_plot(traced_data, output_dir)
                else:
                    self.logger.warning(f"⚠️ Unsupported EventID {event_id} for plotting")
                    continue
                    
                if plot_file:
                    plot_files.append(plot_file)
                    # Update plot counters by EventID type
                    if event_id == 1:
                        self.stats['eventid1_plots_created'] += 1
                    elif event_id == 11:
                        self.stats['eventid11_plots_created'] += 1
                    elif event_id == 23:
                        self.stats['eventid23_plots_created'] += 1
            except Exception as e:
                self.logger.error(f"❌ Error creating plot for EventID {event_id} row {originator_row}: {e}")
        
        self.logger.info(f"✅ Created {len(plot_files)} individual attack timeline plots")
        return plot_files
    
    
    def _create_timeline_plot(self, traced_data: Dict, output_dir: Path) -> Optional[Path]:
        """Unified timeline plot creation for all EventID types (1, 11, 23)."""
        try:
            # Determine event type and extract data accordingly
            if 'analysis_details' in traced_data:
                # EventID 1 (complex) data structure
                event_id = traced_data['analysis_details']['event_id']
                originator_row = traced_data['analysis_details']['originator_row']
                originator_event = traced_data['originator_event']

                # Use raw events without deduplication to show true traced event count
                # (matches CSV export behavior - no deduplication)
                raw_events = traced_data['all_events']

                # Prepare events for plotting (no deduplication)
                events_to_plot = []
                for event in raw_events:
                    events_to_plot.append({
                        'timestamp': pd.to_datetime(event['timestamp'], unit='ms'),
                        'event_id': event['EventID'],
                        'computer': event['Computer'],
                        'process_guid': event.get('ProcessGuid', ''),
                        'command': event.get('CommandLine', '')
                    })
                is_complex = True
                primary_display_text = originator_event.get('CommandLine', 'N/A')
                
            else:
                # EventID 11/23 (simple) data structure  
                event_id = traced_data['event_id']
                originator_row = traced_data['originator_row']
                traced_events = traced_data['traced_events']
                
                if not traced_events:
                    self.logger.warning(f"⚠️ No events to plot for EventID {event_id} row {originator_row}")
                    return None
                
                # Convert file event to standard format
                file_event = traced_events[0]
                events_to_plot = [{
                    'timestamp': pd.to_datetime(file_event['timestamp'], unit='ms'),  # Convert for plotting
                    'event_id': event_id,
                    'computer': file_event.get('Computer', 'Unknown'),
                    'process_guid': file_event.get('process_guid', ''),
                    'command': file_event.get('target_filename', 'Unknown File')
                }]
                is_complex = False
                primary_display_text = file_event.get('target_filename', 'Unknown File')
            
            if not events_to_plot:
                self.logger.warning(f"⚠️ No events to plot for EventID {event_id} row {originator_row}")
                return None
            
            events_df = pd.DataFrame(events_to_plot)
            
            # Adaptive figure sizing based on complexity
            if is_complex:
                figsize = (16, 8)  # Large for complex EventID 1 events
                y_axis_mode = 'eventid'  # Show EventID numbers on Y-axis
            else:
                figsize = (14, 4)  # Medium for simple EventID 11/23 events  
                y_axis_mode = 'simple'  # Simple Y-axis
            
            fig, ax = plt.subplots(figsize=figsize)
            
            # Unified EventID colors
            eventid_colors = {
                1: '#e41a1c',   # Process Creation - Red
                3: '#377eb8',   # Network Connection - Blue
                5: '#1abc9c',   # Process Terminate - Teal
                7: '#f39c12',   # Image Loaded - Orange
                8: '#8e44ad',   # CreateRemoteThread - Purple
                9: '#d35400',   # RawAccessRead - Dark Orange
                10: '#f1c40f',  # ProcessAccess - Yellow
                11: '#2E8B57',  # File Create - Green
                12: '#3498db',  # Registry Event - Light Blue
                13: '#9b59b6',  # Registry Event - Light Purple
                23: '#DC143C'   # File Delete - Red
            }
            
            # Plot events
            if is_complex:
                # Complex timeline with multiple EventIDs
                event_counts = events_df['event_id'].value_counts().sort_index()
                
                for event_id_plot in sorted(events_df['event_id'].unique()):
                    event_subset = events_df[events_df['event_id'] == event_id_plot]
                    color = eventid_colors.get(event_id_plot, '#000000')
                    count = event_counts[event_id_plot]
                    
                    ax.scatter(event_subset['timestamp'], event_subset['event_id'],
                              color=color, s=60, alpha=0.8, 
                              label=f'EventID {event_id_plot} ({count})',
                              edgecolors='black', linewidths=0.5)
                
                # Y-axis shows EventID numbers
                unique_event_ids = sorted(events_df['event_id'].unique())
                ax.set_yticks(unique_event_ids)
                ax.set_yticklabels(unique_event_ids)
                ax.set_ylim(min(unique_event_ids) - 0.5, max(unique_event_ids) + 0.5)
                ax.set_ylabel('EventID')
                
                # Enhanced legend for complex plots
                legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
                legend.set_title('Event Types & Counts', prop={'size': 10, 'weight': 'bold'})
                
            else:
                # Simple timeline with single event
                event_type_name = "File Create" if event_id == 11 else "File Delete"
                color = eventid_colors.get(event_id, '#000000')
                timestamp = events_to_plot[0]['timestamp']
                
                ax.scatter([timestamp], [event_id], c=color, s=100, alpha=0.8, 
                          zorder=3, edgecolors='black', linewidths=1)
                ax.text(timestamp, event_id + 0.3, event_type_name, ha='center', va='bottom', 
                       fontsize=10, weight='bold')
                
                # Y-axis shows just the single EventID
                ax.set_yticks([event_id])
                ax.set_yticklabels([f'EventID {event_id}'])
                ax.set_ylim(event_id - 1, event_id + 1)
                ax.set_ylabel('Event Type')
            
            # DYNAMIC TIME AXIS ADJUSTMENT based on event time span
            time_min = events_df['timestamp'].min()
            time_max = events_df['timestamp'].max()
            time_range = time_max - time_min
            total_seconds = time_range.total_seconds()
            
            # Dynamic time axis formatting based on actual time span
            if total_seconds <= 0:
                # Single event: add ±30 seconds padding for context
                padding = pd.Timedelta(seconds=30)
                ax.set_xlim(time_min - padding, time_max + padding)
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S.%f'))
                ax.xaxis.set_major_locator(mdates.SecondLocator(interval=10))
            elif total_seconds <= 1:
                # Events within 1 second: show millisecond precision
                padding = pd.Timedelta(milliseconds=500)
                ax.set_xlim(time_min - padding, time_max + padding)
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S.%f'))
                ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
            elif total_seconds <= 10:
                # Events within 10 seconds: show second precision
                padding = pd.Timedelta(seconds=2)
                ax.set_xlim(time_min - padding, time_max + padding)
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                ax.xaxis.set_major_locator(mdates.SecondLocator(interval=1))
            elif total_seconds <= 60:
                # Events within 1 minute: show second precision with wider ticks
                padding = pd.Timedelta(seconds=5)
                ax.set_xlim(time_min - padding, time_max + padding)
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                ax.xaxis.set_major_locator(mdates.SecondLocator(interval=10))
            elif total_seconds <= 3600:
                # Events within 1 hour: show minute precision
                padding = pd.Timedelta(minutes=2)
                ax.set_xlim(time_min - padding, time_max + padding)
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
            else:
                # Longer time spans: use default formatting
                ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            
            plt.setp(ax.xaxis.get_ticklabels(), rotation=45, ha='right')
            ax.set_xlabel('Timeline')
            
            # UNIFIED TITLE FORMATTING
            event_type_desc = {1: "Process Creation", 11: "File Create", 23: "File Delete"}
            title_lines = [f'EventID {event_id} ({event_type_desc.get(event_id, "Unknown")}) - Row {originator_row}']
            
            # Add tactic and technique if available
            if traced_data.get('tactic_label'):
                title_lines.append(f'Tactic: {traced_data["tactic_label"]}')
            if traced_data.get('technique_label'):
                title_lines.append(f'Technique: {traced_data["technique_label"]}')
            
            # Add primary display text (Command for EventID 1, TargetFilename for EventID 11/23)
            display_label = "Command" if event_id == 1 else "Target File"
            text_display = str(primary_display_text)[:120] + ('...' if len(str(primary_display_text)) > 120 else '')
            title_lines.append(f'{display_label}: {text_display}')
            
            if is_complex:
                title_lines.append(f'Total Events: {len(events_to_plot)}')
            
            ax.set_title('\n'.join(title_lines), fontsize=11, pad=20)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot with consistent naming
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"eventid{event_id}_timeline_row_{originator_row}.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"📊 EventID {event_id} plot saved: {output_file}")
            return output_file
            
        except Exception as e:
            event_id = traced_data.get('event_id', traced_data.get('analysis_details', {}).get('event_id', 'Unknown'))
            originator_row = traced_data.get('originator_row', traced_data.get('analysis_details', {}).get('originator_row', 'Unknown'))
            self.logger.error(f"❌ Error creating EventID {event_id} plot for row {originator_row}: {e}")
            return None
    
    def create_group_timeline_plot(self, output_dir: Path) -> Optional[Path]:
        """Create group timeline plot showing all traced attack events by computer."""
        if not self.all_traced_events:
            self.logger.warning("⚠️ No traced attack events for group timeline")
            return None
        
        self.logger.info("📊 Creating attack events group timeline plot...")
        
        # Organize events by computer and sort by event count (descending)
        computers_with_counts = []
        for computer in self.computer_events.keys():
            event_count = len(self.computer_events[computer])
            computers_with_counts.append((computer, event_count))
        
        # Sort by event count (descending) for top-to-bottom arrangement
        computers_with_counts.sort(key=lambda x: x[1], reverse=True)
        computers = [computer for computer, count in computers_with_counts]
        
        if not computers:
            self.logger.warning("⚠️ No computers found for group timeline")
            return None
        
        # Create subplots for each computer with increased vertical space
        fig, axes = plt.subplots(len(computers), 1, figsize=(16, 8 * len(computers)), sharex=True)
        if len(computers) == 1:
            axes = [axes]
        
        # EventID colors (added 11 and 23 for file events)
        eventid_colors = {
            1: '#e41a1c', 3: '#377eb8', 5: '#1abc9c', 7: '#f39c12', 8: '#8e44ad',
            9: '#d35400', 10: '#f1c40f', 11: '#2E8B57', 12: '#3498db', 13: '#9b59b6', 23: '#DC143C'
        }
        
        # Plot each computer's events
        for i, (computer, ax) in enumerate(zip(computers, axes)):
            computer_events = pd.DataFrame([
                {
                    'timestamp': event['timestamp'],
                    'event_id': event['event_data']['EventID'],
                    'originator_row': event['originator_row']
                } for event in self.computer_events[computer]
            ])
            
            if len(computer_events) == 0:
                continue
            
            # Plot by originator row
            for originator_row in sorted(computer_events['originator_row'].unique()):
                row_events = computer_events[computer_events['originator_row'] == originator_row]
                
                # Convert timestamps to datetime for plotting
                timestamps = pd.to_datetime(row_events['timestamp'], unit='ms', errors='coerce')
                event_ids = row_events['event_id']
                
                # Use different color for each originator
                color = eventid_colors.get(originator_row % len(eventid_colors) + 1, '#000000')
                
                ax.scatter(timestamps, event_ids,
                         color=color, 
                         s=60, 
                         alpha=0.8, 
                         label=f'Row {originator_row}',
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
                unique_event_ids = sorted(computer_events['event_id'].unique())
                ax.set_yticks(unique_event_ids)
                ax.set_yticklabels(unique_event_ids)
                ax.set_ylim(min(unique_event_ids) - 0.5, max(unique_event_ids) + 0.5)
        
        # Format time axis
        if len(computers) > 0:
            axes[-1].set_xlabel('Time')
            axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
            axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.setp(axes[-1].xaxis.get_ticklabels(), rotation=45, ha='right')
        
        # plt.suptitle('Multi-EventID Attack Progression - Group Timeline\n'
                    # f'Traced from {len(self.selected_originators)} Selected Originators', 
                    # fontsize=16, y=0.98)
        plt.suptitle('Multi-EventID Attack Progression - Group Timeline\n\n', 
                    fontsize=16, y=0.98)
        plt.tight_layout()
        
        # Save plot
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "timeline_all_malicious_events.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"📊 Attack events group timeline saved: {output_file}")
        return output_file
    
    def _generate_basic_labeling_stats(self, total_sysmon_events: int, traced_df: pd.DataFrame):
        """Generate basic labeling statistics from traced events (no labeled dataset needed)."""
        try:
            malicious_events = len(traced_df)
            benign_events = total_sysmon_events - malicious_events
            malicious_percentage = (malicious_events / total_sysmon_events * 100) if total_sysmon_events > 0 else 0.0
            
            # Calculate tactic distribution from traced events
            tactic_distribution = {}
            if not traced_df.empty and 'Tactic' in traced_df.columns:
                tactic_counts = traced_df['Tactic'].value_counts()
                for tactic, count in tactic_counts.items():
                    if pd.notna(tactic):
                        tactic_distribution[tactic] = int(count)
            
            self.labeling_stats = {
                'total_sysmon_events': total_sysmon_events,
                'malicious_events_labeled': malicious_events,
                'benign_events': benign_events,
                'malicious_percentage': round(malicious_percentage, 2),
                'tactic_distribution': tactic_distribution
            }
            
            self.logger.info(f"📊 Basic Statistics Generated: {total_sysmon_events:,} total | {malicious_events:,} malicious | {benign_events:,} benign | {malicious_percentage:.2f}% malicious")
            
        except Exception as e:
            self.logger.error(f"❌ Error generating basic labeling statistics: {e}")
            # Initialize empty stats to avoid JSON errors
            self.labeling_stats = {
                'total_sysmon_events': total_sysmon_events,
                'malicious_events_labeled': 0,
                'benign_events': total_sysmon_events,
                'malicious_percentage': 0.0,
                'tactic_distribution': {}
            }

    def _create_labeled_dataset_OLD_UNUSED(self, traced_events_file: Path, output_file: Path, apt_run_dir: Path) -> bool:
        """
        Create labeled Sysmon dataset with Tactic/Technique columns.
        
        Args:
            traced_events_file: Path to traced_sysmon_events_with_tactics.csv
            output_file: Path for labeled dataset output
            apt_run_dir: APT run directory containing original sysmon CSV
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.logger.info("📋 Creating labeled Sysmon dataset for tactics timeline...")
            
            # Get original sysmon file path dynamically
            run_dir_name = apt_run_dir.name  # e.g., "apt-1-run-04"
            apt_type = run_dir_name.split('-run-')[0]  # e.g., "apt-1" 
            run_id = run_dir_name.split('-run-')[1]    # e.g., "04"
            
            original_sysmon_file = apt_run_dir / f"sysmon-run-{run_id}.csv"
            if not original_sysmon_file.exists():
                self.logger.error(f"❌ Original Sysmon file not found: {original_sysmon_file}")
                return False
            
            # Read original sysmon dataset
            self.logger.info(f"📖 Reading original Sysmon dataset: {original_sysmon_file}")
            import pandas as pd
            sysmon_df = pd.read_csv(original_sysmon_file)
            self.logger.info(f"Original dataset shape: {sysmon_df.shape}")
            
            # Add default Tactic and Technique columns
            self.logger.info("➕ Adding default Tactic/Technique columns")
            sysmon_df['Tactic'] = 'no_attack_tactic'
            sysmon_df['Technique'] = 'no_technique'
            
            # Read traced malicious events
            self.logger.info(f"📖 Reading traced malicious events: {traced_events_file}")
            traced_df = pd.read_csv(traced_events_file)
            self.logger.info(f"Traced events shape: {traced_df.shape}")
            
            # Create mapping keys for event matching
            self.logger.info("🔗 Creating event mapping keys")
            
            def create_mapping_key(row):
                """Create robust unique key matching deduplication logic."""
                def safe_str(value):
                    return str(value) if pd.notna(value) else 'NULL'

                event_id = safe_str(row.get('EventID', ''))
                computer = safe_str(row.get('Computer', ''))
                timestamp = safe_str(row.get('timestamp', ''))
                process_guid = safe_str(row.get('ProcessGuid', ''))
                process_id = safe_str(row.get('ProcessId', ''))
                image = safe_str(row.get('Image', ''))

                # Event-specific unique identifiers matching deduplication logic
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
            
            # Create mapping keys for both datasets
            sysmon_df['mapping_key'] = sysmon_df.apply(create_mapping_key, axis=1)
            traced_df['mapping_key'] = traced_df.apply(create_mapping_key, axis=1)
            
            # Create mapping dictionary from traced events
            self.logger.info("📋 Creating Tactic/Technique mapping")
            tactic_technique_map = {}
            for _, row in traced_df.iterrows():
                key = row['mapping_key']
                tactic_technique_map[key] = {
                    'Tactic': row['Tactic'],
                    'Technique': row['Technique']
                }
            
            self.logger.info(f"Found {len(tactic_technique_map)} unique malicious event keys")
            
            # Apply labels to matching events
            self.logger.info("🏷️ Applying malicious event labels")
            labeled_count = 0
            
            for idx, row in sysmon_df.iterrows():
                mapping_key = row['mapping_key']
                if mapping_key in tactic_technique_map:
                    sysmon_df.at[idx, 'Tactic'] = tactic_technique_map[mapping_key]['Tactic']
                    sysmon_df.at[idx, 'Technique'] = tactic_technique_map[mapping_key]['Technique']
                    labeled_count += 1
            
            self.logger.info(f"✅ Labeled {labeled_count} malicious events")
            
            # Remove temporary mapping key column
            sysmon_df.drop('mapping_key', axis=1, inplace=True)
            
            # Reorder columns to put Tactic and Technique at the beginning
            cols = sysmon_df.columns.tolist()
            cols.remove('Tactic')
            cols.remove('Technique')
            new_cols = ['Tactic', 'Technique'] + cols
            sysmon_df = sysmon_df[new_cols]
            
            # Save labeled dataset
            self.logger.info(f"💾 Saving labeled dataset: {output_file}")
            output_file.parent.mkdir(parents=True, exist_ok=True)
            sysmon_df.to_csv(output_file, index=False)
            
            # Store labeling statistics for JSON output
            total_events = len(sysmon_df)
            benign_events = total_events - labeled_count
            malicious_percentage = (labeled_count / total_events * 100) if total_events > 0 else 0.0
            
            self.labeling_stats = {
                'total_sysmon_events': total_events,
                'malicious_events_labeled': labeled_count,
                'benign_events': benign_events,
                'malicious_percentage': round(malicious_percentage, 2),
                'unique_malicious_keys': len(tactic_technique_map)
            }
            
            # Show tactic distribution
            tactic_counts = sysmon_df['Tactic'].value_counts()
            tactic_distribution = {}
            self.logger.info("📋 Tactic Distribution:")
            for tactic, count in tactic_counts.items():
                if tactic != 'no_attack_tactic':
                    tactic_distribution[tactic] = int(count)
                    self.logger.info(f"  {tactic}: {count:,} events")
            
            self.labeling_stats['tactic_distribution'] = tactic_distribution
            
            self.logger.info(f"📊 Labeling Summary: {total_events:,} total | {labeled_count:,} malicious | {benign_events:,} benign | {malicious_percentage:.2f}% malicious")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error creating labeled dataset: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def _generate_labeling_stats_from_existing_OLD_UNUSED(self, labeled_sysmon_file: Path, traced_events_file: Path):
        """Generate labeling statistics from existing labeled dataset."""
        try:
            # Read the existing labeled dataset
            sysmon_df = pd.read_csv(labeled_sysmon_file)
            traced_df = pd.read_csv(traced_events_file)
            
            total_events = len(sysmon_df)
            # Count events that have actual tactics (not 'no_attack_tactic')
            labeled_count = len(sysmon_df[sysmon_df['Tactic'] != 'no_attack_tactic'])
            benign_events = total_events - labeled_count
            malicious_percentage = (labeled_count / total_events * 100) if total_events > 0 else 0.0
            
            # Count unique malicious mapping keys from traced events
            unique_malicious_keys = len(traced_df) if not traced_df.empty else 0
            
            self.labeling_stats = {
                'total_sysmon_events': total_events,
                'malicious_events_labeled': labeled_count,
                'benign_events': benign_events,
                'malicious_percentage': round(malicious_percentage, 2),
                'unique_malicious_keys': unique_malicious_keys
            }
            
            # Generate tactic distribution
            tactic_counts = sysmon_df['Tactic'].value_counts()
            tactic_distribution = {}
            for tactic, count in tactic_counts.items():
                if tactic != 'no_attack_tactic':
                    tactic_distribution[tactic] = int(count)
            
            self.labeling_stats['tactic_distribution'] = tactic_distribution
            
            self.logger.info(f"📊 Labeling Statistics Generated: {total_events:,} total | {labeled_count:,} malicious | {benign_events:,} benign | {malicious_percentage:.2f}% malicious")
            
        except Exception as e:
            self.logger.error(f"❌ Error generating labeling statistics: {e}")
            # Initialize empty stats to avoid JSON errors
            self.labeling_stats = {
                'total_sysmon_events': 0,
                'malicious_events_labeled': 0,
                'benign_events': 0,
                'malicious_percentage': 0.0,
                'unique_malicious_keys': 0,
                'tactic_distribution': {}
            }

    def create_tactics_timeline_plot(self, output_dir: Path) -> Optional[Path]:
        """Create timeline plot showing all Sysmon events with malicious events highlighted by MITRE tactics."""
        # Read tactics information from the CSV file
        csv_file = output_dir / "traced_sysmon_events_with_tactics.csv"
        if not csv_file.exists():
            self.logger.warning(f"⚠️ Tactics CSV file not found: {csv_file}")
            return None
        
        self.logger.info("📊 Creating complete Sysmon timeline with tactics highlighting...")
        
        # Get APT run directory (parent of output directory)
        apt_run_dir = output_dir.parent
        
        # Read original sysmon dataset directly (no labeled dataset needed)
        run_dir_name = apt_run_dir.name  # e.g., "apt-1-run-04"
        run_id = run_dir_name.split('-run-')[1]    # e.g., "04"
        original_sysmon_file = apt_run_dir / f"sysmon-run-{run_id}.csv"
        
        if not original_sysmon_file.exists():
            self.logger.error(f"❌ Original Sysmon file not found: {original_sysmon_file}")
            return None
            
        try:
            # Read original sysmon dataset
            all_sysmon_df = pd.read_csv(original_sysmon_file)
            self.logger.info(f"📖 Read {len(all_sysmon_df):,} total Sysmon events from original dataset")
            
            # Read traced malicious events 
            traced_df = pd.read_csv(csv_file)
            self.logger.info(f"📖 Read {len(traced_df)} traced malicious events from CSV")
            
            # Generate basic labeling statistics from traced events (no labeled dataset needed)
            self._generate_basic_labeling_stats(len(all_sysmon_df), traced_df)
        except Exception as e:
            self.logger.error(f"❌ Error reading CSV files: {e}")
            return None
        
        # Create single plot for all events
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))

        # PLOT DIRECTLY FROM TRACED CSV (includes all multi-originator associations)
        # No need to match back to sysmon dataset - traced CSV has everything we need

        # Convert timestamps to datetime for plotting
        self.logger.info("🎨 Preparing traced events for plotting...")
        try:
            traced_df['datetime'] = pd.to_datetime(traced_df['timestamp'], unit='ms', errors='coerce')
            # Remove any invalid timestamps
            invalid_timestamps = traced_df['datetime'].isna().sum()
            if invalid_timestamps > 0:
                self.logger.warning(f"⚠️ Removed {invalid_timestamps} traced events with invalid timestamps")
                traced_df = traced_df.dropna(subset=['datetime'])
        except Exception as e:
            self.logger.error(f"❌ Error converting timestamps: {e}")
            return None

        # Also plot benign events as background from original sysmon dataset
        self.logger.info("🎨 Plotting all Sysmon events as background...")
        try:
            all_sysmon_df['datetime'] = pd.to_datetime(all_sysmon_df['timestamp'], unit='ms', errors='coerce')
            all_sysmon_df = all_sysmon_df.dropna(subset=['datetime'])
        except Exception as e:
            self.logger.error(f"❌ Error converting timestamps: {e}")
            return None

        # Plot ALL sysmon events as pale gray background
        ax.scatter(all_sysmon_df['datetime'], all_sysmon_df['EventID'],
                  c='#d0d0d0', alpha=0.4, s=20,
                  label=f'Benign Events ({len(all_sysmon_df):,})', zorder=1)
        self.logger.info(f"📊 Plotted {len(all_sysmon_df):,} total Sysmon events as background")

        # Organize traced malicious events by tactic (directly from CSV)
        tactics_events = {}
        for tactic in traced_df['Tactic'].unique():
            if pd.notna(tactic):  # Skip NaN tactics
                tactics_events[tactic] = traced_df[traced_df['Tactic'] == tactic]
        
        if not tactics_events:
            self.logger.warning("⚠️ No malicious tactics found for timeline")
            return None
        
        # Define tactic colors (diverse, high-contrast palette)
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
        
        # Plot malicious events by tactic (highlighting them over the gray background)
        # Sort by event count (descending) so larger groups are plotted first, smaller groups appear on top
        total_malicious_events = 0

        # Sort tactics by count (largest first) so smaller groups appear on top
        tactics_sorted_by_count = sorted(tactics_events.keys(),
                                         key=lambda t: len(tactics_events[t]),
                                         reverse=True)

        for idx, tactic in enumerate(tactics_sorted_by_count):
            tactic_df = tactics_events[tactic]

            if len(tactic_df) == 0:
                continue

            # Get color for this tactic
            tactic_color = tactic_colors.get(tactic, '#000000')

            # Assign z-order: larger groups get lower z-order (appear behind)
            # smaller groups get higher z-order (appear on top)
            # Formula: smallest count gets highest z-order
            zorder_value = 2 + idx

            # Create scatter plot for this tactic
            scatter = ax.scatter(tactic_df['datetime'], tactic_df['EventID'],
                               c=tactic_color, alpha=0.8, s=60,
                               label=f'{tactic.title()} ({len(tactic_df)} events)',
                               zorder=zorder_value)

            total_malicious_events += len(tactic_df)
            self.logger.info(f"📊 Plotted {len(tactic_df)} events for tactic: {tactic} (z-order: {zorder_value})")
        
        self.logger.info(f"📊 Total malicious events plotted: {total_malicious_events}")
        
        # Customize single plot
        ax.set_ylabel('EventID')
        ax.set_xlabel('Timeline')
        ax.grid(True, alpha=0.3)
        
        # Set y-axis ticks based on actual EventIDs present in the data
        all_eventids = set(all_sysmon_df['EventID'].unique())
        if len(traced_df) > 0:
            all_eventids.update(traced_df['EventID'].unique())
        
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
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.ConciseDateFormatter(plt.matplotlib.dates.AutoDateLocator()))
        ax.xaxis.set_major_locator(plt.matplotlib.dates.AutoDateLocator())
        plt.setp(ax.xaxis.get_ticklabels(), rotation=45, ha='right')
        
        # Set main title with event counts
        total_events = len(all_sysmon_df)
        total_malicious = len(traced_df)  # All traced events from CSV
        benign_count = total_events - len(traced_df['mapping_key'].unique()) if 'mapping_key' in traced_df.columns else total_events
        ax.set_title(f'Complete Sysmon Timeline with MITRE Tactics Highlighting\n'
                    f'Total Events: {total_events:,} | Malicious: {total_malicious:,} | Benign: ~{benign_count:,}',
                    fontsize=14, fontweight='bold')
        
        # Add legend to the plot
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0),
                 title='MITRE Tactics', title_fontsize=12, fontsize=10)
        
        plt.tight_layout()
        
        # Save plot
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "timeline_all_malicious_events_with_tactics.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"📊 Attack events tactics timeline saved: {output_file}")
        return output_file
    
    # ==================== V2 METHODS USING SHARED UTILITIES ====================
    
    def create_group_timeline_plot_v2(self, output_dir: Path) -> Optional[Path]:
        """Create group timeline plot using shared plotting utilities (v2)."""
        if not SHARED_UTILS_AVAILABLE or not FeatureFlags.USE_SHARED_PLOTTING:
            return self.create_group_timeline_plot(output_dir)  # Fallback to original
        
        if not self.all_traced_events:
            self.logger.warning("⚠️ No traced attack events for group timeline (v2)")
            return None
        
        self.logger.info("📊 Creating attack events group timeline plot (v2 - shared utilities)...")
        
        # Prepare malicious events DataFrame
        all_events = []
        for originator_row, traced_data in self.traced_events.items():
            tactic = traced_data.get('tactic_label', '')
            technique = traced_data.get('technique_label', '')
            event_id = traced_data.get('event_id', 1)
            
            if event_id == 1:
                # EventID 1: Process events with complex structure
                for event in traced_data.get('all_events', []):
                    event_dict = dict(event)
                    event_dict['Tactic'] = tactic
                    event_dict['Technique'] = technique
                    all_events.append(event_dict)
            else:
                # EventID 11/23: File events with simpler structure
                for event in traced_data.get('all_events', []):
                    event_dict = dict(event)
                    event_dict['Tactic'] = tactic
                    event_dict['Technique'] = technique
                    all_events.append(event_dict)
        
        if not all_events:
            self.logger.warning("⚠️ No events to plot (v2)")
            return None
        
        malicious_df = pd.DataFrame(all_events)
        
        # Use shared plotting utility
        output_file = output_dir / PlottingConfig.TIMELINE_SIMPLE
        success = plot_simple_timeline_v1(malicious_df, output_dir, self.logger)
        
        return output_file if success else None
    
    def create_tactics_timeline_plot_v2(self, output_dir: Path) -> Optional[Path]:
        """Create tactics timeline plot using shared plotting utilities (v2)."""
        if not SHARED_UTILS_AVAILABLE or not FeatureFlags.USE_SHARED_PLOTTING:
            return self.create_tactics_timeline_plot(output_dir)  # Fallback to original
        
        # Read tactics information from the CSV file
        csv_file = output_dir / "traced_sysmon_events_with_tactics.csv"
        if not csv_file.exists():
            self.logger.warning(f"⚠️ Tactics CSV file not found: {csv_file} (v2)")
            return None
        
        self.logger.info("📊 Creating complete Sysmon timeline with tactics highlighting (v2 - shared utilities)...")
        
        # Get APT run directory (parent of output directory)
        apt_run_dir = output_dir.parent
        
        # Read original sysmon dataset directly
        run_dir_name = apt_run_dir.name  # e.g., "apt-1-run-04"
        run_id = run_dir_name.split('-run-')[1]    # e.g., "04"
        original_sysmon_file = apt_run_dir / f"sysmon-run-{run_id}.csv"
        
        if not original_sysmon_file.exists():
            self.logger.error(f"❌ Original Sysmon file not found: {original_sysmon_file} (v2)")
            return None
            
        try:
            # Read original sysmon dataset
            all_sysmon_df = pd.read_csv(original_sysmon_file)
            self.logger.info(f"📖 Read {len(all_sysmon_df):,} total Sysmon events from original dataset (v2)")
            
            # Read traced malicious events 
            traced_df = pd.read_csv(csv_file)
            self.logger.info(f"📖 Read {len(traced_df)} traced malicious events from CSV (v2)")
            
            # Generate basic labeling statistics
            self._generate_basic_labeling_stats(len(all_sysmon_df), traced_df)
            
        except Exception as e:
            self.logger.error(f"❌ Error reading CSV files (v2): {e}")
            return None
        
        # Create mapping key function for matching traced events to original dataset
        def create_mapping_key(row):
            """Create robust unique key matching deduplication logic."""
            def safe_str(value):
                return str(value) if pd.notna(value) else 'NULL'

            event_id = safe_str(row.get('EventID', ''))
            computer = safe_str(row.get('Computer', ''))
            timestamp = safe_str(row.get('timestamp', ''))
            process_guid = safe_str(row.get('ProcessGuid', ''))
            process_id = safe_str(row.get('ProcessId', ''))
            image = safe_str(row.get('Image', ''))

            # Event-specific unique identifiers matching deduplication logic
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

        # Add default labels to all events
        all_sysmon_df['Label'] = 'Benign'
        all_sysmon_df['Tactic'] = 'Benign'
        
        # Create mapping keys for both datasets
        all_sysmon_df['mapping_key'] = all_sysmon_df.apply(create_mapping_key, axis=1)
        traced_df['mapping_key'] = traced_df.apply(create_mapping_key, axis=1)
        
        # Apply malicious labels to matching events
        malicious_mapping = traced_df.set_index('mapping_key')['Tactic'].to_dict()
        
        for mapping_key, tactic in malicious_mapping.items():
            matching_mask = all_sysmon_df['mapping_key'] == mapping_key
            if matching_mask.any():
                all_sysmon_df.loc[matching_mask, 'Label'] = 'Malicious'
                all_sysmon_df.loc[matching_mask, 'Tactic'] = tactic
        
        # Remove temporary mapping key column
        all_sysmon_df.drop('mapping_key', axis=1, inplace=True)
        
        # Use shared plotting utility
        output_file = output_dir / PlottingConfig.TIMELINE_TACTICS
        success = plot_tactics_timeline_v1(all_sysmon_df, output_dir, self.logger)
        
        return output_file if success else None
    
    # ==================== END V2 METHODS ====================
    
    def export_traced_events_to_csv(self, output_dir: Path) -> Optional[Path]:
        """Export all traced Sysmon events to CSV with Tactic/Technique columns."""
        if not self.traced_events:
            self.logger.warning("⚠️ No traced events to export to CSV")
            return None
        
        self.logger.info("📄 Exporting traced Sysmon events to CSV...")
        
        all_events = []
        
        # Process each traced originator
        for originator_row, traced_data in self.traced_events.items():
            try:
                # Get tactic and technique for this originator
                tactic = traced_data.get('tactic_label', '')
                technique = traced_data.get('technique_label', '')
                event_id = traced_data.get('event_id', 1)
                
                if event_id == 1:
                    # EventID 1: Process events with complex structure
                    for event in traced_data.get('all_events', []):
                        # Convert event to dict and add metadata
                        event_dict = dict(event)
                        event_dict['Tactic'] = tactic
                        event_dict['Technique'] = technique
                        event_dict['OriginatorRow'] = originator_row
                        all_events.append(event_dict)
                        
                elif event_id in [11, 23]:
                    # EventID 11/23: File events with simple structure
                    for event in traced_data.get('traced_events', []):
                        # Convert file event to standard Sysmon format
                        event_dict = {
                            'EventID': event_id,
                            'Computer': event.get('Computer', ''),  # Fixed: uppercase field name
                            'timestamp': event.get('timestamp', ''),
                            'Image': event.get('Image', ''),  # Fixed: uppercase field name
                            'TargetFilename': event.get('TargetFilename', ''),  # Fixed: uppercase field name
                            'Tactic': tactic,
                            'Technique': technique,
                            'OriginatorRow': originator_row
                        }
                        
                        # Add any other fields that might be present
                        for key, value in event.items():
                            if key not in event_dict:
                                event_dict[key] = value
                        
                        all_events.append(event_dict)
                        
            except Exception as e:
                self.logger.error(f"❌ Error processing events for Row {originator_row}: {e}")
                if self.debug:
                    import traceback
                    self.logger.error(traceback.format_exc())
        
        if not all_events:
            self.logger.warning("⚠️ No events collected for CSV export")
            return None

        # NO DEDUPLICATION - keep all traced events including multi-originator associations
        # Script #7 will handle corrections via Correct_SeedRowNumber column
        self.logger.info(f"📊 Exporting {len(all_events)} traced events (no deduplication)")

        # Update statistics
        self.stats['total_traced_events'] = len(all_events)

        # Calculate events_by_computer and events_by_eventid
        self.stats['events_by_computer'] = {}
        self.stats['events_by_eventid'] = {}
        for event in all_events:
            computer = event.get('Computer', 'Unknown')
            event_id = event.get('EventID', 'Unknown')

            if computer not in self.stats['events_by_computer']:
                self.stats['events_by_computer'][computer] = 0
            self.stats['events_by_computer'][computer] += 1

            if event_id not in self.stats['events_by_eventid']:
                self.stats['events_by_eventid'][event_id] = 0
            self.stats['events_by_eventid'][event_id] += 1
        
        # Convert to DataFrame
        events_df = pd.DataFrame(all_events)
        
        # Convert timestamp to datetime for sorting (handle both formats)
        try:
            # Try millisecond timestamp format first
            events_df['timestamp_dt'] = pd.to_datetime(events_df['timestamp'], unit='ms', errors='coerce')
            # If that fails, try string format
            mask = events_df['timestamp_dt'].isna()
            if mask.any():
                events_df.loc[mask, 'timestamp_dt'] = pd.to_datetime(events_df.loc[mask, 'timestamp'], errors='coerce')
        except Exception as e:
            self.logger.warning(f"⚠️ Timestamp conversion warning: {e}")
            # Fallback to string sorting
            events_df['timestamp_dt'] = events_df['timestamp']
        
        # Sort by timestamp (oldest to newest), then by OriginatorRow
        events_df = events_df.sort_values(['timestamp_dt', 'OriginatorRow'])
        
        # Drop the helper columns
        events_df = events_df.drop(columns=['timestamp_dt', '_sysmon_row_index'], errors='ignore')
        
        # Add human-readable timestamp column while preserving original Unix milliseconds
        if 'timestamp' in events_df.columns and events_df['timestamp'].notna().any():
            # Handle Unix timestamps in milliseconds, filtering out negative values
            valid_mask = events_df['timestamp'] > 0
            events_df['timestamp_h'] = ''

            # Convert valid timestamps to human-readable format
            events_df.loc[valid_mask, 'timestamp_h'] = pd.to_datetime(
                events_df.loc[valid_mask, 'timestamp'],
                unit='ms',
                errors='coerce'
            ).dt.strftime('%Y-%m-%d %H:%M:%S.%f').str[:-3]

            # Handle negative/invalid timestamps
            invalid_count = (~valid_mask).sum()
            if invalid_count > 0:
                events_df.loc[~valid_mask, 'timestamp_h'] = 'INVALID_TIMESTAMP'
                self.logger.warning(f"⚠️ Found {invalid_count} invalid/negative timestamps, marked as INVALID_TIMESTAMP")

            self.logger.info("🕒 Created human-readable timestamp column (timestamp_h) with millisecond precision")

        # Add manual reassignment column
        events_df['Correct_SeedRowNumber'] = ''  # Empty for manual filling
        
        # Reorder columns to specified sequence
        specified_columns = [
            'Tactic', 'Technique', 'OriginatorRow', 'Correct_SeedRowNumber', 'EventID', 'Computer',
            'timestamp_h', 'CommandLine', 'TargetFilename', 'ParentCommandLine', 'ProcessGuid',
            'ParentProcessGuid', 'ProcessId', 'ParentProcessId'
        ]
        # Only include columns that exist in the DataFrame
        available_specified_columns = [col for col in specified_columns if col in events_df.columns]
        remaining_columns = [col for col in events_df.columns if col not in specified_columns]
        new_column_order = available_specified_columns + remaining_columns
        events_df = events_df[new_column_order]
        
        # Save to CSV
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_file = output_dir / "traced_sysmon_events_with_tactics.csv"
        
        try:
            events_df.to_csv(csv_file, index=False)
            self.logger.info(f"📄 Traced Sysmon events exported to CSV: {csv_file}")
            self.logger.info(f"   📊 Total events exported: {len(events_df)}")
            self.logger.info(f"   📋 EventIDs included: {sorted(events_df['EventID'].unique())}")
            return csv_file
            
        except Exception as e:
            self.logger.error(f"❌ Error saving CSV file: {e}")
            return None
    
    def save_analysis_results(self, output_dir: Path) -> bool:
        """Save analysis results to JSON file."""
        if not self.traced_events:
            self.logger.warning("⚠️ No traced events to save")
            return False
        
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            results_file = output_dir / "multi_eventid_analysis_results.json"
            
            # Count unique originators (avoid Seed_Event+Tactic duplication)
            unique_originators = len(set(
                traced_data.get('originator_row', traced_data.get('analysis_details', {}).get('originator_row', 0)) 
                for traced_data in self.traced_events.values()
            ))
            
            # Build results structure
            results = {
                'metadata': {
                    'script_name': '6_sysmon_attack_lifecycle_tracer.py',
                    'analysis_type': 'Multi-EventID Attack Lifecycle Analysis (EventID 1, 11, 23)',
                    'processing_timestamp': datetime.now().isoformat(),
                    'total_originators_processed': unique_originators,
                    'focus': 'EventID 1 (Process Creation), EventID 11 (File Create), EventID 23 (File Delete)'
                },
                'tracing_results': self._build_tracing_results(),
                'statistics': self.stats,
                'labeling_statistics': self.labeling_stats if hasattr(self, 'labeling_stats') and self.labeling_stats else None
            }
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self.logger.info(f"💾 Multi-EventID analysis results saved: {results_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error saving multi-EventID analysis results: {e}")
            return False
    
    def _build_tracing_results(self) -> Dict:
        """Build tracing results for JSON output."""
        results = {}
        
        for row, traced_data in self.traced_events.items():
            try:
                # Determine EventID and get appropriate data structure
                if 'analysis_details' in traced_data:
                    # EventID 1 (complex) data structure
                    event_id = traced_data['analysis_details']['event_id']
                    originator_row = traced_data['analysis_details']['originator_row']
                    timestamp = traced_data['analysis_details']['timestamp']
                    
                    results[str(row)] = {
                        'analysis_details': {
                            'originator_row': int(originator_row),
                            'event_id': int(event_id),
                            'processing_type': traced_data.get('processing_type', 'Seed_Event'),
                            'tactic_label': traced_data.get('tactic_label', ''),
                            'technique_label': traced_data.get('technique_label', ''),
                            'timestamp': str(timestamp)
                        },
                        'event_counts': {
                            'total_events': len(traced_data['all_events']),
                            'spawned_events': len(traced_data['spawned_events']),
                            'child_processes': len(traced_data['child_process_create_events']),
                            'injection_events': len(traced_data['eight_and_ten_events'])
                        },
                        'computers_affected': list(set(str(event['Computer']) for event in traced_data['all_events'])),
                        'event_ids_found': list(set(int(event['EventID']) for event in traced_data['all_events'])),
                        'command_line': str(traced_data['originator_event'].get('CommandLine', ''))
                    }
                else:
                    # EventID 11/23 (simple) data structure
                    event_id = traced_data['event_id']
                    originator_row = traced_data['originator_row']
                    
                    # Get first traced event for details
                    file_event = traced_data['traced_events'][0] if traced_data['traced_events'] else {}
                    
                    results[str(row)] = {
                        'analysis_details': {
                            'originator_row': int(originator_row),
                            'event_id': int(event_id),
                            'processing_type': traced_data.get('processing_type', 'Seed_Event'),
                            'tactic_label': traced_data.get('tactic_label', ''),
                            'technique_label': traced_data.get('technique_label', ''),
                            'timestamp': str(file_event.get('timestamp', ''))
                        },
                        'file_details': {
                            'target_filename': str(file_event.get('target_filename', '')),
                            'process_image': str(file_event.get('image', '')),
                            'computer': str(file_event.get('Computer', ''))
                        },
                        'event_counts': {
                            'total_events': len(traced_data['traced_events'])
                        },
                        'event_type': 'File Create' if event_id == 11 else 'File Delete'
                    }
                
            except Exception as e:
                self.logger.error(f"❌ Error building JSON results for Row {row}: {e}")
        
        return results
    
    def print_summary(self) -> None:
        """Print analysis summary."""
        self.logger.info("📋 MULTI-EVENTID ATTACK LIFECYCLE ANALYSIS SUMMARY")
        self.logger.info("=" * 70)
        
        # EventID breakdown
        self.logger.info(f"📊 Selected Attack Events by Type:")
        if self.stats['eventid1_selected'] > 0:
            self.logger.info(f"   EventID 1 (Process Creation): {self.stats['eventid1_selected']}")
        if self.stats['eventid11_selected'] > 0:
            self.logger.info(f"   EventID 11 (File Create): {self.stats['eventid11_selected']}")
        if self.stats['eventid23_selected'] > 0:
            self.logger.info(f"   EventID 23 (File Delete): {self.stats['eventid23_selected']}")
        
        self.logger.info(f"Total originators: {self.stats['total_originators_selected']}")
        self.logger.info(f"Successfully traced: {self.stats['successfully_traced']}")
        self.logger.info(f"Total traced events: {self.stats['total_traced_events']:,} (post-deduplication)")
        if 'events_removed_by_deduplication' in self.stats and self.stats['events_removed_by_deduplication'] > 0:
            self.logger.info(f"Events removed by deduplication: {self.stats['events_removed_by_deduplication']:,} ({self.stats['deduplication_rate_percent']}%)")
        self.logger.info(f"Computers affected: {len(self.stats['events_by_computer'])}")
        
        if self.stats['events_by_computer']:
            self.logger.info("📊 Events by Computer:")
            for computer, count in sorted(self.stats['events_by_computer'].items()):
                short_name = computer.replace('.boombox.local', '').replace('.local', '')
                self.logger.info(f"   {short_name}: {count:,} events")
        
        if self.stats['events_by_eventid']:
            self.logger.info("📊 Events by EventID:")
            for event_id, count in sorted(self.stats['events_by_eventid'].items()):
                self.logger.info(f"   EventID {event_id}: {count:,} events")
        
        duration = self.stats['processing_duration_seconds']
        self.logger.info(f"⏱️ Processing duration: {duration:.2f} seconds")
        self.logger.info("=" * 70)


def create_apt_file_paths(apt_type: str, run_id: str) -> Tuple[Path, Path]:
    """Create file paths for APT dataset files."""
    # Use the correct dataset directory structure: dataset/{apt_type}/{apt_type}-run-{run_id}/
    # We're in scripts/exploratory/, go up to project root then to dataset
    scripts_root = Path(__file__).parent.parent  # Go up to scripts/
    project_root = scripts_root.parent  # Go to research/
    dataset_root = project_root / "dataset"  # Point to dataset folder
    apt_dir = dataset_root / apt_type / f"{apt_type}-run-{run_id:0>2}"
    
    # Sysmon file patterns
    sysmon_patterns = [
        f"sysmon-run-{run_id:0>2}.csv",
        f"sysmon-run-{run_id:0>2}-OLD.csv",
        f"sysmon-{apt_type}-run-{run_id:0>2}.csv"
    ]
    
    sysmon_file = None
    for pattern in sysmon_patterns:
        candidate = apt_dir / pattern
        if candidate.exists():
            sysmon_file = candidate
            break
    
    # Originators file (all_target_events) - in the apt run directory
    originators_file = apt_dir / f"all_target_events_run-{run_id:0>2}.csv"
    
    return sysmon_file, originators_file


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='EventID 1 Attack Lifecycle Tracer - Process-Centric Analysis Only'
    )
    
    # APT dataset options
    parser.add_argument('--apt-type', choices=['apt-1', 'apt-2', 'apt-3', 'apt-4', 'apt-5', 'apt-6'],
                       help='APT dataset type')
    parser.add_argument('--run-id', help='APT run ID (e.g., 04, 15)')
    
    # Direct file options
    parser.add_argument('--sysmon-csv', type=Path, help='Path to Sysmon CSV file')
    parser.add_argument('--originators-csv', type=Path, help='Path to all_target_events CSV file')
    
    # Output options
    parser.add_argument('--output-dir', type=Path, default=Path('./sysmon_event_tracing_analysis_results'),
                       help='Output directory for results (default: ./sysmon_event_tracing_analysis_results)')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.apt_type and args.run_id:
        sysmon_file, originators_file = create_apt_file_paths(args.apt_type, args.run_id)
        
        # Set default output directory to APT run folder if not explicitly specified
        if args.output_dir == Path('./sysmon_event_tracing_analysis_results'):
            apt_run_dir = sysmon_file.parent if sysmon_file else originators_file.parent
            args.output_dir = apt_run_dir / 'sysmon_event_tracing_analysis_results'
        
        if not sysmon_file or not sysmon_file.exists():
            print(f"❌ Error: Sysmon file not found for {args.apt_type} run {args.run_id}")
            return 1
        
        if not originators_file.exists():
            print(f"❌ Error: Originators file not found: {originators_file}")
            return 1
            
    elif args.sysmon_csv and args.originators_csv:
        sysmon_file = args.sysmon_csv
        originators_file = args.originators_csv
        
        if not sysmon_file.exists():
            print(f"❌ Error: Sysmon file not found: {sysmon_file}")
            return 1
        
        if not originators_file.exists():
            print(f"❌ Error: Originators file not found: {originators_file}")
            return 1
    else:
        print("❌ Error: Must provide either --apt-type/--run-id or --sysmon-csv/--originators-csv")
        parser.print_help()
        return 1
    
    # Initialize tracer
    tracer = MultiEventIDAttackLifecycleTracer(debug=args.debug)
    
    print("🚀 MULTI-EVENTID ATTACK LIFECYCLE TRACER")
    print("=" * 50)
    print(f"📊 Sysmon file: {sysmon_file}")
    print(f"📋 Originators file: {originators_file}")
    print(f"📁 Output directory: {args.output_dir}")
    print("=" * 50)
    
    try:
        # Load data
        if not tracer.load_selected_originators(originators_file):
            return 1
        
        if not tracer.load_sysmon_data(sysmon_file):
            return 1
        
        # Process originators
        if not tracer.process_all_originators():
            return 1
        
        # Create visualizations
        plot_files = tracer.create_individual_timeline_plots(args.output_dir)
        group_plot = tracer.create_group_timeline_plot(args.output_dir)
        
        # Export traced events to CSV FIRST (required by tactics timeline)
        csv_file = tracer.export_traced_events_to_csv(args.output_dir)
        
        # Create tactics timeline plot (requires CSV to exist)
        tactics_plot = tracer.create_tactics_timeline_plot(args.output_dir)
        
        # Save results
        tracer.save_analysis_results(args.output_dir)
        
        # Print summary
        tracer.print_summary()
        
        print(f"✅ Multi-EventID attack analysis completed successfully!")
        print(f"📁 Results saved to: {args.output_dir}")
        print(f"📊 Individual plots: {len(plot_files)}")
        print(f"📊 Group plot: {'✅' if group_plot else '❌'}")
        print(f"📊 Tactics plot: {'✅' if tactics_plot else '❌'}")
        print(f"📄 CSV export: {'✅' if csv_file else '❌'}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️ Analysis interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())