#!/usr/bin/env python3
"""
SIMPLE Event Extractor - APT Attack Analysis

Extract all EventID 1, 11, 23 events from Sysmon data for manual selection.
No keyword filtering - just raw events with selection columns.

Features:
- Extracts all process execution (EventID 1) events
- Extracts all file creation (EventID 11) events  
- Extracts all file deletion (EventID 23) events
- Adds Seed_Event/Tactic/Technique selection columns for manual marking
- Preserves original row numbers for traceability
- Chronological ordering for attack timeline analysis

Usage:
    python3 SIMPLE_event_extractor.py --apt-type apt-1 --run-id 04
    python3 SIMPLE_event_extractor.py --sysmon-csv /path/to/sysmon.csv
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

try:
    import pandas as pd
except ImportError as e:
    print(f"❌ Error: Required library not installed: {e}")
    print("   Install with: pip install pandas")
    sys.exit(1)


class SimpleEventExtractor:
    """
    Simple extractor for all target EventID events from Sysmon data.
    
    No filtering, no keyword matching - just extract everything for manual review.
    """
    
    # Target EventIDs for attack analysis
    TARGET_EVENT_IDS = {1, 11, 23}
    
    # APT run ranges per dataset type
    APT_RUN_RANGES = {
        'apt-1': list(range(1, 21)) + [51],  # 01-20, 51
        'apt-2': list(range(21, 31)),        # 21-30
        'apt-3': list(range(31, 39)),        # 31-38
        'apt-4': list(range(39, 45)),        # 39-44
        'apt-5': list(range(45, 48)),        # 45-47
        'apt-6': list(range(48, 51)),        # 48-50
        'apt-7': [52],                       # 52 (test dataset)
    }
    
    def __init__(self, debug: bool = False):
        """Initialize the simple event extractor."""
        self.debug = debug
        self.logger = self._setup_logging()
        
        # Path configuration
        self.scripts_dir = Path(__file__).parent.parent  # Go up to scripts/
        self.project_root = self.scripts_dir.parent  # Go to research/
        self.dataset_dir = self.project_root / "dataset"  # Point to dataset folder
        
        # Statistics
        self.stats = {
            'total_sysmon_events': 0,
            'target_events_found': 0,
            'events_by_eventid': {1: 0, 11: 0, 23: 0},
        }
        
        self.extracted_events = []
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger('SimpleEventExtractor')
        logger.setLevel(logging.DEBUG if self.debug else logging.INFO)
        
        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Create console handler
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG if self.debug else logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
        return logger
    
    def find_sysmon_file(self, apt_dir: Path, run_id: str) -> Optional[Path]:
        """
        Find raw Sysmon CSV file using multiple naming patterns.
        
        Handles various naming conventions:
        - Standard: sysmon-run-XX.csv
        - OLD suffix: sysmon-run-XX-OLD.csv
        - Organized structure: 02_data_processing/processed_data/sysmon-run-XX.csv
        """
        run_id_padded = f"{int(run_id):02d}"  # Ensure 2-digit format
        
        # Try multiple file patterns
        patterns = [
            apt_dir / f"sysmon-run-{run_id_padded}.csv",
            apt_dir / f"sysmon-run-{run_id_padded}-OLD.csv",
            apt_dir / "02_data_processing" / "processed_data" / f"sysmon-run-{run_id_padded}.csv",
            # Try single digit for legacy compatibility
            apt_dir / f"sysmon-run-{run_id}.csv",
            apt_dir / f"sysmon-run-{run_id}-OLD.csv",
        ]
        
        for file_path in patterns:
            if file_path.exists():
                self.logger.info(f"📄 Found Sysmon file: {file_path}")
                return file_path
        
        # Try glob patterns as fallback
        glob_patterns = [
            apt_dir.glob(f"sysmon*{run_id_padded}*.csv"),
            apt_dir.glob(f"sysmon*{run_id}*.csv")
        ]
        
        for glob_pattern in glob_patterns:
            files = list(glob_pattern)
            if files:
                self.logger.info(f"📄 Found Sysmon file via glob: {files[0]}")
                return files[0]  # Return first match
        
        return None
    
    def extract_target_events(self, sysmon_file: Path) -> bool:
        """
        Extract all target EventID events from Sysmon CSV file.
        
        Args:
            sysmon_file: Path to raw sysmon-run-XX.csv file
            
        Returns:
            True if extraction successful, False otherwise
        """
        if not sysmon_file.exists():
            self.logger.error(f"❌ Sysmon file not found: {sysmon_file}")
            return False
        
        self.logger.info(f"📊 Extracting target events from: {sysmon_file}")
        
        try:
            # Read the raw Sysmon CSV
            self.logger.info("📊 Loading Sysmon dataset...")
            df = pd.read_csv(sysmon_file, low_memory=False)
            self.stats['total_sysmon_events'] = len(df)
            
            self.logger.info(f"📊 Loaded {len(df):,} total Sysmon events")
            
            # Validate required columns
            if 'EventID' not in df.columns:
                self.logger.error(f"❌ EventID column not found in {sysmon_file}")
                return False
            
            # Filter for target EventIDs first
            target_events = df[df['EventID'].isin(self.TARGET_EVENT_IDS)].copy()
            self.stats['target_events_found'] = len(target_events)
            
            if len(target_events) == 0:
                self.logger.warning(f"⚠️ No target EventIDs (1, 11, 23) found in {sysmon_file}")
                return False
            
            self.logger.info(f"🎯 Found {len(target_events):,} target events (EventID 1, 11, 23)")
            
            # Add original row numbers (1-indexed like CSV readers)
            target_events.insert(0, 'RawDatasetRowNumber', target_events.index + 2)  # +2 for header and 1-indexing
            
            # Add selection columns at the beginning
            target_events.insert(0, 'Seed_Event', '')
            target_events.insert(1, 'Tactic', '')
            target_events.insert(2, 'Technique', '')
            
            # Ensure required columns exist
            optional_columns = ['timestamp', 'Computer', 'CommandLine', 'TargetFilename', 'ProcessGuid', 'ProcessId', 'ParentProcessGuid', 'ParentProcessId', 'Image', 'ParentImage']
            
            for col in optional_columns:
                if col not in target_events.columns:
                    target_events[col] = None
                    if self.debug:
                        self.logger.debug(f"⚠️ Column {col} not found, will use null values")
            
            # Create human-readable timestamp column while preserving original Unix milliseconds
            if 'timestamp' in target_events.columns and target_events['timestamp'].notna().any():
                # Handle Unix timestamps in milliseconds, filtering out negative values
                valid_mask = target_events['timestamp'] > 0
                target_events['timestamp_h'] = ''

                # Convert valid timestamps to human-readable format
                target_events.loc[valid_mask, 'timestamp_h'] = pd.to_datetime(
                    target_events.loc[valid_mask, 'timestamp'],
                    unit='ms',
                    errors='coerce'
                ).dt.strftime('%Y-%m-%d %H:%M:%S.%f').str[:-3]

                # Handle negative/invalid timestamps
                invalid_count = (~valid_mask).sum()
                if invalid_count > 0:
                    target_events.loc[~valid_mask, 'timestamp_h'] = 'INVALID_TIMESTAMP'
                    self.logger.warning(f"⚠️ Found {invalid_count} invalid/negative timestamps, marked as INVALID_TIMESTAMP")

                self.logger.info("🕒 Created human-readable timestamp column (timestamp_h) with millisecond precision")

            # Sort by timestamp for chronological analysis
            if 'timestamp' in target_events.columns:
                target_events = target_events.sort_values('timestamp')
                self.logger.info("📅 Events sorted chronologically by timestamp")
            
            # Count events by EventID
            for event_id in self.TARGET_EVENT_IDS:
                count = len(target_events[target_events['EventID'] == event_id])
                self.stats['events_by_eventid'][event_id] = count
            
            # Store extracted events
            self.extracted_events = target_events.to_dict('records')
            
            self.logger.info(f"✅ Extracted {len(self.extracted_events)} target events")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error extracting events: {e}")
            import traceback
            if self.debug:
                traceback.print_exc()
            return False
    
    def save_events(self, output_file: Path) -> bool:
        """
        Save extracted events to CSV file for manual selection.
        
        Args:
            output_file: Path for output CSV file
            
        Returns:
            True if saved successfully, False otherwise
        """
        if not self.extracted_events:
            self.logger.warning("⚠️ No extracted events to save")
            return False
        
        try:
            # Convert to DataFrame
            events_df = pd.DataFrame(self.extracted_events)
            
            # Select and order key columns for easy review with new column order
            output_columns = [
                'Seed_Event', 'Tactic', 'Technique', 'RawDatasetRowNumber', 'timestamp_h', 'EventID', 'Computer',
                'CommandLine', 'TargetFilename', 'ProcessGuid', 'ProcessId', 'ParentProcessGuid', 'ParentProcessId',
                'Image', 'ParentImage', 'timestamp'
            ]
            
            # Only include columns that exist
            available_columns = [col for col in output_columns if col in events_df.columns]
            events_df = events_df[available_columns]
            
            # Preserve any existing selections if file already exists
            existing_selections = {}
            if output_file.exists():
                try:
                    existing_df = pd.read_csv(output_file)
                    if 'RawDatasetRowNumber' in existing_df.columns:
                        for _, row in existing_df.iterrows():
                            row_num = row['RawDatasetRowNumber']
                            seed_event_val = row.get('Seed_Event', '')
                            tactic_val = row.get('Tactic', '')
                            technique_val = row.get('Technique', '')
                            if seed_event_val or tactic_val or technique_val:
                                existing_selections[row_num] = (seed_event_val, tactic_val, technique_val)
                        
                        self.logger.info(f"📋 Found existing file, preserving {len(existing_selections)} selections...")
                except Exception as e:
                    self.logger.warning(f"⚠️ Could not read existing selections: {e}")
            
            # Apply existing selections
            preserved_count = 0
            if existing_selections:
                for idx, row in events_df.iterrows():
                    row_num = row['RawDatasetRowNumber']
                    if row_num in existing_selections:
                        seed_event_val, tactic_val, technique_val = existing_selections[row_num]
                        events_df.loc[idx, 'Seed_Event'] = seed_event_val
                        events_df.loc[idx, 'Tactic'] = tactic_val
                        events_df.loc[idx, 'Technique'] = technique_val
                        preserved_count += 1
                
                if preserved_count > 0:
                    self.logger.info(f"   ✅ Preserved {preserved_count} existing selections")
            
            # Save to CSV
            events_df.to_csv(output_file, index=False)
            
            # Count selections
            seed_event_count = len(events_df[events_df['Seed_Event'].str.strip() != ''])
            tactic_count = len(events_df[events_df['Tactic'].str.strip() != ''])
            technique_count = len(events_df[events_df['Technique'].str.strip() != ''])
            total_selected = len(events_df[(events_df['Seed_Event'].str.strip() != '') | (events_df['Tactic'].str.strip() != '') | (events_df['Technique'].str.strip() != '')])
            unselected_count = len(events_df) - total_selected
            
            self.logger.info(f"💾 Saved {len(events_df)} events to: {output_file}")
            self.logger.info(f"📋 MANUAL SELECTION WORKFLOW:")
            self.logger.info(f"   File: {output_file}")
            self.logger.info(f"   🌱 Seed Event selections: {seed_event_count}")
            self.logger.info(f"   🎯 Tactic selections: {tactic_count}")
            self.logger.info(f"   🔧 Technique selections: {technique_count}")
            self.logger.info(f"   ✅ Total selected: {total_selected} events")
            self.logger.info(f"   ⭕ Unselected: {unselected_count} events")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error saving events: {e}")
            import traceback
            if self.debug:
                traceback.print_exc()
            return False
    
    def process_apt_run(self, apt_type: str, run_id: str) -> bool:
        """
        Process a specific APT run to extract all target events.
        
        Args:
            apt_type: APT dataset type (apt-1, apt-2, etc.)
            run_id: Run identifier (04, 05, etc.)
            
        Returns:
            True if processing successful, False otherwise
        """
        self.logger.info(f"🎯 Simple event extraction for {apt_type}-run-{run_id}")
        
        # Validate APT type
        if apt_type not in self.APT_RUN_RANGES:
            self.logger.error(f"❌ Invalid APT type: {apt_type}. Valid types: {list(self.APT_RUN_RANGES.keys())}")
            return False
        
        # Validate run ID
        try:
            run_id_int = int(run_id)
            if run_id_int not in self.APT_RUN_RANGES[apt_type]:
                self.logger.error(f"❌ Invalid run ID {run_id} for {apt_type}. Valid range: {self.APT_RUN_RANGES[apt_type]}")
                return False
        except ValueError:
            self.logger.error(f"❌ Invalid run ID format: {run_id}")
            return False
        
        # Setup paths
        apt_dir = self.dataset_dir / apt_type / f"{apt_type}-run-{run_id:0>2}"
        
        if not apt_dir.exists():
            self.logger.error(f"❌ APT directory not found: {apt_dir}")
            return False
        
        # Find Sysmon file
        sysmon_file = self.find_sysmon_file(apt_dir, run_id)
        if not sysmon_file:
            self.logger.error(f"❌ Sysmon file not found in: {apt_dir}")
            return False
        
        # Setup output file
        output_file = apt_dir / f"all_target_events_run-{run_id:0>2}.csv"
        
        self.logger.info(f"   Raw Sysmon file: {sysmon_file}")
        self.logger.info(f"   Output file: {output_file}")
        
        # Extract events
        if not self.extract_target_events(sysmon_file):
            return False
        
        # Save events
        if not self.save_events(output_file):
            return False
        
        # Print summary statistics
        self.logger.info(f"📊 EXTRACTION SUMMARY:")
        self.logger.info(f"   Total Sysmon events: {self.stats['total_sysmon_events']:,}")
        self.logger.info(f"   Target events extracted: {self.stats['target_events_found']:,}")
        self.logger.info(f"   Events by EventID:")
        for event_id in sorted(self.TARGET_EVENT_IDS):
            count = self.stats['events_by_eventid'][event_id]
            percentage = (count / self.stats['target_events_found'] * 100) if self.stats['target_events_found'] > 0 else 0
            self.logger.info(f"     EventID {event_id}: {count:,} events ({percentage:.1f}%)")
        
        extraction_rate = (self.stats['target_events_found'] / self.stats['total_sysmon_events'] * 100) if self.stats['total_sysmon_events'] > 0 else 0
        self.logger.info(f"   Extraction rate: {extraction_rate:.1f}% of total events")
        
        self.logger.info(f"📝 SELECTION INSTRUCTIONS:")
        self.logger.info(f"   1. Open the CSV file in Excel/LibreOffice")
        self.logger.info(f"   2. Review the CommandLine/TargetFilename columns")
        self.logger.info(f"   3. Mark potential attack operations:")
        self.logger.info(f"      - Seed_Event column: 'X' for significant attack events")
        self.logger.info(f"      - Tactic column: MITRE ATT&CK tactic name")
        self.logger.info(f"      - Technique column: MITRE ATT&CK technique ID")
        self.logger.info(f"   4. Save the file (preserve CSV format)")
        self.logger.info(f"   5. Re-run this script to preserve your selections")
        
        self.logger.info(f"✅ Simple extraction complete for {apt_type}-run-{run_id}")
        return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Simple Event Extractor - Extract all target EventID events for manual selection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python3 SIMPLE_event_extractor.py --apt-type apt-1 --run-id 04
    python3 SIMPLE_event_extractor.py --sysmon-csv /path/to/sysmon.csv --output /path/to/output.csv
        """
    )
    
    # Input options
    parser.add_argument(
        '--sysmon-csv',
        type=str,
        help='Path to raw Sysmon CSV file'
    )
    
    # APT dataset options
    parser.add_argument(
        '--apt-type',
        type=str,
        choices=['apt-1', 'apt-2', 'apt-3', 'apt-4', 'apt-5', 'apt-6', 'apt-7'],
        help='APT dataset type'
    )
    parser.add_argument(
        '--run-id',
        type=str,
        help='Run identifier (e.g., 04, 05, 06)'
    )
    
    # Output options
    parser.add_argument(
        '--output',
        type=str,
        help='Output CSV file path (auto-generated if not specified)'
    )
    
    # Debug options
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.sysmon_csv and not (args.apt_type and args.run_id):
        parser.error("Either --sysmon-csv OR both --apt-type and --run-id must be provided")
    
    if args.apt_type and not args.run_id:
        parser.error("--run-id is required when using --apt-type")
    if args.run_id and not args.apt_type:
        parser.error("--apt-type is required when using --run-id")
    
    # Create extractor
    extractor = SimpleEventExtractor(debug=args.debug)
    
    try:
        if args.apt_type and args.run_id:
            # APT dataset mode
            success = extractor.process_apt_run(args.apt_type, args.run_id)
        else:
            # Direct file mode
            sysmon_file = Path(args.sysmon_csv)
            if not sysmon_file.exists():
                extractor.logger.error(f"❌ Sysmon file not found: {sysmon_file}")
                sys.exit(1)
            
            # Determine output file
            if args.output:
                output_file = Path(args.output)
            else:
                output_file = sysmon_file.parent / f"all_target_events_{sysmon_file.stem}.csv"
            
            extractor.logger.info(f"   Raw Sysmon file: {sysmon_file}")
            extractor.logger.info(f"   Output file: {output_file}")
            
            # Extract and save
            success = extractor.extract_target_events(sysmon_file) and extractor.save_events(output_file)
        
        if success:
            extractor.logger.info("🎉 Simple event extraction completed successfully!")
            sys.exit(0)
        else:
            extractor.logger.error("❌ Simple event extraction failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        extractor.logger.info("⏹️ Extraction interrupted by user")
        sys.exit(1)
    except Exception as e:
        extractor.logger.error(f"❌ Unexpected error: {e}")
        import traceback
        if args.debug:
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()