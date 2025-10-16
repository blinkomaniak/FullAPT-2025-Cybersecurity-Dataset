#!/usr/bin/env python3
"""
Generate Verification Matrix with Refined Causality/Attribution Logic

This script:
1. Interactive IP configuration (attacker IP, internal network, external whitelist)
2. Loads seed events and netflow data
3. Aggregates netflow by community_id
4. Creates netflow-seed_event pairs within correlation window
5. Applies refined causality and attribution logic with filtering
6. Generates verification_matrix_run-XX.csv with 42 columns

Schema includes:
- NetFlow metadata (IPs, ports, protocol, processes, hostnames)
- Seed event metadata (EventID, Computer, ProcessGuid, Image, etc.)
- Computed causality and attribution fields

Filtering rules (applied sequentially):
1. IP Scope Filtering (during pair creation - entities NOT in scope excluded from CSV):
   - RESTRICTED MODE: Only entities with in-scope IPs appear in CSV
     (internal network + attacker IP + whitelisted external IPs)
   - UNRESTRICTED MODE: All entities appear in CSV except excluded IPs
2. Computer matching: Only pairs where seed_event.Computer ∈ netflow.host_hostname (case-insensitive)
3. Temporal correlation: ±correlation_window_seconds (default: 10s, configurable)
4. ICMP filtering: ICMP traffic only x-marked if attacker IP is present

Interactive Configuration:
- Scope Mode: Restricted (whitelist) or Unrestricted (blacklist)
- Attacker IP (default: 192.168.0.4)
- In-scope IPs (restricted mode): Internal network + attacker + optional external IPs
- Excluded IPs (unrestricted mode): Optional IP exclusion list

Usage:
    python3 generate_verification_matrix.py --apt-type apt-1 --run-id 04
    python3 generate_verification_matrix.py --apt-type apt-1 --run-id 04 --correlation-window-seconds 20

APT Run ID Ranges:
    apt-1: 01-20, 51
    apt-2: 21-30
    apt-3: 31-38
    apt-4: 39-44
    apt-5: 45-47
    apt-6: 48-50
    apt-7: 52
"""

import pandas as pd
import numpy as np
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Set
from collections import defaultdict
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches

# APT run ID ranges mapping
APT_RUN_RANGES = {
    'apt-1': list(range(1, 21)) + [51],
    'apt-2': list(range(21, 31)),
    'apt-3': list(range(31, 39)),
    'apt-4': list(range(39, 45)),
    'apt-5': list(range(45, 48)),
    'apt-6': list(range(48, 51)),
    'apt-7': [52],
}

# Constants
THRESHOLD_START = 5000  # milliseconds (diff_set_nst_max)
THRESHOLD_END = 5000    # milliseconds (diff_set_net_max)
DEFAULT_CORRELATION_WINDOW_SECONDS = 10  # Default ±10 second correlation window
MAX_EVENTS_FOR_VISUALIZATION = 200_000  # Maximum events before sampling for timeline plots (reduced for memory safety)

class VerificationMatrixGenerator:
    """Generate verification matrix with refined causality logic."""

    def __init__(self, apt_type: str, run_id: str,
                 correlation_window_seconds: int = None,
                 scope_mode: str = 'restricted',
                 attacker_ip: str = None,
                 in_scope_ips: List[str] = None,
                 excluded_ips: List[str] = None,
                 dc_ip: str = '10.1.0.4',
                 filter_dc_tcp: bool = True,
                 debug: bool = False):
        """
        Initialize generator.

        Parameters:
        -----------
        dc_ip : str
            Domain Controller IP address (default: 10.1.0.4)
        filter_dc_tcp : bool
            If True, filter persistent TCP where DC is involved (default: True)
            If False, include all persistent TCP for x-marking
        """
        self.apt_type = apt_type
        self.run_id = run_id

        # Setup logging
        log_level = logging.DEBUG if debug else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)

        # Setup paths
        self.base_path = Path("/home/researcher/Downloads/research/dataset")
        self.dataset_dir = self.base_path / apt_type / f"{apt_type}-run-{run_id}"
        self.results_dir = self.dataset_dir / "netflow_event_tracing_analysis_results"
        self.results_dir.mkdir(exist_ok=True)

        # Correlation window (convert seconds to milliseconds)
        self.correlation_window_seconds = correlation_window_seconds or DEFAULT_CORRELATION_WINDOW_SECONDS
        self.correlation_window_ms = self.correlation_window_seconds * 1000

        # IP filtering configuration
        self.scope_mode = scope_mode
        self.attacker_ip = attacker_ip or '192.168.0.4'
        self.in_scope_ips = in_scope_ips or []
        self.excluded_ips = excluded_ips or []

        # TCP filtering configuration
        self.dc_ip = dc_ip
        self.filter_dc_tcp = filter_dc_tcp

        self.logger.info(f"🚀 Verification Matrix Generator")
        self.logger.info(f"   APT Type: {apt_type}")
        self.logger.info(f"   Run ID: {run_id}")
        self.logger.info(f"   Dataset: {self.dataset_dir}")
        self.logger.info(f"   Correlation Window: ±{self.correlation_window_seconds}s")
        self.logger.info(f"   Scope Mode: {scope_mode.upper()}")
        self.logger.info(f"   Attacker IP: {self.attacker_ip}")
        self.logger.info(f"   Domain Controller IP: {self.dc_ip}")
        self.logger.info(f"   Filter DC-initiated TCP: {self.filter_dc_tcp}")

        if scope_mode == 'restricted':
            self.logger.info(f"   In-Scope IPs: {self.in_scope_ips}")
        else:
            if self.excluded_ips:
                self.logger.info(f"   Excluded IPs: {self.excluded_ips}")
            else:
                self.logger.info(f"   Excluded IPs: None (all IPs allowed)")

    def load_seed_events(self) -> pd.DataFrame:
        """Load seed events from all_target_events_run-XX.csv."""
        seed_file = self.dataset_dir / f"all_target_events_run-{self.run_id}.csv"

        if not seed_file.exists():
            raise FileNotFoundError(f"Seed events file not found: {seed_file}")

        self.logger.info(f"📥 Loading seed events: {seed_file.name}")
        df = pd.read_csv(seed_file)

        # Filter to manually selected seed events (Seed_Event == 'x')
        if 'Seed_Event' in df.columns:
            df = df[df['Seed_Event'].astype(str).str.strip().str.lower() == 'x'].copy()

        self.logger.info(f"✅ Loaded {len(df)} seed events")
        return df

    def load_netflow_data(self) -> pd.DataFrame:
        """Load netflow data from netflow-run-XX.csv."""
        netflow_file = self.dataset_dir / f"netflow-run-{self.run_id}.csv"

        if not netflow_file.exists():
            raise FileNotFoundError(f"NetFlow file not found: {netflow_file}")

        self.logger.info(f"📥 Loading NetFlow data: {netflow_file.name}")
        df = pd.read_csv(netflow_file)

        self.logger.info(f"✅ Loaded {len(df):,} NetFlow events")
        return df

    def aggregate_netflow_by_community(self, netflow_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Aggregate netflow events by network_community_id.

        Returns: dict[community_id] -> aggregated_data
        """
        self.logger.info("🔗 Aggregating NetFlow by community ID...")

        aggregated = {}

        for community_id, group in netflow_df.groupby('network_community_id'):
            if pd.isna(community_id):
                continue

            # Helper function to collect unique non-null values
            def unique_values(series):
                return sorted([x for x in series.dropna().unique() if pd.notna(x)])

            # Helper function to collect process args as array of arrays
            def collect_process_args(series):
                args_list = []
                for val in series.dropna().unique():
                    if pd.notna(val) and val != '':
                        # Parse if string representation of list
                        if isinstance(val, str) and val.startswith('['):
                            try:
                                parsed = json.loads(val.replace("'", '"'))
                                if parsed:
                                    args_list.append(parsed)
                            except:
                                args_list.append([str(val)])
                        else:
                            args_list.append([str(val)])
                return args_list

            # Aggregate data
            agg_data = {
                'nci': community_id,
                'nst': int(group['event_start'].min()) if 'event_start' in group.columns else None,
                'net': int(group['event_end'].max()) if 'event_end' in group.columns else None,
                'nts': int(group['timestamp'].iloc[0]) if 'timestamp' in group.columns else None,
                'nsip': unique_values(group['source_ip']) if 'source_ip' in group.columns else [],
                'nsp': unique_values(group['source_port']) if 'source_port' in group.columns else [],
                'ndip': unique_values(group['destination_ip']) if 'destination_ip' in group.columns else [],
                'ndp': unique_values(group['destination_port']) if 'destination_port' in group.columns else [],
                'ntr': group['network_transport'].iloc[0] if 'network_transport' in group.columns else '',
                'nby': '',  # Empty for future implementation
                'npack': '',  # Empty for future implementation
                'nhhost': unique_values(group['host_hostname']) if 'host_hostname' in group.columns else [],
                'npe': unique_values(group['process_executable']) if 'process_executable' in group.columns else [],
                'npid': unique_values(group['process_pid']) if 'process_pid' in group.columns else [],
                'npn': unique_values(group['process_name']) if 'process_name' in group.columns else [],
                'nparg': collect_process_args(group['process_args']) if 'process_args' in group.columns else [],
                'ndpe': unique_values(group['destination_process_executable']) if 'destination_process_executable' in group.columns else [],
                'ndpid': unique_values(group['destination_process_pid']) if 'destination_process_pid' in group.columns else [],
                'ndpn': unique_values(group['destination_process_name']) if 'destination_process_name' in group.columns else [],
                'ndparg': collect_process_args(group['destination_process_args']) if 'destination_process_args' in group.columns else [],
            }

            aggregated[community_id] = agg_data

        self.logger.info(f"✅ Aggregated {len(aggregated)} NetFlow communities")
        return aggregated

    def create_nfw_se_pairs(self, netflow_aggregated: Dict, seed_events: pd.DataFrame) -> List[Dict]:
        """
        Create netflow-seed_event pairs within correlation window.

        Returns: list of nfw_se pair dictionaries
        """
        self.logger.info(f"🔗 Creating nfw-se pairs (correlation window: ±{self.correlation_window_seconds}s)...")

        pairs = []

        for _, seed_row in seed_events.iterrows():
            set_time = seed_row['timestamp']

            # Get seed event computer (strip .boombox.local)
            seed_computer_original = seed_row.get('Computer', '').replace('.boombox.local', '')
            seed_computer_lower = seed_computer_original.lower()

            # Find netflows within correlation window
            for community_id, nfw_data in netflow_aggregated.items():
                nst = nfw_data['nst']
                net = nfw_data['net']
                nhhost = nfw_data['nhhost']

                if nst is None or net is None:
                    continue

                # Check if seed event computer is in netflow hostnames (case-insensitive)
                nhhost_lower = [h.lower() for h in nhhost]
                if seed_computer_lower not in nhhost_lower:
                    continue

                # Check if within correlation window
                # Netflow overlaps with [set - window, set + window]
                if (nst <= set_time + self.correlation_window_ms and
                    net >= set_time - self.correlation_window_ms):

                    # IP SCOPE FILTERING: Check if entity should be included in CSV
                    all_ips = set(nfw_data['nsip'] + nfw_data['ndip'])

                    if self.scope_mode == 'restricted':
                        # RESTRICTED: Only include if ALL IPs are in scope
                        if not all_ips.issubset(set(self.in_scope_ips)):
                            continue  # Skip this entity (do NOT add to CSV)
                    else:
                        # UNRESTRICTED: Include unless contains excluded IP
                        if self.excluded_ips and any(ip in self.excluded_ips for ip in all_ips):
                            continue  # Skip this entity (do NOT add to CSV)

                    # Create pair
                    pair = {
                        # NetFlow fields
                        'nci': nfw_data['nci'],
                        'nst': nfw_data['nst'],
                        'net': nfw_data['net'],
                        'nts': nfw_data['nts'],
                        'nsip': nfw_data['nsip'],
                        'nsp': nfw_data['nsp'],
                        'ndip': nfw_data['ndip'],
                        'ndp': nfw_data['ndp'],
                        'ntr': nfw_data['ntr'],
                        'nby': nfw_data['nby'],
                        'npack': nfw_data['npack'],
                        'nhhost': nfw_data['nhhost'],
                        'npe': nfw_data['npe'],
                        'npid': nfw_data['npid'],
                        'npn': nfw_data['npn'],
                        'nparg': nfw_data['nparg'],
                        'ndpe': nfw_data['ndpe'],
                        'ndpid': nfw_data['ndpid'],
                        'ndpn': nfw_data['ndpn'],
                        'ndparg': nfw_data['ndparg'],

                        # Seed event fields
                        'sern': seed_row.get('RawDatasetRowNumber', ''),
                        'set': seed_row['timestamp'],
                        'seid': seed_row.get('EventID', ''),
                        'secomp': seed_row.get('Computer', '').replace('.boombox.local', ''),
                        'secline': seed_row.get('CommandLine', ''),
                        'setf': seed_row.get('TargetFilename', ''),
                        'sepguid': seed_row.get('ProcessGuid', ''),
                        'sepid': seed_row.get('ProcessId', ''),
                        'seppguid': seed_row.get('ParentProcessGuid', ''),
                        'seppid': seed_row.get('ParentProcessId', ''),
                        'seim': seed_row.get('Image', ''),
                        'sepim': seed_row.get('ParentImage', ''),
                        'setac': seed_row.get('Tactic', ''),
                        'setech': seed_row.get('Technique', ''),

                        # Computed fields (initialized)
                        'diff_set_nst': int(set_time - nst),
                        'diff_set_nst_max': THRESHOLD_START,
                        'diff_set_net': int(net - set_time),
                        'diff_set_net_max': THRESHOLD_END,
                        'netflow_duration': int(net - nst),
                        'nfw_se_attrib': 'none',
                        'causality_type': 'none',
                        'netflow_attribution': '',
                        'subnetflow_attribution': '',
                    }

                    pairs.append(pair)

        self.logger.info(f"✅ Created {len(pairs)} nfw-se pairs")
        return pairs

    def determine_causality_and_attribution(self, nfw_se_pairs: List[Dict]) -> List[Dict]:
        """
        Apply refined causality and attribution logic.

        Processes pairs chronologically by seed event timestamp.

        Special filtering rules (applied after temporal causality determination):
        1. ICMP traffic (v4/v6): Only x-marked if attacker IP is present
        2. External IPs: Entities with non-whitelisted external IPs are not x-marked
           - Allowed IPs: internal network + attacker IP + user-defined whitelist
        """
        self.logger.info("🎯 Determining causality and attribution...")

        # Sort chronologically by seed event timestamp
        nfw_se_pairs.sort(key=lambda x: x['set'])

        self.logger.info(f"   Processing {len(nfw_se_pairs)} pairs in chronological order...")

        for idx, pair in enumerate(nfw_se_pairs):
            nst = pair['nst']
            net = pair['net']
            set_time = pair['set']
            nci = pair['nci']

            # Get other entities with same nci that have been processed already
            other_entities = [
                p for i, p in enumerate(nfw_se_pairs[:idx])
                if p['nci'] == nci
            ]

            # Check if all other entities have attribution 'none'
            all_others_none = all(
                e['nfw_se_attrib'] == 'none'
                for e in other_entities
            )

            # Calculate netflow duration
            netflow_duration = net - nst

            # Calculate absolute differences
            abs_diff_start = abs(set_time - nst)
            abs_diff_end = abs(set_time - net)

            # SPECIAL CASE: Very short netflows (< 10s duration)
            # For short flows, seed can be within ±5s of netflow boundaries
            if netflow_duration < (THRESHOLD_START + THRESHOLD_END):
                # Seed event inside netflow
                if nst <= set_time <= net:
                    pair['causality_type'] = 'ct_nfw_se'
                    if all_others_none:
                        pair['nfw_se_attrib'] = 'netflow'
                    else:
                        pair['nfw_se_attrib'] = 'subnetflow'
                # Seed event within 5s BEFORE netflow starts (seed causes netflow)
                elif set_time < nst and (nst - set_time) <= THRESHOLD_START:
                    pair['causality_type'] = 'ct_se_nfw'
                    if all_others_none:
                        pair['nfw_se_attrib'] = 'netflow'
                    else:
                        pair['nfw_se_attrib'] = 'subnetflow'
                # Seed event within 5s AFTER netflow ends (netflow causes seed)
                elif set_time > net and (set_time - net) <= THRESHOLD_END:
                    pair['causality_type'] = 'ct_nfw_se'
                    if all_others_none:
                        pair['nfw_se_attrib'] = 'netflow'
                    else:
                        pair['nfw_se_attrib'] = 'subnetflow'
                else:
                    pair['nfw_se_attrib'] = 'none'
                    pair['causality_type'] = 'none'

            # CONDITION 1: Seed Event Near NetFlow Start
            elif abs_diff_start <= THRESHOLD_START and (net - set_time) >= THRESHOLD_END:
                pair['causality_type'] = 'ct_se_nfw'
                if all_others_none:
                    pair['nfw_se_attrib'] = 'netflow'
                else:
                    pair['nfw_se_attrib'] = 'subnetflow'

            # CONDITION 2: Seed Event Near NetFlow End
            elif abs_diff_end <= THRESHOLD_END and (set_time - nst) >= THRESHOLD_START:
                pair['causality_type'] = 'ct_nfw_se'
                if all_others_none:
                    pair['nfw_se_attrib'] = 'netflow'
                else:
                    pair['nfw_se_attrib'] = 'subnetflow'

            # CONDITION 3: Seed Event In Middle (Persistent NetFlow)
            elif (net - set_time) > THRESHOLD_END and (set_time - nst) > THRESHOLD_START:
                pair['causality_type'] = 'ct_nfw_se'
                if all_others_none:
                    pair['nfw_se_attrib'] = 'netflow'
                else:
                    pair['nfw_se_attrib'] = 'subnetflow'

            # DEFAULT: No correlation
            else:
                pair['nfw_se_attrib'] = 'none'
                pair['causality_type'] = 'none'

            # ===================================================================
            # WHITELIST: Attacker IP involvement bypasses ALL filtering
            # ===================================================================
            source_ips = pair['nsip'] if isinstance(pair['nsip'], list) else []
            dest_ips = pair['ndip'] if isinstance(pair['ndip'], list) else []

            involves_attacker = (self.attacker_ip in source_ips or
                                self.attacker_ip in dest_ips)

            if involves_attacker:
                # WHITELISTED: Skip all filtering rules below
                # Attribution already set by causality logic above
                pass
            else:
                # NOT WHITELISTED: Apply filtering rules

                # ===================================================================
                # ICMP FILTERING RULE: Don't x-mark ICMP unless attacker IP present
                # ===================================================================
                is_icmp = pair['ntr'].lower() in ['icmp', 'ipv6-icmp']

                if is_icmp:
                    # Since we already checked involves_attacker=False above,
                    # all ICMP here has NO attacker IP → filter it
                    pair['nfw_se_attrib'] = 'none'
                    pair['causality_type'] = 'none'

                # ===================================================================
                # UDP PERSISTENCE FILTERING RULE
                # ===================================================================
                is_udp = pair['ntr'].lower() == 'udp'

                if is_udp:
                    netflow_duration = pair['net'] - pair['nst']
                    persistence_threshold_ms = 2 * self.correlation_window_seconds * 1000

                    if netflow_duration > persistence_threshold_ms:
                        # Override attribution: Don't x-mark persistent UDP flows
                        pair['nfw_se_attrib'] = 'none'
                        pair['causality_type'] = 'none'

                # ===================================================================
                # TCP PERSISTENCE FILTERING RULE (Conditional on user setting)
                # ===================================================================
                is_tcp = pair['ntr'].lower() == 'tcp'

                if is_tcp and self.filter_dc_tcp:
                    netflow_duration = pair['net'] - pair['nst']
                    persistence_threshold_ms = 2 * self.correlation_window_seconds * 1000

                    if netflow_duration > persistence_threshold_ms:
                        # Check if DC is involved (source, dest, or bidirectional)
                        dc_involved = (self.dc_ip in source_ips or
                                      self.dc_ip in dest_ips)

                        # Filter if DC involved (attacker already excluded by whitelist above)
                        if dc_involved:
                            pair['nfw_se_attrib'] = 'none'
                            pair['causality_type'] = 'none'

            # ===================================================================
            # Set attribution markers
            # ===================================================================
            if pair['nfw_se_attrib'] == 'netflow':
                pair['netflow_attribution'] = 'x'
            elif pair['nfw_se_attrib'] == 'subnetflow':
                pair['subnetflow_attribution'] = 'x'

        # Count attributions
        netflow_count = sum(1 for p in nfw_se_pairs if p['nfw_se_attrib'] == 'netflow')
        subnetflow_count = sum(1 for p in nfw_se_pairs if p['nfw_se_attrib'] == 'subnetflow')
        none_count = sum(1 for p in nfw_se_pairs if p['nfw_se_attrib'] == 'none')

        self.logger.info(f"✅ Attribution summary:")
        self.logger.info(f"   Netflow: {netflow_count}")
        self.logger.info(f"   Subnetflow: {subnetflow_count}")
        self.logger.info(f"   None: {none_count}")

        return nfw_se_pairs

    def generate_verification_matrix_csv(self, nfw_se_pairs: List[Dict]) -> Tuple[Path, Path]:
        """
        Generate verification matrix CSV files.

        Returns: (original_file_path, v2_file_path)
        """
        self.logger.info("📄 Generating verification matrix CSV...")

        # Convert to DataFrame
        df = pd.DataFrame(nfw_se_pairs)

        # Serialize arrays as JSON strings
        array_columns = ['nsip', 'nsp', 'ndip', 'ndp', 'nhhost', 'npe', 'npid', 'npn', 'nparg',
                        'ndpe', 'ndpid', 'ndpn', 'ndparg']

        for col in array_columns:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, (list, tuple)) else '')

        # Define column order (43 columns)
        # First 8 columns: nci, sern, netflow_attribution, subnetflow_attribution, setac, setech, nfw_se_attrib, causality_type for easy reference
        columns = [
            'nci', 'sern', 'netflow_attribution', 'subnetflow_attribution', 'setac', 'setech', 'nfw_se_attrib', 'causality_type',
            'diff_set_nst', 'diff_set_net', 'netflow_duration',
            'nst', 'net', 'nts', 'nsip', 'nsp', 'ndip', 'ndp', 'ntr', 'nby', 'npack',
            'nhhost',
            'npe', 'npid', 'npn', 'nparg', 'ndpe', 'ndpid', 'ndpn', 'ndparg',
            'set', 'seid', 'secomp', 'secline', 'setf', 'sepguid', 'sepid',
            'seppguid', 'seppid', 'seim', 'sepim',
            'diff_set_nst_max', 'diff_set_net_max'
        ]

        # Reorder columns
        df = df[columns]

        # Save original file
        original_file = self.results_dir / f"verification_matrix_run-{self.run_id}.csv"
        df.to_csv(original_file, index=False)
        self.logger.info(f"✅ Saved original: {original_file.name}")

        # Save v2 file (copy for manual editing)
        v2_file = self.results_dir / f"verification_matrix_v2_run-{self.run_id}.csv"
        df.to_csv(v2_file, index=False)
        self.logger.info(f"✅ Saved v2 (manual): {v2_file.name}")

        return original_file, v2_file

    def run(self):
        """Execute complete verification matrix generation pipeline."""
        try:
            # Step 1: Load data
            seed_events = self.load_seed_events()
            netflow_data = self.load_netflow_data()

            # Step 2: Aggregate netflow
            netflow_aggregated = self.aggregate_netflow_by_community(netflow_data)

            # Step 3: Create nfw-se pairs
            nfw_se_pairs = self.create_nfw_se_pairs(netflow_aggregated, seed_events)

            # Step 4: Determine causality and attribution
            nfw_se_pairs = self.determine_causality_and_attribution(nfw_se_pairs)

            # Step 5: Generate CSV files
            original_file, v2_file = self.generate_verification_matrix_csv(nfw_se_pairs)

            # Step 6: Generate correlation plots
            self.generate_correlation_plots(original_file, seed_events)

            # ===================================================================
            # ⏸️  MANUAL CHECKPOINT: Review and modify verification matrix
            # ===================================================================
            print("\n" + "="*80)
            print("⏸️  MANUAL CHECKPOINT REQUIRED")
            print("="*80)
            print(f"\n📋 Verification Matrix Generated:")
            print(f"   📁 File: {v2_file}")
            print(f"\n📊 Correlation Plots Generated:")
            print(f"   📁 Directory: {self.results_dir / 'correlation_plots'}")
            print(f"\n✏️  INSTRUCTIONS:")
            print(f"   1. Open: {v2_file}")
            print(f"   2. Review automated x-marks in columns:")
            print(f"      • netflow_attribution (Tier 1: whole community labeling)")
            print(f"      • subnetflow_attribution (Tier 2: temporal segment labeling)")
            print(f"   3. Use correlation plots to validate/modify attributions")
            print(f"   4. Save changes to the same file")
            print(f"\n💡 TIP: Green bars in plots = exclusive attribution (netflow)")
            print(f"         Orange bars = shared attribution (subnetflow)")
            print(f"         Gray bars = no attribution")
            print("="*80)
            input("\n⏵  Press ENTER when you have finished reviewing/editing the file...")
            print("\n✅ Resuming pipeline with your modified verification matrix...\n")

            # Step 7: Three-tier labeling (NEW PHASE 2)
            self.logger.info("🏷️  STEP 7: Applying three-tier labeling...")
            labeled_df = self.apply_three_tier_labeling(v2_file, seed_events, netflow_data)

            # Save labeled dataset (in root dataset directory)
            labeled_file = self.dataset_dir / f"netflow-run-{self.run_id}-labeled.csv"
            labeled_df.to_csv(labeled_file, index=False)
            self.logger.info(f"✅ Saved labeled dataset: {labeled_file}")

            # Step 8: Generate timeline visualizations (NEW PHASE 3)
            self.logger.info("📊 STEP 8: Creating timeline visualizations...")
            self.create_timeline_visualizations(labeled_df, seed_events)

            self.logger.info("="*80)
            self.logger.info("✅ COMPLETE PIPELINE FINISHED!")
            self.logger.info("="*80)
            self.logger.info(f"📁 Output files:")
            self.logger.info(f"   Original: {original_file}")
            self.logger.info(f"   V2 (manual): {v2_file}")
            self.logger.info("")
            self.logger.info(f"📊 Total pairs: {len(nfw_se_pairs)}")
            self.logger.info(f"📊 Unique communities: {len(set(p['nci'] for p in nfw_se_pairs))}")
            self.logger.info(f"📊 Unique seed events: {len(set(p['sern'] for p in nfw_se_pairs))}")

            return True

        except Exception as e:
            self.logger.error(f"❌ Pipeline failed: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return False

    def generate_correlation_plots(self, verification_matrix_path: Path, seed_df: pd.DataFrame) -> bool:
        """
        Generate correlation hotspot plots grouped by connection pattern.

        Args:
            verification_matrix_path: Path to verification_matrix_run-XX.csv
            seed_df: DataFrame of seed events

        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info("📊 Generating correlation hotspot plots...")

            # Load verification matrix
            df = pd.read_csv(verification_matrix_path)

            if len(df) == 0:
                self.logger.warning("⚠️ No pairs in verification matrix - skipping plots")
                return True

            # Create plots directory
            plots_dir = self.results_dir / "correlation_plots"
            plots_dir.mkdir(exist_ok=True)

            # Group pairs by connection pattern (source_ip, dest_ip, protocol)
            connection_groups = self._group_by_connection(df)

            self.logger.info(f"   Found {len(connection_groups)} unique connection patterns")

            # Generate plots for each connection group
            plot_count = 0
            for connection_key, group_data in connection_groups.items():
                parts_generated = self._generate_connection_plots(
                    connection_key, group_data, seed_df, plots_dir
                )
                plot_count += parts_generated

            # Generate summary report
            self._generate_plot_summary(connection_groups, plots_dir)

            self.logger.info(f"✅ Generated {plot_count} correlation plots")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to generate correlation plots: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return False

    def _group_by_connection(self, df: pd.DataFrame) -> Dict:
        """
        Group verification matrix pairs by connection pattern.

        Returns dict: {connection_key: {'pairs': df_subset, 'stats': {...}}}
        """
        from ast import literal_eval

        groups = defaultdict(list)

        for idx, row in df.iterrows():
            # Extract ALL unique IPs from both source and destination lists
            try:
                nsip_list = literal_eval(row['nsip']) if isinstance(row['nsip'], str) else row['nsip']
                ndip_list = literal_eval(row['ndip']) if isinstance(row['ndip'], str) else row['ndip']

                # Ensure lists
                if not isinstance(nsip_list, list):
                    nsip_list = [nsip_list]
                if not isinstance(ndip_list, list):
                    ndip_list = [ndip_list]

                # Get all unique IPs from both source and destination
                all_ips = set(nsip_list + ndip_list)

                # Sort IPs to create canonical order (interchangeable source/dest)
                sorted_ips = sorted(all_ips)

                # Create connection key using sorted IPs
                if len(sorted_ips) >= 2:
                    ip1, ip2 = sorted_ips[0], sorted_ips[1]
                elif len(sorted_ips) == 1:
                    # Self-connection (same IP as source and dest)
                    ip1 = ip2 = sorted_ips[0]
                else:
                    # Fallback
                    ip1 = ip2 = 'unknown'

            except Exception as e:
                # Fallback to string representation
                ip1 = str(row['nsip'])
                ip2 = str(row['ndip'])

            protocol = row['ntr']
            connection_key = f"{ip1}__{ip2}__{protocol}"

            groups[connection_key].append(idx)

        # Build structured groups with stats
        connection_groups = {}
        for conn_key, indices in groups.items():
            group_df = df.loc[indices].copy()

            # Parse IPs and protocol from key
            parts = conn_key.split('__')
            source_ip, dest_ip, protocol = parts[0], parts[1], parts[2]

            connection_groups[conn_key] = {
                'pairs': group_df,
                'source_ip': source_ip,
                'dest_ip': dest_ip,
                'protocol': protocol,
                'stats': {
                    'total_pairs': len(group_df),
                    'netflow_attribution': (group_df['netflow_attribution'] == 'x').sum(),
                    'subnetflow_attribution': (group_df['subnetflow_attribution'] == 'x').sum(),
                    'none_attribution': ((group_df['netflow_attribution'] != 'x') &
                                        (group_df['subnetflow_attribution'] != 'x')).sum(),
                    'unique_ncis': group_df['nci'].nunique(),
                    'unique_seeds': group_df['sern'].nunique()
                }
            }

        return connection_groups

    def _generate_connection_plots(self, connection_key: str, group_data: Dict,
                                   seed_df: pd.DataFrame, plots_dir: Path) -> int:
        """
        Generate plots for a specific connection pattern.

        Returns: Number of plot parts generated
        """
        pairs_df = group_data['pairs']
        source_ip = group_data['source_ip']
        dest_ip = group_data['dest_ip']
        protocol = group_data['protocol']

        # Create connection-specific directory
        conn_dir = plots_dir / connection_key
        conn_dir.mkdir(exist_ok=True)

        # Apply pagination logic
        parts = self._paginate_pairs(pairs_df)

        self.logger.info(f"   {connection_key}: {len(pairs_df)} pairs → {len(parts)} part(s)")

        # Generate plot for each part
        for part_num, part_df in enumerate(parts, 1):
            self._create_hotspot_plot(
                part_df, seed_df, conn_dir, part_num, len(parts),
                source_ip, dest_ip, protocol
            )

            # Export events CSV
            events_csv = conn_dir / f"events_part{part_num}.csv"
            part_df.to_csv(events_csv, index=False)

            # Export metadata JSON
            self._generate_part_metadata(
                part_df, conn_dir, part_num, len(parts),
                source_ip, dest_ip, protocol
            )

        return len(parts)

    def _paginate_pairs(self, df: pd.DataFrame,
                        max_seeds_per_plot: int = 36,
                        max_time_window_hours: int = 2) -> List[pd.DataFrame]:
        """
        Split pairs into multiple parts based on pagination criteria.

        Returns: List of DataFrames (one per part)
        """
        # If small enough, return as single part
        unique_seeds = df['sern'].nunique()

        if unique_seeds <= max_seeds_per_plot:
            # Check time span
            min_time = df['set'].min()
            max_time = df['set'].max()
            time_span_hours = (max_time - min_time) / (1000 * 3600)

            if time_span_hours <= max_time_window_hours:
                return [df]

        # Need to paginate - split by time windows
        parts = []
        df_sorted = df.sort_values('set').copy()

        window_ms = max_time_window_hours * 3600 * 1000  # Convert to milliseconds

        start_idx = 0
        while start_idx < len(df_sorted):
            start_time = df_sorted.iloc[start_idx]['set']
            end_time = start_time + window_ms

            # Get all rows in this time window
            window_mask = (df_sorted['set'] >= start_time) & (df_sorted['set'] < end_time)
            window_df = df_sorted[window_mask].copy()

            # Check if this window has too many unique seeds
            if window_df['sern'].nunique() > max_seeds_per_plot:
                # Further split by seed count
                unique_serms = window_df['sern'].unique()[:max_seeds_per_plot]
                window_df = window_df[window_df['sern'].isin(unique_serms)].copy()

            parts.append(window_df)

            # Move to next window
            start_idx += len(window_df)

        return parts if len(parts) > 0 else [df]

    def _create_hotspot_plot(self, df: pd.DataFrame, seed_df: pd.DataFrame,
                            conn_dir: Path, part_num: int, total_parts: int,
                            source_ip: str, dest_ip: str, protocol: str):
        """
        Create correlation hotspot plot matching INTEGRATED_netflow_labeler style.
        """
        # Get unique seed events in this part
        seed_serms = df['sern'].unique()
        n_seeds = len(seed_serms)

        if n_seeds == 0:
            return

        # Determine grid layout (matching INTEGRATED_netflow_labeler logic)
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
        else:
            rows, cols = 8, 8
            figsize = (32, 32)

        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        plt.subplots_adjust(wspace=0.4, hspace=0.6)

        if n_seeds == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        # Plot each seed event
        for i, sern in enumerate(seed_serms):
            if i >= len(axes):
                break

            ax = axes[i]

            # Get seed event info
            seed_pairs = df[df['sern'] == sern]
            seed_info = seed_df[seed_df['RawDatasetRowNumber'] == sern]

            if len(seed_info) == 0:
                # Fallback: get from verification matrix
                seed_row = seed_pairs.iloc[0]
                seed_timestamp = seed_row['set']
                event_id = seed_row.get('seid', 1)
                command_line = seed_row.get('secline', 'N/A')
                target_filename = seed_row.get('setf', 'N/A')
            else:
                seed_row = seed_info.iloc[0]
                seed_timestamp = seed_row['timestamp']
                event_id = seed_row['EventID']
                command_line = seed_row.get('CommandLine', 'N/A')
                target_filename = seed_row.get('TargetFilename', 'N/A')

            # Convert timestamp to datetime
            seed_datetime = pd.to_datetime(seed_timestamp, unit='ms')

            # Plot NetFlow bars for this seed
            unique_ncis = seed_pairs['nci'].unique()

            # Limit NCIs per subplot to prevent matplotlib size errors (max 50)
            max_ncis_per_subplot = 50
            if len(unique_ncis) > max_ncis_per_subplot:
                # Show only first N NCIs with x-marks, then others
                x_marked_ncis = seed_pairs[
                    (seed_pairs['netflow_attribution'] == 'x') |
                    (seed_pairs['subnetflow_attribution'] == 'x')
                ]['nci'].unique()

                if len(x_marked_ncis) <= max_ncis_per_subplot:
                    unique_ncis = x_marked_ncis
                else:
                    unique_ncis = x_marked_ncis[:max_ncis_per_subplot]

            for j, nci in enumerate(unique_ncis):
                nci_rows = seed_pairs[seed_pairs['nci'] == nci]

                # Use first row for netflow times
                row = nci_rows.iloc[0]
                nst_datetime = pd.to_datetime(row['nst'], unit='ms')
                net_datetime = pd.to_datetime(row['net'], unit='ms')

                # Determine color based on attribution
                has_netflow = (nci_rows['netflow_attribution'] == 'x').any()
                has_subnetflow = (nci_rows['subnetflow_attribution'] == 'x').any()

                if has_netflow:
                    color = '#2ecc71'  # Green - netflow attribution
                    label_suffix = " (netflow)"
                elif has_subnetflow:
                    color = '#f39c12'  # Orange - subnetflow attribution
                    label_suffix = " (subnetflow)"
                else:
                    color = '#95a5a6'  # Gray - none
                    label_suffix = " (none)"

                # Draw horizontal bar
                duration_mpl = mdates.date2num(net_datetime) - mdates.date2num(nst_datetime)
                ax.barh(j, duration_mpl, left=mdates.date2num(nst_datetime),
                       height=0.6, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)

            # Get tactic and technique for title (from first row of seed_pairs)
            tactic = seed_pairs.iloc[0].get('setac', 'N/A') if len(seed_pairs) > 0 else 'N/A'
            technique = seed_pairs.iloc[0].get('setech', 'N/A') if len(seed_pairs) > 0 else 'N/A'

            # Plot seed event as vertical line
            if event_id == 1:
                title_detail = command_line[:35] + '...' if len(str(command_line)) > 35 else command_line
                seed_color = '#e74c3c'  # Red
            elif event_id in [11, 23]:
                import os
                title_detail = os.path.basename(str(target_filename))
                seed_color = '#3498db'  # Blue
            else:
                title_detail = f"EventID {event_id}"
                seed_color = '#666666'

            ax.axvline(x=seed_datetime, color=seed_color, alpha=0.8, linewidth=2.0,
                      linestyle='-', zorder=10)

            # Format subplot with NCI labels on y-axis
            if len(unique_ncis) > 0:
                ax.set_yticks(range(len(unique_ncis)))
                # Strip "1:" prefix and use first 8 chars of hash for y-axis labels
                nci_labels = []
                for nci in unique_ncis:
                    # Remove "1:" prefix if present
                    nci_hash = nci[2:] if nci.startswith('1:') else nci
                    # Take first 8 chars of hash
                    label = nci_hash[:8] + "..." if len(nci_hash) > 8 else nci_hash
                    nci_labels.append(label)
                ax.set_yticklabels(nci_labels, fontsize=6)
            ax.set_ylabel('NCI', fontsize=7)
            ax.set_xlabel(f'Time (±{self.correlation_window_seconds}s)', fontsize=7)

            # Multi-line title with tactic and technique
            title_line1 = f"Seed {sern} | EventID {event_id}"
            title_line2 = f"{title_detail[:40]}"
            title_line3 = f"Tactic: {tactic} | Technique: {technique}" if tactic != 'N/A' or technique != 'N/A' else ""
            full_title = f"{title_line1}\n{title_line2}\n{title_line3}" if title_line3 else f"{title_line1}\n{title_line2}"
            ax.set_title(full_title, fontsize=7, pad=8)

            # CRITICAL: Set time limits to center seed event (matching INTEGRATED_netflow_labeler)
            from datetime import timedelta
            display_padding = timedelta(seconds=self.correlation_window_seconds)
            display_start = seed_datetime - display_padding
            display_end = seed_datetime + display_padding
            ax.set_xlim(display_start, display_end)

            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            ax.tick_params(axis='x', labelsize=7)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            ax.grid(True, alpha=0.3, axis='x')

        # Hide unused subplots
        for i in range(n_seeds, len(axes)):
            axes[i].axis('off')

        # Overall title
        title = f"Correlation Hotspots: {source_ip} → {dest_ip} ({protocol.upper()}) - APT-{self.apt_type} Run-{self.run_id} (Part {part_num}/{total_parts})"
        fig.suptitle(title, fontsize=14, y=0.995)

        # Save plot with adaptive DPI to prevent size errors
        plot_path = conn_dir / f"correlation_hotspots_run-{self.run_id}_part{part_num}.png"

        try:
            # Try with standard DPI first
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"      ✅ Part {part_num}: {plot_path.name}")
        except Exception as e:
            if "too large" in str(e):
                # Reduce DPI if image is too large
                self.logger.warning(f"      ⚠️  Reducing DPI due to size limit...")
                plt.savefig(plot_path, dpi=100, bbox_inches='tight')
                self.logger.info(f"      ✅ Part {part_num}: {plot_path.name} (reduced DPI)")
            else:
                raise
        finally:
            plt.close()

    def _generate_part_metadata(self, df: pd.DataFrame, conn_dir: Path,
                                part_num: int, total_parts: int,
                                source_ip: str, dest_ip: str, protocol: str):
        """Generate metadata JSON for a plot part."""
        metadata = {
            "connection": {
                "source_ip": source_ip,
                "destination_ip": dest_ip,
                "protocol": protocol
            },
            "time_range": {
                "start": int(df['nst'].min()),
                "end": int(df['net'].max()),
                "duration_seconds": float((df['net'].max() - df['nst'].min()) / 1000)
            },
            "statistics": {
                "total_pairs": len(df),
                "netflow_attribution": int((df['netflow_attribution'] == 'x').sum()),
                "subnetflow_attribution": int((df['subnetflow_attribution'] == 'x').sum()),
                "none_attribution": int(((df['netflow_attribution'] != 'x') &
                                        (df['subnetflow_attribution'] != 'x')).sum()),
                "unique_ncis": int(df['nci'].nunique()),
                "unique_seed_events": int(df['sern'].nunique())
            },
            "ncis_included": df['nci'].unique().tolist(),
            "pagination": {
                "part_number": part_num,
                "total_parts": total_parts,
                "reason": "time_window_split" if total_parts > 1 else "single_part"
            }
        }

        metadata_path = conn_dir / f"metadata_part{part_num}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def _generate_plot_summary(self, connection_groups: Dict, plots_dir: Path):
        """Generate summary report JSON."""
        summary = {
            "run_id": self.run_id,
            "apt_type": self.apt_type,
            "generation_timestamp": datetime.now().isoformat(),
            "total_connections": len(connection_groups),
            "connections": []
        }

        total_plots = 0
        for conn_key, group_data in connection_groups.items():
            conn_dir = plots_dir / conn_key
            plot_files = list(conn_dir.glob("correlation_hotspots_run-*.png"))

            summary["connections"].append({
                "pattern": conn_key,
                "parts": len(plot_files),
                "total_pairs": int(group_data['stats']['total_pairs']),
                "x_marked_pairs": int(group_data['stats']['netflow_attribution'] +
                                     group_data['stats']['subnetflow_attribution'])
            })
            total_plots += len(plot_files)

        summary["total_plots_generated"] = total_plots

        summary_path = plots_dir / "summary_report.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        self.logger.info(f"   📄 Summary: {summary_path}")

    # ========================================================================
    # PHASE 2: THREE-TIER LABELING (Automated Mode)
    # ========================================================================

    def apply_three_tier_labeling(self, verification_matrix_path: Path,
                                   seed_df: pd.DataFrame,
                                   netflow_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply three-tier labeling system to NetFlow dataset.

        Tier 0: Baseline - Label all 192.168.0.4 traffic as malicious/No-Tactic/No-Technique
        Tier 1: Direct NetFlow Attribution - Label entire communities
        Tier 2: Sub-NetFlow Attribution - Label specific time segments (overrides Tier 1)
        """
        self.logger.info("🏷️ Applying three-tier labeling system...")

        # Build tactic/technique lookup from seed events
        tactic_lookup = {}
        for _, row in seed_df.iterrows():
            tactic_lookup[row['RawDatasetRowNumber']] = {
                'Tactic': row.get('Tactic', 'unknown'),
                'Technique': row.get('Technique', 'unknown')
            }

        # Load attribution mappings
        direct_mapping = self._load_direct_mapping(verification_matrix_path)
        subnetflow_mapping = self._load_subnetflow_mapping_automated(
            verification_matrix_path, netflow_df, seed_df
        )

        # Initialize label columns
        result_df = netflow_df.copy()
        result_df['Tactic'] = ''
        result_df['Technique'] = ''
        result_df['Label'] = 'benign'
        result_df['attribution_source'] = 'none'
        result_df['subnetflow_id'] = ''

        # Statistics counters
        baseline_count = 0
        direct_count = 0
        subnetflow_count = 0

        # TIER 0: Baseline Attacker IP Labeling
        self.logger.info("🎯 Tier 0: Baseline Attacker IP labeling...")
        attacker_mask = (
            (result_df['source_ip'] == self.attacker_ip) |
            (result_df['destination_ip'] == self.attacker_ip)
        )
        baseline_count = attacker_mask.sum()
        if baseline_count > 0:
            result_df.loc[attacker_mask, 'Label'] = 'malicious'
            result_df.loc[attacker_mask, 'Tactic'] = 'No-Tactic'
            result_df.loc[attacker_mask, 'Technique'] = 'No-Technique'
            result_df.loc[attacker_mask, 'attribution_source'] = 'attacker_ip'
            self.logger.info(f"   📊 Labeled {baseline_count:,} events (Attacker IP: {self.attacker_ip})")

        # Apply selective subnetflow segmentation (only for communities needing it)
        if subnetflow_mapping:
            result_df = self._apply_selective_subnetflow_segmentation(
                result_df, subnetflow_mapping.keys()
            )

        # TIER 1: Direct NetFlow Attribution
        self.logger.info("🎯 Tier 1: Direct NetFlow attribution...")
        for community_id, seed_event in sorted(direct_mapping.items(), key=lambda x: x[1]):
            if seed_event in tactic_lookup:
                mask = result_df['network_community_id'] == community_id
                affected = mask.sum()

                if affected > 0:
                    result_df.loc[mask, 'Tactic'] = tactic_lookup[seed_event]['Tactic']
                    result_df.loc[mask, 'Technique'] = tactic_lookup[seed_event]['Technique']
                    result_df.loc[mask, 'Label'] = 'malicious'
                    result_df.loc[mask, 'attribution_source'] = 'direct'
                    direct_count += affected

        self.logger.info(f"   📊 Labeled {direct_count:,} events via direct attribution")

        # TIER 2: Sub-NetFlow Attribution
        self.logger.info("🎯 Tier 2: Sub-NetFlow attribution...")
        for (community_id, subnetflow_id), seed_event in sorted(subnetflow_mapping.items(), key=lambda x: x[1]):
            if seed_event in tactic_lookup:
                mask = (
                    (result_df['network_community_id'] == community_id) &
                    (result_df['subnetflow_id'] == subnetflow_id)
                )
                affected = mask.sum()

                if affected > 0:
                    result_df.loc[mask, 'Tactic'] = tactic_lookup[seed_event]['Tactic']
                    result_df.loc[mask, 'Technique'] = tactic_lookup[seed_event]['Technique']
                    result_df.loc[mask, 'Label'] = 'malicious'
                    result_df.loc[mask, 'attribution_source'] = 'subnetflow'
                    subnetflow_count += affected

        self.logger.info(f"   📊 Labeled {subnetflow_count:,} events via subnetflow attribution")

        # Summary
        unlabeled = (result_df['attribution_source'] == 'none').sum()
        self.logger.info(f"\n📊 THREE-TIER LABELING SUMMARY:")
        self.logger.info(f"   Tier 0 (Attacker IP): {baseline_count:,}")
        self.logger.info(f"   Tier 1 (Direct):      {direct_count:,}")
        self.logger.info(f"   Tier 2 (Sub-NetFlow): {subnetflow_count:,}")
        self.logger.info(f"   Unlabeled (Benign):   {unlabeled:,}")
        self.logger.info(f"   Total:                {len(result_df):,}")

        # Drop temporary attribution_source column
        result_df = result_df.drop(['attribution_source'], axis=1)

        return result_df

    def _load_direct_mapping(self, verification_matrix_path: Path) -> Dict[str, int]:
        """Load direct NetFlow attribution from verification matrix."""
        if not verification_matrix_path.exists():
            self.logger.warning("⚠️ Verification matrix not found")
            return {}

        try:
            df = pd.read_csv(verification_matrix_path)
            # Use netflow_attribution column, nci (network community id), sern (seed event row number)
            direct_mask = df['netflow_attribution'].astype(str).str.strip().str.lower() == 'x'
            direct_df = df[direct_mask].sort_values('sern', ascending=True)

            mapping = {}
            for _, row in direct_df.iterrows():
                mapping[row['nci']] = int(row['sern'])

            self.logger.info(f"   📊 Loaded {len(mapping)} direct attributions")
            return mapping
        except Exception as e:
            self.logger.error(f"❌ Error loading direct mapping: {e}")
            return {}

    def _load_subnetflow_mapping_automated(self, verification_matrix_path: Path,
                                          netflow_df: pd.DataFrame,
                                          seed_df: pd.DataFrame) -> Dict[Tuple[str, int], int]:
        """
        Load subnetflow mapping with automated on-the-fly assignment.
        Uses end-time proximity assignment (±10s window).
        """
        if not verification_matrix_path.exists():
            return {}

        try:
            df = pd.read_csv(verification_matrix_path)
            # Use subnetflow_attribution column, nci (network community id)
            subnetflow_mask = df['subnetflow_attribution'].astype(str).str.strip().str.lower() == 'x'
            subnetflow_communities = df[subnetflow_mask]['nci'].unique()

            if len(subnetflow_communities) == 0:
                self.logger.info("   ✅ No communities marked for subnetflow attribution")
                return {}

            self.logger.info(f"   🔄 Processing {len(subnetflow_communities)} communities for automated subnetflow assignment...")

            # Process each community
            all_assignments = {}

            for community_id in subnetflow_communities:
                community_data = netflow_df[netflow_df['network_community_id'] == community_id].copy()

                if len(community_data) == 0:
                    continue

                # Parse timestamps
                community_data['event_start_parsed'] = pd.to_datetime(community_data['event_start'], unit='ms')
                community_data['event_end_parsed'] = pd.to_datetime(community_data['event_end'], unit='ms')

                # Group by (start, end) to create subnetflows
                grouped = community_data.groupby(['event_start_parsed', 'event_end_parsed'])

                # Create subnetflows DataFrame
                subnetflows_data = []
                for i, ((start_time, end_time), group) in enumerate(grouped):
                    subnetflows_data.append({
                        'subnetflow_id': i + 1,
                        'start_timestamp': start_time,
                        'end_timestamp': end_time,
                        'event_count': len(group),
                        'community_id': community_id
                    })

                subnetflows_df = pd.DataFrame(subnetflows_data)

                # Get seed events correlated with this community
                community_seed_events = df[
                    (df['nci'] == community_id) &
                    ((df['netflow_attribution'].astype(str).str.lower() == 'x') |
                     (df['subnetflow_attribution'].astype(str).str.lower() == 'x'))
                ]['sern'].unique()

                community_seed_df = seed_df[seed_df['RawDatasetRowNumber'].isin(community_seed_events)].copy()

                # Ensure seed_timestamp column exists - check if Timestamp or timestamp column exists
                if 'seed_timestamp' not in community_seed_df.columns:
                    # Try both Timestamp and timestamp columns
                    if 'Timestamp' in community_seed_df.columns:
                        community_seed_df['seed_timestamp'] = pd.to_datetime(community_seed_df['Timestamp'], unit='ms')
                    elif 'timestamp' in community_seed_df.columns:
                        community_seed_df['seed_timestamp'] = pd.to_datetime(community_seed_df['timestamp'], unit='ms')

                # Apply end-time proximity assignment
                assignments = self._apply_end_time_proximity_assignment(
                    subnetflows_df, community_seed_df
                )

                all_assignments.update(assignments)

                # Generate end_time_proximity_timeline plot for this community
                self._create_subnetflow_timeline_plot(
                    community_id, community_data, subnetflows_df,
                    community_seed_df, assignments
                )

            self.logger.info(f"   📊 Generated {len(all_assignments)} subnetflow assignments")
            return all_assignments

        except Exception as e:
            self.logger.error(f"❌ Error in automated subnetflow mapping: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return {}

    def _apply_end_time_proximity_assignment(self, subnetflows_df: pd.DataFrame,
                                            seed_events_df: pd.DataFrame) -> Dict[Tuple[str, int], int]:
        """
        Assign subnetflows to seed events based on end-time proximity (±10s window).
        """
        assignments = {}
        time_window_seconds = 10  # ±10 seconds

        for _, seed in seed_events_df.iterrows():
            seed_time = seed['seed_timestamp']
            seed_id = seed['RawDatasetRowNumber']

            # Find subnetflows within time window
            for _, subnetflow in subnetflows_df.iterrows():
                end_time = subnetflow['end_timestamp']
                distance = abs((seed_time - end_time).total_seconds())

                assignment_key = (subnetflow['community_id'], subnetflow['subnetflow_id'])

                # Assign if within window and not already assigned
                if distance <= time_window_seconds and assignment_key not in assignments:
                    assignments[assignment_key] = seed_id

        return assignments

    def _create_subnetflow_timeline_plot(self, community_id: str, community_data: pd.DataFrame,
                                        subnetflows_df: pd.DataFrame, seed_events_df: pd.DataFrame,
                                        assignments: Dict):
        """Create end_time_proximity_timeline plot showing subnetflows colored by tactic."""
        try:
            # Create pattern_analysis directory structure
            source_ips = '|'.join(map(str, community_data['source_ip'].unique()))
            dest_ips = '|'.join(map(str, community_data['destination_ip'].unique()))
            protocols = '|'.join(map(str, community_data['network_transport'].unique()))
            protocol = protocols.split('|')[0] if '|' in str(protocols) else str(protocols)

            # Create canonical IP pair folder name
            all_source_ips = set(str(ip) for ip in community_data['source_ip'].unique())
            all_dest_ips = set(str(ip) for ip in community_data['destination_ip'].unique())
            all_ips = sorted(all_source_ips | all_dest_ips)
            if len(all_ips) >= 2:
                ip_pair_folder = f"{all_ips[0]}__{all_ips[1]}__{protocol}"
            elif len(all_ips) == 1:
                ip_pair_folder = f"{all_ips[0]}__{all_ips[0]}__{protocol}"
            else:
                ip_pair_folder = "unknown_connection"

            pattern_analysis_dir = self.results_dir / "pattern_analysis" / ip_pair_folder
            pattern_analysis_dir.mkdir(parents=True, exist_ok=True)

            clean_id = community_id.replace('/', '_').replace(':', '_').replace('=', '').replace('+', 'plus')[:20]
            output_path = pattern_analysis_dir / f'end_time_proximity_timeline_{clean_id}.png'

            # Set up plot
            fig, ax = plt.subplots(figsize=(20, 12))

            # Tactic colors
            tactic_colors = {
                'initial-access': '#000000', 'execution': '#4169E1',
                'persistence': '#228B22', 'privilege-escalation': '#B22222',
                'defense-evasion': '#FF8C00', 'credential-access': '#FFD700',
                'discovery': '#8B4513', 'lateral-movement': '#FF1493',
                'collection': '#9932CC', 'command-and-control': '#00CED1',
                'exfiltration': '#32CD32', 'impact': '#DC143C',
                'unassigned': '#CCCCCC'
            }

            # Plot subnetflows as horizontal bars
            bar_height = 0.8
            y_labels = []

            for i, (_, subnetflow) in enumerate(subnetflows_df.iterrows()):
                start_time = subnetflow['start_timestamp']
                end_time = subnetflow['end_timestamp']
                sub_id = subnetflow['subnetflow_id']

                # Ordinal number for y-axis
                ordinal_number = i + 1

                # Get assignment and color by tactic
                assigned_seed = assignments.get((community_id, sub_id), None)
                if assigned_seed:
                    # Find seed info to get tactic
                    seed_info = seed_events_df[seed_events_df['RawDatasetRowNumber'] == assigned_seed]
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

                # Calculate duration for bar width
                start_time_parsed = pd.to_datetime(start_time)
                end_time_parsed = pd.to_datetime(end_time)
                duration_mpl = mdates.date2num(end_time_parsed) - mdates.date2num(start_time_parsed)

                # Minimum bar width for visibility
                min_bar_width = 1.0 / 86400.0  # 1 second in matplotlib date units
                if duration_mpl <= 0 or duration_mpl < min_bar_width:
                    duration_mpl = min_bar_width

                # Plot horizontal bar
                ax.barh(i, duration_mpl,
                       left=mdates.date2num(start_time_parsed),
                       height=bar_height,
                       color=color,
                       alpha=alpha,
                       edgecolor='none',
                       linewidth=0)

                # Create y-axis label
                if assigned_seed:
                    y_labels.append(f'{ordinal_number} → {assigned_seed}')
                else:
                    y_labels.append(f'{ordinal_number}')

            # Format plot
            ax.set_yticks(range(len(subnetflows_df)))
            ax.set_yticklabels(y_labels)
            ax.set_ylabel('SubNetFlow ID → Seed Event', fontsize=10)
            ax.set_xlabel('Timeline', fontsize=10)
            ax.set_title(f'End-Time Proximity Assignment\nCommunity: {clean_id}\n'
                        f'Connection: {ip_pair_folder}',
                        fontsize=12)
            ax.grid(True, alpha=0.3, axis='x')
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

            # Add legend
            legend_elements = []
            used_tactics = set()
            for (comm_id, sub_id), seed_id in assignments.items():
                if comm_id == community_id:
                    seed_info = seed_events_df[seed_events_df['RawDatasetRowNumber'] == seed_id]
                    if len(seed_info) > 0:
                        tactic = seed_info.iloc[0]['Tactic']
                        if tactic not in used_tactics:
                            color = tactic_colors.get(tactic, '#666666')
                            legend_elements.append(plt.Rectangle((0, 0), 1, 1, fc=color, alpha=0.8,
                                                                label=tactic.title()))
                            used_tactics.add(tactic)

            if legend_elements:
                ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()

        except Exception as e:
            self.logger.error(f"❌ Error creating subnetflow timeline for {community_id[:20]}...: {e}")

    def _apply_selective_subnetflow_segmentation(self, netflow_df: pd.DataFrame,
                                                subnetflow_keys: Set[Tuple[str, int]]) -> pd.DataFrame:
        """Apply subnetflow segmentation only to communities that need it."""
        communities_needing_segmentation = set(key[0] for key in subnetflow_keys)

        if not communities_needing_segmentation:
            return netflow_df

        self.logger.info(f"   🔧 Segmenting {len(communities_needing_segmentation)} communities into subnetflows...")

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

            # Assign subnetflow IDs
            for i, ((start_time, end_time), group) in enumerate(grouped):
                subnetflow_id = i + 1
                netflow_df.loc[group.index, 'subnetflow_id'] = subnetflow_id

        return netflow_df

    # ========================================================================
    # PHASE 3: TIMELINE VISUALIZATIONS
    # ========================================================================

    def _smart_sample_for_visualization(self, labeled_df: pd.DataFrame, max_events: int = MAX_EVENTS_FOR_VISUALIZATION) -> pd.DataFrame:
        """
        Smart sampling for large datasets while preserving temporal boundaries.

        Strategy:
        - For each (Label, Tactic) group:
          1. Always include first event (earliest timestamp)
          2. Always include last event (latest timestamp)
          3. Sample remaining middle events proportionally

        This preserves the temporal span of each tactic in the timeline.
        """
        if len(labeled_df) <= max_events:
            self.logger.info(f"   Dataset size ({len(labeled_df):,}) within visualization limit - no sampling needed")
            return labeled_df

        self.logger.warning(f"⚠️  Large dataset detected: {len(labeled_df):,} events")
        self.logger.warning(f"   Applying smart sampling to {max_events:,} events for visualization...")

        # Ensure timestamp column exists
        if 'timestamp_parsed' not in labeled_df.columns:
            labeled_df = labeled_df.copy()
            labeled_df['timestamp_parsed'] = pd.to_datetime(labeled_df['timestamp'], unit='ms')

        sampled_events = []

        # Group by Label and Tactic
        labeled_df['Tactic'] = labeled_df['Tactic'].fillna('no-tactic')

        # Separate benign and malicious for different allocation strategies
        benign_df = labeled_df[labeled_df['Label'] == 'benign']
        malicious_df = labeled_df[labeled_df['Label'] == 'malicious']

        # Allocate 10% of budget to benign (background noise), 90% to malicious (important attack data)
        benign_budget = int(max_events * 0.1)
        malicious_budget = max_events - benign_budget

        # Sample benign events (simple random sample with boundary preservation)
        if len(benign_df) > 0:
            if len(benign_df) <= benign_budget:
                sampled_events.append(benign_df)
                self.logger.info(f"      benign: {len(benign_df):,} events (all kept)")
            else:
                benign_sorted = benign_df.sort_values('timestamp_parsed')
                first_benign = benign_sorted.iloc[[0]]
                last_benign = benign_sorted.iloc[[-1]]
                middle_benign = benign_sorted.iloc[1:-1]

                n_middle = min(len(middle_benign), benign_budget - 2)
                if n_middle > 0:
                    middle_sample = middle_benign.sample(n=n_middle, random_state=42)
                else:
                    middle_sample = pd.DataFrame()

                benign_sample = pd.concat([first_benign, middle_sample, last_benign])
                sampled_events.append(benign_sample)
                self.logger.info(f"      benign: {len(benign_df):,} → {len(benign_sample):,} events (first/last preserved)")

        # Sample malicious events by tactic (preserve boundaries per tactic)
        if len(malicious_df) > 0:
            malicious_groups = malicious_df.groupby('Tactic')
            total_malicious_groups = len(malicious_groups)
            events_per_tactic = max(10, malicious_budget // total_malicious_groups)

            for tactic, group_df in malicious_groups:
                group_size = len(group_df)

                if group_size <= 2:
                    sampled_events.append(group_df)
                else:
                    group_sorted = group_df.sort_values('timestamp_parsed')
                    first_event = group_sorted.iloc[[0]]
                    last_event = group_sorted.iloc[[-1]]
                    middle_events = group_sorted.iloc[1:-1]

                    n_middle_sample = min(len(middle_events), max(0, events_per_tactic - 2))

                    if n_middle_sample > 0 and len(middle_events) > n_middle_sample:
                        middle_sample = middle_events.sample(n=n_middle_sample, random_state=42)
                    else:
                        middle_sample = middle_events

                    group_sample = pd.concat([first_event, middle_sample, last_event])
                    sampled_events.append(group_sample)

                    self.logger.info(f"      malicious/{tactic}: {group_size:,} → {len(group_sample):,} events (first/last preserved)")

        # Combine all sampled groups
        sampled_df = pd.concat(sampled_events, ignore_index=True)

        # If still over limit, do final proportional sampling
        if len(sampled_df) > max_events:
            self.logger.warning(f"   Still {len(sampled_df):,} events - applying final proportional sampling...")
            sampled_df = sampled_df.sample(n=max_events, random_state=42)

        self.logger.info(f"   ✅ Sampled {len(sampled_df):,} events for visualization (from {len(labeled_df):,})")

        return sampled_df

    def create_timeline_visualizations(self, labeled_df: pd.DataFrame, seed_df: pd.DataFrame):
        """Create timeline visualizations."""
        self.logger.info("📊 Creating timeline visualizations...")

        # Apply smart sampling for large datasets
        labeled_df_viz = self._smart_sample_for_visualization(labeled_df)
        was_sampled = len(labeled_df_viz) < len(labeled_df)

        # Multi-Track Timeline (pass both original and sampled for accurate legend)
        self._create_multi_track_timeline(labeled_df_viz, labeled_df, was_sampled)

        # Dual-Domain Timeline
        self._create_dual_domain_timeline(labeled_df_viz, seed_df, was_sampled, len(labeled_df))

    def _create_multi_track_timeline(self, labeled_df_viz: pd.DataFrame, labeled_df_original: pd.DataFrame, was_sampled: bool = False):
        """
        Create Multi-Track Timeline showing individual NetFlow events by tactic.

        Parameters:
        -----------
        labeled_df_viz : pd.DataFrame
            Sampled/filtered data for visualization (plotted points)
        labeled_df_original : pd.DataFrame
            Original full dataset (for accurate legend counts)
        was_sampled : bool
            Whether sampling was applied
        """
        self.logger.info("   📊 Generating Multi-Track Timeline...")

        # Parse timestamps for visualization data
        labeled_df_viz = labeled_df_viz.copy()
        labeled_df_viz['timestamp_parsed'] = pd.to_datetime(labeled_df_viz['timestamp'], unit='ms')

        # Separate visualization data (for plotting)
        malicious_df_viz = labeled_df_viz[labeled_df_viz['Label'] == 'malicious'].copy()
        benign_df_viz = labeled_df_viz[labeled_df_viz['Label'] == 'benign'].copy()

        # Separate original data (for legend counts)
        malicious_df_original = labeled_df_original[labeled_df_original['Label'] == 'malicious'].copy()
        benign_df_original = labeled_df_original[labeled_df_original['Label'] == 'benign'].copy()

        # Fill missing tactics
        if len(malicious_df_viz) > 0:
            malicious_df_viz['Tactic'] = malicious_df_viz['Tactic'].fillna('no-tactic')
            malicious_df_viz['Tactic'] = malicious_df_viz['Tactic'].replace('', 'no-tactic')

        if len(malicious_df_original) > 0:
            malicious_df_original['Tactic'] = malicious_df_original['Tactic'].fillna('no-tactic')
            malicious_df_original['Tactic'] = malicious_df_original['Tactic'].replace('', 'no-tactic')

        # Create figure
        fig, ax = plt.subplots(figsize=(16, 10))

        # Y-level assignments (based on visualization data)
        y_levels = {'benign': 1}
        if len(malicious_df_viz) > 0:
            unique_tactics = sorted(malicious_df_viz['Tactic'].unique())
            for i, tactic in enumerate(unique_tactics):
                y_levels[tactic] = i + 2

        # Plot benign (use visualization data)
        benign_color = '#CCCCCC'
        if len(benign_df_viz) > 0:
            ax.scatter(benign_df_viz['timestamp_parsed'],
                      [y_levels['benign']] * len(benign_df_viz),
                      c=benign_color, alpha=0.4, s=15)

        # Plot malicious by tactic (use visualization data)
        tactic_colors = {
            'initial-access': '#000000', 'execution': '#4169E1',
            'persistence': '#228B22', 'privilege-escalation': '#B22222',
            'defense-evasion': '#FF8C00', 'credential-access': '#FFD700',
            'discovery': '#8B4513', 'lateral-movement': '#FF1493',
            'collection': '#9932CC', 'command-and-control': '#00CED1',
            'exfiltration': '#32CD32', 'impact': '#DC143C',
            'No-Tactic': '#666666', 'no-tactic': '#666666'
        }

        if len(malicious_df_viz) > 0:
            for tactic in sorted(malicious_df_viz['Tactic'].unique()):
                tactic_events = malicious_df_viz[malicious_df_viz['Tactic'] == tactic]
                color = tactic_colors.get(tactic, '#696969')
                y_level = y_levels[tactic]

                ax.scatter(tactic_events['timestamp_parsed'],
                          [y_level] * len(tactic_events),
                          c=color, alpha=0.7, s=30)

        # Formatting
        ax.set_ylabel('Tactic Levels')
        ax.set_xlabel('Timeline')
        # Title with sampling note if applicable
        title = f'Multi-Track Timeline Analysis\nAPT-{self.apt_type} Run-{self.run_id} - NetFlow Events by MITRE Tactic'
        if was_sampled:
            title += f'\n(Sampled: {len(labeled_df_viz):,} events shown from {len(labeled_df_original):,} total - first/last per tactic preserved)'
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='x')

        # Y-axis
        sorted_y_items = sorted(y_levels.items(), key=lambda x: x[1])
        y_ticks = [item[1] for item in sorted_y_items]
        y_labels = [item[0].replace('-', ' ').title() if item[0] != 'benign' else 'Benign'
                   for item in sorted_y_items]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)
        ax.set_ylim(0.5, max(y_ticks) + 0.5)

        # X-axis
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
        plt.xticks(rotation=45)

        # Create comprehensive legend with event counts (from ORIGINAL dataset)
        legend_elements = []

        # Section 1: Label Distribution
        legend_elements.append(plt.Line2D([0], [0], color='none', linestyle='',
                                        label='── EVENT LABELS ──'))
        legend_elements.append(plt.Line2D([0], [0], marker='o', color=benign_color,
                                        markersize=8, linestyle='', alpha=0.7,
                                        label=f'Benign: {len(benign_df_original):,} events'))
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='red',
                                        markersize=8, linestyle='', alpha=0.7,
                                        label=f'Malicious: {len(malicious_df_original):,} events'))

        # Section 2: Tactic Distribution
        if len(malicious_df_original) > 0:
            legend_elements.append(plt.Line2D([0], [0], color='none', linestyle='',
                                            label='── MITRE TACTICS ──'))
            for tactic in sorted(malicious_df_original['Tactic'].unique()):
                tactic_events = malicious_df_original[malicious_df_original['Tactic'] == tactic]
                color = tactic_colors.get(tactic, '#696969')
                tactic_count = len(tactic_events)
                legend_elements.append(plt.Line2D([0], [0], marker='o', color=color,
                                                markersize=8, linestyle='', alpha=0.7,
                                                label=f'{tactic.title()}: {tactic_count:,} events'))

        # Add legend
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.02, 1.0), loc='upper left',
                 fontsize=9, frameon=True, fancybox=True, shadow=True)

        plt.tight_layout()

        # Save (use lower DPI for large datasets to reduce memory)
        output_path = self.results_dir / f"multi_track_timeline_run-{self.run_id}.png"
        dpi = 100 if was_sampled else 150
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()

        self.logger.info(f"      ✅ Saved: {output_path.name}")

    def _create_dual_domain_timeline(self, labeled_df: pd.DataFrame, seed_df: pd.DataFrame, was_sampled: bool = False, original_count: int = 0):
        """Create Dual-Domain Timeline combining Sysmon + NetFlow."""
        self.logger.info("   📊 Generating Dual-Domain Timeline...")

        # Load Sysmon labeled dataset (not just seed events!)
        sysmon_labeled_df = self._load_sysmon_labeled_dataset()
        if sysmon_labeled_df is None:
            self.logger.warning("⚠️ Sysmon labeled dataset not found - skipping dual-domain timeline")
            return

        # Parse timestamps
        labeled_df['timestamp_parsed'] = pd.to_datetime(labeled_df['timestamp'], unit='ms')
        sysmon_labeled_df['timestamp_parsed'] = pd.to_datetime(sysmon_labeled_df['timestamp'], unit='ms')

        # Determine unified time range
        all_times = pd.concat([
            labeled_df['timestamp_parsed'],
            sysmon_labeled_df['timestamp_parsed']
        ]).dropna()
        time_min, time_max = all_times.min(), all_times.max()

        # Create figure with 2 panels
        fig, (ax_sysmon, ax_netflow) = plt.subplots(2, 1, figsize=(20, 12),
                                                     sharex=True,
                                                     gridspec_kw={'height_ratios': [1.2, 1]})

        # Tactic colors
        tactic_colors = {
            'initial-access': '#000000', 'execution': '#4169E1',
            'persistence': '#228B22', 'privilege-escalation': '#B22222',
            'defense-evasion': '#FF8C00', 'credential-access': '#FFD700',
            'discovery': '#8B4513', 'lateral-movement': '#FF1493',
            'collection': '#9932CC', 'command-and-control': '#00CED1',
            'exfiltration': '#32CD32', 'impact': '#DC143C',
            'No-Tactic': '#666666'
        }

        # ========== TOP PANEL: Sysmon malicious events grouped by computer ==========
        # Group by computer (use Computer column)
        if 'Computer' in sysmon_labeled_df.columns:
            computers = sysmon_labeled_df.groupby('Computer')
        else:
            # Fallback: create generic grouping
            sysmon_copy = sysmon_labeled_df.copy()
            sysmon_copy['computer_id'] = 'Host'
            computers = sysmon_copy.groupby('computer_id')

        # Create y-levels for each computer
        computer_names = sorted(computers.groups.keys())
        y_positions = {comp: i for i, comp in enumerate(computer_names)}

        # Plot events for each computer, colored by tactic
        for computer_name, computer_events in computers:
            y_pos = y_positions[computer_name]

            # Group by tactic and plot with different colors
            if 'Tactic' in computer_events.columns:
                for tactic, tactic_events in computer_events.groupby('Tactic'):
                    if pd.notna(tactic) and tactic != '':
                        color = tactic_colors.get(tactic, '#696969')
                        ax_sysmon.scatter(tactic_events['timestamp_parsed'],
                                         [y_pos] * len(tactic_events),
                                         c=color, alpha=0.7, s=25, marker='o')
            else:
                # No tactic info - plot all as generic
                ax_sysmon.scatter(computer_events['timestamp_parsed'],
                                 [y_pos] * len(computer_events),
                                 c='#FF0000', alpha=0.7, s=25, marker='o')

        # Customize top panel
        ax_sysmon.set_ylabel('Host Systems', fontsize=12, fontweight='bold')
        ax_sysmon.set_title('🖥️  HOST-LEVEL MALICIOUS EVENTS (Sysmon)',
                           fontsize=12, fontweight='bold', pad=10)
        ax_sysmon.set_yticks(list(y_positions.values()))
        ax_sysmon.set_yticklabels([f'{comp}' for comp in computer_names])
        ax_sysmon.set_xlim(time_min, time_max)
        ax_sysmon.grid(True, alpha=0.3, axis='x')

        # Legend for top panel (deduplicate tactics)
        handles_sysmon = []
        seen_tactics = set()
        # Filter out NaN values before sorting
        unique_tactics = [t for t in sysmon_labeled_df['Tactic'].unique() if pd.notna(t)]
        for tactic in sorted(unique_tactics):
            if tactic not in seen_tactics:
                color = tactic_colors.get(tactic, '#696969')
                handles_sysmon.append(plt.Line2D([0], [0], marker='o', color=color,
                                                markersize=8, linestyle='', alpha=0.7,
                                                label=tactic.title()))
                seen_tactics.add(tactic)
        if handles_sysmon:
            ax_sysmon.legend(handles=handles_sysmon, loc='upper right', fontsize=10)

        # ========== BOTTOM PANEL: NetFlow events by tactic ==========
        # Separate malicious and benign
        malicious_netflow = labeled_df[labeled_df['Label'] == 'malicious'].copy()
        benign_netflow = labeled_df[labeled_df['Label'] == 'benign']

        # Y-level assignments: Benign at level 0, tactics at levels 1+
        y_levels = {}
        if len(benign_netflow) > 0:
            y_levels['benign'] = 0

        if len(malicious_netflow) > 0:
            malicious_netflow['Tactic'] = malicious_netflow['Tactic'].fillna('No-Tactic')
            unique_tactics = sorted(malicious_netflow['Tactic'].unique())
            for i, tactic in enumerate(unique_tactics):
                y_levels[tactic] = i + 1

        # Plot benign events (sample if too many)
        if len(benign_netflow) > 0:
            if len(benign_netflow) > 5000:
                benign_sample = benign_netflow.sample(n=5000, random_state=42)
            else:
                benign_sample = benign_netflow

            ax_netflow.scatter(benign_sample['timestamp_parsed'],
                              [y_levels['benign']] * len(benign_sample),
                              c='#CCCCCC', alpha=0.3, s=10)

        # Plot malicious events by tactic (each tactic on its own level)
        if len(malicious_netflow) > 0:
            for tactic in sorted(malicious_netflow['Tactic'].unique()):
                tactic_flows = malicious_netflow[malicious_netflow['Tactic'] == tactic]
                color = tactic_colors.get(tactic, '#666666')
                y_level = y_levels[tactic]

                ax_netflow.scatter(tactic_flows['timestamp_parsed'],
                                  [y_level] * len(tactic_flows),
                                  c=color, s=20, alpha=0.8)

        # Format bottom panel
        ax_netflow.set_ylabel('Network Tactics', fontsize=12, fontweight='bold')
        ax_netflow.set_xlabel('Timeline', fontsize=12, fontweight='bold')
        ax_netflow.set_title('🌐 NETWORK-LEVEL MALICIOUS EVENTS (NetFlow)',
                            fontsize=12, fontweight='bold', pad=15)

        # Set y-axis
        if y_levels:
            y_ticks = list(y_levels.values())
            y_labels = [key.title() if key != 'benign' else 'Benign' for key in y_levels.keys()]
            ax_netflow.set_yticks(y_ticks)
            ax_netflow.set_yticklabels(y_labels)
            ax_netflow.set_ylim(-0.5, max(y_ticks) + 0.5)

        ax_netflow.set_xlim(time_min, time_max)
        ax_netflow.grid(True, alpha=0.3, axis='x')

        # Legend for bottom panel
        handles_netflow = []
        if 'benign' in y_levels:
            handles_netflow.append(plt.Line2D([0], [0], marker='o', color='#CCCCCC',
                                             markersize=8, linestyle='', alpha=0.6,
                                             label='Benign Events'))
        if len(malicious_netflow) > 0:
            for tactic in sorted(malicious_netflow['Tactic'].unique()):
                color = tactic_colors.get(tactic, '#666666')
                handles_netflow.append(plt.Line2D([0], [0], marker='o', color=color,
                                                  markersize=8, linestyle='', alpha=0.8,
                                                  label=tactic.title()))
        if handles_netflow:
            ax_netflow.legend(handles=handles_netflow, loc='upper right', fontsize=10)

        # X-axis formatting
        ax_netflow.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
        plt.setp(ax_netflow.xaxis.get_majorticklabels(), rotation=45)

        # Overall title with sampling note if applicable
        title = f'Dual-Domain Attack Timeline: APT-{self.apt_type} Run-{self.run_id}'
        if was_sampled:
            title += f'\n(NetFlow Sampled: {len(labeled_df):,} events from {original_count:,} total - first/last per tactic preserved)'
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)

        plt.tight_layout()

        # Save (use lower DPI for large datasets to reduce memory)
        output_path = self.results_dir / f"dual_domain_attack_timeline.png"
        dpi = 100 if was_sampled else 300
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()

        self.logger.info(f"      ✅ Saved: {output_path.name} (DPI: {dpi})")

    def _load_sysmon_labeled_dataset(self) -> pd.DataFrame:
        """Load Sysmon labeled dataset if available."""
        sysmon_labeled_file = self.dataset_dir / f"sysmon-run-{self.run_id}-labeled.csv"

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
            self.logger.info(f"   📊 Loaded {len(malicious_sysmon):,} malicious Sysmon events from sysmon-run-{self.run_id}-labeled.csv")
            return malicious_sysmon
        except Exception as e:
            self.logger.error(f"❌ Error loading Sysmon dataset: {e}")
            return None


def configure_ip_addresses():
    """
    Interactive IP configuration with scope mode selection.
    Returns: (scope_mode, attacker_ip, in_scope_ips, excluded_ips, filter_dc_tcp, dc_ip)

    scope_mode: 'restricted' or 'unrestricted'
    in_scope_ips: List of IPs that should appear in CSV (restricted mode)
    excluded_ips: List of IPs that should NOT appear in CSV (unrestricted mode)
    filter_dc_tcp: bool - True to filter persistent DC-involved TCP flows
    dc_ip: str - Domain Controller IP address
    """
    print("\n" + "="*80)
    print("IP FILTERING & SCOPE WIZARD")
    print("="*80)

    # STEP 1: Choose scope mode
    print(f"\n📊 STEP 1: Analysis Scope Mode")
    print(f"\nDefine which NetFlow entities should be included in analysis:\n")
    print(f"[1] RESTRICTED SCOPE (Default - Recommended)")
    print(f"    → ONLY entities with specific IPs appear in CSV")
    print(f"    → Scope: Internal network (10.1.0.0/24) + Attacker IP")
    print(f"    → All other entities EXCLUDED from CSV entirely")
    print(f"    → Best for focused APT attack analysis\n")
    print(f"[2] UNRESTRICTED SCOPE")
    print(f"    → ALL entities appear in CSV (no IP filtering)")
    print(f"    → Optional: Exclude specific IPs from CSV")
    print(f"    → Best for comprehensive network-wide analysis")

    mode_choice = input("\nSelect mode [1/2]: ").strip()

    if mode_choice == '2':
        scope_mode = 'unrestricted'
        print(f"\n✅ Selected: UNRESTRICTED SCOPE")
    else:
        scope_mode = 'restricted'
        print(f"\n✅ Selected: RESTRICTED SCOPE")

    # STEP 2: Configure attacker IP (both modes)
    print(f"\n🎯 STEP 2: Attacker IP Configuration")
    print(f"Default: 192.168.0.4")
    attacker_response = input("Keep default attacker IP? [Y/n]: ").strip().lower()

    if attacker_response in ['n', 'no']:
        attacker_ip = input("Enter attacker IP address: ").strip()
    else:
        attacker_ip = '192.168.0.4'

    print(f"✅ Attacker IP set to: {attacker_ip}")

    # Mode-specific configuration
    if scope_mode == 'restricted':
        # RESTRICTED MODE: Configure in-scope IPs
        print(f"\n🏠 STEP 3: Internal Network Scope (10.1.0.0/24)")
        default_internal = ['10.1.0.4', '10.1.0.5', '10.1.0.6', '10.1.0.7', '10.1.0.8']
        print(f"Default IPs that WILL appear in CSV:")
        print(f"  {', '.join(default_internal)}")

        print(f"\nModify internal network scope?")
        print(f"[1] Keep all defaults (10.1.0.4-10.1.0.8)")
        print(f"[2] Remove specific IPs (exclude from CSV)")
        print(f"[3] Add additional internal IPs (include in CSV)")
        print(f"[4] Both remove and add IPs")

        modify_choice = input("\nSelect option [1/2/3/4]: ").strip()

        internal_ips = default_internal.copy()

        if modify_choice in ['2', '4']:
            remove_input = input("Enter IPs to REMOVE from scope (comma-separated): ").strip()
            if remove_input:
                to_remove = [ip.strip() for ip in remove_input.split(',')]
                internal_ips = [ip for ip in internal_ips if ip not in to_remove]
                print(f"✅ Removed: {', '.join(to_remove)}")

        if modify_choice in ['3', '4']:
            add_input = input("Enter IPs to ADD to scope (comma-separated): ").strip()
            if add_input:
                to_add = [ip.strip() for ip in add_input.split(',')]
                internal_ips.extend(to_add)
                print(f"✅ Added: {', '.join(to_add)}")

        print(f"✅ Internal Network: {', '.join(internal_ips)}")

        # External IPs inclusion
        print(f"\n🌐 STEP 4: Include External IPs in Analysis (Optional)")
        print(f"\nBy default, ALL external IPs are EXCLUDED from CSV.")
        print(f"Their entities will NOT appear in the verification matrix.")

        external_response = input("\nInclude specific external IPs in analysis? [y/N]: ").strip().lower()

        external_ips = []
        if external_response in ['y', 'yes']:
            external_input = input("Enter external IPs to INCLUDE (comma-separated): ").strip()
            if external_input:
                external_ips = [ip.strip() for ip in external_input.split(',')]
                print(f"✅ External IPs included: {', '.join(external_ips)}")
        else:
            print(f"✅ No external IPs included")

        # Build in_scope_ips
        in_scope_ips = internal_ips + [attacker_ip] + external_ips
        excluded_ips = []

    else:
        # UNRESTRICTED MODE: Configure exclusions
        print(f"\n🚫 STEP 3: Exclude Specific IPs from CSV (Optional)")
        print(f"\nIn unrestricted mode, ALL IPs are included by default.")
        print(f"You can exclude specific IPs to filter them out of CSV.")

        exclude_response = input("\nExclude specific IPs from analysis? [y/N]: ").strip().lower()

        excluded_ips = []
        if exclude_response in ['y', 'yes']:
            exclude_input = input("Enter IPs to EXCLUDE (comma-separated): ").strip()
            if exclude_input:
                excluded_ips = [ip.strip() for ip in exclude_input.split(',')]
                print(f"✅ IPs excluded: {', '.join(excluded_ips)}")
        else:
            print(f"✅ No IPs excluded")

        in_scope_ips = []  # Not used in unrestricted mode

    # STEP 4/5: TCP Persistence Filtering (both modes)
    step_num = "4" if scope_mode == 'unrestricted' else "5"
    print(f"\n📊 STEP {step_num}: TCP Persistence Filtering")
    print(f"\nConfigure filtering for persistent TCP connections involving Domain Controller:\n")
    print(f"⚡ NOTE: Flows involving the attacker IP ({attacker_ip}) are ALWAYS x-marked")
    print(f"         regardless of protocol, duration, or DC involvement.\n")
    print(f"[1] FILTER DC-INVOLVED TCP (Default - Recommended)")
    print(f"    → Filter persistent TCP (>20s) involving DC (10.1.0.4) as source/destination")
    print(f"    → Includes bidirectional flows (e.g., [10.1.0.4, 10.1.0.5] ↔ [10.1.0.4, 10.1.0.5])")
    print(f"    → Preserves flows with attacker IP involvement (auto-whitelisted)")
    print(f"    → Reduces false positives from benign DC infrastructure traffic")
    print(f"    → Best for: Standard APT analysis\n")
    print(f"[2] INCLUDE ALL PERSISTENT TCP")
    print(f"    → Do NOT filter any persistent TCP flows")
    print(f"    → All TCP flows eligible for x-marking")
    print(f"    → Flows with attacker IP always x-marked (auto-whitelisted)")
    print(f"    → Best for: Comprehensive DC activity analysis")

    tcp_filter_choice = input("\nSelect option [1/2]: ").strip()

    if tcp_filter_choice == '2':
        filter_dc_tcp = False
        print(f"\n✅ Selected: INCLUDE ALL PERSISTENT TCP")
    else:
        filter_dc_tcp = True
        print(f"\n✅ Selected: FILTER DC-INVOLVED TCP (Default)")

    # Confirm DC IP
    dc_ip_input = input("\nEnter Domain Controller IP (default: 10.1.0.4): ").strip()
    if dc_ip_input:
        dc_ip = dc_ip_input
    else:
        dc_ip = '10.1.0.4'

    print(f"✅ Domain Controller IP: {dc_ip}")

    # Configuration summary
    print("\n" + "="*80)
    print("CONFIGURATION SUMMARY")
    print("="*80)
    print(f"Analysis Mode: {scope_mode.upper()} SCOPE")
    print(f"Attacker IP: {attacker_ip}")
    print(f"Domain Controller IP: {dc_ip}")
    print(f"TCP Filtering: {'FILTER DC-INVOLVED' if filter_dc_tcp else 'INCLUDE ALL'}")

    if scope_mode == 'restricted':
        print(f"\n✅ Entities WILL appear in CSV if they contain ONLY:")
        print(f"   • Internal Network: {', '.join([ip for ip in in_scope_ips if ip.startswith('10.1.0.')])}")
        print(f"   • Attacker IP: {attacker_ip}")
        if any(not ip.startswith('10.1.0.') and ip != attacker_ip for ip in in_scope_ips):
            external_included = [ip for ip in in_scope_ips if not ip.startswith('10.1.0.') and ip != attacker_ip]
            print(f"   • External IPs: {', '.join(external_included)}")
        print(f"\n❌ Entities will NOT appear in CSV if they contain:")
        print(f"   • Any other IP address")
        print(f"\n   Example: NetFlow with 8.8.8.8 → NOT in CSV")
        print(f"            NetFlow with 10.1.0.5 → IN CSV")
    else:
        print(f"\n✅ ALL entities will appear in CSV")
        if excluded_ips:
            print(f"❌ EXCEPT entities containing:")
            print(f"   • Excluded IPs: {', '.join(excluded_ips)}")
            print(f"\n   Example: NetFlow with {excluded_ips[0]} → NOT in CSV")
        else:
            print(f"   No IP exclusions")

    proceed = input("\nProceed with this configuration? [Y/n]: ").strip().lower()
    if proceed in ['n', 'no']:
        print("\nRestarting configuration...")
        return configure_ip_addresses()

    print("="*80 + "\n")

    return scope_mode, attacker_ip, in_scope_ips, excluded_ips, filter_dc_tcp, dc_ip


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate Verification Matrix with Refined Causality Logic',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
APT Run ID Ranges:
  apt-1: 01-20, 51
  apt-2: 21-30
  apt-3: 31-38
  apt-4: 39-44
  apt-5: 45-47
  apt-6: 48-50
  apt-7: 52

Examples:
  python3 generate_verification_matrix.py --apt-type apt-1 --run-id 04
  python3 generate_verification_matrix.py --apt-type apt-2 --run-id 22
  python3 generate_verification_matrix.py --apt-type apt-3 --run-id 35
  python3 generate_verification_matrix.py --apt-type apt-1 --run-id 04 --correlation-window-seconds 20
        """
    )

    parser.add_argument('--apt-type', required=True,
                       choices=['apt-1', 'apt-2', 'apt-3', 'apt-4', 'apt-5', 'apt-6', 'apt-7'],
                       help='APT type')
    parser.add_argument('--run-id', required=True,
                       help='Run ID (e.g., 04, 22, 35)')
    parser.add_argument('--correlation-window-seconds', type=int, default=None,
                       help='Correlation time window for attribution (default: 10 seconds)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')

    args = parser.parse_args()

    # Pad run_id with leading zero if needed
    run_id = args.run_id.zfill(2)

    # Interactive IP configuration
    scope_mode, attacker_ip, in_scope_ips, excluded_ips, filter_dc_tcp, dc_ip = configure_ip_addresses()

    # Create generator and run
    generator = VerificationMatrixGenerator(
        apt_type=args.apt_type,
        run_id=run_id,
        correlation_window_seconds=args.correlation_window_seconds,
        scope_mode=scope_mode,
        attacker_ip=attacker_ip,
        in_scope_ips=in_scope_ips,
        excluded_ips=excluded_ips,
        dc_ip=dc_ip,
        filter_dc_tcp=filter_dc_tcp,
        debug=args.debug
    )
    success = generator.run()

    exit(0 if success else 1)


if __name__ == "__main__":
    main()
