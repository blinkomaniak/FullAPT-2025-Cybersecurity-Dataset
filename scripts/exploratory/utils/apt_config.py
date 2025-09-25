#!/usr/bin/env python3
"""
Configuration management for APT analysis scripts.

This module provides centralized configuration for:
- MITRE ATT&CK tactic colors
- File patterns and paths
- Common constants

Used by: 6_sysmon_attack_lifecycle_tracer.py, 7_create_labeled_sysmon_dataset.py
"""

from pathlib import Path
from typing import Dict, Any


class TacticColors:
    """MITRE ATT&CK tactic color definitions - single source of truth."""
    
    STANDARD = {
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
        'Unknown': '#696969',             # Dim Gray (neutral)
        'no_attack_tactic': '#D3D3D3',    # Light Gray
    }
    
    # Background colors
    BENIGN_EVENT_COLOR = '#d0d0d0'        # Light gray for benign events
    BENIGN_EVENT_ALPHA = 0.4              # Transparency for background


class FilePaths:
    """Path construction utilities for APT analysis."""
    
    @staticmethod
    def get_base_path(apt_type: str, run_id: str) -> Path:
        """Get base path for APT run directory."""
        return Path(f"/home/researcher/Downloads/research/dataset/{apt_type}/{apt_type}-run-{run_id}")
    
    @staticmethod  
    def get_sysmon_file(apt_type: str, run_id: str) -> Path:
        """Get path to original sysmon CSV file."""
        base_path = FilePaths.get_base_path(apt_type, run_id)
        return base_path / f"sysmon-run-{run_id}.csv"
    
    @staticmethod
    def get_master_tactics_file(apt_type: str, run_id: str) -> Path:
        """Get path to master tactics file."""
        base_path = FilePaths.get_base_path(apt_type, run_id)
        return base_path / f"all_target_events_run-{run_id}.csv"
    
    @staticmethod
    def get_traced_events_file(apt_type: str, run_id: str, version: str = "v2") -> Path:
        """Get path to traced events file."""
        base_path = FilePaths.get_base_path(apt_type, run_id)
        results_dir = base_path / "sysmon_event_tracing_analysis_results"
        
        if version == "v2":
            return results_dir / "traced_sysmon_events_with_tactics_v2.csv"
        else:
            return results_dir / "traced_sysmon_events_with_tactics.csv"
    
    @staticmethod
    def get_results_dir(apt_type: str, run_id: str) -> Path:
        """Get results directory path."""
        base_path = FilePaths.get_base_path(apt_type, run_id)
        return base_path / "sysmon_event_tracing_analysis_results"
    
    @staticmethod
    def get_labeled_dataset_file(apt_type: str, run_id: str, version: str = "v2") -> Path:
        """Get path to labeled dataset file."""
        base_path = FilePaths.get_base_path(apt_type, run_id)
        
        if version == "v2":
            return base_path / f"sysmon-run-{run_id}-labeled-v2.csv"
        else:
            return base_path / f"sysmon-run-{run_id}-labeled.csv"


class PlottingConfig:
    """Configuration for plotting and visualization."""
    
    # Common plot settings
    DPI = 300
    FIGURE_SIZE_TIMELINE = (16, 10)
    FIGURE_SIZE_GROUP = (16, 8)
    
    # Point sizes and styling
    MALICIOUS_POINT_SIZE = 60
    BENIGN_POINT_SIZE = 20
    POINT_ALPHA = 0.8
    POINT_EDGE_COLOR = 'black'
    POINT_EDGE_WIDTH = 0.5
    
    # Grid and axis settings
    GRID_ALPHA = 0.3
    ROTATION_ANGLE = 45
    
    # File naming patterns
    TIMELINE_SIMPLE = "timeline_all_malicious_events.png"
    TIMELINE_SIMPLE_V2 = "timeline_all_malicious_events_v2.png"
    TIMELINE_TACTICS = "timeline_all_malicious_events_with_tactics.png"
    TIMELINE_TACTICS_V2 = "timeline_all_malicious_events_with_tactics_v2.png"
    
    # Common EventIDs (fallback if no data available)
    DEFAULT_EVENT_IDS = [1, 3, 5, 7, 8, 9, 10, 11, 12, 13, 23]


class LoggingConfig:
    """Logging configuration settings."""
    
    FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
    LEVEL = 'INFO'


class ValidationConfig:
    """Configuration for data validation and checks."""
    
    # Required columns for different file types
    REQUIRED_SYSMON_COLUMNS = ['EventID', 'Computer', 'timestamp']
    REQUIRED_TRACED_COLUMNS = ['EventID', 'Computer', 'timestamp', 'OriginatorRow']
    REQUIRED_MASTER_COLUMNS = ['OriginalRowNumber', 'Tactic', 'Technique']
    
    # File size limits (MB)
    MAX_FILE_SIZE_MB = 1000
    
    # Data quality thresholds
    MIN_MALICIOUS_EVENTS = 1
    MAX_MISSING_DATA_PERCENT = 10.0


# Feature flags for gradual rollout
class FeatureFlags:
    """Feature flags for enabling/disabling new functionality."""
    
    USE_SHARED_PLOTTING = True      # Enable shared plotting utilities
    USE_SHARED_PATHS = True         # Enable shared path utilities  
    USE_SHARED_VALIDATION = True    # Enable shared validation
    ENABLE_CACHING = False          # Enable data caching (future)
    ENABLE_WORKFLOW_ORCHESTRATOR = False  # Enable workflow manager (future)


# Version information
__version__ = "1.0.0"
__author__ = "APT Analysis Framework"
__description__ = "Shared configuration for APT analysis scripts"