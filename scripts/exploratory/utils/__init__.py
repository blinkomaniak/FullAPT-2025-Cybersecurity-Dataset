"""
APT Analysis Utilities Package

This package contains shared utilities for APT analysis scripts to eliminate
code duplication and improve maintainability.

Modules:
- apt_config: Configuration management (colors, paths, feature flags)
- apt_plotting_utils: Shared plotting functions
- apt_path_utils: Path management and file validation utilities  
- apt_workflow_manager: Complete workflow orchestration

Usage:
    from utils.apt_config import TacticColors, PlottingConfig
    from utils.apt_plotting_utils import plot_simple_timeline_v2
    from utils.apt_path_utils import PathManager
    from utils.apt_workflow_manager import APTWorkflowManager
"""

__version__ = "1.0.0"
__author__ = "APT Analysis Framework"

# Import main classes for convenience
from .apt_config import TacticColors, PlottingConfig, FeatureFlags, FilePaths
from .apt_path_utils import PathManager
from .apt_workflow_manager import APTWorkflowManager

__all__ = [
    'TacticColors',
    'PlottingConfig', 
    'FeatureFlags',
    'FilePaths',
    'PathManager',
    'APTWorkflowManager'
]