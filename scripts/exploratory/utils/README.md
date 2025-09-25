# APT Analysis Utilities Package

This package contains shared utilities for APT analysis scripts to eliminate code duplication and improve maintainability.

## Overview

The utilities package was created as part of a comprehensive refactoring effort in August 2025 to modernize the APT analysis scripts while maintaining 100% backward compatibility.

## Modules

### 🔧 `apt_config.py` - Configuration Management
**Purpose**: Centralized configuration for all APT analysis scripts

**Key Components**:
- `TacticColors`: MITRE ATT&CK tactic color schemes (16 colors)
- `FilePaths`: Standardized path construction utilities
- `PlottingConfig`: Plot settings, DPI, sizes, and styling constants
- `FeatureFlags`: Enable/disable shared functionality for gradual rollout
- `ValidationConfig`: Data validation thresholds and requirements

**Usage**:
```python
from utils.apt_config import TacticColors, PlottingConfig, FeatureFlags

# Get tactic colors
lateral_movement_color = TacticColors.STANDARD['lateral-movement']  # '#FF1493'

# Plot configuration
dpi = PlottingConfig.DPI  # 300
figure_size = PlottingConfig.FIGURE_SIZE_TIMELINE  # (16, 10)

# Feature control
if FeatureFlags.USE_SHARED_PLOTTING:
    # Use shared utilities
    pass
```

### 📊 `apt_plotting_utils.py` - Shared Plotting Functions
**Purpose**: Eliminate 95% code duplication in timeline plotting between scripts

**Key Functions**:
- `create_simple_timeline_plot()`: Multi-computer group timeline visualization
- `create_tactics_timeline_plot()`: Complete tactics timeline with benign background
- `plot_simple_timeline_v2()`, `plot_tactics_timeline_v2()`: Convenience functions
- `validate_plotting_data()`: Data validation for plotting functions

**Features**:
- Intelligent timestamp handling (numeric vs datetime detection)
- Consistent styling across all plots
- Built-in error handling and logging
- Fallback mechanisms for missing data

**Usage**:
```python
from utils.apt_plotting_utils import create_tactics_timeline_plot

# Create tactics timeline with shared utility
success = create_tactics_timeline_plot(
    full_df=labeled_dataset,
    tactic_colors=TacticColors.STANDARD,
    output_path=output_dir / "timeline.png",
    title_suffix=" (Enhanced)",
    logger=logger
)
```

### 🗂️ `apt_path_utils.py` - Path Management Utilities
**Purpose**: Standardized file path handling and validation

**Key Components**:
- `PathManager`: Complete path management for APT analysis workflows
- File existence validation with size checking
- Workflow step suggestions based on current file state
- Safe file operations with automatic backup

**Features**:
```python
from utils.apt_path_utils import PathManager

# Initialize path manager for APT run
path_manager = PathManager('apt-1', '04', logger)

# Validate required input files
is_valid, missing_files = path_manager.validate_input_files(['sysmon', 'master_tactics'])

# Get suggested next steps
suggestions = path_manager.suggest_workflow_next_steps()
# Returns: ['🚀 Run Script 6 to create traced events']

# Get comprehensive file information
file_info = path_manager.get_file_info()
```

### 🚀 `apt_workflow_manager.py` - Complete Workflow Orchestration
**Purpose**: Automated execution and monitoring of the complete APT analysis pipeline

**Key Features**:
- **Complete Automation**: Executes Script 6 → Manual corrections → Script 7
- **Dependency Validation**: Automatic file checking and validation
- **Progress Tracking**: Real-time step execution with timing
- **Comprehensive Reporting**: JSON reports and console summaries
- **Error Recovery**: Graceful handling of failures with detailed diagnostics

**Workflow Steps**:
1. **Validate Inputs** - Check required files exist and are accessible
2. **Run Script 6** - Execute lifecycle tracer (optional, can be skipped)
3. **Check Manual Corrections** - Validate v2 corrections file exists
4. **Run Script 7** - Execute labeled dataset creator
5. **Validate Outputs** - Verify all expected files were generated
6. **Generate Report** - Create comprehensive workflow report

**Usage**:
```bash
# Complete automated workflow
python3 utils/apt_workflow_manager.py --apt-type apt-1 --run-id 04

# Skip Script 6 if outputs already exist
python3 utils/apt_workflow_manager.py --apt-type apt-1 --run-id 04 --skip-script6

# Force rebuild all outputs
python3 utils/apt_workflow_manager.py --apt-type apt-1 --run-id 04 --force-rebuild

# Debug mode with detailed logging
python3 utils/apt_workflow_manager.py --apt-type apt-1 --run-id 04 --debug
```

**Example Output**:
```
============================================================
📊 APT WORKFLOW SUMMARY - APT-1-Run-04
============================================================
🎉 Status: SUCCESSFUL
⏱️  Duration: 42.3 seconds
✅ Completed: 6/6 steps
📁 Generated files: 8

📋 Step Details:
  ✅ Validate input files and dependencies (0.1s)
  ⏩ Execute Script 6 (Lifecycle Tracer) (skipped)
  ✅ Check for manual corrections needed (0.0s)
  ✅ Execute Script 7 (Labeled Dataset Creator) (31.2s)
  ✅ Validate all output files (0.1s)
  ✅ Generate workflow report (0.2s)
============================================================
```

## Integration with Main Scripts

### Script 6 Integration
Script 6 (`6_sysmon_attack_lifecycle_tracer.py`) now includes:
- V2 methods using shared plotting utilities
- Automatic fallback to original implementation
- Feature flag control for gradual rollout

```python
# In Script 6
if SHARED_UTILS_AVAILABLE and FeatureFlags.USE_SHARED_PLOTTING:
    return self.create_tactics_timeline_plot_v2(output_dir)  # Uses shared utilities
else:
    return self.create_tactics_timeline_plot(output_dir)     # Original implementation
```

### Script 7 Integration
Script 7 (`7_create_labeled_sysmon_dataset.py`) now includes:
- V2 methods using shared plotting utilities
- Enhanced timestamp handling
- Automatic workflow integration

```python
# In Script 7
if SHARED_UTILS_AVAILABLE and FeatureFlags.USE_SHARED_PLOTTING:
    self.create_simple_timeline_plot_v2(malicious_df)
    self.create_tactics_timeline_plot_v2(labeled_df)
else:
    self.create_simple_timeline_plot(malicious_df)    # Original
    self.create_tactics_timeline_plot(labeled_df)     # Original
```

## Safety Mechanisms

### Feature Flags
All new functionality is controlled by feature flags in `apt_config.py`:
```python
class FeatureFlags:
    USE_SHARED_PLOTTING = True   # Can be disabled for rollback
    USE_SHARED_PATHS = True      # Path utilities
    USE_SHARED_VALIDATION = True # Validation features
```

### Import Safety
Shared utilities are imported with try/catch to handle missing dependencies:
```python
try:
    from utils.apt_config import TacticColors, PlottingConfig, FeatureFlags
    SHARED_UTILS_AVAILABLE = True
except ImportError:
    print("⚠️ Warning: Shared utilities not available - using original implementation")
    SHARED_UTILS_AVAILABLE = False
```

### Fallback Mechanisms
Every enhanced script maintains its original functionality:
```python
if not SHARED_UTILS_AVAILABLE or not FeatureFlags.USE_SHARED_PLOTTING:
    return self.original_method(df)  # Original method
else:
    return shared_utility_method(df, output_dir, logger)  # Shared utility
```

## Benefits Achieved

### Code Quality
- **95% Reduction**: Eliminated duplicate plotting code between scripts
- **Single Source of Truth**: Centralized configuration for all settings
- **Consistent Styling**: Unified color schemes and formatting
- **Enhanced Reliability**: Better error handling and validation

### Maintainability
- **Modular Architecture**: Clear separation of concerns
- **Easy Updates**: Single point of modification for improvements
- **Professional Structure**: Following Python package conventions
- **Comprehensive Documentation**: Detailed usage examples and API docs

### Workflow Efficiency
- **Complete Automation**: One-command execution of entire pipeline
- **Intelligent Validation**: Automatic dependency checking
- **Progress Visibility**: Real-time status updates and reporting
- **Error Recovery**: Graceful handling of failures with detailed diagnostics

## Usage Examples

### Basic Import and Usage
```python
# Import shared components
from utils.apt_config import TacticColors, PlottingConfig
from utils.apt_plotting_utils import create_tactics_timeline_plot
from utils.apt_path_utils import PathManager
from utils.apt_workflow_manager import APTWorkflowManager

# Use shared plotting
success = create_tactics_timeline_plot(
    full_df=dataset,
    tactic_colors=TacticColors.STANDARD,
    output_path=Path("timeline.png"),
    logger=logger
)

# Path management
path_manager = PathManager('apt-1', '04')
suggestions = path_manager.suggest_workflow_next_steps()

# Workflow automation
workflow = APTWorkflowManager('apt-1', '04', skip_script6=True)
success = workflow.run_workflow()
```

### Configuration Management
```python
from utils.apt_config import TacticColors, PlottingConfig, FeatureFlags

# Access centralized colors
colors = TacticColors.STANDARD
discovery_color = colors['discovery']  # '#8B4513'

# Plot configuration
fig_size = PlottingConfig.FIGURE_SIZE_TIMELINE  # (16, 10)
dpi = PlottingConfig.DPI  # 300

# Feature control
if FeatureFlags.USE_SHARED_PLOTTING:
    # Use enhanced features
    pass
```

### Workflow Automation
```bash
# From exploratory directory
python3 utils/apt_workflow_manager.py --apt-type apt-1 --run-id 04

# With options
python3 utils/apt_workflow_manager.py \
  --apt-type apt-1 \
  --run-id 04 \
  --skip-script6 \
  --debug
```

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure you're running from the main exploratory directory
2. **Missing Dependencies**: All original script dependencies still required
3. **Feature Flag Issues**: Check `FeatureFlags.USE_SHARED_PLOTTING` setting
4. **Path Problems**: Workflow manager handles path detection automatically

### Debug Mode
Enable detailed logging in any utility:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Rollback Strategy
Disable shared utilities if needed:
```python
# In apt_config.py
class FeatureFlags:
    USE_SHARED_PLOTTING = False  # Disable shared utilities
```

## Version Information
- **Version**: 1.0.0
- **Created**: August 2025
- **Compatibility**: Python 3.7+
- **Dependencies**: pandas, numpy, matplotlib, seaborn

## Future Enhancements
- **Data Caching**: Cache frequently loaded datasets
- **Memory Optimization**: Chunk processing for large datasets
- **Enhanced Validation**: More comprehensive data quality checks
- **Parallel Processing**: Multi-threaded workflow execution