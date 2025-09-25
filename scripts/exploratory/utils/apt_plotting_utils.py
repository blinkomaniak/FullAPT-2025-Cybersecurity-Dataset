#!/usr/bin/env python3
"""
Shared plotting utilities for APT analysis scripts.

This module provides centralized plotting functions to eliminate code duplication
between Script 6 and Script 7, while maintaining identical output behavior.

Used by: 6_sysmon_attack_lifecycle_tracer.py, 7_create_labeled_sysmon_dataset.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging

from .apt_config import TacticColors, PlottingConfig


def setup_plot_style():
    """Setup consistent plot styling across all functions."""
    plt.style.use('default')  # Reset to default style
    

def create_simple_timeline_plot(
    malicious_df: pd.DataFrame,
    tactic_colors: Dict[str, str],
    output_path: Path,
    title_suffix: str = "",
    logger: Optional[logging.Logger] = None
) -> bool:
    """
    Create group timeline plot - extracted from both scripts.
    
    Args:
        malicious_df: DataFrame containing malicious events only
        tactic_colors: Dictionary mapping tactics to colors
        output_path: Path where to save the plot
        title_suffix: Optional suffix for the plot title
        logger: Optional logger for status messages
        
    Returns:
        bool: True if plot created successfully, False otherwise
    """
    if logger:
        logger.info(f"📈 Creating group timeline plot{title_suffix}")
    
    # Ensure datetime column exists (preserve if already correctly converted)
    if 'datetime' not in malicious_df.columns:
        if 'timestamp' in malicious_df.columns:
            malicious_df = malicious_df.copy()
            # Check if timestamp is already datetime or needs conversion from milliseconds
            if pd.api.types.is_numeric_dtype(malicious_df['timestamp']):
                # Numeric timestamps - assume milliseconds
                malicious_df['datetime'] = pd.to_datetime(malicious_df['timestamp'], unit='ms', errors='coerce')
            else:
                # String or already converted timestamps
                malicious_df['datetime'] = pd.to_datetime(malicious_df['timestamp'], errors='coerce')
        else:
            if logger:
                logger.error("❌ No timestamp/datetime column found")
            return False
    else:
        # Datetime column already exists - validate it's not corrupted
        if malicious_df['datetime'].dtype == 'object':
            # Might be string datetimes that need parsing
            malicious_df = malicious_df.copy()
            malicious_df['datetime'] = pd.to_datetime(malicious_df['datetime'], errors='coerce')
        # If already datetime64, leave it alone
    
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
        if logger:
            logger.warning("⚠️ No computers found for group timeline")
        return False
    
    # Create subplots for each computer
    fig_height = max(8 * len(computers), 6)  # Minimum height of 6
    fig, axes = plt.subplots(len(computers), 1, 
                           figsize=(PlottingConfig.FIGURE_SIZE_GROUP[0], fig_height), 
                           sharex=True)
    if len(computers) == 1:
        axes = [axes]
    
    # Plot each computer's events
    for i, (computer, ax) in enumerate(zip(computers, axes)):
        computer_events = malicious_df[malicious_df['Computer'] == computer].copy()
        
        if len(computer_events) == 0:
            continue
        
        # Group by tactic for visual distinction
        unique_tactics = computer_events['Tactic'].unique()
        
        for j, tactic in enumerate(sorted(unique_tactics)):
            tactic_events = computer_events[computer_events['Tactic'] == tactic]
            
            # Use consistent tactic color
            color = tactic_colors.get(tactic, '#000000')  # Default to black if tactic not found
            
            ax.scatter(tactic_events['datetime'], tactic_events['EventID'],
                     color=color, 
                     s=PlottingConfig.MALICIOUS_POINT_SIZE, 
                     alpha=PlottingConfig.POINT_ALPHA, 
                     label=f'{tactic} ({len(tactic_events)})',
                     edgecolors=PlottingConfig.POINT_EDGE_COLOR, 
                     linewidths=PlottingConfig.POINT_EDGE_WIDTH)
        
        # Customize subplot
        computer_short = computer.replace('.boombox.local', '').replace('.local', '')
        ax.set_ylabel('EventID')
        ax.set_title(f"Attack Events Timeline: {computer_short}")
        ax.grid(True, alpha=PlottingConfig.GRID_ALPHA)
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
        plt.setp(axes[-1].xaxis.get_ticklabels(), 
                rotation=PlottingConfig.ROTATION_ANGLE, ha='right')
    
    plt.suptitle(f'Multi-EventID Attack Progression - Group Timeline{title_suffix}\n\n', 
                fontsize=16, y=0.98)
    plt.tight_layout()
    
    # Save plot
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=PlottingConfig.DPI, bbox_inches='tight')
    plt.close()
    
    if logger:
        logger.info(f"✅ Group timeline plot saved: {output_path}")
    
    return True


def create_tactics_timeline_plot(
    full_df: pd.DataFrame,
    tactic_colors: Dict[str, str],
    output_path: Path,
    title_suffix: str = "",
    logger: Optional[logging.Logger] = None
) -> bool:
    """
    Create tactics timeline plot with benign background - extracted from both scripts.
    
    Args:
        full_df: DataFrame containing all events (malicious + benign)
        tactic_colors: Dictionary mapping tactics to colors
        output_path: Path where to save the plot
        title_suffix: Optional suffix for the plot title
        logger: Optional logger for status messages
        
    Returns:
        bool: True if plot created successfully, False otherwise
    """
    if logger:
        logger.info(f"📈 Creating tactics timeline plot{title_suffix}")
    
    # Create single plot for all events
    fig, ax = plt.subplots(1, 1, figsize=PlottingConfig.FIGURE_SIZE_TIMELINE)
    
    # Ensure datetime column exists (preserve if already correctly converted)
    full_df = full_df.copy()
    if 'datetime' not in full_df.columns:
        if 'timestamp' in full_df.columns:
            # Check if timestamp is already datetime or needs conversion from milliseconds
            if pd.api.types.is_numeric_dtype(full_df['timestamp']):
                # Numeric timestamps - assume milliseconds
                full_df['datetime'] = pd.to_datetime(full_df['timestamp'], unit='ms', errors='coerce')
            else:
                # String or already converted timestamps
                full_df['datetime'] = pd.to_datetime(full_df['timestamp'], errors='coerce')
        else:
            if logger:
                logger.error("❌ No timestamp/datetime column found")
            return False
    else:
        # Datetime column already exists - validate it's not corrupted
        if full_df['datetime'].dtype == 'object':
            # Might be string datetimes that need parsing
            full_df['datetime'] = pd.to_datetime(full_df['datetime'], errors='coerce')
        # If already datetime64, leave it alone
    
    # Remove any invalid timestamps
    invalid_timestamps = full_df['datetime'].isna().sum()
    if invalid_timestamps > 0:
        if logger:
            logger.warning(f"⚠️ Removed {invalid_timestamps} events with invalid timestamps")
        full_df = full_df.dropna(subset=['datetime'])
    
    # First, plot ALL Sysmon events as pale gray background
    if logger:
        logger.info("🎨 Plotting all Sysmon events as background...")
    
    benign_events = full_df[full_df['Label'] == 'Benign'] if 'Label' in full_df.columns else pd.DataFrame()
    if len(benign_events) > 0:
        ax.scatter(benign_events['datetime'], benign_events['EventID'], 
                  c=TacticColors.BENIGN_EVENT_COLOR, 
                  alpha=TacticColors.BENIGN_EVENT_ALPHA, 
                  s=PlottingConfig.BENIGN_POINT_SIZE, 
                  label=f'Benign Events ({len(benign_events):,})', zorder=1)
        if logger:
            logger.info(f"📊 Plotted {len(benign_events):,} benign events as background")
    
    # Now organize and plot malicious events by tactic
    malicious_events = full_df[full_df['Label'] == 'Malicious'] if 'Label' in full_df.columns else full_df
    tactics_events = {}
    for tactic in malicious_events['Tactic'].unique():
        tactics_events[tactic] = malicious_events[malicious_events['Tactic'] == tactic]
    
    if not tactics_events:
        if logger:
            logger.warning("⚠️ No malicious tactics found for timeline")
        return False
    
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
                   c=tactic_color, 
                   alpha=PlottingConfig.POINT_ALPHA, 
                   s=PlottingConfig.MALICIOUS_POINT_SIZE, 
                   label=f'{tactic.title()} ({len(tactic_df)} events)', zorder=2)
        
        total_malicious_events += len(tactic_df)
        if logger:
            logger.info(f"📊 Plotted {len(tactic_df)} events for tactic: {tactic}")
    
    if logger:
        logger.info(f"📊 Total malicious events plotted: {total_malicious_events}")
    
    # Customize plot
    ax.set_ylabel('EventID')
    ax.set_xlabel('Timeline')
    ax.grid(True, alpha=PlottingConfig.GRID_ALPHA)
    
    # Set y-axis ticks based on actual EventIDs present in the data
    all_eventids = set(full_df['EventID'].unique())
    
    if all_eventids:
        sorted_eventids = sorted(all_eventids)
        ax.set_yticks(sorted_eventids)
        ax.set_yticklabels(sorted_eventids)
        ax.set_ylim(min(sorted_eventids) - 0.5, max(sorted_eventids) + 0.5)
    else:
        # Fallback to common EventIDs if no data
        ax.set_yticks(PlottingConfig.DEFAULT_EVENT_IDS)
        ax.set_ylim(0, 25)
    
    # Format time axis
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax.xaxis.get_ticklabels(), rotation=PlottingConfig.ROTATION_ANGLE, ha='right')
    
    # Set main title with event counts
    total_events = len(full_df)
    benign_count = len(benign_events) if len(benign_events) > 0 else total_events - total_malicious_events
    
    ax.set_title(f'Complete Sysmon Timeline with MITRE Tactics Highlighting{title_suffix}\n'
                f'Total Events: {total_events:,} | Malicious: {total_malicious_events:,} | Benign: {benign_count:,}', 
                fontsize=14, fontweight='bold')
    
    # Add legend
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0),
             title='MITRE Tactics', title_fontsize=12, fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=PlottingConfig.DPI, bbox_inches='tight')
    plt.close()
    
    if logger:
        logger.info(f"✅ Tactics timeline plot saved: {output_path}")
    
    return True


def validate_plotting_data(df: pd.DataFrame, required_columns: List[str]) -> Tuple[bool, List[str]]:
    """
    Validate that DataFrame has required columns for plotting.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        
    Returns:
        Tuple of (is_valid, missing_columns)
    """
    if df is None or df.empty:
        return False, ["DataFrame is empty"]
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    return len(missing_cols) == 0, missing_cols


def get_plot_filename(base_name: str, version: str = "") -> str:
    """
    Generate consistent plot filenames.
    
    Args:
        base_name: Base filename without extension
        version: Version suffix (e.g., "v2")
        
    Returns:
        Filename with appropriate suffix
    """
    if version:
        return f"{base_name}_{version}.png"
    else:
        return f"{base_name}.png"


# Convenience functions for common plot types
def plot_simple_timeline_v2(malicious_df: pd.DataFrame, output_dir: Path, 
                           logger: Optional[logging.Logger] = None) -> bool:
    """Create v2 simple timeline plot with standard settings."""
    output_path = output_dir / PlottingConfig.TIMELINE_SIMPLE_V2
    return create_simple_timeline_plot(
        malicious_df, TacticColors.STANDARD, output_path, " (v2)", logger
    )


def plot_tactics_timeline_v2(full_df: pd.DataFrame, output_dir: Path,
                           logger: Optional[logging.Logger] = None) -> bool:
    """Create v2 tactics timeline plot with standard settings."""
    output_path = output_dir / PlottingConfig.TIMELINE_TACTICS_V2  
    return create_tactics_timeline_plot(
        full_df, TacticColors.STANDARD, output_path, " (v2)", logger
    )


def plot_simple_timeline_v1(malicious_df: pd.DataFrame, output_dir: Path,
                           logger: Optional[logging.Logger] = None) -> bool:
    """Create v1 simple timeline plot with standard settings."""
    output_path = output_dir / PlottingConfig.TIMELINE_SIMPLE
    return create_simple_timeline_plot(
        malicious_df, TacticColors.STANDARD, output_path, "", logger
    )


def plot_tactics_timeline_v1(full_df: pd.DataFrame, output_dir: Path,
                           logger: Optional[logging.Logger] = None) -> bool:
    """Create v1 tactics timeline plot with standard settings."""
    output_path = output_dir / PlottingConfig.TIMELINE_TACTICS
    return create_tactics_timeline_plot(
        full_df, TacticColors.STANDARD, output_path, "", logger
    )