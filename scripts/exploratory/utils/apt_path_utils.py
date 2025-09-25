#!/usr/bin/env python3
"""
Path utilities for APT analysis scripts.

This module provides centralized path management and file validation
to reduce code duplication and improve consistency.

Used by: 6_sysmon_attack_lifecycle_tracer.py, 7_create_labeled_sysmon_dataset.py
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
import os

from .apt_config import FilePaths, ValidationConfig


class PathManager:
    """Manages all file paths for APT analysis workflow."""
    
    def __init__(self, apt_type: str, run_id: str, logger: Optional[logging.Logger] = None):
        """
        Initialize path manager for specific APT run.
        
        Args:
            apt_type: APT type (e.g., 'apt-1')
            run_id: Run ID (e.g., '04')
            logger: Optional logger for status messages
        """
        self.apt_type = apt_type
        self.run_id = run_id
        self.logger = logger
        
        # Initialize all paths
        self.base_path = FilePaths.get_base_path(apt_type, run_id)
        self.results_dir = FilePaths.get_results_dir(apt_type, run_id)
        
        # Input files
        self.sysmon_file = FilePaths.get_sysmon_file(apt_type, run_id)
        self.master_tactics_file = FilePaths.get_master_tactics_file(apt_type, run_id)
        self.traced_events_file = FilePaths.get_traced_events_file(apt_type, run_id, "v1")
        self.traced_events_v2_file = FilePaths.get_traced_events_file(apt_type, run_id, "v2")
        
        # Output files
        self.labeled_dataset_file = FilePaths.get_labeled_dataset_file(apt_type, run_id, "v1")
        self.labeled_dataset_v2_file = FilePaths.get_labeled_dataset_file(apt_type, run_id, "v2")
        
        # Ensure results directory exists
        self.results_dir.mkdir(exist_ok=True, parents=True)
    
    def validate_input_files(self, required_files: Optional[List[str]] = None) -> Tuple[bool, List[str]]:
        """
        Validate that required input files exist.
        
        Args:
            required_files: List of file types to check. If None, checks all standard files.
            
        Returns:
            Tuple of (all_valid, missing_files)
        """
        if required_files is None:
            required_files = ['sysmon', 'master_tactics']
        
        file_map = {
            'sysmon': self.sysmon_file,
            'master_tactics': self.master_tactics_file,
            'traced_events': self.traced_events_file,
            'traced_events_v2': self.traced_events_v2_file,
            'labeled_dataset': self.labeled_dataset_file,
            'labeled_dataset_v2': self.labeled_dataset_v2_file
        }
        
        missing_files = []
        for file_type in required_files:
            if file_type in file_map:
                file_path = file_map[file_type]
                if not file_path.exists():
                    missing_files.append(f"{file_type}: {file_path}")
                    if self.logger:
                        self.logger.warning(f"⚠️ Missing file: {file_path}")
                else:
                    # Check file size
                    try:
                        size_mb = file_path.stat().st_size / (1024 * 1024)
                        if size_mb > ValidationConfig.MAX_FILE_SIZE_MB:
                            if self.logger:
                                self.logger.warning(f"⚠️ Large file ({size_mb:.1f}MB): {file_path}")
                    except OSError:
                        pass
        
        all_valid = len(missing_files) == 0
        
        if self.logger:
            if all_valid:
                self.logger.info(f"✅ All required files validated for {self.apt_type}-run-{self.run_id}")
            else:
                self.logger.error(f"❌ Missing {len(missing_files)} required files")
        
        return all_valid, missing_files
    
    def get_output_file_paths(self, version: str = "v1") -> Dict[str, Path]:
        """
        Get all output file paths for a specific version.
        
        Args:
            version: Version suffix ('v1' or 'v2')
            
        Returns:
            Dictionary mapping file types to paths
        """
        suffix = f"_{version}" if version == "v2" else ""
        
        return {
            'timeline_simple': self.results_dir / f"timeline_all_malicious_events{suffix}.png",
            'timeline_tactics': self.results_dir / f"timeline_all_malicious_events_with_tactics{suffix}.png",
            'json_results': self.results_dir / "multi_eventid_analysis_results.json",
            'csv_traced_events': self.results_dir / f"traced_sysmon_events_with_tactics{suffix}.csv",
            'labeled_dataset': self.labeled_dataset_v2_file if version == "v2" else self.labeled_dataset_file
        }
    
    def clean_old_outputs(self, version: str = "v1", confirm: bool = True) -> List[Path]:
        """
        Clean old output files for a specific version.
        
        Args:
            version: Version to clean ('v1' or 'v2')
            confirm: Whether to actually delete files (if False, just returns list)
            
        Returns:
            List of files that were (or would be) deleted
        """
        output_files = self.get_output_file_paths(version)
        deleted_files = []
        
        for file_type, file_path in output_files.items():
            if file_path.exists():
                if confirm:
                    try:
                        file_path.unlink()
                        deleted_files.append(file_path)
                        if self.logger:
                            self.logger.info(f"🗑️ Deleted old file: {file_path}")
                    except OSError as e:
                        if self.logger:
                            self.logger.error(f"❌ Failed to delete {file_path}: {e}")
                else:
                    deleted_files.append(file_path)
        
        return deleted_files
    
    def get_file_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get comprehensive information about all files.
        
        Returns:
            Dictionary with file information including existence, size, modification time
        """
        files_to_check = {
            'sysmon_file': self.sysmon_file,
            'master_tactics_file': self.master_tactics_file,
            'traced_events_file': self.traced_events_file,
            'traced_events_v2_file': self.traced_events_v2_file,
            'labeled_dataset_file': self.labeled_dataset_file,
            'labeled_dataset_v2_file': self.labeled_dataset_v2_file
        }
        
        file_info = {}
        for name, path in files_to_check.items():
            info = {
                'path': str(path),
                'exists': path.exists(),
                'size_mb': 0,
                'modified': None
            }
            
            if path.exists():
                try:
                    stat = path.stat()
                    info['size_mb'] = stat.st_size / (1024 * 1024)
                    info['modified'] = stat.st_mtime
                except OSError:
                    pass
            
            file_info[name] = info
        
        return file_info
    
    def suggest_workflow_next_steps(self) -> List[str]:
        """
        Analyze current file state and suggest next workflow steps.
        
        Returns:
            List of suggested actions
        """
        suggestions = []
        file_info = self.get_file_info()
        
        # Check basic requirements
        if not file_info['sysmon_file']['exists']:
            suggestions.append("❌ Missing sysmon dataset - cannot proceed")
            return suggestions
        
        if not file_info['master_tactics_file']['exists']:
            suggestions.append("❌ Missing master tactics file - cannot proceed")
            return suggestions
        
        # Workflow suggestions based on current state
        if not file_info['traced_events_file']['exists']:
            suggestions.append("🚀 Run Script 6 to create traced events")
        
        elif file_info['traced_events_file']['exists'] and not file_info['traced_events_v2_file']['exists']:
            suggestions.append("✏️ Manual step: Copy traced events to v2 file and add Correct_SeedRowNumber column")
            suggestions.append(f"   cp {self.traced_events_file} {self.traced_events_v2_file}")
        
        elif file_info['traced_events_v2_file']['exists'] and not file_info['labeled_dataset_v2_file']['exists']:
            suggestions.append("🏷️ Run Script 7 to create labeled dataset")
        
        else:
            suggestions.append("✅ All files present - workflow complete")
            
            # Check if files need updates
            if (file_info['traced_events_file']['exists'] and 
                file_info['traced_events_v2_file']['exists']):
                
                v1_modified = file_info['traced_events_file']['modified']
                v2_modified = file_info['traced_events_v2_file']['modified']
                
                if v1_modified and v2_modified and v1_modified > v2_modified:
                    suggestions.append("⚠️ v1 traced events newer than v2 - consider updating v2 file")
        
        return suggestions


def create_backup_path(original_path: Path, suffix: str = None) -> Path:
    """
    Create a backup file path with timestamp.
    
    Args:
        original_path: Path to original file
        suffix: Optional suffix (if None, uses current timestamp)
        
    Returns:
        Path for backup file
    """
    if suffix is None:
        from datetime import datetime
        suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    stem = original_path.stem
    extension = original_path.suffix
    backup_name = f"{stem}_BACKUP_{suffix}{extension}"
    
    return original_path.parent / backup_name


def safe_file_operation(source_path: Path, dest_path: Path, operation: str = "copy",
                       backup: bool = True, logger: Optional[logging.Logger] = None) -> bool:
    """
    Perform safe file operations with optional backup.
    
    Args:
        source_path: Source file path
        dest_path: Destination file path  
        operation: Operation type ('copy', 'move', 'backup')
        backup: Whether to create backup of destination if it exists
        logger: Optional logger
        
    Returns:
        True if operation successful, False otherwise
    """
    import shutil
    
    try:
        # Create backup if destination exists and backup requested
        if backup and dest_path.exists():
            backup_path = create_backup_path(dest_path)
            shutil.copy2(dest_path, backup_path)
            if logger:
                logger.info(f"📋 Created backup: {backup_path}")
        
        # Perform operation
        if operation == "copy":
            shutil.copy2(source_path, dest_path)
        elif operation == "move":
            shutil.move(source_path, dest_path)
        elif operation == "backup":
            backup_path = create_backup_path(source_path)
            shutil.copy2(source_path, backup_path)
            dest_path = backup_path
        
        if logger:
            logger.info(f"✅ {operation.title()} completed: {source_path} → {dest_path}")
        
        return True
        
    except Exception as e:
        if logger:
            logger.error(f"❌ {operation.title()} failed: {e}")
        return False


def find_files_by_pattern(directory: Path, pattern: str, 
                         recursive: bool = True) -> List[Path]:
    """
    Find files matching a pattern in directory.
    
    Args:
        directory: Directory to search
        pattern: Glob pattern (e.g., "*.csv", "sysmon-*.csv")
        recursive: Whether to search subdirectories
        
    Returns:
        List of matching file paths
    """
    if not directory.exists():
        return []
    
    if recursive:
        return list(directory.rglob(pattern))
    else:
        return list(directory.glob(pattern))