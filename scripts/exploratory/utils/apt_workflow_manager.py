#!/usr/bin/env python3
"""
APT Analysis Workflow Manager

This module provides orchestration for the complete APT analysis workflow,
managing the execution of Script 6 and Script 7 in the correct order with
proper dependency checking and error handling.

Features:
- Automated workflow orchestration
- Dependency validation
- Progress tracking
- Error recovery
- File management
- Comprehensive reporting

Usage:
    python3 apt_workflow_manager.py --apt-type apt-1 --run-id 04
    python3 apt_workflow_manager.py --apt-type apt-1 --run-id 04 --skip-script6
    python3 apt_workflow_manager.py --apt-type apt-1 --run-id 04 --force-rebuild
"""

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import time

# Import shared utilities
try:
    from .apt_config import FeatureFlags, ValidationConfig
    from .apt_path_utils import PathManager, safe_file_operation
    SHARED_UTILS_AVAILABLE = True
except ImportError:
    print("⚠️ Warning: Shared utilities not available")
    SHARED_UTILS_AVAILABLE = False


class WorkflowStep:
    """Represents a single step in the workflow."""
    
    def __init__(self, name: str, description: str, required: bool = True):
        self.name = name
        self.description = description
        self.required = required
        self.status = "pending"  # pending, running, completed, failed, skipped
        self.start_time = None
        self.end_time = None
        self.error_message = None
        self.output_files = []
        
    def start(self):
        """Mark step as started."""
        self.status = "running"
        self.start_time = datetime.now()
    
    def complete(self, output_files: List[str] = None):
        """Mark step as completed."""
        self.status = "completed"
        self.end_time = datetime.now()
        self.output_files = output_files or []
    
    def fail(self, error_message: str):
        """Mark step as failed."""
        self.status = "failed"
        self.end_time = datetime.now()
        self.error_message = error_message
    
    def skip(self, reason: str = ""):
        """Mark step as skipped."""
        self.status = "skipped"
        self.error_message = reason
    
    @property
    def duration(self) -> Optional[float]:
        """Get step duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert step to dictionary for JSON serialization."""
        return {
            'name': self.name,
            'description': self.description,
            'required': self.required,
            'status': self.status,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration': self.duration,
            'error_message': self.error_message,
            'output_files': self.output_files
        }


class APTWorkflowManager:
    """Manages the complete APT analysis workflow."""
    
    def __init__(self, apt_type: str, run_id: str, 
                 skip_script6: bool = False, 
                 force_rebuild: bool = False,
                 debug: bool = False):
        """
        Initialize workflow manager.
        
        Args:
            apt_type: APT type (e.g., 'apt-1')
            run_id: Run ID (e.g., '04')
            skip_script6: Skip Script 6 execution
            force_rebuild: Force rebuild of all outputs
            debug: Enable debug logging
        """
        self.apt_type = apt_type
        self.run_id = run_id
        self.skip_script6 = skip_script6
        self.force_rebuild = force_rebuild
        self.debug = debug
        
        # Initialize logger
        self.logger = self._setup_logger()
        
        # Initialize path manager if available
        if SHARED_UTILS_AVAILABLE:
            self.path_manager = PathManager(apt_type, run_id, self.logger)
        else:
            self.path_manager = None
        
        # Define workflow steps
        self.steps = [
            WorkflowStep("validate_inputs", "Validate input files and dependencies"),
            WorkflowStep("run_script6", "Execute Script 6 (Lifecycle Tracer)", not skip_script6),
            WorkflowStep("check_manual_corrections", "Check for manual corrections needed"),
            WorkflowStep("run_script7", "Execute Script 7 (Labeled Dataset Creator)"),
            WorkflowStep("validate_outputs", "Validate all output files"),
            WorkflowStep("generate_report", "Generate workflow report")
        ]
        
        # Workflow state
        self.workflow_start_time = None
        self.workflow_end_time = None
        self.current_step = None
        
    def _setup_logger(self) -> logging.Logger:
        """Set up logging configuration."""
        level = logging.DEBUG if self.debug else logging.INFO
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        return logging.getLogger(f"APTWorkflow_{self.apt_type}_{self.run_id}")
    
    def run_workflow(self) -> bool:
        """
        Execute the complete workflow.
        
        Returns:
            True if workflow completed successfully, False otherwise
        """
        self.workflow_start_time = datetime.now()
        self.logger.info(f"🚀 Starting APT analysis workflow for {self.apt_type.upper()}-Run-{self.run_id}")
        
        try:
            for step in self.steps:
                if not step.required and step.name == "run_script6" and self.skip_script6:
                    step.skip("Skipped by user request")
                    self.logger.info(f"⏩ Skipping step: {step.description}")
                    continue
                
                self.current_step = step
                self.logger.info(f"▶️ Starting step: {step.description}")
                
                step.start()
                success = self._execute_step(step)
                
                if success:
                    self.logger.info(f"✅ Completed step: {step.description} ({step.duration:.1f}s)")
                else:
                    self.logger.error(f"❌ Failed step: {step.description}")
                    if step.required:
                        self.logger.error("🛑 Required step failed - aborting workflow")
                        return False
                    else:
                        self.logger.warning("⚠️ Optional step failed - continuing workflow")
            
            self.workflow_end_time = datetime.now()
            total_duration = (self.workflow_end_time - self.workflow_start_time).total_seconds()
            self.logger.info(f"🎉 Workflow completed successfully in {total_duration:.1f} seconds!")
            return True
            
        except Exception as e:
            self.workflow_end_time = datetime.now()
            self.logger.error(f"💥 Workflow failed with exception: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _execute_step(self, step: WorkflowStep) -> bool:
        """
        Execute a single workflow step.
        
        Args:
            step: WorkflowStep to execute
            
        Returns:
            True if step succeeded, False otherwise
        """
        try:
            if step.name == "validate_inputs":
                return self._validate_inputs(step)
            elif step.name == "run_script6":
                return self._run_script6(step)
            elif step.name == "check_manual_corrections":
                return self._check_manual_corrections(step)
            elif step.name == "run_script7":
                return self._run_script7(step)
            elif step.name == "validate_outputs":
                return self._validate_outputs(step)
            elif step.name == "generate_report":
                return self._generate_report(step)
            else:
                step.fail(f"Unknown step: {step.name}")
                return False
                
        except Exception as e:
            step.fail(str(e))
            return False
    
    def _validate_inputs(self, step: WorkflowStep) -> bool:
        """Validate input files and dependencies."""
        required_files = ['sysmon', 'master_tactics']
        
        if self.path_manager:
            is_valid, missing_files = self.path_manager.validate_input_files(required_files)
            
            if not is_valid:
                step.fail(f"Missing required files: {missing_files}")
                return False
            
            step.complete([str(self.path_manager.sysmon_file), str(self.path_manager.master_tactics_file)])
        else:
            # Basic validation without path manager
            base_path = Path(f"/home/researcher/Downloads/research/dataset/{self.apt_type}/{self.apt_type}-run-{self.run_id}")
            sysmon_file = base_path / f"sysmon-run-{self.run_id}.csv"
            master_file = base_path / f"all_target_events_run-{self.run_id}.csv"
            
            missing = []
            if not sysmon_file.exists():
                missing.append(str(sysmon_file))
            if not master_file.exists():
                missing.append(str(master_file))
            
            if missing:
                step.fail(f"Missing required files: {missing}")
                return False
            
            step.complete([str(sysmon_file), str(master_file)])
        
        return True
    
    def _run_script6(self, step: WorkflowStep) -> bool:
        """Execute Script 6 (Lifecycle Tracer)."""
        # Check if running from utils folder or main folder
        if Path("utils").exists():
            # Running from main exploratory folder
            script_path = Path("6_sysmon_attack_lifecycle_tracer.py")
        else:
            # Running from utils folder
            script_path = Path("../6_sysmon_attack_lifecycle_tracer.py")
            
        if not script_path.exists():
            step.fail(f"Script 6 not found: {script_path}")
            return False
        
        cmd = [
            "python3", 
            str(script_path),
            "--apt-type", self.apt_type,
            "--run-id", self.run_id
        ]
        
        self.logger.info(f"🔧 Running command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=1800  # 30 minutes timeout
            )
            
            if result.returncode == 0:
                # Look for output files
                if self.path_manager:
                    output_paths = self.path_manager.get_output_file_paths("v1")
                    existing_outputs = [str(p) for p in output_paths.values() if p.exists()]
                    step.complete(existing_outputs)
                else:
                    step.complete([])
                
                self.logger.debug(f"Script 6 stdout: {result.stdout}")
                return True
            else:
                step.fail(f"Script 6 failed with code {result.returncode}: {result.stderr}")
                self.logger.error(f"Script 6 stderr: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            step.fail("Script 6 timed out (30 minutes)")
            return False
        except Exception as e:
            step.fail(f"Error running Script 6: {e}")
            return False
    
    def _check_manual_corrections(self, step: WorkflowStep) -> bool:
        """Check if manual corrections are needed or available."""
        if self.path_manager:
            v1_file = self.path_manager.traced_events_file
            v2_file = self.path_manager.traced_events_v2_file
            
            if not v1_file.exists():
                step.fail("Script 6 output file not found - cannot check for manual corrections")
                return False
            
            if v2_file.exists():
                self.logger.info("✅ Manual corrections file (v2) found")
                step.complete([str(v2_file)])
            else:
                self.logger.warning("⚠️ No manual corrections file (v2) found")
                self.logger.info(f"💡 Suggestion: Copy {v1_file} to {v2_file} and add Correct_SeedRowNumber column")
                
                # Auto-create v2 file if v1 exists
                if safe_file_operation(v1_file, v2_file, "copy", backup=False, logger=self.logger):
                    self.logger.info("📋 Auto-created v2 file from v1 for workflow continuity")
                    step.complete([str(v2_file)])
                else:
                    step.fail("Could not create v2 file from v1")
                    return False
        else:
            # Basic check without path manager
            step.complete([])  # Assume it's okay
        
        return True
    
    def _run_script7(self, step: WorkflowStep) -> bool:
        """Execute Script 7 (Labeled Dataset Creator)."""
        # Check if running from utils folder or main folder
        if Path("utils").exists():
            # Running from main exploratory folder
            script_path = Path("7_create_labeled_sysmon_dataset.py")
        else:
            # Running from utils folder
            script_path = Path("../7_create_labeled_sysmon_dataset.py")
            
        if not script_path.exists():
            step.fail(f"Script 7 not found: {script_path}")
            return False
        
        cmd = [
            "python3",
            str(script_path), 
            "--apt-type", self.apt_type,
            "--run-id", self.run_id
        ]
        
        self.logger.info(f"🔧 Running command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutes timeout
            )
            
            if result.returncode == 0:
                # Look for output files
                if self.path_manager:
                    output_paths = self.path_manager.get_output_file_paths("v2")
                    existing_outputs = [str(p) for p in output_paths.values() if p.exists()]
                    step.complete(existing_outputs)
                else:
                    step.complete([])
                
                self.logger.debug(f"Script 7 stdout: {result.stdout}")
                return True
            else:
                step.fail(f"Script 7 failed with code {result.returncode}: {result.stderr}")
                self.logger.error(f"Script 7 stderr: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            step.fail("Script 7 timed out (30 minutes)")
            return False
        except Exception as e:
            step.fail(f"Error running Script 7: {e}")
            return False
    
    def _validate_outputs(self, step: WorkflowStep) -> bool:
        """Validate all output files were created successfully."""
        if not self.path_manager:
            step.complete([])  # Skip validation if no path manager
            return True
        
        expected_files = []
        missing_files = []
        
        # Check v1 outputs (from Script 6)
        if not self.skip_script6:
            v1_paths = self.path_manager.get_output_file_paths("v1")
            for file_type, file_path in v1_paths.items():
                if file_type in ['timeline_simple', 'timeline_tactics', 'csv_traced_events']:
                    expected_files.append(file_path)
                    if not file_path.exists():
                        missing_files.append(str(file_path))
        
        # Check v2 outputs (from Script 7)
        v2_paths = self.path_manager.get_output_file_paths("v2")
        for file_type, file_path in v2_paths.items():
            if file_type in ['timeline_simple', 'timeline_tactics', 'labeled_dataset']:
                expected_files.append(file_path)
                if not file_path.exists():
                    missing_files.append(str(file_path))
        
        if missing_files:
            step.fail(f"Missing output files: {missing_files}")
            return False
        
        step.complete([str(f) for f in expected_files])
        self.logger.info(f"✅ Validated {len(expected_files)} output files")
        return True
    
    def _generate_report(self, step: WorkflowStep) -> bool:
        """Generate comprehensive workflow report."""
        try:
            report = self._create_workflow_report()
            
            # Save report to file
            if self.path_manager:
                report_file = self.path_manager.results_dir / f"workflow_report_{self.apt_type}_run_{self.run_id}.json"
            else:
                report_file = Path(f"workflow_report_{self.apt_type}_run_{self.run_id}.json")
            
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            step.complete([str(report_file)])
            self.logger.info(f"📊 Workflow report saved: {report_file}")
            
            # Print summary
            self._print_workflow_summary(report)
            
            return True
            
        except Exception as e:
            step.fail(f"Error generating report: {e}")
            return False
    
    def _create_workflow_report(self) -> Dict[str, Any]:
        """Create comprehensive workflow report."""
        total_duration = None
        if self.workflow_start_time and self.workflow_end_time:
            total_duration = (self.workflow_end_time - self.workflow_start_time).total_seconds()
        
        report = {
            'workflow_info': {
                'apt_type': self.apt_type,
                'run_id': self.run_id,
                'start_time': self.workflow_start_time.isoformat() if self.workflow_start_time else None,
                'end_time': self.workflow_end_time.isoformat() if self.workflow_end_time else None,
                'total_duration': total_duration,
                'skip_script6': self.skip_script6,
                'force_rebuild': self.force_rebuild
            },
            'steps': [step.to_dict() for step in self.steps],
            'summary': {
                'total_steps': len(self.steps),
                'completed_steps': len([s for s in self.steps if s.status == "completed"]),
                'failed_steps': len([s for s in self.steps if s.status == "failed"]),
                'skipped_steps': len([s for s in self.steps if s.status == "skipped"]),
                'success': all(s.status in ["completed", "skipped"] for s in self.steps if s.required)
            },
            'output_files': []
        }
        
        # Collect all output files
        for step in self.steps:
            if step.output_files:
                report['output_files'].extend(step.output_files)
        
        return report
    
    def _print_workflow_summary(self, report: Dict[str, Any]):
        """Print workflow summary to console."""
        print("\n" + "="*60)
        print(f"📊 APT WORKFLOW SUMMARY - {self.apt_type.upper()}-Run-{self.run_id}")
        print("="*60)
        
        summary = report['summary']
        if summary['success']:
            print("🎉 Status: SUCCESSFUL")
        else:
            print("❌ Status: FAILED")
        
        duration = report['workflow_info']['total_duration']
        if duration:
            print(f"⏱️  Duration: {duration:.1f} seconds")
        print(f"✅ Completed: {summary['completed_steps']}/{summary['total_steps']} steps")
        
        if summary['failed_steps'] > 0:
            print(f"❌ Failed: {summary['failed_steps']} steps")
        if summary['skipped_steps'] > 0:
            print(f"⏩ Skipped: {summary['skipped_steps']} steps")
        
        print(f"📁 Generated files: {len(report['output_files'])}")
        
        print("\n📋 Step Details:")
        for step in self.steps:
            status_icon = {
                "completed": "✅", "failed": "❌", "skipped": "⏩", 
                "running": "🔄", "pending": "⏳"
            }.get(step.status, "❓")
            
            duration_str = f" ({step.duration:.1f}s)" if step.duration else ""
            print(f"  {status_icon} {step.description}{duration_str}")
            
            if step.status == "failed" and step.error_message:
                print(f"      💬 {step.error_message}")
        
        print("="*60)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='APT Analysis Workflow Manager',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete workflow
  python3 apt_workflow_manager.py --apt-type apt-1 --run-id 04
  
  # Skip Script 6 (use existing outputs)
  python3 apt_workflow_manager.py --apt-type apt-1 --run-id 04 --skip-script6
  
  # Force rebuild all outputs
  python3 apt_workflow_manager.py --apt-type apt-1 --run-id 04 --force-rebuild
  
  # Enable debug logging
  python3 apt_workflow_manager.py --apt-type apt-1 --run-id 04 --debug
        """
    )
    
    parser.add_argument('--apt-type', type=str, required=True,
                       help='APT type (e.g., apt-1)')
    parser.add_argument('--run-id', type=str, required=True,
                       help='Run ID (e.g., 04)')
    parser.add_argument('--skip-script6', action='store_true',
                       help='Skip Script 6 execution (use existing outputs)')
    parser.add_argument('--force-rebuild', action='store_true',
                       help='Force rebuild of all outputs')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    
    args = parser.parse_args()
    
    try:
        workflow = APTWorkflowManager(
            apt_type=args.apt_type,
            run_id=args.run_id,
            skip_script6=args.skip_script6,
            force_rebuild=args.force_rebuild,
            debug=args.debug
        )
        
        success = workflow.run_workflow()
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n🛑 Workflow interrupted by user")
        return 130
    except Exception as e:
        print(f"💥 Workflow failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())