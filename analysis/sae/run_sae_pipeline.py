#!/usr/bin/env python3
"""
SAE Pipeline Orchestrator
=========================
This script runs the complete SAE (Stacked Autoencoder) pipeline from start to finish.
It executes all 5 stages in order for specified datasets with proper error handling,
logging, and progress tracking.

Usage:
    python run_sae_pipeline.py --dataset bigbase --version v1
    python run_sae_pipeline.py --dataset unraveled --version v1
    python run_sae_pipeline.py --dataset both --version v1  # Run both datasets

Features:
    - Automatic stage execution in correct order
    - Comprehensive error handling and logging
    - Progress tracking with timing information
    - Validation of prerequisites before each stage
    - Optional stage skipping for resuming interrupted runs
    - Remote server friendly (non-interactive where possible)
"""

import os
import sys
import subprocess
import argparse
import time
import json
from datetime import datetime
from pathlib import Path

# Add project root to path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.append(str(project_root))

from utils.paths import get_processed_path, get_encoded_dir, get_model_dir, get_eval_dir

class SAEPipelineOrchestrator:
    def __init__(self, dataset, version, model="sae", schema_config=None, encoding_config=None, skip_stages=None, verbose=True):
        self.dataset = dataset.lower()
        self.version = version
        self.model = model
        self.schema_config = schema_config
        self.encoding_config = encoding_config
        self.skip_stages = skip_stages or []
        self.verbose = verbose
        self.start_time = time.time()
        self.stage_times = {}
        
        # Validate dataset
        valid_datasets = ["bigbase", "unraveled"]
        if self.dataset not in valid_datasets:
            raise ValueError(f"Dataset must be one of: {valid_datasets}")
        
        # Setup paths
        self.script_dir = Path(__file__).parent
        self.project_root = self.script_dir.parent
        
        # Dataset-specific configurations
        self.configs = {
            "bigbase": {
                "schema": schema_config,
                "encoding": encoding_config,
                "encoding_script": "02-encoding-bigbase-sae.py",
                "subdir": ""  # No subdirectory for bigbase
            },
            "unraveled": {
                "schema": schema_config, 
                "encoding": encoding_config,
                "encoding_script": "02-encoding-unraveled-sae.py",
                "subdir": "network-flows"  # Subdirectory for unraveled
            }
        }
        
        # Initialize logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging for the pipeline run"""
        log_dir = self.project_root / "logs"
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"sae_pipeline_{self.dataset}_{self.version}_{timestamp}.log"
        
        self.log_file = log_file
        self.log(f"🚀 SAE Pipeline started at {datetime.now()}")
        self.log(f"📊 Dataset: {self.dataset}, Version: {self.version}, Model: {self.model}")
        
    def log(self, message, level="INFO"):
        """Log message to both console and file"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {level}: {message}"
        
        if self.verbose:
            print(log_message)
        
        with open(self.log_file, "a") as f:
            f.write(log_message + "\n")
    
    def run_command(self, cmd, stage_name, dataset=None):
        """Execute a command with proper error handling and logging"""
        ds = dataset or self.dataset
        self.log(f"🔧 Running {stage_name} for {ds}...")
        self.log(f"📝 Command: {' '.join(cmd)}")
        
        stage_start = time.time()
        
        try:
            # Change to script directory for relative imports
            result = subprocess.run(
                cmd,
                cwd=self.script_dir,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout per stage
            )
            
            stage_duration = time.time() - stage_start
            self.stage_times[f"{stage_name}_{ds}"] = stage_duration
            
            if result.returncode == 0:
                self.log(f"✅ {stage_name} completed successfully for {ds} ({stage_duration:.1f}s)")
                if result.stdout.strip():
                    self.log(f"📤 Output: {result.stdout.strip()}")
                return True
            else:
                self.log(f"❌ {stage_name} failed for {ds} (exit code: {result.returncode})", "ERROR")
                if result.stderr.strip():
                    self.log(f"🚨 Error: {result.stderr.strip()}", "ERROR")
                if result.stdout.strip():
                    self.log(f"📤 Output: {result.stdout.strip()}")
                return False
                
        except subprocess.TimeoutExpired:
            self.log(f"⏰ {stage_name} timed out for {ds}", "ERROR")
            return False
        except Exception as e:
            self.log(f"💥 Unexpected error in {stage_name} for {ds}: {str(e)}", "ERROR")
            return False
    
    def check_prerequisites(self, stage_num, dataset=None):
        """Check if prerequisites for a stage are met"""
        ds = dataset or self.dataset
        config = self.configs[ds]
        
        if stage_num == 1:
            # Check if raw data directory exists
            raw_data_path = self.project_root / "datasets" / ds
            if config["subdir"]:
                raw_data_path = raw_data_path / config["subdir"]
            
            if not raw_data_path.exists():
                self.log(f"❌ Raw data directory not found: {raw_data_path}", "ERROR")
                return False
            return True
            
        elif stage_num == 2:
            # Check if processed parquet exists
            processed_path = get_processed_path(self.model, ds, self.version)
            if not Path(processed_path).exists():
                self.log(f"❌ Processed data not found: {processed_path}", "ERROR")
                return False
            return True
            
        elif stage_num == 3:
            # Check if encoded data exists
            encoded_dir = get_encoded_dir(self.model, ds, self.version)
            required_files = ["X_train_encoded.npz", "X_val_encoded.npz", "y_train.npy", "y_val.npy"]
            
            for file in required_files:
                file_path = Path(encoded_dir) / file
                if not file_path.exists():
                    self.log(f"❌ Encoded data not found: {file_path}", "ERROR")
                    return False
            return True
            
        elif stage_num in [4, 5]:
            # Check if trained model exists
            model_dir = get_model_dir(self.model, ds, self.version)
            model_path = Path(model_dir) / f"{self.model}-model-{ds}-{self.version}.keras"
            
            if not model_path.exists():
                self.log(f"❌ Trained model not found: {model_path}", "ERROR")
                return False
            
            # For evaluation stages, also check test data
            encoded_dir = get_encoded_dir(self.model, ds, self.version)
            test_files = ["X_test_encoded.npz", "y_test.npy"]
            
            for file in test_files:
                file_path = Path(encoded_dir) / file
                if not file_path.exists():
                    self.log(f"❌ Test data not found: {file_path}", "ERROR")
                    return False
            return True
        
        return True
    
    def run_stage_1(self, dataset=None):
        """Stage 1: Data Aggregation"""
        ds = dataset or self.dataset
        config = self.configs[ds]
        
        if 1 in self.skip_stages:
            self.log(f"⏭️ Skipping Stage 1 for {ds}")
            return True
            
        if not self.check_prerequisites(1, ds):
            return False
        
        cmd = [
            sys.executable, "01-aggregation.py",
            "--model", self.model,
            "--dataset", ds,
            "--version", self.version,
            "--schema", config["schema"]
        ]
        
        if config["subdir"]:
            cmd.extend(["--subdir", config["subdir"]])
        
        return self.run_command(cmd, "Stage 1: Aggregation", ds)
    
    def run_stage_2(self, dataset=None):
        """Stage 2: Data Encoding"""
        ds = dataset or self.dataset
        config = self.configs[ds]
        
        if 2 in self.skip_stages:
            self.log(f"⏭️ Skipping Stage 2 for {ds}")
            return True
            
        if not self.check_prerequisites(2, ds):
            return False
        
        cmd = [
            sys.executable, config["encoding_script"],
            "--dataset", ds,
            "--version", self.version,
            "--model", self.model,
            "--encoding_config", config["encoding"]
        ]
        
        return self.run_command(cmd, "Stage 2: Encoding", ds)
    
    def run_stage_3(self, dataset=None):
        """Stage 3: Model Training"""
        ds = dataset or self.dataset
        
        if 3 in self.skip_stages:
            self.log(f"⏭️ Skipping Stage 3 for {ds}")
            return True
            
        if not self.check_prerequisites(3, ds):
            return False
        
        cmd = [
            sys.executable, "03-training-sae.py",
            "--dataset", ds,
            "--version", self.version,
            "--model", self.model
        ]
        
        return self.run_command(cmd, "Stage 3: Training", ds)
    
    def run_stage_4(self, dataset=None):
        """Stage 4: Model Evaluation"""
        ds = dataset or self.dataset
        
        if 4 in self.skip_stages:
            self.log(f"⏭️ Skipping Stage 4 for {ds}")
            return True
            
        if not self.check_prerequisites(4, ds):
            return False
        
        cmd = [
            sys.executable, "04-model-evaluation-sae.py",
            "--dataset", ds,
            "--version", self.version,
            "--model", self.model
        ]
        
        # Make evaluation non-interactive for remote servers
        env = os.environ.copy()
        env["PYTHON_INTERACTIVE"] = "0"
        
        self.log(f"🔧 Running Stage 4: Evaluation for {ds} (non-interactive mode)...")
        self.log(f"📝 Command: {' '.join(cmd)}")
        
        stage_start = time.time()
        
        try:
            # For evaluation, we need to handle the interactive prompt
            process = subprocess.Popen(
                cmd,
                cwd=self.script_dir,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env
            )
            
            # Send 'y' to recompute if prompted
            stdout, stderr = process.communicate(input='y\n', timeout=3600)
            
            stage_duration = time.time() - stage_start
            self.stage_times[f"Stage 4: Evaluation_{ds}"] = stage_duration
            
            if process.returncode == 0:
                self.log(f"✅ Stage 4: Evaluation completed successfully for {ds} ({stage_duration:.1f}s)")
                if stdout.strip():
                    self.log(f"📤 Output: {stdout.strip()}")
                return True
            else:
                self.log(f"❌ Stage 4: Evaluation failed for {ds} (exit code: {process.returncode})", "ERROR")
                if stderr.strip():
                    self.log(f"🚨 Error: {stderr.strip()}", "ERROR")
                return False
                
        except Exception as e:
            self.log(f"💥 Unexpected error in Stage 4: Evaluation for {ds}: {str(e)}", "ERROR")
            return False
    
    def run_stage_5(self, dataset=None):
        """Stage 5: Result Visualization"""
        ds = dataset or self.dataset
        
        if 5 in self.skip_stages:
            self.log(f"⏭️ Skipping Stage 5 for {ds}")
            return True
            
        if not self.check_prerequisites(5, ds):
            return False
        
        cmd = [
            sys.executable, "05-result-vis.py",
            "--dataset", ds,
            "--version", self.version,
            "--model", self.model
        ]
        
        return self.run_command(cmd, "Stage 5: Visualization", ds)
    
    def run_single_dataset(self, dataset):
        """Run the complete pipeline for a single dataset"""
        self.log(f"🎯 Starting pipeline for dataset: {dataset}")
        
        stages = [
            ("Stage 1: Aggregation", self.run_stage_1),
            ("Stage 2: Encoding", self.run_stage_2),
            ("Stage 3: Training", self.run_stage_3),
            ("Stage 4: Evaluation", self.run_stage_4),
            ("Stage 5: Visualization", self.run_stage_5)
        ]
        
        for stage_name, stage_func in stages:
            if not stage_func(dataset):
                self.log(f"🛑 Pipeline stopped at {stage_name} for {dataset}", "ERROR")
                return False
        
        self.log(f"🎉 Pipeline completed successfully for {dataset}")
        return True
    
    def run_pipeline(self):
        """Run the complete SAE pipeline"""
        try:
            # Run for single dataset
            return self.run_single_dataset(self.dataset)
                
        except KeyboardInterrupt:
            self.log("🚫 Pipeline interrupted by user", "WARNING")
            return False
        except Exception as e:
            self.log(f"💥 Unexpected pipeline error: {str(e)}", "ERROR")
            return False
        finally:
            self.finalize_logging()
    
    def finalize_logging(self):
        """Generate final pipeline report"""
        total_duration = time.time() - self.start_time
        
        self.log("=" * 60)
        self.log("📊 PIPELINE EXECUTION REPORT")
        self.log("=" * 60)
        self.log(f"🕐 Total Duration: {total_duration:.1f}s ({total_duration/60:.1f} minutes)")
        
        if self.stage_times:
            self.log("⏱️ Stage Timings:")
            for stage, duration in self.stage_times.items():
                self.log(f"  {stage}: {duration:.1f}s")
        
        self.log(f"📝 Full log saved to: {self.log_file}")
        self.log("=" * 60)

def main():
    parser = argparse.ArgumentParser(
        description="SAE Pipeline Orchestrator - Run complete SAE training pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run pipeline for bigbase dataset
  python run_sae_pipeline.py --dataset bigbase --version v1 --schema ../config/schema-bigbase-sae-v1.json --encoding_config ../config/encoding-bigbase-sae-v1.json
  
  # Run pipeline for unraveled dataset  
  python run_sae_pipeline.py --dataset unraveled --version v1 --schema ../config/schema-unraveled-sae-v1.json --encoding_config ../config/encoding-unraveled-sae-v1.json
  
  # Skip certain stages (e.g., if rerunning after failure)
  python run_sae_pipeline.py --dataset bigbase --version v1 --schema ../config/schema-bigbase-sae-v1.json --encoding_config ../config/encoding-bigbase-sae-v1.json --skip-stages 1 2
  
  # Run silently (minimal output)
  python run_sae_pipeline.py --dataset bigbase --version v1 --schema ../config/schema-bigbase-sae-v1.json --encoding_config ../config/encoding-bigbase-sae-v1.json --quiet
        """
    )
    
    parser.add_argument("--dataset", required=True, choices=["bigbase", "unraveled"],
                       help="Dataset to process (bigbase or unraveled)")
    parser.add_argument("--version", required=True, help="Dataset version (e.g., v1)")
    parser.add_argument("--model", default="sae", help="Model type (default: sae)")
    parser.add_argument("--schema", required=True, help="Path to schema configuration JSON file")
    parser.add_argument("--encoding_config", required=True, help="Path to encoding configuration JSON file")
    parser.add_argument("--skip-stages", type=int, nargs="*", default=[],
                       help="Stage numbers to skip (1-5)")
    parser.add_argument("--quiet", action="store_true", help="Minimal console output")
    
    args = parser.parse_args()
    
    # Create and run pipeline orchestrator
    orchestrator = SAEPipelineOrchestrator(
        dataset=args.dataset,
        version=args.version,
        model=args.model,
        schema_config=args.schema,
        encoding_config=args.encoding_config,
        skip_stages=args.skip_stages,
        verbose=not args.quiet
    )
    
    success = orchestrator.run_pipeline()
    
    if success:
        print("\n🎉 SAE Pipeline completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ SAE Pipeline failed. Check logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()