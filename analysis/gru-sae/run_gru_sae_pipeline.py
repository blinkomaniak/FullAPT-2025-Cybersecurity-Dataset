#!/usr/bin/env python3
"""
GRU-SAE Pipeline Runner

Automatically executes all 5 stages of the GRU-SAE pipeline:
1. Data Aggregation
2. Sequence Encoding  
3. Model Training
4. Model Evaluation
5. Result Visualization

Usage:
    python run_gru_sae_pipeline.py --dataset bigbase --version v1 --config-preset basic
    python run_gru_sae_pipeline.py --dataset unraveled --version v1 --config-preset network --subdir network-flows
"""

import os
import sys
import subprocess
import argparse
import time
from datetime import datetime

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_stage(stage_num, stage_name):
    print(f"\n{Colors.HEADER}{'='*60}{Colors.ENDC}")
    print(f"{Colors.BOLD}🚀 STAGE {stage_num}: {stage_name.upper()}{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}")

def print_success(message):
    print(f"{Colors.OKGREEN}✅ {message}{Colors.ENDC}")

def print_error(message):
    print(f"{Colors.FAIL}❌ {message}{Colors.ENDC}")

def print_warning(message):
    print(f"{Colors.WARNING}⚠️ {message}{Colors.ENDC}")

def should_show_line(line, stage_name):
    """Filter output lines to reduce verbosity"""
    line = line.strip()
    
    # Always show important messages
    important_patterns = [
        '✅', '❌', '⚠️', '🚀', '📋', '📊', '💾', '🔍', '📥', '🧠', 
        'GPU available', 'Model architecture', 'Training complete', 
        'Evaluation complete', 'Error', 'Failed', 'Exception',
        'Stage', 'Building', 'Loading', 'Saving', 'completed'
    ]
    
    if any(pattern in line for pattern in important_patterns):
        return True
    
    # For training stage, filter out repetitive progress
    if 'training' in stage_name.lower():
        # Show epoch start and end, but not individual batch updates
        if 'Epoch' in line and ('loss:' in line or 'val_loss:' in line):
            # Show every 10th epoch to reduce spam
            if 'Epoch' in line:
                try:
                    epoch_num = int(line.split('Epoch ')[1].split('/')[0])
                    return epoch_num % 10 == 1 or epoch_num % 10 == 0  # Show epochs 1, 10, 20, etc.
                except:
                    return True
        # Show EarlyStopping and callback messages
        elif any(callback in line for callback in ['EarlyStopping', 'ReduceLROnPlateau', 'ModelCheckpoint']):
            return True
        # Show training summary
        elif 'Training Summary' in line or 'Final' in line:
            return True
        # Hide individual batch progress
        elif '/step' in line and 'ETA:' in line:
            return False
    
    # For encoding stage, reduce tqdm progress spam
    if 'encoding' in stage_name.lower():
        # Show progress every 25% or final result
        if '100%' in line or '25%' in line or '50%' in line or '75%' in line:
            return True
        elif 'it/s' in line and '%' in line:
            return False  # Hide intermediate progress
    
    # Show other lines by default
    return True

def run_command(cmd, stage_name):
    """Run a command and handle errors with filtered real-time output streaming"""
    print(f"{Colors.OKCYAN}📋 Command: {' '.join(cmd)}{Colors.ENDC}")
    start_time = time.time()
    
    try:
        # Run command with real-time output streaming
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True
        )
        
        # Stream output in real-time with filtering
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output and should_show_line(output, stage_name):
                print(output.strip())
        
        # Wait for process to complete and get return code
        return_code = process.poll()
        elapsed = time.time() - start_time
        
        if return_code == 0:
            print_success(f"{stage_name} completed in {elapsed:.1f}s")
            return True
        else:
            print_error(f"{stage_name} failed after {elapsed:.1f}s with return code {return_code}")
            return False
            
    except Exception as e:
        elapsed = time.time() - start_time
        print_error(f"{stage_name} failed after {elapsed:.1f}s with exception: {str(e)}")
        return False

def get_config_paths(dataset, config_preset):
    """Get configuration file paths based on dataset and preset"""
    config_dir = "analysis/config"
    
    if dataset == "bigbase":
        if config_preset == "basic":
            schema_config = f"{config_dir}/schema-bigbase-gru-sae-v1.json"
            encoding_config = f"{config_dir}/encoding-bigbase-gru-sae-v1.json"
        elif config_preset == "extended":
            schema_config = f"{config_dir}/schema-bigbase-gru-sae-v2.json"
            encoding_config = f"{config_dir}/encoding-bigbase-gru-sae-v2.json"
        else:
            raise ValueError(f"Unknown config preset for bigbase: {config_preset}")
    
    elif dataset == "unraveled":
        if config_preset == "network":
            schema_config = f"{config_dir}/schema-unraveled-gru-sae-v1.json"
            encoding_config = f"{config_dir}/encoding-unraveled-gru-sae-v1.json"
        else:
            raise ValueError(f"Unknown config preset for unraveled: {config_preset}")
    
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    return schema_config, encoding_config

def main():
    parser = argparse.ArgumentParser(description="Run complete GRU-SAE pipeline")
    
    # Required arguments
    parser.add_argument("--dataset", required=True, choices=["bigbase", "unraveled"], 
                       help="Dataset name")
    parser.add_argument("--version", required=True, help="Dataset version (e.g., v1)")
    parser.add_argument("--config-preset", required=True, 
                       choices=["basic", "extended", "network"],
                       help="Configuration preset (basic/extended for bigbase, network for unraveled)")
    
    # Optional arguments
    parser.add_argument("--subdir", default="", help="Subdirectory within dataset folder")
    parser.add_argument("--model", default="gru-sae", help="Model name")
    
    # Stage 2 parameters
    parser.add_argument("--seq-len", type=int, help="Sequence length (default: 50 for bigbase, 10 for unraveled)")
    parser.add_argument("--sample-size", type=int, default=2000000, help="Sample size for encoding")
    parser.add_argument("--max-features", type=int, default=500, help="Max features after dimensionality reduction")
    
    # Stage 3 parameters
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=150, help="Training epochs")
    parser.add_argument("--encoder-units", type=int, nargs='+', default=[256, 128], help="Encoder LSTM units")
    parser.add_argument("--decoder-units", type=int, nargs='+', default=[128, 256], help="Decoder LSTM units")
    parser.add_argument("--stopping-strategy", choices=['standard', 'smart', 'adaptive'], default='smart',
                       help="Early stopping strategy")
    parser.add_argument("--min-improvement", type=float, default=0.5,
                       help="Minimum improvement percentage for smart/adaptive stopping")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping")
    
    # Pipeline control
    parser.add_argument("--start-from", type=int, choices=[1,2,3,4,5], default=1, 
                       help="Start pipeline from specific stage")
    parser.add_argument("--stop-at", type=int, choices=[1,2,3,4,5], default=5,
                       help="Stop pipeline at specific stage")
    parser.add_argument("--skip-stages", type=int, nargs='*', default=[],
                       help="Skip specific stages (e.g., --skip-stages 1 3)")
    
    args = parser.parse_args()
    
    # Set default sequence length based on dataset
    if args.seq_len is None:
        args.seq_len = 50 if args.dataset == "bigbase" else 10
    
    # Get configuration paths
    try:
        schema_config, encoding_config = get_config_paths(args.dataset, args.config_preset)
    except ValueError as e:
        print_error(str(e))
        sys.exit(1)
    
    # Verify config files exist
    for config_file in [schema_config, encoding_config]:
        if not os.path.exists(config_file):
            print_error(f"Configuration file not found: {config_file}")
            sys.exit(1)
    
    print(f"{Colors.BOLD}🔧 GRU-SAE Pipeline Configuration{Colors.ENDC}")
    print(f"Dataset: {args.dataset}")
    print(f"Version: {args.version}")
    print(f"Config Preset: {args.config_preset}")
    print(f"Model: {args.model}")
    print(f"Sequence Length: {args.seq_len}")
    print(f"Sample Size: {args.sample_size}")
    print(f"Early Stopping: {args.stopping_strategy} (min_improvement={args.min_improvement}%, patience={args.patience})")
    print(f"Architecture: Encoder={args.encoder_units}, Decoder={args.decoder_units}")
    print(f"Schema Config: {schema_config}")
    print(f"Encoding Config: {encoding_config}")
    if args.subdir:
        print(f"Subdirectory: {args.subdir}")
    
    # Pipeline execution
    pipeline_start = time.time()
    failed_stages = []
    
    # Stage 1: Data Aggregation
    if 1 >= args.start_from and 1 <= args.stop_at and 1 not in args.skip_stages:
        print_stage(1, "Data Aggregation")
        cmd = [
            "python", "analysis/gru-sae/01-aggregation.py",
            "--model", args.model,
            "--dataset", args.dataset,
            "--version", args.version,
            "--schema", schema_config
        ]
        if args.subdir:
            cmd.extend(["--subdir", args.subdir])
        
        if not run_command(cmd, "Stage 1: Data Aggregation"):
            failed_stages.append(1)
            if input("Continue with next stage? (y/n): ").lower() != 'y':
                sys.exit(1)
    
    # Stage 2: Sequence Encoding
    if 2 >= args.start_from and 2 <= args.stop_at and 2 not in args.skip_stages:
        print_stage(2, "Sequence Encoding")
        cmd = [
            "python", "analysis/gru-sae/02-encoding-gru-sae.py",
            "--dataset", args.dataset,
            "--version", args.version,
            "--model", args.model,
            "--encoding_config", encoding_config,
            "--seq-len", str(args.seq_len),
            "--sample-size", str(args.sample_size),
            "--max-features", str(args.max_features)
        ]
        
        if not run_command(cmd, "Stage 2: Sequence Encoding"):
            failed_stages.append(2)
            if input("Continue with next stage? (y/n): ").lower() != 'y':
                sys.exit(1)
    
    # Stage 3: Model Training
    if 3 >= args.start_from and 3 <= args.stop_at and 3 not in args.skip_stages:
        print_stage(3, "Model Training")
        cmd = [
            "python", "analysis/gru-sae/03-training-gru-sae.py",
            "--dataset", args.dataset,
            "--version", args.version,
            "--model", args.model,
            "--batch-size", str(args.batch_size),
            "--epochs", str(args.epochs),
            "--encoder-units"] + [str(u) for u in args.encoder_units] + [
            "--decoder-units"] + [str(u) for u in args.decoder_units] + [
            "--stopping-strategy", args.stopping_strategy,
            "--min-improvement", str(args.min_improvement),
            "--patience", str(args.patience)]
        
        if not run_command(cmd, "Stage 3: Model Training"):
            failed_stages.append(3)
            if input("Continue with next stage? (y/n): ").lower() != 'y':
                sys.exit(1)
    
    # Stage 4: Model Evaluation
    if 4 >= args.start_from and 4 <= args.stop_at and 4 not in args.skip_stages:
        print_stage(4, "Model Evaluation")
        cmd = [
            "python", "analysis/gru-sae/04-model-evaluation-gru-sae.py",
            "--dataset", args.dataset,
            "--version", args.version,
            "--model", args.model,
            "--batch-size", "32"
        ]
        
        if not run_command(cmd, "Stage 4: Model Evaluation"):
            failed_stages.append(4)
            if input("Continue with next stage? (y/n): ").lower() != 'y':
                sys.exit(1)
    
    # Stage 5: Result Visualization
    if 5 >= args.start_from and 5 <= args.stop_at and 5 not in args.skip_stages:
        print_stage(5, "Result Visualization")
        cmd = [
            "python", "analysis/gru-sae/05-result-vis-gru-sae.py",
            "--dataset", args.dataset,
            "--version", args.version,
            "--model", args.model
        ]
        
        if not run_command(cmd, "Stage 5: Result Visualization"):
            failed_stages.append(5)
    
    # Pipeline summary
    total_time = time.time() - pipeline_start
    print(f"\n{Colors.HEADER}{'='*60}{Colors.ENDC}")
    print(f"{Colors.BOLD}🏁 LSTM-SAE PIPELINE COMPLETE{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}")
    print(f"Total execution time: {total_time/60:.1f} minutes")
    
    if failed_stages:
        print_warning(f"Failed stages: {failed_stages}")
        print("Please check the error messages above and rerun failed stages manually.")
    else:
        print_success("All stages completed successfully!")
        
        # Show output locations
        print(f"\n{Colors.OKBLUE}📁 Pipeline Artifacts:{Colors.ENDC}")
        print(f"  Processed Data: analysis/experiments/processed/{args.model}-{args.dataset}-{args.version}.parquet")
        print(f"  Encoded Data:   analysis/experiments/encoded/{args.model}-{args.dataset}-{args.version}/")
        print(f"  Trained Model:  analysis/experiments/models/{args.model}-{args.dataset}-{args.version}/")
        print(f"  Evaluation:     analysis/experiments/eval/{args.model}-{args.dataset}-{args.version}/")
        print(f"  Visualizations: analysis/experiments/vis/{args.model}-{args.dataset}-{args.version}/")

if __name__ == "__main__":
    main()
