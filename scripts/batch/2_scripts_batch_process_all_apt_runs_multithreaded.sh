#!/bin/bash
#
# Multithreaded Batch Process All APT Runs - Cross-Domain Correlation Analysis
# ============================================================================
#
# High-performance batch processor using multithreaded correlation analysis.
# Optimized for 32-CPU servers with parallel run processing.
#
# Directory Structure:
# research/
# ├── dataset/
# │   ├── scripts/batch/
# │   │   ├── scripts_batch_correlation_analyzer_multithreaded.py     # Multithreaded analysis script
# │   │   └── scripts_batch_process_all_apt_runs_multithreaded.sh    # This script
# │   ├── apt-1/apt-1-run-XX/
# │   ├── apt-2/apt-2-run-XX/
# │   └── ...
# └── analysis/correlation-analysis/correlation_analysis_results/
#
# Usage:
#     cd dataset/scripts/batch/
#     bash scripts_batch_process_all_apt_runs_multithreaded.sh
#     bash scripts_batch_process_all_apt_runs_multithreaded.sh --apt-type apt-1 --workers 32
#     bash scripts_batch_process_all_apt_runs_multithreaded.sh --apt-type apt-1 --exclude-runs "01 02 03" --workers 16
#     bash scripts_batch_process_all_apt_runs_multithreaded.sh --parallel-runs 4 --workers 8  # 4 runs in parallel, 8 workers each
#

# Temporarily disabled to prevent early exit on minor errors
# set -e

# Configuration - paths relative to scripts directory
SCRIPT_NAME="2_scripts_batch_correlation_analyzer_multithreaded.py"
SCRIPTS_DIR="$(pwd)"
# Go up to dataset/ (which is 2 levels up from batch/)
DATASET_DIR="$(dirname "$(dirname "$SCRIPTS_DIR")")"
# Go up to research/ root for analysis directory
RESEARCH_ROOT="$(dirname "$(dirname "$(dirname "$SCRIPTS_DIR")")")"
ANALYSIS_DIR="$RESEARCH_ROOT/analysis/correlation-analysis"
RESULTS_DIR="$ANALYSIS_DIR/correlation_analysis_results"
LOG_DIR="$ANALYSIS_DIR/correlation_analysis_logs"

# Default settings for multithreading
DEFAULT_WORKERS=16  # Conservative default for correlation workers
DEFAULT_PARALLEL_RUNS=2  # Number of APT runs to process simultaneously

# Parse command line arguments
APT_TYPE_FILTER=""
RUN_RANGE=""
EXCLUDE_RUNS=""
WORKERS="$DEFAULT_WORKERS"
PARALLEL_RUNS="$DEFAULT_PARALLEL_RUNS"

while [[ $# -gt 0 ]]; do
  case $1 in
    --apt-type)
      APT_TYPE_FILTER="$2"
      shift 2
      ;;
    --run-range)
      RUN_RANGE="$2"
      shift 2
      ;;
    --exclude-runs)
      EXCLUDE_RUNS="$2"
      shift 2
      ;;
    --workers)
      WORKERS="$2"
      shift 2
      ;;
    --parallel-runs)
      PARALLEL_RUNS="$2"
      shift 2
      ;;
    *)
      echo "Unknown option $1"
      echo "Usage: $0 [--apt-type TYPE] [--run-range X-Y] [--exclude-runs \"X Y Z\"] [--workers N] [--parallel-runs N]"
      exit 1
      ;;
  esac
done

# Verify we're in the batch directory
if [[ ! "$(basename "$PWD")" == "batch" ]]; then
    echo "❌ Error: This script must be run from the scripts/batch/ directory"
    echo "   Current directory: $PWD"
    echo "   Expected: .../dataset/scripts/batch/"
    exit 1
fi

# Verify the analysis script exists
if [[ ! -f "$SCRIPT_NAME" ]]; then
    echo "❌ Error: Analysis script not found: $SCRIPT_NAME"
    echo "   Make sure you're running from the scripts/ directory with the multithreaded script"
    exit 1
fi

# Create results and log directories
mkdir -p "$RESULTS_DIR"
mkdir -p "$LOG_DIR"

echo "🚀 Multithreaded Batch Cross-Domain Correlation Analysis"
echo "========================================================"
echo "Scripts directory: $SCRIPTS_DIR"
echo "Dataset directory: $DATASET_DIR"
echo "Analysis directory: $ANALYSIS_DIR"
echo "Results directory: $RESULTS_DIR"
echo "Log directory: $LOG_DIR"
echo "💻 Workers per run: $WORKERS"
echo "🔄 Parallel runs: $PARALLEL_RUNS"

if [[ -n "$APT_TYPE_FILTER" ]]; then
    echo "APT type filter: $APT_TYPE_FILTER"
fi

if [[ -n "$RUN_RANGE" ]]; then
    echo "Run range filter: $RUN_RANGE"
fi

if [[ -n "$EXCLUDE_RUNS" ]]; then
    echo "Exclude runs: $EXCLUDE_RUNS"
fi

echo ""

# Initialize counters
total_runs=0
successful_runs=0
failed_runs=0
skipped_runs=0

# Clear previous batch summary
if [[ -f "$RESULTS_DIR/batch_summary_results_multithreaded.csv" ]]; then
    echo "🗑️  Clearing previous batch summary results..."
    rm "$RESULTS_DIR/batch_summary_results_multithreaded.csv"
fi

# Function to process a single run (enhanced for multithreading)
process_run() {
    local apt_type=$1
    local run_id=$2
    local run_dir=$3
    
    echo "📊 Processing $apt_type-run-$run_id (MT: $WORKERS workers)..."
    
    # Check if run directory exists
    if [[ ! -d "$run_dir" ]]; then
        echo "❌ Directory not found: $run_dir"
        return 1
    fi
    
    # Check if required files exist - handle multiple naming patterns
    sysmon_file=""
    network_file=""
    
    # Try organized structure first
    if [[ -f "$run_dir/02_data_processing/processed_data/sysmon-run-$run_id.csv" ]]; then
        sysmon_file="02_data_processing/processed_data/sysmon-run-$run_id.csv"
    # Try standard naming
    elif [[ -f "$run_dir/sysmon-run-$run_id.csv" ]]; then
        sysmon_file="sysmon-run-$run_id.csv"
    # Try OLD suffix pattern
    elif [[ -f "$run_dir/sysmon-run-$run_id-OLD.csv" ]]; then
        sysmon_file="sysmon-run-$run_id-OLD.csv"
    # Try any sysmon file with the run ID
    else
        for candidate in "$run_dir"/sysmon*"$run_id"*.csv; do
            if [[ -f "$candidate" ]]; then
                sysmon_file="$(basename "$candidate")"
                break
            fi
        done
    fi
    
    if [[ -f "$run_dir/02_data_processing/processed_data/network_traffic_flow-run-$run_id.csv" ]]; then
        network_file="02_data_processing/processed_data/network_traffic_flow-run-$run_id.csv"
    elif [[ -f "$run_dir/network_traffic_flow-run-$run_id.csv" ]]; then
        network_file="network_traffic_flow-run-$run_id.csv"
    fi
    
    if [[ -z "$sysmon_file" || -z "$network_file" ]]; then
        echo "⚠️  Required CSV files not found in $run_dir - SKIPPING"
        echo "   Looking for: sysmon-run-$run_id.csv and network_traffic_flow-run-$run_id.csv"
        echo "   Available CSV files:"
        ls -1 "$run_dir"/*.csv 2>/dev/null | head -3 || echo "     (no CSV files found)"
        return 2  # Return 2 to indicate skip (not failure)
    fi
    
    echo "   Found: $sysmon_file"
    echo "   Found: $network_file"
    
    # Run multithreaded analysis from scripts directory
    log_file="$LOG_DIR/${apt_type}_run_${run_id}_multithreaded.log"
    
    echo "   Running multithreaded correlation analysis ($WORKERS workers)..."
    start_time=$(date +%s)
    
    if python3 "$SCRIPT_NAME" --run-id "$run_id" --apt-type "$apt_type" --workers "$WORKERS" > "$log_file" 2>&1; then
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        echo "   ✅ Analysis completed successfully (${duration}s)"
        return 0
    else
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        echo "   ❌ Analysis failed (${duration}s) - check log: $log_file"
        return 1
    fi
}

# Define APT types and their actual run ranges (based on directory structure)
declare -A APT_RUNS=(
    ["apt-1"]="04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 51"  # No runs 01-03 exist
    ["apt-2"]="21 22 23 24 25 26 27 28 29 30"
    ["apt-3"]="31 32 33 34 35 36 37 38"
    ["apt-4"]="39 40 41 42 43 44"
    ["apt-5"]="45 46 47"
    ["apt-6"]="48 49 50"
)

# Filter APT types if specified
if [[ -n "$APT_TYPE_FILTER" ]]; then
    if [[ -v APT_RUNS[$APT_TYPE_FILTER] ]]; then
        declare -A FILTERED_RUNS=(["$APT_TYPE_FILTER"]="${APT_RUNS[$APT_TYPE_FILTER]}")
        APT_RUNS=()
        for key in "${!FILTERED_RUNS[@]}"; do
            APT_RUNS[$key]="${FILTERED_RUNS[$key]}"
        done
    else
        echo "❌ Invalid APT type: $APT_TYPE_FILTER"
        echo "Available: ${!APT_RUNS[@]}"
        exit 1
    fi
fi

# Create job queue for parallel processing
declare -a JOB_QUEUE=()

# Build job queue
for apt_type in "${!APT_RUNS[@]}"; do
    runs="${APT_RUNS[$apt_type]}"
    
    for run_id in $runs; do
        # Skip if run range filter is specified and run is not in range
        if [[ -n "$RUN_RANGE" ]]; then
            start_run=$(echo "$RUN_RANGE" | cut -d'-' -f1)
            end_run=$(echo "$RUN_RANGE" | cut -d'-' -f2)
            
            # Convert to numbers for comparison
            run_num=$((10#$run_id))
            start_num=$((10#$start_run))
            end_num=$((10#$end_run))
            
            if [[ $run_num -lt $start_num || $run_num -gt $end_num ]]; then
                continue
            fi
        fi
        
        # Skip if run is in exclude list
        if [[ -n "$EXCLUDE_RUNS" ]]; then
            if [[ " $EXCLUDE_RUNS " == *" $run_id "* ]]; then
                echo "⏭️  Skipping excluded run: $apt_type-run-$run_id"
                continue
            fi
        fi
        
        run_dir="$DATASET_DIR/$apt_type/$apt_type-run-$run_id"
        JOB_QUEUE+=("$apt_type:$run_id:$run_dir")
    done
done

total_jobs=${#JOB_QUEUE[@]}
echo "📋 Job queue: $total_jobs runs to process"
echo "🔄 Processing $PARALLEL_RUNS runs in parallel..."
echo ""

# Process jobs in parallel batches
job_index=0
active_jobs=()

# Function to wait for any job to complete
wait_for_job() {
    local pids=("$@")
    local completed_pid
    
    while true; do
        for i in "${!pids[@]}"; do
            pid=${pids[i]}
            if ! kill -0 "$pid" 2>/dev/null; then
                wait "$pid"
                local exit_code=$?
                echo "🏁 Background job completed (PID: $pid, Exit: $exit_code)"
                return $exit_code
            fi
        done
        sleep 1
    done
}

# Main processing loop
declare -a all_job_results=()

while [[ $job_index -lt $total_jobs ]] || [[ ${#active_jobs[@]} -gt 0 ]]; do
    # Start new jobs if we have capacity and jobs remaining
    while [[ ${#active_jobs[@]} -lt $PARALLEL_RUNS ]] && [[ $job_index -lt $total_jobs ]]; do
        job=${JOB_QUEUE[$job_index]}
        IFS=':' read -r apt_type run_id run_dir <<< "$job"
        
        echo ""
        echo "🎯 Starting $apt_type-run-$run_id in background..."
        
        # Start job in background
        (
            process_run "$apt_type" "$run_id" "$run_dir"
            exit_code=$?
            
            if [[ $exit_code -eq 0 ]]; then
                echo "✅ $apt_type-run-$run_id completed successfully"
            elif [[ $exit_code -eq 2 ]]; then
                echo "⏭️  $apt_type-run-$run_id skipped (missing files)"
            else
                echo "❌ $apt_type-run-$run_id failed"
            fi
            
            exit $exit_code
        ) &
        
        bg_pid=$!
        active_jobs+=($bg_pid)
        echo "🚀 Started $apt_type-run-$run_id (PID: $bg_pid)"
        
        ((job_index++))
    done
    
    # Wait for at least one job to complete if we're at capacity
    if [[ ${#active_jobs[@]} -ge $PARALLEL_RUNS ]]; then
        wait_for_job "${active_jobs[@]}"
        
        # Remove completed jobs from active list and collect results
        new_active_jobs=()
        for pid in "${active_jobs[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                new_active_jobs+=($pid)
            else
                # Job completed, collect result
                wait "$pid"
                exit_code=$?
                all_job_results+=($exit_code)
            fi
        done
        active_jobs=("${new_active_jobs[@]}")
    fi
done

# Wait for any remaining jobs and collect results
echo ""
echo "⏳ Waiting for remaining jobs to complete..."

for pid in "${active_jobs[@]}"; do
    wait "$pid"
    exit_code=$?
    all_job_results+=($exit_code)
done

# Count results
total_runs=${#all_job_results[@]}
successful_runs=0
skipped_runs=0
failed_runs=0

for result in "${all_job_results[@]}"; do
    if [[ $result -eq 0 ]]; then
        ((successful_runs++))
    elif [[ $result -eq 2 ]]; then
        ((skipped_runs++))
    else
        ((failed_runs++))
    fi
done

# Generate final summary
echo ""
echo "🎯 MULTITHREADED BATCH PROCESSING SUMMARY"
echo "========================================"
echo "Total runs processed: $total_runs"
echo "Successful: $successful_runs"
echo "Skipped (missing files): $skipped_runs"
echo "Failed: $failed_runs"
if [[ $total_runs -gt 0 ]]; then
    echo "Success rate: $(( successful_runs * 100 / total_runs ))%"
    echo "Skip rate: $(( skipped_runs * 100 / total_runs ))%"
else
    echo "No runs processed"
fi

echo ""
echo "📊 Results located in:"
echo "  - Summary table: $RESULTS_DIR/batch_summary_results_multithreaded.csv"
echo "  - Individual results: $RESULTS_DIR/"
echo "  - Processing logs: $LOG_DIR/"

# Generate summary statistics
if [[ -f "$RESULTS_DIR/batch_summary_results_multithreaded.csv" ]]; then
    echo ""
    echo "📈 QUICK STATISTICS:"
    echo "==================="
    
    # Count by APT type
    echo "Runs per APT type:"
    tail -n +2 "$RESULTS_DIR/batch_summary_results_multithreaded.csv" | cut -d',' -f1 | sort | uniq -c | sort -nr
    
    echo ""
    echo "Average attribution rates and performance:"
    python3 -c "
import pandas as pd
import sys
try:
    df = pd.read_csv('$RESULTS_DIR/batch_summary_results_multithreaded.csv')
    print(f'Overall average attribution rate: {df[\"attributed_pct\"].mean():.1f}%')
    print(f'Average processing speed: {df[\"throughput_flows_per_second\"].mean():,.0f} flows/second')
    print(f'Average processing time: {df[\"processing_time_seconds\"].mean():.1f} seconds per run')
    print()
    
    for apt_type in sorted(df['apt_type'].unique()):
        apt_data = df[df['apt_type'] == apt_type]
        avg_rate = apt_data['attributed_pct'].mean()
        avg_speed = apt_data['throughput_flows_per_second'].mean()
        run_count = len(apt_data)
        print(f'{apt_type}: {avg_rate:.1f}% attribution, {avg_speed:,.0f} flows/s (n={run_count})')
except Exception as e:
    print(f'Could not generate statistics: {e}')
    sys.exit(0)
"
fi

echo ""
echo "✅ Multithreaded batch processing complete!"
echo "💻 Configuration used: $WORKERS workers per run, $PARALLEL_RUNS parallel runs"

# Return to scripts directory (maintain working directory)
cd "$SCRIPTS_DIR"
