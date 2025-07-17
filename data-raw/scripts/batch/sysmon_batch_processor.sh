#!/bin/bash
# Sysmon Batch Processor - Interactive Script
# Run Sysmon Event Analysis across all APT runs
# 
# Usage: ./sysmon_batch_processor.sh
# 
# Author: Generated for APT sysmon batch processing
# Date: $(date)

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_header() {
    echo -e "${PURPLE}════════════════════════════════════════${NC}"
    echo -e "${PURPLE} $1 ${NC}"
    echo -e "${PURPLE}════════════════════════════════════════${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Function to check environment
check_environment() {
    print_header "Environment Check"
    
    # Check if we're in the right directory
    if [ ! -d "../" ] || [ ! -d "../../apt-1" ] && [ ! -d "../../apt-2" ] && [ ! -d "../../apt-3" ]; then
        print_error "Not in the correct directory!"
        echo "Expected to be in: /data-raw/scripts/batch/"
        echo "Current directory: $(pwd)"
        echo "Please cd to the correct directory and run again."
        exit 1
    fi
    
    # Check Python dependencies
    print_info "Checking Python dependencies..."
    if ! python3 -c "import pandas, yaml, json" 2>/dev/null; then
        print_warning "Missing Python dependencies. Installing..."
        pip3 install pandas pyyaml --user
    else
        print_success "Python dependencies OK"
    fi
    
    # Check script permissions
    if [ ! -x "../pipeline/sysmon_event_analysis.py" ]; then
        print_info "Making scripts executable..."
        chmod +x ../pipeline/*.py
    fi
    
    # Count available runs across all APT folders
    total_runs=$(find ../../apt-* -name "*sysmon*.csv" 2>/dev/null | sed 's|/[^/]*$||' | sort -u | wc -l)
    total_analysis=$(find ../../apt-* -name "5_entry-events-plots" -type d 2>/dev/null | wc -l)
    existing_csv=$(find ../../apt-* -name "sysmon-*.csv" 2>/dev/null | wc -l)
    
    print_info "APT Data Summary:"
    echo "   📂 Runs with Sysmon CSV files: $total_runs"
    echo "   📊 Available CSV files: $existing_csv"
    echo "   📈 Existing analysis outputs: $total_analysis"
    
    if [ "$total_analysis" -gt 0 ]; then
        print_warning "Found $total_analysis existing analysis directories (will be overwritten)"
    fi
    
    echo ""
}

# Function to run single test
run_single_test() {
    print_header "Single Test Run"
    
    # Find smallest run for testing
    smallest_run=""
    smallest_size=999999999999
    
    for run_dir in ../../apt-*/apt*-*-run-*; do
        if [ -d "$run_dir" ] && [ "$(find "$run_dir" -name "*sysmon*.csv" | wc -l)" -gt 0 ] && [ "$(find "$run_dir" -name "*extracted_information.json" | wc -l)" -gt 0 ]; then
            size=$(find "$run_dir" -name "*sysmon*.csv" -exec du -sb {} + | awk '{sum+=$1} END {print sum}')
            if [ "$size" -lt "$smallest_size" ]; then
                smallest_size=$size
                smallest_run=$run_dir
            fi
        fi
    done
    
    if [ -z "$smallest_run" ]; then
        print_error "No runs with both Sysmon CSV and Caldera JSON found!"
        return 1
    fi
    
    human_size=$(find "$smallest_run" -name "*sysmon*.csv" -exec du -ch {} + | tail -1 | cut -f1)
    print_info "Testing with smallest run: $smallest_run ($human_size)"
    echo ""
    
    # Find the required files
    sysmon_csv=$(find "$smallest_run" -name "*sysmon*.csv" | head -1)
    caldera_json=$(find "$smallest_run" -name "*extracted_information.json" | head -1)
    output_dir="$smallest_run/5_entry-events-plots"
    
    start_time=$(date +%s)
    print_info "Running: python3 ../pipeline/sysmon_event_analysis.py --caldera_json \"$caldera_json\" --sysmon_csv \"$sysmon_csv\" --output_dir \"$output_dir\""
    
    if echo -e "n\nn\nn" | python3 ../pipeline/sysmon_event_analysis.py --caldera_json "$caldera_json" --sysmon_csv "$sysmon_csv" --output_dir "$output_dir"; then
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        
        # Check output
        if [ -d "$output_dir" ] && [ "$(find "$output_dir" -name "*.png" | wc -l)" -gt 0 ]; then
            plot_count=$(find "$output_dir" -name "*.png" | wc -l)
            output_size=$(du -sh "$output_dir" | cut -f1)
            print_success "Test completed successfully!"
            echo "   📊 Output: $plot_count plots in $(basename "$output_dir") ($output_size)"
            echo "   ⏱️  Duration: ${duration}s"
        else
            print_warning "Completed but no plots found"
        fi
    else
        print_error "Test failed!"
        return 1
    fi
    
    echo ""
}

# Function to run batch processing
run_batch_processing() {
    print_header "Batch Processing All APT-1 Runs"
    
    # Get list of runs to process
    runs_to_process=()
    for run_dir in ../../apt-*/apt*-*-run-*; do
        if [ -d "$run_dir" ] && [ "$(find "$run_dir" -name "*sysmon*.csv" | wc -l)" -gt 0 ] && [ "$(find "$run_dir" -name "*extracted_information.json" | wc -l)" -gt 0 ]; then
            runs_to_process+=("$run_dir")
        fi
    done
    
    total_runs=${#runs_to_process[@]}
    
    if [ "$total_runs" -eq 0 ]; then
        print_error "No runs with both Sysmon CSV and Caldera JSON found!"
        return 1
    fi
    
    print_info "Processing $total_runs runs..."
    echo ""
    
    # Initialize counters
    current=0
    success_count=0
    failed_count=0
    start_time_total=$(date +%s)
    
    # Create log file
    log_file="../../processing_log_$(date +%Y%m%d_%H%M%S).txt"
    failed_log="../../failed_runs_$(date +%Y%m%d_%H%M%S).txt"
    
    echo "APT-1 Batch Processing Log - $(date)" > "$log_file"
    echo "═══════════════════════════════════════" >> "$log_file"
    
    # Process each run
    for run_dir in "${runs_to_process[@]}"; do
        current=$((current + 1))
        run_size=$(find "$run_dir" -name "*sysmon*.jsonl" -exec du -ch {} + 2>/dev/null | tail -1 | cut -f1)
        
        echo ""
        echo -e "${CYAN}[$current/$total_runs] 🔄 Processing: $run_dir${NC}"
        echo "   📊 Size: $run_size"
        
        # Log to file
        echo "[$current/$total_runs] Processing: $run_dir ($run_size)" >> "$log_file"
        
        # Check existing CSV
        existing_csv=$(find "$run_dir" -name "sysmon-*.csv" | head -1)
        if [ -n "$existing_csv" ]; then
            csv_size=$(du -h "$existing_csv" | cut -f1)
            print_warning "CSV exists: $(basename "$existing_csv") ($csv_size) - will overwrite"
        fi
        
        # Process the run
        start_time=$(date +%s)
        if python3 ../pipeline/2_sysmon_csv_creator.py --apt-dir "$run_dir" --no-validate; then
            end_time=$(date +%s)
            duration=$((end_time - start_time))
            
            # Check output
            output_csv=$(find "$run_dir" -name "sysmon-*.csv" | head -1)
            if [ -n "$output_csv" ]; then
                csv_size=$(du -h "$output_csv" | cut -f1)
                print_success "Success: $(basename "$output_csv") ($csv_size) in ${duration}s"
                echo "   Success: $(basename "$output_csv") ($csv_size) in ${duration}s" >> "$log_file"
                success_count=$((success_count + 1))
            else
                print_warning "Completed but no CSV found"
                echo "   Warning: Completed but no CSV found" >> "$log_file"
                echo "$run_dir - No CSV output" >> "$failed_log"
                failed_count=$((failed_count + 1))
            fi
        else
            end_time=$(date +%s)
            duration=$((end_time - start_time))
            print_error "Failed after ${duration}s"
            echo "   Failed after ${duration}s" >> "$log_file"
            echo "$run_dir - Processing failed" >> "$failed_log"
            failed_count=$((failed_count + 1))
        fi
        
        # Progress update
        echo -e "${BLUE}   📊 Progress: $success_count success, $failed_count failed${NC}"
    done
    
    # Final summary
    end_time_total=$(date +%s)
    total_duration=$((end_time_total - start_time_total))
    
    echo ""
    print_header "Batch Processing Complete"
    echo -e "${GREEN}🎉 Final Results:${NC}"
    echo "   ✅ Successful: $success_count runs"
    echo "   ❌ Failed: $failed_count runs"
    echo "   ⏱️  Total time: ${total_duration}s ($((total_duration / 60))m)"
    echo "   📝 Log file: $log_file"
    
    # Log final results
    echo "" >> "$log_file"
    echo "Final Results:" >> "$log_file"
    echo "Successful: $success_count runs" >> "$log_file"
    echo "Failed: $failed_count runs" >> "$log_file"
    echo "Total time: ${total_duration}s" >> "$log_file"
    
    if [ "$failed_count" -gt 0 ]; then
        print_warning "Failed runs logged in: $failed_log"
        echo "   📋 Failed runs:"
        cat "$failed_log" 2>/dev/null | while read line; do
            echo "      $line"
        done
    fi
    
    # Show generated files
    echo ""
    print_info "Generated CSV files:"
    csv_count=$(find ../../apt-* -name "sysmon-*.csv" | wc -l)
    find ../../apt-* -name "sysmon-*.csv" -exec ls -lh {} \; | head -5
    if [ "$csv_count" -gt 5 ]; then
        echo "   ... and $((csv_count - 5)) more CSV files"
    fi
    
    echo ""
    print_info "Total CSV size:"
    find ../../apt-* -name "sysmon-*.csv" -exec du -ch {} + 2>/dev/null | tail -1 || echo "No CSV files found"
}

# Function to show status
show_status() {
    print_header "Current Status"
    
    total_runs=$(find ../../apt-* -name "*sysmon*.jsonl" | sed 's|/[^/]*$||' | sort -u | wc -l)
    processed_runs=$(find ../../apt-* -name "sysmon-*.csv" | sed 's|/[^/]*$||' | sort -u | wc -l)
    total_jsonl=$(find ../../apt-* -name "*sysmon*.jsonl" | wc -l)
    total_csv=$(find ../../apt-* -name "sysmon-*.csv" | wc -l)
    
    echo "📊 APT-1 Processing Status:"
    echo "   📂 Total runs with Sysmon data: $total_runs"
    echo "   ✅ Processed runs: $processed_runs"
    echo "   📄 Total Sysmon JSONL files: $total_jsonl"
    echo "   📊 Generated CSV files: $total_csv"
    echo "   📈 Progress: $((processed_runs * 100 / total_runs))%"
    
    if [ "$total_csv" -gt 0 ]; then
        echo ""
        echo "💾 CSV File Sizes:"
        find ../../apt-* -name "sysmon-*.csv" -exec du -ch {} + 2>/dev/null | tail -1
        
        echo ""
        echo "📋 Recent CSV files:"
        find ../../apt-* -name "sysmon-*.csv" -exec ls -lt {} \; | head -3
    fi
    
    echo ""
}

# Function to cleanup files
cleanup_files() {
    print_header "Cleanup Options"
    
    echo "Select cleanup option:"
    echo "1) Remove all CSV files"
    echo "2) Remove log files only"
    echo "3) Remove failed run logs only"
    echo "4) Cancel"
    echo ""
    
    read -p "Enter choice [1-4]: " cleanup_choice
    
    case $cleanup_choice in
        1)
            csv_count=$(find ../../apt-* -name "sysmon-*.csv" | wc -l)
            if [ "$csv_count" -gt 0 ]; then
                echo "Found $csv_count CSV files"
                read -p "Are you sure you want to delete all CSV files? [y/N]: " confirm
                if [[ $confirm =~ ^[Yy]$ ]]; then
                    find ../../apt-* -name "sysmon-*.csv" -delete
                    print_success "Deleted $csv_count CSV files"
                else
                    print_info "Cancelled"
                fi
            else
                print_info "No CSV files found"
            fi
            ;;
        2)
            log_count=$(find ../../ -maxdepth 1 -name "processing_log_*.txt" | wc -l)
            if [ "$log_count" -gt 0 ]; then
                find ../../ -maxdepth 1 -name "processing_log_*.txt" -delete
                print_success "Deleted $log_count log files"
            else
                print_info "No log files found"
            fi
            ;;
        3)
            failed_count=$(find ../../ -maxdepth 1 -name "failed_runs_*.txt" | wc -l)
            if [ "$failed_count" -gt 0 ]; then
                find ../../ -maxdepth 1 -name "failed_runs_*.txt" -delete
                print_success "Deleted $failed_count failed run logs"
            else
                print_info "No failed run logs found"
            fi
            ;;
        4)
            print_info "Cleanup cancelled"
            ;;
        *)
            print_error "Invalid choice"
            ;;
    esac
    echo ""
}

# Main menu function
show_menu() {
    clear
    print_header "APT-1 Batch Processor"
    echo -e "${BLUE}Interactive Script for Sysmon CSV Processing${NC}"
    echo ""
    echo "Options:"
    echo "1) 🧪 Single Test Run (smallest dataset)"
    echo "2) 🚀 Batch Process All Runs"
    echo "3) 📊 Show Current Status"
    echo "4) 🧹 Cleanup Files"
    echo "5) ❓ Show Help"
    echo "6) 🚪 Exit"
    echo ""
}

# Help function
show_help() {
    print_header "Help Information"
    echo "This script processes APT-1 Sysmon JSONL files using script #2"
    echo ""
    echo "🧪 Single Test Run:"
    echo "   - Processes the smallest run first"
    echo "   - Good for testing before full batch"
    echo "   - Takes 1-5 minutes typically"
    echo ""
    echo "🚀 Batch Process All Runs:"
    echo "   - Processes all 17 runs with Sysmon data"
    echo "   - Takes 2-4 hours total"
    echo "   - Creates detailed logs"
    echo "   - Can be interrupted and resumed"
    echo ""
    echo "📊 Show Current Status:"
    echo "   - Shows processing progress"
    echo "   - Lists generated files"
    echo "   - Shows disk usage"
    echo ""
    echo "🧹 Cleanup Files:"
    echo "   - Remove generated CSV files"
    echo "   - Remove log files"
    echo "   - Start fresh if needed"
    echo ""
    echo "💡 Tips:"
    echo "   - Use screen/tmux for long processing"
    echo "   - Monitor with: watch 'find ../../apt-* -name \"sysmon-*.csv\" | wc -l'"
    echo "   - Each run produces ~100-500MB CSV from ~2-10GB JSONL"
    echo ""
}

# Main execution
main() {
    # Check environment first
    check_environment
    
    # Main loop
    while true; do
        show_menu
        read -p "Enter your choice [1-6]: " choice
        echo ""
        
        case $choice in
            1)
                run_single_test
                read -p "Press Enter to continue..."
                ;;
            2)
                echo "⚠️  This will process all APT-1 runs (estimated 2-4 hours)"
                read -p "Continue? [y/N]: " confirm
                if [[ $confirm =~ ^[Yy]$ ]]; then
                    run_batch_processing
                else
                    print_info "Batch processing cancelled"
                fi
                read -p "Press Enter to continue..."
                ;;
            3)
                show_status
                read -p "Press Enter to continue..."
                ;;
            4)
                cleanup_files
                read -p "Press Enter to continue..."
                ;;
            5)
                show_help
                read -p "Press Enter to continue..."
                ;;
            6)
                print_success "Goodbye!"
                exit 0
                ;;
            *)
                print_error "Invalid choice. Please enter 1-6."
                read -p "Press Enter to continue..."
                ;;
        esac
    done
}

# Run main function
main "$@"