#!/bin/bash
# Unified JSONL to CSV Processor - Interactive Script
# Processes all JSONL files across all APT runs using pipeline scripts #2 and #3
# 
# Usage: ./unified_jsonl_processor.sh
# 
# Author: Generated for comprehensive APT JSONL processing
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
    if [ ! -d "../pipeline" ] || [ ! -d "../../apt-1" ]; then
        print_error "Not in the correct directory!"
        echo "Expected to be in: /data-raw/scripts/batch/"
        echo "Current directory: $(pwd)"
        echo "Please cd to the correct directory and run again."
        exit 1
    fi
    
    # Check Python dependencies
    print_info "Checking Python dependencies..."
    if ! python3 -c "import pandas, yaml, json, numpy" 2>/dev/null; then
        print_warning "Missing Python dependencies. Installing..."
        pip3 install pandas pyyaml numpy --user
    else
        print_success "Python dependencies OK"
    fi
    
    # Check script permissions
    if [ ! -x "../pipeline/2_sysmon_csv_creator.py" ]; then
        print_info "Making scripts executable..."
        chmod +x ../pipeline/*.py
    fi
    
    # Count available JSONL files
    sysmon_jsonl=$(find ../../apt-* -name "*sysmon_operational*.jsonl" 2>/dev/null | wc -l)
    network_jsonl=$(find ../../apt-* -name "*network_traffic*.jsonl" 2>/dev/null | wc -l)
    total_runs=$(find ../../apt-* -name "config.yaml" | wc -l)
    
    # Count existing CSV files
    existing_sysmon_csv=$(find ../../apt-* -name "sysmon*.csv" 2>/dev/null | wc -l)
    existing_network_csv=$(find ../../apt-* -name "network_traffic*.csv" 2>/dev/null | wc -l)
    
    print_info "APT Data Summary:"
    echo "   📂 Total APT runs: $total_runs"
    echo "   📄 Sysmon JSONL files: $sysmon_jsonl"
    echo "   📄 Network Traffic JSONL files: $network_jsonl"
    echo "   📊 Existing Sysmon CSVs: $existing_sysmon_csv"
    echo "   📊 Existing Network CSVs: $existing_network_csv"
    
    if [ "$existing_sysmon_csv" -gt 0 ] || [ "$existing_network_csv" -gt 0 ]; then
        print_warning "Found existing CSV files (will be overwritten during processing)"
    fi
    
    echo ""
}

# Function to get run info
get_run_info() {
    local run_dir="$1"
    local run_name=$(basename "$run_dir")
    local sysmon_jsonl=$(find "$run_dir" -name "*sysmon_operational*.jsonl" | head -1)
    local network_jsonl=$(find "$run_dir" -name "*network_traffic*.jsonl" | head -1)
    local config_file="$run_dir/config.yaml"
    
    local has_sysmon="false"
    local has_network="false"
    local has_config="false"
    
    [ -f "$sysmon_jsonl" ] && has_sysmon="true"
    [ -f "$network_jsonl" ] && has_network="true"
    [ -f "$config_file" ] && has_config="true"
    
    echo "$run_name|$has_sysmon|$has_network|$has_config|$sysmon_jsonl|$network_jsonl|$config_file"
}

# Function to check if CSV files already exist
check_existing_csvs() {
    local run_dir="$1"
    local process_type="$2"
    
    local has_sysmon_csv="false"
    local has_network_csv="false"
    
    # Check for existing Sysmon CSV (multiple patterns for backward compatibility)
    if find "$run_dir" -name "sysmon*.csv" -type f | head -1 > /dev/null; then
        has_sysmon_csv="true"
    fi
    
    # Check for existing Network CSV (multiple patterns for backward compatibility)
    if find "$run_dir" -name "network_traffic*.csv" -type f | head -1 > /dev/null; then
        has_network_csv="true"
    fi
    
    echo "$has_sysmon_csv|$has_network_csv"
}

# Function to process single run
process_single_run() {
    local run_dir="$1"
    local process_type="$2"  # "sysmon", "network", or "both"
    local skip_existing="$3"  # "true" or "false"
    local run_info=$(get_run_info "$run_dir")
    
    IFS='|' read -r run_name has_sysmon has_network has_config sysmon_jsonl network_jsonl config_file <<< "$run_info"
    
    local success_count=0
    local skipped_count=0
    local total_count=0
    
    echo -e "${CYAN}Processing: $run_name${NC}"
    
    if [ "$has_config" = "false" ]; then
        print_warning "No config.yaml found in $run_name - skipping"
        return 0
    fi
    
    # Check existing CSV files if skip_existing is enabled
    local existing_csvs=""
    if [ "$skip_existing" = "true" ]; then
        existing_csvs=$(check_existing_csvs "$run_dir" "$process_type")
    fi
    IFS='|' read -r has_sysmon_csv has_network_csv <<< "$existing_csvs"
    
    # Process Sysmon if requested and available
    if [ "$process_type" = "sysmon" ] || [ "$process_type" = "both" ]; then
        if [ "$has_sysmon" = "true" ]; then
            total_count=$((total_count + 1))
            
            # Skip if CSV already exists and skip_existing is enabled
            if [ "$skip_existing" = "true" ] && [ "$has_sysmon_csv" = "true" ]; then
                local existing_csv=$(find "$run_dir" -name "sysmon*.csv" -type f | head -1)
                if [ -n "$existing_csv" ] && [ -f "$existing_csv" ]; then
                    local csv_size=$(du -h "$existing_csv" | cut -f1)
                    print_info "Sysmon CSV already exists: $(basename "$existing_csv") ($csv_size) - skipping"
                    skipped_count=$((skipped_count + 1))
                    success_count=$((success_count + 1))  # Count as success since it's already done
                else
                    print_warning "Sysmon CSV detection failed - processing normally"
                    print_info "Processing Sysmon: $(basename "$sysmon_jsonl")"
                    
                    if python3 ../pipeline/2_sysmon_csv_creator.py --apt-dir "$run_dir" --no-validate 2>/dev/null; then
                        # Check if CSV was created
                        if find "$run_dir" -name "sysmon*.csv" -type f | head -1 > /dev/null; then
                            local csv_file=$(find "$run_dir" -name "sysmon*.csv" -type f | head -1)
                            local csv_size=$(du -h "$csv_file" | cut -f1)
                            print_success "Sysmon CSV created: $(basename "$csv_file") ($csv_size)"
                            success_count=$((success_count + 1))
                        else
                            print_warning "Sysmon processing completed but no CSV found"
                        fi
                    else
                        print_error "Sysmon processing failed"
                    fi
                fi
            else
                print_info "Processing Sysmon: $(basename "$sysmon_jsonl")"
                
                if python3 ../pipeline/2_sysmon_csv_creator.py --apt-dir "$run_dir" --no-validate 2>/dev/null; then
                    # Check if CSV was created
                    if find "$run_dir" -name "sysmon*.csv" -type f | head -1 > /dev/null; then
                        local csv_file=$(find "$run_dir" -name "sysmon*.csv" -type f | head -1)
                        local csv_size=$(du -h "$csv_file" | cut -f1)
                        print_success "Sysmon CSV created: $(basename "$csv_file") ($csv_size)"
                        success_count=$((success_count + 1))
                    else
                        print_warning "Sysmon processing completed but no CSV found"
                    fi
                else
                    print_error "Sysmon processing failed"
                fi
            fi
        else
            print_info "No Sysmon JSONL file found - skipping"
        fi
    fi
    
    # Process Network Traffic if requested and available
    if [ "$process_type" = "network" ] || [ "$process_type" = "both" ]; then
        if [ "$has_network" = "true" ]; then
            total_count=$((total_count + 1))
            
            # Skip if CSV already exists and skip_existing is enabled
            if [ "$skip_existing" = "true" ] && [ "$has_network_csv" = "true" ]; then
                local existing_csv=$(find "$run_dir" -name "network_traffic*.csv" -type f | head -1)
                if [ -n "$existing_csv" ] && [ -f "$existing_csv" ]; then
                    local csv_size=$(du -h "$existing_csv" | cut -f1)
                    print_info "Network CSV already exists: $(basename "$existing_csv") ($csv_size) - skipping"
                    skipped_count=$((skipped_count + 1))
                    success_count=$((success_count + 1))  # Count as success since it's already done
                else
                    print_warning "Network CSV detection failed - processing normally"
                    print_info "Processing Network: $(basename "$network_jsonl")"
                    
                    if python3 ../pipeline/3_network_traffic_csv_creator.py --apt-dir "$run_dir" --no-validate 2>/dev/null; then
                        # Check if CSV was created
                        if find "$run_dir" -name "network_traffic*.csv" -type f | head -1 > /dev/null; then
                            local csv_file=$(find "$run_dir" -name "network_traffic*.csv" -type f | head -1)
                            local csv_size=$(du -h "$csv_file" | cut -f1)
                            print_success "Network CSV created: $(basename "$csv_file") ($csv_size)"
                            success_count=$((success_count + 1))
                        else
                            print_warning "Network processing completed but no CSV found"
                        fi
                    else
                        print_error "Network processing failed"
                    fi
                fi
            else
                print_info "Processing Network: $(basename "$network_jsonl")"
                
                if python3 ../pipeline/3_network_traffic_csv_creator.py --apt-dir "$run_dir" --no-validate 2>/dev/null; then
                    # Check if CSV was created
                    if find "$run_dir" -name "network_traffic*.csv" -type f | head -1 > /dev/null; then
                        local csv_file=$(find "$run_dir" -name "network_traffic*.csv" -type f | head -1)
                        local csv_size=$(du -h "$csv_file" | cut -f1)
                        print_success "Network CSV created: $(basename "$csv_file") ($csv_size)"
                        success_count=$((success_count + 1))
                    else
                        print_warning "Network processing completed but no CSV found"
                    fi
                else
                    print_error "Network processing failed"
                fi
            fi
        else
            print_info "No Network JSONL file found - skipping"
        fi
    fi
    
    if [ "$skipped_count" -gt 0 ]; then
        echo "   📊 Success: $success_count/$total_count (skipped: $skipped_count)"
    else
        echo "   📊 Success: $success_count/$total_count"
    fi
    echo ""
    
    return $success_count
}

# Function to run batch processing
run_batch_processing() {
    local process_type="$1"
    local type_name="$2"
    local skip_existing="$3"  # "true" or "false"
    
    if [ "$skip_existing" = "true" ]; then
        print_header "Batch Processing All APT Runs - $type_name (Skip Existing)"
    else
        print_header "Batch Processing All APT Runs - $type_name (Reprocess All)"
    fi
    
    # Get list of all run directories
    local run_dirs=()
    for run_dir in ../../apt-*/apt-*-run-*; do
        if [ -d "$run_dir" ]; then
            run_dirs+=("$run_dir")
        fi
    done
    
    local total_runs=${#run_dirs[@]}
    
    if [ "$total_runs" -eq 0 ]; then
        print_error "No APT run directories found!"
        return 1
    fi
    
    print_info "Found $total_runs APT runs to process"
    echo ""
    
    # Initialize counters
    local current=0
    local total_success=0
    local total_failed=0
    local total_skipped=0
    local start_time_total=$(date +%s)
    
    # Create log files
    local skip_suffix=""
    if [ "$skip_existing" = "true" ]; then
        skip_suffix="_skip"
    fi
    local log_file="../../unified_processing_log_${process_type}${skip_suffix}_$(date +%Y%m%d_%H%M%S).txt"
    local failed_log="../../unified_failed_runs_${process_type}${skip_suffix}_$(date +%Y%m%d_%H%M%S).txt"
    
    echo "Unified APT JSONL Processing Log - $type_name - $(date)" > "$log_file"
    if [ "$skip_existing" = "true" ]; then
        echo "Mode: Skip existing CSV files" >> "$log_file"
    else
        echo "Mode: Reprocess all files" >> "$log_file"
    fi
    echo "═══════════════════════════════════════════════════════" >> "$log_file"
    
    # Process each run
    for run_dir in "${run_dirs[@]}"; do
        current=$((current + 1))
        
        echo ""
        echo -e "${CYAN}[$current/$total_runs] Processing: $(basename "$run_dir")${NC}"
        
        # Log to file
        echo "[$current/$total_runs] Processing: $(basename "$run_dir")" >> "$log_file"
        
        # Process the run
        local start_time=$(date +%s)
        local run_success=$(process_single_run "$run_dir" "$process_type" "$skip_existing")
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        if [ "$run_success" -gt 0 ]; then
            total_success=$((total_success + run_success))
            echo "   Success: $run_success files processed in ${duration}s" >> "$log_file"
        else
            total_failed=$((total_failed + 1))
            echo "   Failed: No files processed in ${duration}s" >> "$log_file"
            echo "$(basename "$run_dir") - Processing failed" >> "$failed_log"
        fi
        
        # Progress update
        echo -e "${BLUE}   📊 Overall Progress: $total_success files processed, $total_failed runs failed${NC}"
    done
    
    # Final summary
    local end_time_total=$(date +%s)
    local total_duration=$((end_time_total - start_time_total))
    
    echo ""
    print_header "Batch Processing Complete - $type_name"
    echo -e "${GREEN}🎉 Final Results:${NC}"
    echo "   ✅ Total files processed: $total_success"
    echo "   ❌ Failed runs: $total_failed"
    echo "   ⏱️  Total time: ${total_duration}s ($((total_duration / 60))m)"
    echo "   📝 Log file: $log_file"
    
    # Log final results
    echo "" >> "$log_file"
    echo "Final Results:" >> "$log_file"
    echo "Total files processed: $total_success" >> "$log_file"
    echo "Failed runs: $total_failed" >> "$log_file"
    echo "Total time: ${total_duration}s" >> "$log_file"
    
    if [ "$total_failed" -gt 0 ]; then
        print_warning "Failed runs logged in: $failed_log"
    fi
    
    # Show generated files summary
    echo ""
    print_info "Generated CSV files summary:"
    if [ "$process_type" = "sysmon" ] || [ "$process_type" = "both" ]; then
        local sysmon_count=$(find ../../apt-* -name "sysmon*.csv" 2>/dev/null | wc -l)
        echo "   📊 Sysmon CSV files: $sysmon_count"
    fi
    if [ "$process_type" = "network" ] || [ "$process_type" = "both" ]; then
        local network_count=$(find ../../apt-* -name "network_traffic*.csv" 2>/dev/null | wc -l)
        echo "   📊 Network CSV files: $network_count"
    fi
    
    echo ""
}

# Function to run single test
run_single_test() {
    local process_type="$1"
    local type_name="$2"
    local skip_existing="$3"  # "true" or "false"
    
    if [ "$skip_existing" = "true" ]; then
        print_header "Single Test Run - $type_name (Skip Existing)"
    else
        print_header "Single Test Run - $type_name (Reprocess)"
    fi
    
    # Find a run with the required JSONL files
    local test_run=""
    for run_dir in ../../apt-*/apt-*-run-*; do
        if [ -d "$run_dir" ]; then
            local run_info=$(get_run_info "$run_dir")
            IFS='|' read -r run_name has_sysmon has_network has_config sysmon_jsonl network_jsonl config_file <<< "$run_info"
            
            if [ "$has_config" = "true" ]; then
                if [ "$process_type" = "sysmon" ] && [ "$has_sysmon" = "true" ]; then
                    test_run="$run_dir"
                    break
                elif [ "$process_type" = "network" ] && [ "$has_network" = "true" ]; then
                    test_run="$run_dir"
                    break
                elif [ "$process_type" = "both" ] && [ "$has_sysmon" = "true" ] && [ "$has_network" = "true" ]; then
                    test_run="$run_dir"
                    break
                fi
            fi
        fi
    done
    
    if [ -z "$test_run" ]; then
        print_error "No suitable test run found for $type_name!"
        return 1
    fi
    
    print_info "Testing with: $(basename "$test_run")"
    echo ""
    
    # Process the test run
    local start_time=$(date +%s)
    local success_count=$(process_single_run "$test_run" "$process_type" "$skip_existing")
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    if [ "$success_count" -gt 0 ]; then
        print_success "Test completed successfully!"
        echo "   📊 Files processed: $success_count"
        echo "   ⏱️  Duration: ${duration}s"
    else
        print_error "Test failed!"
        return 1
    fi
    
    echo ""
}

# Function to show status
show_status() {
    print_header "Current Processing Status"
    
    local total_runs=$(find ../../apt-* -name "config.yaml" | wc -l)
    local sysmon_jsonl=$(find ../../apt-* -name "*sysmon_operational*.jsonl" | wc -l)
    local network_jsonl=$(find ../../apt-* -name "*network_traffic*.jsonl" | wc -l)
    local sysmon_csv=$(find ../../apt-* -name "sysmon*.csv" | wc -l)
    local network_csv=$(find ../../apt-* -name "network_traffic*.csv" | wc -l)
    
    echo "📊 APT Processing Status:"
    echo "   📂 Total runs: $total_runs"
    echo "   📄 Sysmon JSONL files: $sysmon_jsonl"
    echo "   📄 Network JSONL files: $network_jsonl"
    echo "   📊 Generated Sysmon CSVs: $sysmon_csv"
    echo "   📊 Generated Network CSVs: $network_csv"
    
    if [ "$sysmon_jsonl" -gt 0 ]; then
        echo "   📈 Sysmon Progress: $((sysmon_csv * 100 / sysmon_jsonl))%"
    fi
    if [ "$network_jsonl" -gt 0 ]; then
        echo "   📈 Network Progress: $((network_csv * 100 / network_jsonl))%"
    fi
    
    if [ "$sysmon_csv" -gt 0 ] || [ "$network_csv" -gt 0 ]; then
        echo ""
        echo "💾 CSV File Sizes:"
        if [ "$sysmon_csv" -gt 0 ]; then
            find ../../apt-* -name "sysmon*.csv" -exec du -ch {} + 2>/dev/null | tail -1 | sed 's/^/   Sysmon Total: /'
        fi
        if [ "$network_csv" -gt 0 ]; then
            find ../../apt-* -name "network_traffic*.csv" -exec du -ch {} + 2>/dev/null | tail -1 | sed 's/^/   Network Total: /'
        fi
    fi
    
    echo ""
}

# Function to cleanup files
cleanup_files() {
    print_header "Cleanup Options"
    
    echo "Select cleanup option:"
    echo "1) Remove all Sysmon CSV files"
    echo "2) Remove all Network CSV files"
    echo "3) Remove all CSV files (both types)"
    echo "4) Remove log files only"
    echo "5) Cancel"
    echo ""
    
    read -p "Enter choice [1-5]: " cleanup_choice
    
    case $cleanup_choice in
        1)
            local sysmon_count=$(find ../../apt-* -name "sysmon*.csv" | wc -l)
            if [ "$sysmon_count" -gt 0 ]; then
                echo "Found $sysmon_count Sysmon CSV files"
                read -p "Are you sure you want to delete all Sysmon CSV files? [y/N]: " confirm
                if [[ $confirm =~ ^[Yy]$ ]]; then
                    find ../../apt-* -name "sysmon*.csv" -delete
                    print_success "Deleted $sysmon_count Sysmon CSV files"
                else
                    print_info "Cancelled"
                fi
            else
                print_info "No Sysmon CSV files found"
            fi
            ;;
        2)
            local network_count=$(find ../../apt-* -name "network_traffic*.csv" | wc -l)
            if [ "$network_count" -gt 0 ]; then
                echo "Found $network_count Network CSV files"
                read -p "Are you sure you want to delete all Network CSV files? [y/N]: " confirm
                if [[ $confirm =~ ^[Yy]$ ]]; then
                    find ../../apt-* -name "network_traffic*.csv" -delete
                    print_success "Deleted $network_count Network CSV files"
                else
                    print_info "Cancelled"
                fi
            else
                print_info "No Network CSV files found"
            fi
            ;;
        3)
            local total_csv=$(($(find ../../apt-* -name "sysmon*.csv" | wc -l) + $(find ../../apt-* -name "network_traffic*.csv" | wc -l)))
            if [ "$total_csv" -gt 0 ]; then
                echo "Found $total_csv total CSV files"
                read -p "Are you sure you want to delete ALL CSV files? [y/N]: " confirm
                if [[ $confirm =~ ^[Yy]$ ]]; then
                    find ../../apt-* -name "sysmon*.csv" -delete
                    find ../../apt-* -name "network_traffic*.csv" -delete
                    print_success "Deleted $total_csv CSV files"
                else
                    print_info "Cancelled"
                fi
            else
                print_info "No CSV files found"
            fi
            ;;
        4)
            local log_count=$(find ../../ -maxdepth 1 -name "unified_*_log_*.txt" | wc -l)
            if [ "$log_count" -gt 0 ]; then
                find ../../ -maxdepth 1 -name "unified_*_log_*.txt" -delete
                find ../../ -maxdepth 1 -name "unified_failed_runs_*.txt" -delete 2>/dev/null || true
                print_success "Deleted log files"
            else
                print_info "No log files found"
            fi
            ;;
        5)
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
    print_header "Unified APT JSONL Processor"
    echo -e "${BLUE}Interactive Script for Processing All APT JSONL Files${NC}"
    echo ""
    echo "Processing Options:"
    echo "1) 🧪 Single Test Run - Sysmon Only"
    echo "2) 🧪 Single Test Run - Network Only"
    echo "3) 🧪 Single Test Run - Both Types"
    echo "4) 🚀 Batch Process All - Sysmon Only"
    echo "5) 🚀 Batch Process All - Network Only"
    echo "6) 🚀 Batch Process All - Both Types"
    echo ""
    echo "Smart Processing (Skip Existing CSVs):"
    echo "7) 🧪 Single Test Run - Sysmon Only (Skip Existing)"
    echo "8) 🧪 Single Test Run - Network Only (Skip Existing)"
    echo "9) 🧪 Single Test Run - Both Types (Skip Existing)"
    echo "10) 🚀 Batch Process All - Sysmon Only (Skip Existing)"
    echo "11) 🚀 Batch Process All - Network Only (Skip Existing)"
    echo "12) 🚀 Batch Process All - Both Types (Skip Existing)"
    echo ""
    echo "Other Options:"
    echo "13) 📊 Show Current Status"
    echo "14) 🧹 Cleanup Files"
    echo "15) ❓ Show Help"
    echo "16) 🚪 Exit"
    echo ""
}

# Help function
show_help() {
    print_header "Help Information"
    echo "This script processes APT JSONL files using pipeline scripts #2 and #3"
    echo ""
    echo "🧪 Single Test Runs:"
    echo "   - Test processing with one run before full batch"
    echo "   - Takes 1-3 minutes typically"
    echo "   - Good for verifying setup"
    echo ""
    echo "🚀 Batch Process All:"
    echo "   - Processes all runs across apt-1 through apt-6"
    echo "   - Sysmon only: ~2-4 hours"
    echo "   - Network only: ~1-2 hours"
    echo "   - Both types: ~3-5 hours"
    echo "   - Creates detailed logs"
    echo ""
    echo "🎯 Smart Processing (Skip Existing):"
    echo "   - Automatically skips runs that already have CSV files"
    echo "   - Perfect for resuming interrupted processing"
    echo "   - Much faster if many files are already processed"
    echo "   - Uses same pipeline scripts but checks for existing outputs"
    echo "   - Sysmon only: ~0.5-2 hours (depending on completed runs)"
    echo "   - Network only: ~0.5-1 hours (depending on completed runs)"
    echo "   - Both types: ~1-3 hours (depending on completed runs)"
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
    echo "   - Each run produces 100-800MB CSV from 2-10GB JSONL"
    echo "   - Monitor progress with: watch 'find ../../apt-* -name \"*.csv\" | wc -l'"
    echo "   - Config files in each run directory specify output formats"
    echo "   - Use 'Skip Existing' options to resume interrupted processing"
    echo "   - Regular options reprocess all files (overwrites existing CSVs)"
    echo ""
}

# Main execution
main() {
    # Check environment first
    check_environment
    
    # Main loop
    while true; do
        show_menu
        read -p "Enter your choice [1-16]: " choice
        echo ""
        
        case $choice in
            1)
                run_single_test "sysmon" "Sysmon Only" "false"
                read -p "Press Enter to continue..."
                ;;
            2)
                run_single_test "network" "Network Only" "false"
                read -p "Press Enter to continue..."
                ;;
            3)
                run_single_test "both" "Both Types" "false"
                read -p "Press Enter to continue..."
                ;;
            4)
                echo "⚠️  This will process all APT Sysmon runs (estimated 2-4 hours)"
                read -p "Continue? [y/N]: " confirm
                if [[ $confirm =~ ^[Yy]$ ]]; then
                    run_batch_processing "sysmon" "Sysmon Only" "false"
                else
                    print_info "Batch processing cancelled"
                fi
                read -p "Press Enter to continue..."
                ;;
            5)
                echo "⚠️  This will process all APT Network runs (estimated 1-2 hours)"
                read -p "Continue? [y/N]: " confirm
                if [[ $confirm =~ ^[Yy]$ ]]; then
                    run_batch_processing "network" "Network Only" "false"
                else
                    print_info "Batch processing cancelled"
                fi
                read -p "Press Enter to continue..."
                ;;
            6)
                echo "⚠️  This will process all APT runs for both types (estimated 3-5 hours)"
                read -p "Continue? [y/N]: " confirm
                if [[ $confirm =~ ^[Yy]$ ]]; then
                    run_batch_processing "both" "Both Types" "false"
                else
                    print_info "Batch processing cancelled"
                fi
                read -p "Press Enter to continue..."
                ;;
            7)
                run_single_test "sysmon" "Sysmon Only" "true"
                read -p "Press Enter to continue..."
                ;;
            8)
                run_single_test "network" "Network Only" "true"
                read -p "Press Enter to continue..."
                ;;
            9)
                run_single_test "both" "Both Types" "true"
                read -p "Press Enter to continue..."
                ;;
            10)
                echo "⚠️  This will process all APT Sysmon runs, skipping existing CSVs (estimated 0.5-2 hours)"
                read -p "Continue? [y/N]: " confirm
                if [[ $confirm =~ ^[Yy]$ ]]; then
                    run_batch_processing "sysmon" "Sysmon Only" "true"
                else
                    print_info "Batch processing cancelled"
                fi
                read -p "Press Enter to continue..."
                ;;
            11)
                echo "⚠️  This will process all APT Network runs, skipping existing CSVs (estimated 0.5-1 hours)"
                read -p "Continue? [y/N]: " confirm
                if [[ $confirm =~ ^[Yy]$ ]]; then
                    run_batch_processing "network" "Network Only" "true"
                else
                    print_info "Batch processing cancelled"
                fi
                read -p "Press Enter to continue..."
                ;;
            12)
                echo "⚠️  This will process all APT runs for both types, skipping existing CSVs (estimated 1-3 hours)"
                read -p "Continue? [y/N]: " confirm
                if [[ $confirm =~ ^[Yy]$ ]]; then
                    run_batch_processing "both" "Both Types" "true"
                else
                    print_info "Batch processing cancelled"
                fi
                read -p "Press Enter to continue..."
                ;;
            13)
                show_status
                read -p "Press Enter to continue..."
                ;;
            14)
                cleanup_files
                read -p "Press Enter to continue..."
                ;;
            15)
                show_help
                read -p "Press Enter to continue..."
                ;;
            16)
                print_success "Goodbye!"
                exit 0
                ;;
            *)
                print_error "Invalid choice. Please enter 1-16."
                read -p "Press Enter to continue..."
                ;;
        esac
    done
}

# Run main function
main "$@"