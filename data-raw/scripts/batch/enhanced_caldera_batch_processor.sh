#!/bin/bash
# Enhanced Caldera Report Batch Processor - Interactive Script
# Processes all Caldera report files across all APT runs using enhanced pipeline script #4
# 
# Usage: ./enhanced_caldera_batch_processor.sh
# 
# Author: Generated for comprehensive APT Caldera report processing
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
    if ! python3 -c "import json, yaml, re, argparse" 2>/dev/null; then
        print_warning "Missing Python dependencies. Installing..."
        pip3 install pyyaml --user
    else
        print_success "Python dependencies OK"
    fi
    
    # Check script permissions
    if [ ! -x "../pipeline/4_universal_caldera_report_transformer.py" ]; then
        print_info "Making scripts executable..."
        chmod +x ../pipeline/*.py
    fi
    
    # Count available Caldera files and existing processed files
    original_caldera=$(find ../../apt-* -name "*event-logs.json" -not -name "*extracted_information*" 2>/dev/null | wc -l)
    extracted_caldera=$(find ../../apt-* -name "*extracted_information.json" 2>/dev/null | wc -l)
    total_runs=$(find ../../apt-* -name "config.yaml" | wc -l)
    
    print_info "APT Caldera Data Summary:"
    echo "   📂 Total APT runs: $total_runs"
    echo "   📄 Original Caldera files: $original_caldera"
    echo "   📊 Existing extracted files: $extracted_caldera"
    
    if [ "$extracted_caldera" -gt 0 ]; then
        print_warning "Found existing extracted files (will be overwritten during processing)"
    fi
    
    echo ""
}

# Function to get run info
get_run_info() {
    local run_dir="$1"
    local run_name=$(basename "$run_dir")
    local caldera_file=$(find "$run_dir" -name "*event-logs.json" -not -name "*extracted_information*" | head -1)
    local extracted_file=$(find "$run_dir" -name "*extracted_information.json" | head -1)
    local config_file="$run_dir/config.yaml"
    
    local has_caldera="false"
    local has_extracted="false"
    local has_config="false"
    
    [ -f "$caldera_file" ] && has_caldera="true"
    [ -f "$extracted_file" ] && has_extracted="true"
    [ -f "$config_file" ] && has_config="true"
    
    echo "$run_name|$has_caldera|$has_extracted|$has_config|$caldera_file|$extracted_file|$config_file"
}

# Function to check if extracted information file already exists
check_existing_extracted() {
    local run_dir="$1"
    
    local has_extracted="false"
    
    # Check for existing extracted information file
    if find "$run_dir" -name "*extracted_information.json" -type f | head -1 > /dev/null; then
        has_extracted="true"
    fi
    
    echo "$has_extracted"
}

# Function to process single run
process_single_run() {
    local run_dir="$1"
    local skip_existing="$2"  # "true" or "false"
    local run_info=$(get_run_info "$run_dir")
    
    IFS='|' read -r run_name has_caldera has_extracted has_config caldera_file extracted_file config_file <<< "$run_info"
    
    local success_count=0
    local skipped_count=0
    local total_count=0
    
    echo -e "${CYAN}Processing: $run_name${NC}"
    
    if [ "$has_config" = "false" ]; then
        print_warning "No config.yaml found in $run_name - skipping"
        return 0
    fi
    
    if [ "$has_caldera" = "false" ]; then
        print_warning "No Caldera event-logs file found in $run_name - skipping"
        return 0
    fi
    
    total_count=1
    
    # Check existing extracted file if skip_existing is enabled
    local existing_extracted=""
    if [ "$skip_existing" = "true" ]; then
        existing_extracted=$(check_existing_extracted "$run_dir")
    fi
    
    # Process Caldera file
    if [ "$skip_existing" = "true" ] && [ "$existing_extracted" = "true" ]; then
        local existing_file=$(find "$run_dir" -name "*extracted_information.json" -type f | head -1)
        if [ -n "$existing_file" ] && [ -f "$existing_file" ]; then
            local file_size=$(du -h "$existing_file" | cut -f1)
            local entry_count=$(python3 -c "import json; data=json.load(open('$existing_file')); print(len(data))" 2>/dev/null || echo "unknown")
            print_info "Extracted info already exists: $(basename "$existing_file") ($file_size, $entry_count entries) - skipping"
            skipped_count=$((skipped_count + 1))
            success_count=$((success_count + 1))  # Count as success since it's already done
        else
            print_warning "Extracted file detection failed - processing normally"
            print_info "Processing Caldera: $(basename "$caldera_file")"
            
            if python3 ../pipeline/4_universal_caldera_report_transformer.py --apt-dir "$run_dir" 2>/dev/null; then
                # Check if extracted file was created
                if find "$run_dir" -name "*extracted_information.json" -type f | head -1 > /dev/null; then
                    local new_file=$(find "$run_dir" -name "*extracted_information.json" -type f | head -1)
                    local file_size=$(du -h "$new_file" | cut -f1)
                    local entry_count=$(python3 -c "import json; data=json.load(open('$new_file')); print(len(data))" 2>/dev/null || echo "unknown")
                    print_success "Extracted info created: $(basename "$new_file") ($file_size, $entry_count entries)"
                    success_count=$((success_count + 1))
                else
                    print_warning "Caldera processing completed but no extracted file found"
                fi
            else
                print_error "Caldera processing failed"
            fi
        fi
    else
        print_info "Processing Caldera: $(basename "$caldera_file")"
        
        if python3 ../pipeline/4_universal_caldera_report_transformer.py --apt-dir "$run_dir" 2>/dev/null; then
            # Check if extracted file was created
            if find "$run_dir" -name "*extracted_information.json" -type f | head -1 > /dev/null; then
                local new_file=$(find "$run_dir" -name "*extracted_information.json" -type f | head -1)
                local file_size=$(du -h "$new_file" | cut -f1)
                local entry_count=$(python3 -c "import json; data=json.load(open('$new_file')); print(len(data))" 2>/dev/null || echo "unknown")
                print_success "Extracted info created: $(basename "$new_file") ($file_size, $entry_count entries)"
                success_count=$((success_count + 1))
            else
                print_warning "Caldera processing completed but no extracted file found"
            fi
        else
            print_error "Caldera processing failed"
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
    local skip_existing="$1"  # "true" or "false"
    
    if [ "$skip_existing" = "true" ]; then
        print_header "Batch Processing All APT Runs - Caldera Reports (Skip Existing)"
    else
        print_header "Batch Processing All APT Runs - Caldera Reports (Reprocess All)"
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
    local log_file="../../caldera_processing_log${skip_suffix}_$(date +%Y%m%d_%H%M%S).txt"
    local failed_log="../../caldera_failed_runs${skip_suffix}_$(date +%Y%m%d_%H%M%S).txt"
    
    echo "Enhanced Caldera Batch Processing Log - $(date)" > "$log_file"
    if [ "$skip_existing" = "true" ]; then
        echo "Mode: Skip existing extracted files" >> "$log_file"
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
        local run_success=$(process_single_run "$run_dir" "$skip_existing")
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
    print_header "Batch Processing Complete - Caldera Reports"
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
    print_info "Generated extracted information files summary:"
    local extracted_count=$(find ../../apt-* -name "*extracted_information.json" 2>/dev/null | wc -l)
    echo "   📊 Extracted information files: $extracted_count"
    
    # Calculate total size
    if [ "$extracted_count" -gt 0 ]; then
        find ../../apt-* -name "*extracted_information.json" -exec du -ch {} + 2>/dev/null | tail -1 | sed 's/^/   💾 Total size: /'
    fi
    
    echo ""
}

# Function to run single test
run_single_test() {
    local skip_existing="$1"  # "true" or "false"
    
    if [ "$skip_existing" = "true" ]; then
        print_header "Single Test Run - Caldera Reports (Skip Existing)"
    else
        print_header "Single Test Run - Caldera Reports (Reprocess)"
    fi
    
    # Find a run with Caldera file
    local test_run=""
    for run_dir in ../../apt-*/apt-*-run-*; do
        if [ -d "$run_dir" ]; then
            local run_info=$(get_run_info "$run_dir")
            IFS='|' read -r run_name has_caldera has_extracted has_config caldera_file extracted_file config_file <<< "$run_info"
            
            if [ "$has_config" = "true" ] && [ "$has_caldera" = "true" ]; then
                test_run="$run_dir"
                break
            fi
        fi
    done
    
    if [ -z "$test_run" ]; then
        print_error "No suitable test run found for Caldera processing!"
        return 1
    fi
    
    print_info "Testing with: $(basename "$test_run")"
    echo ""
    
    # Process the test run
    local start_time=$(date +%s)
    local success_count=$(process_single_run "$test_run" "$skip_existing")
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
    print_header "Current Caldera Processing Status"
    
    local total_runs=$(find ../../apt-* -name "config.yaml" | wc -l)
    local original_caldera=$(find ../../apt-* -name "*event-logs.json" -not -name "*extracted_information*" | wc -l)
    local extracted_caldera=$(find ../../apt-* -name "*extracted_information.json" | wc -l)
    
    echo "📊 APT Caldera Processing Status:"
    echo "   📂 Total runs: $total_runs"
    echo "   📄 Original Caldera files: $original_caldera"
    echo "   📊 Generated extracted files: $extracted_caldera"
    
    if [ "$original_caldera" -gt 0 ]; then
        echo "   📈 Processing Progress: $((extracted_caldera * 100 / original_caldera))%"
    fi
    
    if [ "$extracted_caldera" -gt 0 ]; then
        echo ""
        echo "💾 Extracted Information File Sizes:"
        find ../../apt-* -name "*extracted_information.json" -exec du -ch {} + 2>/dev/null | tail -1 | sed 's/^/   Total Size: /'
        
        echo ""
        echo "📋 Sample Entry Counts:"
        for file in $(find ../../apt-* -name "*extracted_information.json" | head -3); do
            local entry_count=$(python3 -c "import json; data=json.load(open('$file')); print(len(data))" 2>/dev/null || echo "unknown")
            echo "   $(basename "$file"): $entry_count entries"
        done
        if [ "$(find ../../apt-* -name "*extracted_information.json" | wc -l)" -gt 3 ]; then
            echo "   ... and $((extracted_caldera - 3)) more files"
        fi
    fi
    
    echo ""
}

# Function to cleanup files
cleanup_files() {
    print_header "Cleanup Options"
    
    echo "Select cleanup option:"
    echo "1) Remove all extracted information files"
    echo "2) Remove log files only"
    echo "3) Cancel"
    echo ""
    
    read -p "Enter choice [1-3]: " cleanup_choice
    
    case $cleanup_choice in
        1)
            local extracted_count=$(find ../../apt-* -name "*extracted_information.json" | wc -l)
            if [ "$extracted_count" -gt 0 ]; then
                echo "Found $extracted_count extracted information files"
                read -p "Are you sure you want to delete all extracted information files? [y/N]: " confirm
                if [[ $confirm =~ ^[Yy]$ ]]; then
                    find ../../apt-* -name "*extracted_information.json" -delete
                    print_success "Deleted $extracted_count extracted information files"
                else
                    print_info "Cancelled"
                fi
            else
                print_info "No extracted information files found"
            fi
            ;;
        2)
            local log_count=$(find ../../ -maxdepth 1 -name "caldera_*_log_*.txt" | wc -l)
            if [ "$log_count" -gt 0 ]; then
                find ../../ -maxdepth 1 -name "caldera_*_log_*.txt" -delete
                find ../../ -maxdepth 1 -name "caldera_failed_runs_*.txt" -delete 2>/dev/null || true
                print_success "Deleted log files"
            else
                print_info "No log files found"
            fi
            ;;
        3)
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
    print_header "Enhanced APT Caldera Report Processor"
    echo -e "${BLUE}Interactive Script for Processing All APT Caldera Reports${NC}"
    echo ""
    echo "Processing Options:"
    echo "1) 🧪 Single Test Run (Reprocess)"
    echo "2) 🚀 Batch Process All (Reprocess All)"
    echo ""
    echo "Smart Processing (Skip Existing Extracted Files):"
    echo "3) 🧪 Single Test Run (Skip Existing)"
    echo "4) 🚀 Batch Process All (Skip Existing)"
    echo ""
    echo "Other Options:"
    echo "5) 📊 Show Current Status"
    echo "6) 🧹 Cleanup Files"
    echo "7) ❓ Show Help"
    echo "8) 🚪 Exit"
    echo ""
}

# Help function
show_help() {
    print_header "Help Information"
    echo "This script processes APT Caldera event-logs files using enhanced pipeline script #4"
    echo ""
    echo "🧪 Single Test Runs:"
    echo "   - Test processing with one run before full batch"
    echo "   - Takes 10-30 seconds typically"
    echo "   - Good for verifying setup"
    echo ""
    echo "🚀 Batch Process All:"
    echo "   - Processes all runs across apt-1 through apt-6"
    echo "   - Estimated time: 15-45 minutes (depending on dataset size)"
    echo "   - Creates detailed logs"
    echo ""
    echo "🎯 Smart Processing (Skip Existing):"
    echo "   - Automatically skips runs that already have extracted information files"
    echo "   - Perfect for resuming interrupted processing"
    echo "   - Much faster if many files are already processed"
    echo "   - Uses enhanced pipeline script #4 with config.yaml support"
    echo ""
    echo "📊 Show Current Status:"
    echo "   - Shows processing progress"
    echo "   - Lists generated files and their sizes"
    echo "   - Shows entry counts for sample files"
    echo ""
    echo "🧹 Cleanup Files:"
    echo "   - Remove generated extracted information files"
    echo "   - Remove log files"
    echo "   - Start fresh if needed"
    echo ""
    echo "💡 Tips:"
    echo "   - Each Caldera file typically produces 50-200 extracted entries"
    echo "   - Original files are 1-5MB, extracted files are typically 200-800KB"
    echo "   - Monitor progress with: find ../../apt-* -name '*extracted_information.json' | wc -l"
    echo "   - Config files specify output filenames"
    echo "   - Use 'Skip Existing' options to resume interrupted processing"
    echo ""
    echo "🔧 Enhanced Features:"
    echo "   - Uses config.yaml from each APT run directory"
    echo "   - Automatic file detection and naming"
    echo "   - Improved error handling and logging"
    echo "   - Progress tracking with entry counts"
    echo ""
}

# Main execution
main() {
    # Check environment first
    check_environment
    
    # Main loop
    while true; do
        show_menu
        read -p "Enter your choice [1-8]: " choice
        echo ""
        
        case $choice in
            1)
                run_single_test "false"
                read -p "Press Enter to continue..."
                ;;
            2)
                echo "⚠️  This will process all APT Caldera reports (estimated 15-45 minutes)"
                read -p "Continue? [y/N]: " confirm
                if [[ $confirm =~ ^[Yy]$ ]]; then
                    run_batch_processing "false"
                else
                    print_info "Batch processing cancelled"
                fi
                read -p "Press Enter to continue..."
                ;;
            3)
                run_single_test "true"
                read -p "Press Enter to continue..."
                ;;
            4)
                echo "⚠️  This will process all APT Caldera reports, skipping existing extracted files (estimated 5-30 minutes)"
                read -p "Continue? [y/N]: " confirm
                if [[ $confirm =~ ^[Yy]$ ]]; then
                    run_batch_processing "true"
                else
                    print_info "Batch processing cancelled"
                fi
                read -p "Press Enter to continue..."
                ;;
            5)
                show_status
                read -p "Press Enter to continue..."
                ;;
            6)
                cleanup_files
                read -p "Press Enter to continue..."
                ;;
            7)
                show_help
                read -p "Press Enter to continue..."
                ;;
            8)
                print_success "Goodbye!"
                exit 0
                ;;
            *)
                print_error "Invalid choice. Please enter 1-8."
                read -p "Press Enter to continue..."
                ;;
        esac
    done
}

# Run main function
main "$@"