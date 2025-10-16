#!/usr/bin/env python3
"""
Caldera Command Analysis Script
Analyzes all original Caldera event logs from APT-1 runs to extract and analyze command patterns.
"""

import json
import os
import re
import pandas as pd
from collections import defaultdict, Counter
from pathlib import Path

def find_original_caldera_files(base_path):
    """Find all original Caldera event log files (not extracted_information)"""
    files = []
    base_dir = Path(base_path)
    
    for json_file in base_dir.rglob("*event-logs.json"):
        if "extracted_information" not in str(json_file):
            # Extract run info from path
            run_match = re.search(r'apt-1-.*?run-(\d+)', str(json_file))
            if not run_match:
                run_match = re.search(r'apt-1-run-(\d+)', str(json_file))
            if not run_match:
                # Try to extract from obsolete format
                run_match = re.search(r'apt-1-.*?-(\d+)', str(json_file))
            
            run_number = run_match.group(1) if run_match else "unknown"
            files.append({
                'file_path': str(json_file),
                'run_number': run_number,
                'filename': json_file.name
            })
    
    return sorted(files, key=lambda x: x['run_number'])

def categorize_command(command):
    """Categorize commands based on their content"""
    if not command:
        return "empty"
    
    command_lower = command.lower()
    
    if "curl" in command_lower and "--data" in command_lower:
        return "webshell"
    elif "curl" in command_lower:
        return "curl"
    elif "exec-background" in command_lower:
        return "exec-background"
    elif command_lower.startswith("hostname"):
        return "hostname"
    elif command_lower.startswith("whoami"):
        return "whoami"
    elif command_lower.startswith("ipconfig"):
        return "ipconfig"
    elif "net " in command_lower:
        return "net_command"
    elif "cmd.exe" in command_lower:
        return "cmd_exe"
    elif "ps.exe" in command_lower:
        return "ps_exe"
    elif "mom64.exe" in command_lower:
        return "mom64_exe"
    elif ".exe" in command_lower:
        return "executable"
    elif command_lower.startswith("c:\\"):
        return "file_path"
    else:
        return "other"

def analyze_commands():
    """Main function to analyze all commands from Caldera reports"""
    
    # Find all original Caldera files
    apt1_path = "../../apt-1/"
    caldera_files = find_original_caldera_files(apt1_path)
    
    print(f"Found {len(caldera_files)} original Caldera files")
    
    # Data structures for analysis
    command_data = []  # List of all commands with metadata
    command_runs = defaultdict(set)  # command -> set of runs
    command_frequency = Counter()  # command -> total frequency
    run_commands = defaultdict(list)  # run -> list of commands
    
    # Process each file
    for file_info in caldera_files:
        print(f"Processing: {file_info['filename']} (Run {file_info['run_number']})")
        
        try:
            with open(file_info['file_path'], 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract commands from each entry
            for idx, entry in enumerate(data):
                command = entry.get("command", "")
                
                # Store detailed info
                command_data.append({
                    'command': command,
                    'run_number': file_info['run_number'],
                    'entry_index': idx,
                    'filename': file_info['filename'],
                    'command_type': categorize_command(command)
                })
                
                # Update aggregated data
                command_runs[command].add(file_info['run_number'])
                command_frequency[command] += 1
                run_commands[file_info['run_number']].append(command)
                
        except Exception as e:
            print(f"Error processing {file_info['filename']}: {e}")
    
    # Create comprehensive analysis table
    analysis_rows = []
    
    for command, runs in command_runs.items():
        runs_list = sorted(runs, key=lambda x: int(x) if x.isdigit() else 999)
        runs_str = ",".join(runs_list)
        
        # Format runs for readability (compress ranges)
        runs_formatted = format_run_ranges(runs_list)
        
        analysis_rows.append({
            'command': command,
            'runs': runs_formatted,
            'runs_raw': runs_str,
            'frequency': command_frequency[command],
            'run_count': len(runs),
            'command_type': categorize_command(command),
            'first_appearance': min(runs_list, key=lambda x: int(x) if x.isdigit() else 999),
            'unique_to_run': len(runs) == 1,
            'appears_in_all': len(runs) == len(caldera_files)
        })
    
    # Sort by frequency (most common first)
    analysis_rows.sort(key=lambda x: x['frequency'], reverse=True)
    
    # Create DataFrames
    df_analysis = pd.DataFrame(analysis_rows)
    df_detailed = pd.DataFrame(command_data)
    
    # Save results
    df_analysis.to_csv('caldera_commands_analysis.csv', index=False)
    df_detailed.to_csv('caldera_commands_detailed.csv', index=False)
    
    # Generate summary
    generate_summary(df_analysis, df_detailed, caldera_files)
    
    print(f"\nAnalysis complete!")
    print(f"Total unique commands: {len(df_analysis)}")
    print(f"Total command instances: {len(df_detailed)}")
    print(f"Files processed: {len(caldera_files)}")
    
    return df_analysis, df_detailed

def format_run_ranges(runs_list):
    """Format run list with ranges (e.g., 01,03,05-10,15-20)"""
    if not runs_list:
        return ""
    
    # Convert to integers for range detection
    nums = []
    for run in runs_list:
        try:
            nums.append(int(run))
        except ValueError:
            continue
    
    if not nums:
        return ",".join(runs_list)
    
    nums.sort()
    ranges = []
    start = nums[0]
    end = nums[0]
    
    for i in range(1, len(nums)):
        if nums[i] == end + 1:
            end = nums[i]
        else:
            if start == end:
                ranges.append(f"{start:02d}")
            else:
                ranges.append(f"{start:02d}-{end:02d}")
            start = end = nums[i]
    
    # Add final range
    if start == end:
        ranges.append(f"{start:02d}")
    else:
        ranges.append(f"{start:02d}-{end:02d}")
    
    return ",".join(ranges)

def generate_summary(df_analysis, df_detailed, caldera_files):
    """Generate summary statistics and insights"""
    
    with open('caldera_analysis_summary.txt', 'w') as f:
        f.write("CALDERA COMMAND ANALYSIS SUMMARY\n")
        f.write("="*50 + "\n\n")
        
        # Basic statistics
        f.write("BASIC STATISTICS:\n")
        f.write(f"Total files processed: {len(caldera_files)}\n")
        f.write(f"Total unique commands: {len(df_analysis)}\n")
        f.write(f"Total command instances: {len(df_detailed)}\n\n")
        
        # Commands appearing in all runs
        all_runs = df_analysis[df_analysis['appears_in_all']]
        f.write(f"COMMANDS APPEARING IN ALL RUNS ({len(all_runs)}):\n")
        for _, row in all_runs.head(10).iterrows():
            f.write(f"  - {row['command'][:80]}{'...' if len(row['command']) > 80 else ''} (freq: {row['frequency']})\n")
        f.write("\n")
        
        # Unique commands per run
        unique_per_run = df_analysis[df_analysis['unique_to_run']]
        f.write(f"COMMANDS UNIQUE TO SPECIFIC RUNS ({len(unique_per_run)}):\n")
        for _, row in unique_per_run.head(10).iterrows():
            f.write(f"  Run {row['first_appearance']}: {row['command'][:60]}{'...' if len(row['command']) > 60 else ''}\n")
        f.write("\n")
        
        # Command types distribution
        f.write("COMMAND TYPES DISTRIBUTION:\n")
        type_counts = df_analysis['command_type'].value_counts()
        for cmd_type, count in type_counts.items():
            f.write(f"  {cmd_type}: {count} commands\n")
        f.write("\n")
        
        # Most frequent commands
        f.write("TOP 10 MOST FREQUENT COMMANDS:\n")
        for i, (_, row) in enumerate(df_analysis.head(10).iterrows(), 1):
            f.write(f"  {i}. {row['command'][:60]}{'...' if len(row['command']) > 60 else ''} (freq: {row['frequency']}, runs: {row['run_count']})\n")
        f.write("\n")
        
        # Webshell commands analysis
        webshell_commands = df_analysis[df_analysis['command_type'] == 'webshell']
        f.write(f"WEBSHELL COMMANDS ({len(webshell_commands)}):\n")
        for _, row in webshell_commands.head(10).iterrows():
            f.write(f"  - {row['command'][:80]}{'...' if len(row['command']) > 80 else ''} (runs: {row['runs']})\n")
        f.write("\n")
        
        # File paths
        file_paths = df_analysis[df_analysis['command_type'] == 'file_path']
        f.write(f"FILE PATH COMMANDS ({len(file_paths)}):\n")
        for _, row in file_paths.head(10).iterrows():
            f.write(f"  - {row['command'][:80]}{'...' if len(row['command']) > 80 else ''} (runs: {row['runs']})\n")

if __name__ == "__main__":
    df_analysis, df_detailed = analyze_commands()
    print("\nOutput files created:")
    print("- caldera_commands_analysis.csv (main analysis table)")
    print("- caldera_commands_detailed.csv (detailed command instances)")
    print("- caldera_analysis_summary.txt (summary insights)")