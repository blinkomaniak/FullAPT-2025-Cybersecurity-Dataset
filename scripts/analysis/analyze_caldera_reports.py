#!/usr/bin/env python3
"""
Caldera Reports Analysis Script
Analyzes the structure and content of original Caldera reports across all APT-1 runs
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd

def analyze_caldera_report(report_path: str) -> Dict[str, Any]:
    """Analyze a single Caldera report and return statistics."""
    
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Basic file info
        file_size = os.path.getsize(report_path)
        filename = os.path.basename(report_path)
        run_dir = os.path.basename(os.path.dirname(report_path))
        
        # Analyze JSON structure
        analysis = {
            'file_info': {
                'filename': filename,
                'run_directory': run_dir,
                'file_size_bytes': file_size,
                'file_size_kb': round(file_size / 1024, 2)
            },
            'json_structure': {
                'is_list': isinstance(data, list),
                'is_dict': isinstance(data, dict),
                'top_level_type': type(data).__name__
            }
        }
        
        # Count entries based on structure
        if isinstance(data, list):
            # If it's a list, count items
            analysis['content'] = {
                'total_entries': len(data),
                'entry_types': {},
                'structure_analysis': 'list_of_items'
            }
            
            # Analyze first few entries to understand structure
            if data:
                first_entry = data[0]
                analysis['content']['first_entry_type'] = type(first_entry).__name__
                analysis['content']['first_entry_keys'] = list(first_entry.keys()) if isinstance(first_entry, dict) else None
                
                # Count different types if mixed
                type_counts = {}
                for item in data[:100]:  # Sample first 100 for performance
                    item_type = type(item).__name__
                    type_counts[item_type] = type_counts.get(item_type, 0) + 1
                analysis['content']['entry_types'] = type_counts
                
        elif isinstance(data, dict):
            # If it's a dict, analyze keys and potential nested lists
            analysis['content'] = {
                'top_level_keys': list(data.keys()),
                'structure_analysis': 'dictionary'
            }
            
            # Look for arrays within the dictionary
            nested_counts = {}
            for key, value in data.items():
                if isinstance(value, list):
                    nested_counts[key] = len(value)
                elif isinstance(value, dict):
                    nested_counts[key] = f"dict_with_{len(value)}_keys"
                else:
                    nested_counts[key] = f"scalar_{type(value).__name__}"
            
            analysis['content']['nested_structure'] = nested_counts
            
            # If there are lists, find the largest one (likely the main content)
            list_keys = [k for k, v in data.items() if isinstance(v, list)]
            if list_keys:
                largest_list_key = max(list_keys, key=lambda k: len(data[k]))
                analysis['content']['main_content_key'] = largest_list_key
                analysis['content']['total_entries'] = len(data[largest_list_key])
                
                # Analyze structure of items in main list
                main_list = data[largest_list_key]
                if main_list:
                    first_item = main_list[0]
                    analysis['content']['main_entry_type'] = type(first_item).__name__
                    analysis['content']['main_entry_keys'] = list(first_item.keys()) if isinstance(first_item, dict) else None
            else:
                analysis['content']['total_entries'] = 0
        
        return analysis
        
    except Exception as e:
        return {
            'file_info': {
                'filename': os.path.basename(report_path),
                'run_directory': os.path.basename(os.path.dirname(report_path)),
                'file_size_bytes': 0,
                'file_size_kb': 0
            },
            'error': str(e),
            'content': {'total_entries': 0}
        }

def main():
    """Main analysis function."""
    print("🔍 Caldera Original Reports Analysis")
    print("=" * 50)
    
    # Find all original Caldera reports
    report_files = []
    for root, dirs, files in os.walk("../../apt-1"):
        for file in files:
            if file.endswith("_event-logs.json") and "extracted" not in file:
                report_files.append(os.path.join(root, file))
    
    report_files.sort()
    
    if not report_files:
        print("❌ No original Caldera reports found!")
        return
    
    print(f"📂 Found {len(report_files)} original Caldera reports")
    print()
    
    # Analyze each report
    all_analyses = []
    
    for i, report_path in enumerate(report_files, 1):
        print(f"[{i:2d}/{len(report_files)}] 🔍 Analyzing: {os.path.basename(report_path)}")
        
        analysis = analyze_caldera_report(report_path)
        all_analyses.append(analysis)
        
        # Print basic info
        run_dir = analysis['file_info']['run_directory']
        file_size = analysis['file_info']['file_size_kb']
        total_entries = analysis.get('content', {}).get('total_entries', 0)
        
        if 'error' in analysis:
            print(f"    ❌ Error: {analysis['error']}")
        else:
            print(f"    📊 Entries: {total_entries:,}")
            print(f"    📁 Size: {file_size} KB")
            
            # Show structure info
            if analysis['content'].get('structure_analysis') == 'list_of_items':
                print(f"    📋 Structure: Direct list of {total_entries} items")
            elif analysis['content'].get('structure_analysis') == 'dictionary':
                main_key = analysis['content'].get('main_content_key')
                if main_key:
                    print(f"    📋 Structure: Dictionary with main content in '{main_key}' ({total_entries} items)")
                else:
                    print(f"    📋 Structure: Dictionary with no main list")
    
    print()
    
    # Summary statistics
    print("📊 SUMMARY STATISTICS")
    print("=" * 50)
    
    # Create summary table
    summary_data = []
    total_entries_sum = 0
    total_size_sum = 0
    
    for analysis in all_analyses:
        if 'error' not in analysis:
            entries = analysis.get('content', {}).get('total_entries', 0)
            size_kb = analysis['file_info']['file_size_kb']
            
            summary_data.append({
                'Run Directory': analysis['file_info']['run_directory'],
                'Filename': analysis['file_info']['filename'],
                'Entries': entries,
                'Size (KB)': size_kb,
                'Structure': analysis['content'].get('structure_analysis', 'unknown')
            })
            
            total_entries_sum += entries
            total_size_sum += size_kb
    
    # Convert to DataFrame for nice display
    df = pd.DataFrame(summary_data)
    
    if not df.empty:
        print(f"📈 Total Reports Analyzed: {len(df)}")
        print(f"📊 Total Entries Across All Reports: {total_entries_sum:,}")
        print(f"💾 Total Size: {total_size_sum:.1f} KB ({total_size_sum/1024:.1f} MB)")
        print()
        
        print("📋 ENTRY COUNT STATISTICS:")
        entries_stats = df['Entries'].describe()
        print(f"   • Mean entries per report: {entries_stats['mean']:.1f}")
        print(f"   • Median entries per report: {entries_stats['50%']:.1f}")
        print(f"   • Min entries: {entries_stats['min']:.0f}")
        print(f"   • Max entries: {entries_stats['max']:.0f}")
        print(f"   • Std deviation: {entries_stats['std']:.1f}")
        print()
        
        print("🔝 TOP 5 REPORTS BY ENTRY COUNT:")
        top_5 = df.nlargest(5, 'Entries')[['Run Directory', 'Entries', 'Size (KB)']]
        for _, row in top_5.iterrows():
            print(f"   • {row['Run Directory']:20s}: {row['Entries']:6,} entries ({row['Size (KB)']:6.1f} KB)")
        print()
        
        print("🔻 BOTTOM 5 REPORTS BY ENTRY COUNT:")
        bottom_5 = df.nsmallest(5, 'Entries')[['Run Directory', 'Entries', 'Size (KB)']]
        for _, row in bottom_5.iterrows():
            print(f"   • {row['Run Directory']:20s}: {row['Entries']:6,} entries ({row['Size (KB)']:6.1f} KB)")
        print()
        
        print("📝 DETAILED REPORT TABLE:")
        print("-" * 80)
        print(f"{'Run Directory':<25} {'Entries':<8} {'Size(KB)':<10} {'Structure':<15}")
        print("-" * 80)
        for _, row in df.iterrows():
            print(f"{row['Run Directory']:<25} {row['Entries']:<8,} {row['Size (KB)']:<10.1f} {row['Structure']:<15}")
        
        # Check for any structural differences
        structure_types = df['Structure'].value_counts()
        print()
        print("🏗️  STRUCTURE ANALYSIS:")
        for structure, count in structure_types.items():
            print(f"   • {structure}: {count} reports")
        
        # Save detailed results
        output_file = "../../caldera_reports_analysis.csv"
        df.to_csv(output_file, index=False)
        print()
        print(f"💾 Detailed results saved to: {output_file}")
    
    else:
        print("❌ No valid reports found for analysis")

if __name__ == "__main__":
    main()