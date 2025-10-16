#!/usr/bin/env python3
"""
Tactic Analyzer for Caldera Data
Extracts and analyzes tactics from APT simulation event logs to identify:
1. All unique tactic names
2. Count of entries per tactic
3. Potential duplications or naming inconsistencies
4. Naming patterns and case variations
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

def analyze_tactics(json_file_path):
    """
    Analyze tactics from Caldera event logs JSON file
    
    Args:
        json_file_path (str): Path to the JSON file containing event logs
        
    Returns:
        dict: Analysis results including tactics, counts, and patterns
    """
    
    print(f"Analyzing tactics from: {json_file_path}")
    print("=" * 60)
    
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return None
    
    if not isinstance(data, list):
        print("Error: Expected JSON array at root level")
        return None
    
    # Extract tactics
    tactics = []
    tactic_details = []
    
    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            print(f"Warning: Entry {i} is not a dictionary, skipping")
            continue
            
        # Look for attack_metadata.tactic
        attack_metadata = entry.get('attack_metadata', {})
        if isinstance(attack_metadata, dict):
            tactic = attack_metadata.get('tactic')
            if tactic:
                tactics.append(tactic)
                tactic_details.append({
                    'entry_index': i,
                    'tactic': tactic,
                    'technique_name': attack_metadata.get('technique_name', 'N/A'),
                    'technique_id': attack_metadata.get('technique_id', 'N/A'),
                    'ability_name': entry.get('ability_metadata', {}).get('ability_name', 'N/A'),
                    'command': entry.get('command', 'N/A')[:50] + ('...' if len(entry.get('command', '')) > 50 else '')
                })
    
    # Analysis results
    results = {
        'total_entries': len(data),
        'entries_with_tactics': len(tactics),
        'unique_tactics': list(set(tactics)),
        'tactic_counts': dict(Counter(tactics)),
        'tactic_details': tactic_details
    }
    
    # Print summary
    print(f"Total entries in dataset: {results['total_entries']}")
    print(f"Entries with tactics: {results['entries_with_tactics']}")
    print(f"Number of unique tactics: {len(results['unique_tactics'])}")
    print()
    
    # Display unique tactics and counts
    print("UNIQUE TACTICS AND COUNTS:")
    print("-" * 40)
    for tactic, count in sorted(results['tactic_counts'].items()):
        print(f"{tactic:<20} : {count:>3} entries")
    print()
    
    # Check for potential naming inconsistencies
    print("POTENTIAL NAMING PATTERN ANALYSIS:")
    print("-" * 40)
    
    # Group by lowercase to find case variations
    lowercase_groups = defaultdict(list)
    for tactic in results['unique_tactics']:
        lowercase_groups[tactic.lower()].append(tactic)
    
    # Check for case variations
    case_variations = {k: v for k, v in lowercase_groups.items() if len(v) > 1}
    if case_variations:
        print("Case variations found:")
        for lower_tactic, variations in case_variations.items():
            print(f"  '{lower_tactic}' appears as: {variations}")
    else:
        print("No case variations detected")
    
    # Check for similar tactics (potential typos or variations)
    print("\nTactic similarity analysis:")
    unique_tactics_sorted = sorted(results['unique_tactics'])
    for i, tactic1 in enumerate(unique_tactics_sorted):
        for tactic2 in unique_tactics_sorted[i+1:]:
            # Simple similarity check: edit distance or substring
            if (tactic1 in tactic2 or tactic2 in tactic1 or 
                abs(len(tactic1) - len(tactic2)) <= 2):
                print(f"  Potentially similar: '{tactic1}' and '{tactic2}'")
    
    print("\nNAMING PATTERNS:")
    print("-" * 20)
    
    # Analyze naming patterns
    patterns = {
        'all_lowercase': [t for t in results['unique_tactics'] if t.islower()],
        'all_uppercase': [t for t in results['unique_tactics'] if t.isupper()],
        'title_case': [t for t in results['unique_tactics'] if t.istitle()],
        'mixed_case': [t for t in results['unique_tactics'] if not t.islower() and not t.isupper() and not t.istitle()],
        'with_hyphens': [t for t in results['unique_tactics'] if '-' in t],
        'with_underscores': [t for t in results['unique_tactics'] if '_' in t],
        'with_spaces': [t for t in results['unique_tactics'] if ' ' in t]
    }
    
    for pattern_name, pattern_tactics in patterns.items():
        if pattern_tactics:
            print(f"{pattern_name}: {pattern_tactics}")
    
    # Show detailed breakdown by tactic
    print("\nDETAILED TACTIC BREAKDOWN:")
    print("=" * 60)
    
    for tactic in sorted(results['unique_tactics']):
        tactic_entries = [d for d in tactic_details if d['tactic'] == tactic]
        print(f"\nTACTIC: {tactic} ({len(tactic_entries)} entries)")
        print("-" * 50)
        
        # Show unique techniques for this tactic
        techniques = set((d['technique_name'], d['technique_id']) for d in tactic_entries)
        print(f"Techniques used:")
        for tech_name, tech_id in sorted(techniques):
            count = sum(1 for d in tactic_entries if d['technique_name'] == tech_name and d['technique_id'] == tech_id)
            print(f"  {tech_id}: {tech_name} ({count} times)")
        
        # Show sample commands
        sample_commands = list(set(d['command'] for d in tactic_entries))[:3]
        print(f"Sample commands:")
        for cmd in sample_commands:
            print(f"  {cmd}")
    
    return results

def main():
    """Main function to run the tactic analysis"""
    
    # Default file path
    json_file = "/home/researcher/Downloads/research/data-raw/apt-1/apt-1-05-04-run-05/apt34-05-04-test-1_event-logs.json"
    
    # Check if file exists
    if not Path(json_file).exists():
        print(f"Error: File not found: {json_file}")
        return 1
    
    # Run analysis
    results = analyze_tactics(json_file)
    
    if results is None:
        print("Analysis failed")
        return 1
    
    # Save results to a summary file
    output_file = "/home/researcher/Downloads/research/data-raw/apt-1/apt-1-05-04-run-05/tactic_analysis_results.json"
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed results saved to: {output_file}")
    except Exception as e:
        print(f"Warning: Could not save results file: {e}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())