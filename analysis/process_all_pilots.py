#!/usr/bin/env python3
"""
Script to process all pilots and scenarios from the Data_Organized directory.
This will iterate through each pilot's metadata and process all scenarios.
"""

import os
import json
import subprocess
import argparse
from pathlib import Path


def find_metadata_files(pilot_data_dir):
    """
    Find all metadata JSON files in the pilot_data directory.
    
    Args:
        pilot_data_dir: Path to the pilot_data directory
        
    Returns:
        List of tuples (pilot_name, metadata_path)
    """
    metadata_files = []
    
    for pilot_folder in os.listdir(pilot_data_dir):
        pilot_path = os.path.join(pilot_data_dir, pilot_folder)
        
        if not os.path.isdir(pilot_path):
            continue
        
        # Look for metadata_{pilot_id}.json
        for filename in os.listdir(pilot_path):
            if filename.startswith('metadata_') and filename.endswith('.json'):
                metadata_path = os.path.join(pilot_path, filename)
                metadata_files.append((pilot_folder, metadata_path))
                break
    
    return sorted(metadata_files)


def get_scenario_count(metadata_path):
    """Get the number of scenarios from metadata file."""
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return len(metadata.get('scenario_order', []))


def process_pilot_scenario(metadata_path, scenario_num, base_path, dry_run=False):
    """
    Process a single pilot scenario.
    
    Args:
        metadata_path: Path to the metadata JSON file
        scenario_num: Scenario number (1 or 2)
        base_path: Base path to flight test data
        dry_run: If True, only print the command without executing
        
    Returns:
        True if successful, False otherwise
    """
    cmd = [
        'python3',
        'main.py',
        '--metadata', metadata_path,
        '--scenario', str(scenario_num),
        '--base_path', base_path
    ]
    
    if dry_run:
        print(f"  [DRY RUN] Would execute: {' '.join(cmd)}")
        return True
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Error processing scenario {scenario_num}:")
        print(f"    {e.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Process all pilots and scenarios')
    parser.add_argument('--pilot_data_dir', type=str, 
                       default='/Users/ianlasic/Downloads/Data_Organized_2/pilot_data',
                       help='Path to the pilot_data directory')
    parser.add_argument('--base_path', type=str,
                       default='/Users/ianlasic/Empatica-raw-data-analysis/Flight_test_data',
                       help='Base path to flight test data')
    parser.add_argument('--pilot', type=str, default=None,
                       help='Process only this specific pilot (by folder name)')
    parser.add_argument('--scenario', type=int, default=None,
                       help='Process only this specific scenario number (1 or 2)')
    parser.add_argument('--dry_run', action='store_true',
                       help='Print commands without executing')
    args = parser.parse_args()
    
    # Find all metadata files
    print(f"Scanning for metadata files in: {args.pilot_data_dir}\n")
    metadata_files = find_metadata_files(args.pilot_data_dir)
    
    if not metadata_files:
        print("No metadata files found!")
        return
    
    print(f"Found {len(metadata_files)} pilots with metadata files\n")
    
    # Process each pilot
    total_processed = 0
    total_failed = 0
    
    for pilot_folder, metadata_path in metadata_files:
        # Skip if specific pilot requested and this isn't it
        if args.pilot and pilot_folder != args.pilot:
            continue
        
        print(f"{'='*60}")
        print(f"Processing: {pilot_folder}")
        print(f"Metadata: {metadata_path}")
        
        # Get number of scenarios
        scenario_count = get_scenario_count(metadata_path)
        print(f"Scenarios: {scenario_count}")
        print(f"{'='*60}\n")
        
        # Process each scenario
        for scenario_num in range(1, scenario_count + 1):
            # Skip if specific scenario requested and this isn't it
            if args.scenario and scenario_num != args.scenario:
                continue
            
            print(f"  Processing Scenario {scenario_num}...")
            success = process_pilot_scenario(
                metadata_path, 
                scenario_num, 
                args.base_path,
                args.dry_run
            )
            
            if success:
                total_processed += 1
                print(f"  ✓ Scenario {scenario_num} completed\n")
            else:
                total_failed += 1
                print(f"  ✗ Scenario {scenario_num} failed\n")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total scenarios processed: {total_processed}")
    print(f"Total scenarios failed: {total_failed}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
