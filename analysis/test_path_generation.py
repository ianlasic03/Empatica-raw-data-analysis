#!/usr/bin/env python3
"""
Test script to verify path generation logic works correctly
"""

from dataLoader2 import DataLoader2

# Test with aldi metadata
metadata_path = "/Users/ianlasic/Downloads/Data_Organized/pilot_data/aldi_gotwald/metadata_aldi.json"

print("=" * 60)
print("Testing Path Generation")
print("=" * 60)

# Parse metadata
metadata = DataLoader2.parse_metadata(metadata_path)

print(f"\nParsed Metadata:")
print(f"  Pilot ID: {metadata['pilot_id']}")
print(f"  Flight Date: {metadata['flight_date']}")
print(f"  Number of Scenarios: {len(metadata['scenarios'])}")

for i, scenario in enumerate(metadata['scenarios'], start=1):
    print(f"\n  Scenario {i}:")
    print(f"    Sequence: {scenario['sequence']}")
    print(f"    Condition: {scenario['condition']}")
    print(f"    Clippy Suffix: {scenario['clippy_suffix']}")
    print(f"    Start Time: {scenario['start_time']}")
    print(f"    End Time: {scenario['end_time']}")

# Test raw data path generation
print(f"\n{'=' * 60}")
print("Testing Raw Data Path Generation")
print("=" * 60)

raw_data_path = DataLoader2.get_raw_data_path(metadata['flight_date'])
print(f"\nRaw Data Path: {raw_data_path}")

import os
if raw_data_path and os.path.exists(raw_data_path):
    print(f"✓ Path exists!")
    avro_files = [f for f in os.listdir(raw_data_path) if f.endswith('.avro')]
    print(f"  Found {len(avro_files)} .avro files")
else:
    print(f"✗ Path does not exist!")

# Test output path generation
print(f"\n{'=' * 60}")
print("Testing Output Path Generation")
print("=" * 60)

output_path = DataLoader2.get_output_path(metadata['flight_date'], metadata['pilot_id'])
print(f"\nOutput Path: {output_path}")

if os.path.exists(output_path):
    print(f"✓ Path exists!")
else:
    print(f"  Path does not exist yet (will be created when saving)")

# Test file naming for each scenario
print(f"\n{'=' * 60}")
print("Testing File Naming")
print("=" * 60)

for i, scenario in enumerate(metadata['scenarios'], start=1):
    print(f"\nScenario {i} ({scenario['sequence']}):")
    
    # Format date from YYYY-MM-DD to MM_DD
    date_parts = metadata['flight_date'].split('-')
    formatted_date = f"{date_parts[1]}_{date_parts[2]}"
    
    # Example file names
    sys_peaks_file = f"{metadata['pilot_id']}_{scenario['sequence']}{scenario['clippy_suffix']}_sys_peaks_{formatted_date}.json"
    sdnn_file = f"{metadata['pilot_id']}_{scenario['sequence']}{scenario['clippy_suffix']}_SDNN(1_shot)_{formatted_date}.json"
    rmssd_file = f"{metadata['pilot_id']}_{scenario['sequence']}{scenario['clippy_suffix']}_RMSSD(1_shot)_{formatted_date}.json"
    hr_file = f"{metadata['pilot_id']}_{scenario['sequence']}{scenario['clippy_suffix']}_heart_rate_{formatted_date}.json"
    
    print(f"  Systolic Peaks: {sys_peaks_file}")
    print(f"  SDNN (1-shot):  {sdnn_file}")
    print(f"  RMSSD (1-shot): {rmssd_file}")
    print(f"  Heart Rate:     {hr_file}")

print(f"\n{'=' * 60}")
print("Test Complete!")
print("=" * 60)
