# Dynamic Data Pipeline Usage Guide

## Overview

The data pipeline has been refactored to dynamically determine:
- Which raw wearable data to load based on flight date
- Output file paths and names based on pilot ID and scenario
- Clippy condition from metadata

## Key Changes

### 1. Metadata-Driven Processing

The pipeline now reads pilot metadata JSON files to extract:
- `pilot_id`: Pilot identifier (e.g., 'aldi', 'japh')
- `flight_date`: Flight date in YYYY-MM-DD format
- `scenario_order`: List of scenarios (e.g., ['A', 'B'])
- `scenario_A` and `scenario_B`: Timing and condition info for each scenario

### 2. Automatic Path Resolution

**Input Path (Raw Data)**:
- Automatically resolved from flight date
- Format: `/Users/ianlasic/Empatica-raw-data-analysis/1/1/participant_data/{flight_date}/BETATEST-*/raw_data/v6`
- Example: `2025-09-08` → `.../2025-09-08/BETATEST-3YK9T1L1K5/raw_data/v6`

**Output Path**:
- Automatically constructed from flight date and pilot ID
- Format: `/Users/ianlasic/Empatica-raw-data-analysis/Flight_test_participant_data/output_data_{MM}_{DD}/{pilot_id}_data`
- Example: `aldi`, `2025-09-08` → `.../output_data_09_08/aldi_data`

### 3. Dynamic File Naming

Output files are named using the pattern:
```
{pilot_id}_{scenario_sequence}{clippy_suffix}_{metric}_{date}.json
```

Examples:
- `aldi_Seq1_A_sys_peaks_09_08.json` (without Clippy)
- `aldi_Seq2_B_Clippy_heart_rate_09_08.json` (with Clippy)

## Usage

### Process a Single Scenario

```bash
python3 analysis/main.py \
  --metadata /path/to/metadata.json \
  --scenario 1
```

Arguments:
- `--metadata`: Path to pilot metadata JSON file (required)
- `--scenario`: Scenario number (1 or 2) (required)
- `--input`: Override raw data path (optional, auto-detected from metadata)
- `--output`: Override output path (optional, auto-detected from metadata)
- `--base_path`: Override base path for flight test data (optional)

### Example: Process Aldi's First Scenario

```bash
python3 analysis/main.py \
  --metadata /Users/ianlasic/Downloads/Data_Organized/pilot_data/aldi_gotwald/metadata_aldi.json \
  --scenario 1
```

This will:
1. Parse metadata for pilot `aldi` on `2025-09-08`
2. Load raw data from `.../2025-09-08/BETATEST-3YK9T1L1K5/raw_data/v6`
3. Process Scenario 1 (Seq1_A, without Clippy)
4. Save outputs to `.../output_data_09_08/aldi_data/` with names like:
   - `aldi_Seq1_A_sys_peaks_09_08.json`
   - `aldi_Seq1_A_heart_rate_09_08.json`
   - etc.

### Example: Process Aldi's Second Scenario

```bash
python3 analysis/main.py \
  --metadata /Users/ianlasic/Downloads/Data_Organized/pilot_data/aldi_gotwald/metadata_aldi.json \
  --scenario 2
```

This will process Scenario 2 (Seq2_B, with Clippy) and save outputs with names like:
- `aldi_Seq2_B_Clippy_sys_peaks_09_08.json`
- `aldi_Seq2_B_Clippy_heart_rate_09_08.json`
- etc.

## Testing

### Test Path Generation

To verify the path generation logic works correctly:

```bash
python3 analysis/test_path_generation.py
```

This will:
- Parse metadata for a test pilot
- Show the raw data path that will be used
- Show the output path that will be created
- Display example file names for each scenario

## Metadata File Format

Your metadata JSON should follow this structure:

```json
{
  "pilot_id": "aldi",
  "flight_date": "2025-09-08",
  "scenario_order": ["A", "B"],
  "scenario_A": {
    "start_time": "2025-09-08T20:24:49+00:00",
    "end_time": "2025-09-08T20:30:55+00:00",
    "condition": "without_clippy"
  },
  "scenario_B": {
    "start_time": "2025-09-08T20:37:20+00:00",
    "end_time": "...",
    "condition": "with_clippy"
  }
}
```

## Output Files

For each scenario, the following files are generated:

1. **Systolic Peaks**: `{pilot_id}_{scenario}{clippy}_sys_peaks_{date}.json`
2. **Heart Rate**: `{pilot_id}_{scenario}{clippy}_heart_rate_{date}.json`
3. **SDNN (1-shot)**: `{pilot_id}_{scenario}{clippy}_SDNN(1_shot)_{date}.json`
4. **RMSSD (1-shot)**: `{pilot_id}_{scenario}{clippy}_RMSSD(1_shot)_{date}.json`
5. **SDNN (continuous)**: `{pilot_id}_{scenario}{clippy}_SDNN_{date}.json`
6. **RMSSD (continuous)**: `{pilot_id}_{scenario}{clippy}_RMSSD_{date}.json`
7. **EDA**: `{pilot_id}_{scenario}{clippy}_eda_{date}.json`
8. **Temperature**: `{pilot_id}_{scenario}{clippy}_temperature_{date}.json`

## Processing Multiple Pilots

To process all pilots, you can create a simple bash script:

```bash
#!/bin/bash

METADATA_DIR="/Users/ianlasic/Downloads/Data_Organized/pilot_data"

for pilot_dir in "$METADATA_DIR"/*; do
  if [ -d "$pilot_dir" ]; then
    metadata_file="$pilot_dir/metadata_*.json"
    
    if [ -f $metadata_file ]; then
      echo "Processing pilot: $(basename $pilot_dir)"
      
      # Process scenario 1
      python3 analysis/main.py --metadata "$metadata_file" --scenario 1
      
      # Process scenario 2
      python3 analysis/main.py --metadata "$metadata_file" --scenario 2
    fi
  fi
done
```

## Troubleshooting

### Raw data path not found
- Verify the flight date in metadata matches a folder in `1/1/participant_data/`
- Check that the BETATEST folder exists for that date
- Ensure `raw_data/v6` subdirectory exists

### Output directory issues
- The output directory will be created automatically if it doesn't exist
- Ensure you have write permissions to the base path

### Scenario number invalid
- Check that the scenario number (1 or 2) matches the number of scenarios in `scenario_order`
- Scenario numbers are 1-indexed (1 for first scenario, 2 for second)
