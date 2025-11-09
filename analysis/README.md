# Empatica Data Analysis Pipeline

This pipeline processes raw Empatica watch data and generates physiological metrics (heart rate, HRV, EDA, temperature) for flight test scenarios.

## Dynamic Output Pipeline

The pipeline now automatically determines:
- Which raw data file to load based on flight date
- Pilot ID from metadata
- Scenario information (Seq1_A, Seq2_B, etc.)
- Clippy condition (with/without)
- Output file naming and paths

## Usage

### Process a Single Pilot Scenario

```bash
python3 main.py --metadata /path/to/metadata_pilot.json --scenario 1
```

**Arguments:**
- `--metadata` (required): Path to the pilot's metadata JSON file
- `--scenario` (required): Scenario number (1 or 2)
- `--input` (optional): Override the raw data directory path
- `--output` (optional): Override the output directory path
- `--base_path` (optional): Base path to flight test data (default: `/Users/ianlasic/Empatica-raw-data-analysis/Flight_test_participant_data`)

**Example:**
```bash
python3 main.py \
  --metadata /Users/ianlasic/Downloads/Data_Organized/pilot_data/aldi_gotwald/metadata_aldi.json \
  --scenario 1
```

This will:
1. Read metadata to extract pilot_id, flight_date, scenario info
2. Automatically determine raw data path: `Flight_test_participant_data/output_data_09_08/aldi_data/raw_data/v6`
3. Process scenario 1 (Seq1_A) for pilot "aldi"
4. Save output to: `Flight_test_participant_data/output_data_09_08/aldi_data/`
5. Generate files like: `physio_aldi_seq1_a_heart_rate_09_08.json`

### Process All Pilots and Scenarios

```bash
python3 process_all_pilots.py
```

**Arguments:**
- `--pilot_data_dir` (optional): Path to the pilot_data directory (default: `/Users/ianlasic/Downloads/Data_Organized/pilot_data`)
- `--base_path` (optional): Base path to flight test data
- `--pilot` (optional): Process only a specific pilot by folder name
- `--scenario` (optional): Process only a specific scenario number (1 or 2)
- `--dry_run` (optional): Print commands without executing

**Examples:**

Process everything:
```bash
python3 process_all_pilots.py
```

Process only pilot "aldi_gotwald":
```bash
python3 process_all_pilots.py --pilot aldi_gotwald
```

Process only scenario 2 for all pilots:
```bash
python3 process_all_pilots.py --scenario 2
```

Dry run to see what would be executed:
```bash
python3 process_all_pilots.py --dry_run
```

## Metadata File Structure

The metadata JSON file should contain:

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

The pipeline generates the following files for each scenario:

- `physio_{pilot_id}_{scenario}_{clippy}_sys_peaks_{date}.json` - Systolic peaks
- `physio_{pilot_id}_{scenario}_{clippy}_heart_rate_{date}.json` - Heart rate
- `physio_{pilot_id}_{scenario}_{clippy}_SDNN_{date}.json` - Continuous SDNN
- `physio_{pilot_id}_{scenario}_{clippy}_RMSSD_{date}.json` - Continuous RMSSD
- `physio_{pilot_id}_{scenario}_{clippy}_SDNN(1_shot)_{date}.json` - One-shot SDNN
- `physio_{pilot_id}_{scenario}_{clippy}_RMSSD(1_shot)_{date}.json` - One-shot RMSSD
- `physio_{pilot_id}_{scenario}_{clippy}_eda_{date}.json` - Electrodermal activity
- `physio_{pilot_id}_{scenario}_{clippy}_temperature_{date}.json` - Temperature

**Example filenames:**
- `physio_aldi_seq1_a_heart_rate_09_08.json`
- `physio_aldi_seq2_b_clippy_SDNN_09_08.json`

## Directory Structure

```
Flight_test_participant_data/
  output_data_09_08/
    aldi_data/
      raw_data/v6/          # Input: Empatica .avro files
      physio_*.json         # Output: Processed metrics
  output_data_09_09/
    ...

Data_Organized/
  pilot_data/
    aldi_gotwald/
      metadata_aldi.json    # Input: Pilot metadata
    ...
```

## Key Functions

### DataLoader2 Class

- `parse_metadata(metadata_path)` - Parse metadata JSON and extract pilot info
- `get_raw_data_path(flight_date, pilot_id, base_path)` - Construct raw data path
- `get_output_path(flight_date, pilot_id, base_path)` - Construct output path
- `output_data(...)` - Save processed data with dynamic naming

## Notes

- Start and end times are automatically extracted from metadata
- If end_time is a duration string (e.g., "13 min 48 sec in"), it's currently ignored
- The pipeline handles both "with_clippy" and "without_clippy" conditions
- File naming follows the convention: `physio_{pilot}_{seq}_{condition}_{metric}_{date}.json`
