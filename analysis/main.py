from dataLoader2 import DataLoader2
import argparse
import matplotlib.pyplot as plt
import csv
import pandas as pd
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo


def plot_heart_rate(rr_intervals_timestamps, hrt, smooth_hrt):
    # Convert timestamps to datetime
    rr_datetimes = pd.to_datetime(rr_intervals_timestamps)

    plt.figure(figsize=(12, 5))

    # Plot raw and smoothed HR using the same time axis
    plt.plot(rr_datetimes, hrt, color='blue', label='Raw HR')
    plt.plot(rr_datetimes, smooth_hrt, color='red', label='Smooth HR')

    # Plot x-ticks every 1000 steps
    tick_indices = list(range(0, len(rr_datetimes), 1000))
    tick_labels = [rr_datetimes[i].strftime('%H:%M:%S') for i in tick_indices]

    plt.xticks(ticks=[rr_datetimes[i] for i in tick_indices], labels=tick_labels, rotation=45)

    plt.xlabel('Time')
    plt.ylabel('HR (BPM)')
    plt.title('Heart Rate Over Time')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_heart_rate_SDNN_dual_axis(heart_rate, SDNN, RMSSD, SDNN_times=None, RMSSD_times=None):
    """
    Plot heart rate with SDNN and RMSSD on a dual-axis chart.

    Parameters:
    - heart_rate: array-like, full heart rate time series (1 value per second or sample)
    - SDNN: array-like, SDNN values (e.g., every 30s)
    - RMSSD: array-like, RMSSD values (e.g., every 30s)
    - SDNN_times: array-like or None, optional time values for SDNN (default: evenly spaced)
    - RMSSD_times: array-like or None, optional time values for RMSSD (default: evenly spaced)
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Left Y-Axis for Heart Rate
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Heart Rate (bpm)', color='tab:red')
    ax1.plot(heart_rate, label='Heart Rate', color='tab:red', linewidth=1.5)
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Right Y-Axis for HRV metrics
    ax2 = ax1.twinx()
    if SDNN_times is None:
        SDNN_times = list(range(len(SDNN)))
    if RMSSD_times is None:
        RMSSD_times = list(range(len(RMSSD)))
    ax2.plot(SDNN_times, SDNN, label='SDNN', color='tab:blue', linewidth=2, linestyle='--')
    ax2.plot(RMSSD_times, RMSSD, label='RMSSD', color='tab:green', linewidth=2, linestyle='-.')
    ax2.set_ylabel('HRV (ms)', color='black')
    ax2.tick_params(axis='y', labelcolor='black')

    ax1.set_ylim(min(heart_rate) - 2, max(heart_rate) + 2)  # HR axis
    ax2.set_ylim(min(min(SDNN), min(RMSSD)) - 0.15, max(max(SDNN), max(RMSSD)) + 0.15)
    # Legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.title('Heart Rate and HRV (SDNN, RMSSD) Over Time')
    fig.tight_layout()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

def plot_RR_distribution(rr_intervals_values):
     plt.hist(rr_intervals_values, 13, color='skyblue', edgecolor='black')

def plot_RR_intervals(rr_intervals_timestamps, rr_intervals_values):
    # Convert timestamps to datetime
    rr_datetimes = pd.to_datetime(rr_intervals_timestamps)

    # Set up figure
    plt.figure(figsize=(12, 5))
    plt.plot(rr_datetimes, rr_intervals_values, color='green', label='RR Intervals')

    # Plot x-ticks every 1000 steps
    tick_indices = list(range(0, len(rr_datetimes), 1000))
    tick_labels = [rr_datetimes[i].strftime('%H:%M:%S') for i in tick_indices]

    plt.xticks(ticks=[rr_datetimes[i] for i in tick_indices], labels=tick_labels, rotation=45)

    plt.xlabel('Time')
    plt.ylabel('RR Interval (seconds)')
    plt.title('RR Intervals Over Time')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Load and process raw data')
    parser.add_argument("--metadata", type=str, required=True, help='Path to the pilot metadata JSON file')
    parser.add_argument("--scenario", type=int, required=True, help='Scenario number (1 or 2)')
    parser.add_argument("--input", type=str, required=False, default=None, help='Path to the raw data directory (optional, will be determined from metadata)')
    parser.add_argument("--output", type=str, required=False, default=None, help='Path to the output directory (optional, will be determined from metadata)')
    parser.add_argument("--base_path", type=str, required=False, default="/Users/ianlasic/Empatica-raw-data-analysis/Flight_test_data", help='Base path to flight test data')
    args = parser.parse_args()
    
    # Parse metadata
    print(f"Loading metadata from: {args.metadata}")
    metadata = DataLoader2.parse_metadata(args.metadata)
    
    pilot_id = metadata['pilot_id']
    flight_date = metadata['flight_date']
    
    # Validate scenario number
    if args.scenario < 1 or args.scenario > len(metadata['scenarios']):
        raise ValueError(f"Invalid scenario number. Must be between 1 and {len(metadata['scenarios'])}")
    
    # Get scenario info (scenarios are 1-indexed)
    scenario_info = metadata['scenarios'][args.scenario - 1]
    scenario_sequence = scenario_info['sequence']
    clippy_suffix = scenario_info['clippy_suffix']
    start_time_iso = scenario_info['start_time']
    end_time_iso = scenario_info['end_time']
    
    print(f"\nProcessing:")
    print(f"  Pilot: {pilot_id}")
    print(f"  Flight Date: {flight_date}")
    print(f"  Scenario: {scenario_sequence} {clippy_suffix}")
    print(f"  Start Time: {start_time_iso}")
    print(f"  End Time: {end_time_iso}")
    
    # Determine input path (raw data directory)
    if args.input:
        input_path = args.input
    else:
        input_path = DataLoader2.get_raw_data_path(flight_date)
    
    if not input_path:
        raise ValueError(f"Could not find raw data path for flight date: {flight_date}")
    
    print(f"  Input Path: {input_path}")
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = DataLoader2.get_output_path(flight_date, pilot_id, args.base_path)
    
    print(f"  Output Path: {output_path}\n")
    
    # Parse start and end times from ISO format
    if start_time_iso and end_time_iso:
        try:
            # Parse ISO timestamp and extract time component
            start_dt = datetime.fromisoformat(start_time_iso)
            start_time = start_dt.time()
            
            # Handle end_time - might be ISO format or might be duration string
            if end_time_iso.startswith('20'):  # Looks like ISO timestamp
                end_dt = datetime.fromisoformat(end_time_iso)
                end_time = end_dt.time()
            else:
                # It's a duration string like "13 min 48 sec in" - set end_time to None for now
                print(f"  Warning: End time is a duration string: {end_time_iso}")
                end_time = None
        except Exception as e:
            print(f"  Warning: Could not parse timestamps: {e}")
            start_time = None
            end_time = None
    else:
        start_time = None
        end_time = None
            
    data_loader = DataLoader2(input_path)

    # Process the avro files
    data_loader.process_avro_files_test(start_time, end_time)
    
    """ metrics_data now contains accelerometer, gyroscope, 
    eda (timestamps, values), temperature (timsestamps, values), 
    tags, bvp, systolic peaks, steps, rr_intervals (timestamps, values),
    SDNN (timestamps, values), and RMSSD(timestamps, values)
    """
    metrics_data = data_loader.metrics_data
    # Convert systolic peaks from nanosecs to seconds 
    systolic_peaks_secs = np.array(metrics_data['systolicPeaks']) / 1e9

    # RR intervals with cleaned systolic peaks (Lipponen, J. A., & Tarvainen autobeat correction)
    _, clean_peaks = data_loader.clean_systolic_peaks(systolic_peaks_secs)

    RRis, sys_peaks = data_loader.calculate_RR_intervals(clean_peaks)
   
    # RRis with auto beat correction (Good for testing threshold correction in Kubios)
    RRis, sys_peaks = data_loader.apply_threshold_correction(sys_peaks)    
    
    HR = data_loader.calculate_heartrate(RRis)
    smooth_HR = data_loader.smooth_heart_rate(HR)

    RR = np.array(RRis)  # in seconds
    mean_HR_star = 60.0 / RR.mean()
    #print("mean RR intervals: ", np.mean(RRis) * 1000)
    #print("mean HR start: ", mean_HR_star)
    
    SDNN = data_loader.calculate_SDNN(RRis)
    #print("SDNN: ", SDNN * 1000)
    RMSSD = data_loader.calculate_RMSSD(RRis)
    #print("RMSSD: ", RMSSD * 1000)
    
    SDNN_cont = data_loader.smoothing_SDNN(RRis)

    RMSSD_cont = data_loader.smoothing_RMSSD(RRis)

    # Save data with dynamic naming
    data_loader.output_data(
        end_time=end_time_iso,
        metrics_data=metrics_data,
        sys_peaks=sys_peaks,
        RRis=RRis,
        hrt=smooth_HR,
        output_path=output_path,
        pilot_id=pilot_id,
        scenario_sequence=scenario_sequence,
        clippy_suffix=clippy_suffix,
        flight_date=flight_date
    )

    #print("SDNN kernel avg: ", np.mean(SDNN_cont) * 1000)
    #print("RMSSD kernel avg: ", np.mean(RMSSD_cont) * 1000)
    
    # Plot important metrics 
    #plot_RR_distribution(RRis)
    #plot_RR_intervals(metrics_data['rr_intervals_clean']['timestamps'], RRis)
    #plot_heart_rate(metrics_data['rr_intervals_clean']['timestamps'], HR, smooth_HR)
    #plot_heart_rate_SDNN_dual_axis(smooth_HR, SDNN_cont, RMSSD_cont)

 
if __name__ == "__main__":
    main()