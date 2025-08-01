from dataLoader import DataLoader
import argparse
import matplotlib.pyplot as plt
import csv
import pandas as pd
import numpy as np
from datetime import datetime, timezone, time, timedelta
import pytz
from zoneinfo import ZoneInfo
import neurokit2 as nk

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

def plot_heart_rate_SDNN(heart_rate, SDNN):
    plt.figure(figsize=(10, 5))
    #plt.plot(heart_rate, label='Heart Rate', color='blue')
    plt.plot(SDNN, label='SDNN (aligned)', color='red')
    plt.title('Heart Rate and SDNN Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Heart Rate (bpm) / SDNN (ms)')
    plt.legend()
    plt.grid()
    plt.show()

def plot_bvp_data(bvp_data):
    plt.figure(figsize=(10, 5))
    plt.plot(bvp_data, label='BVP Signal', color='blue')
    plt.title('BVP Signal Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('BVP Signal')
    plt.legend()
    plt.grid()
    plt.show()

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


def plot_sys_peaks(peaks):
    plt.figure(figsize=(10, 5))
    plt.plot(peaks, label='syspeak Signal', color='blue')
    plt.title('sys peaks Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('syspeaks Signal')
    plt.legend()
    plt.grid()
    plt.show()

"""PACIFIC = timezone(timedelta(hours=-7))

def parse_time(timestr: str, fallback_tz: timezone = PACIFIC) -> time:
    try:
        t = time.fromisoformat(timestr)
    except ValueError:
        # no offset present, parse naive and attach your zone
        fmt = "%H:%M:%S.%f" if "." in timestr else "%H:%M:%S"
        dt = datetime.strptime(timestr, fmt)
        return time(dt.hour, dt.minute, dt.second, dt.microsecond,
                    tzinfo=fallback_tz)
    else:
        # time.fromisoformat gave us either tzinfo=None or a datetime.timezone
        if t.tzinfo is None:
            return t.replace(tzinfo=fallback_tz)
        # normalize any timezone(timedelta) into a single instance
        off = t.utcoffset()
        return t.replace(tzinfo=timezone(off))"""

    #t = datetime.strptime(timestr, "%H:%M" if len(timestr) == 5 else "%H:%M:%S").time()
    #return t.replace(tzinfo=tzinfo)

US_TZ = pytz.timezone("US/Pacific")

def parse_time(timestr: str) -> time:
    """
    Accepts:
      - "11:00:00-07:00"
      - "10:38:07.685881-07:00"
      - "2:30:00"
      - "14:05:03.123"
    and always returns an aware time in US/Pacific.
    """
    t = time.fromisoformat(timestr)
    if t.tzinfo is None:
        # no offset in the string → attach our local tz
        return t.replace(tzinfo=US_TZ)
    # it already had an offset → leave it
    return t

def main():
    parser = argparse.ArgumentParser(description='Load and process raw data')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the raw data directory')
    parser.add_argument("--start_time", required=True, help="Start time in HH:MM or HH:MM:SS")
    parser.add_argument("--end_time", required=True, help="End time in HH:MM or HH:MM:SS")
    #parser.add_argument("--timezone", required=True, help="Specify timezone for start and end time")
    #parser.add_argument('--output_path', type=str, required=True, help='Path to save the processed data')
    args = parser.parse_args()
    
    start_time = (datetime.strptime(args.start_time, '%H:%M:%S')).time()
    end_time = (datetime.strptime(args.end_time, '%H:%M:%S')).time()
    
    data_loader = DataLoader(args.data_path)

    # Process the avro files
    data_loader.process_avro_files_test(start_time, end_time)
    
    """ metrics_data now contains accelerometer, gyroscope, 
    eda (timestamps, values), temperature (timsestamps, values), 
    tags, bvp, systolic peaks, steps, rr_intervals (timestamps, values)
    """
    metrics_data = data_loader.metrics_data

    systolic_peaks_secs = np.array(data_loader.metrics_data['systolicPeaks']) / 1e9

    # RR intervals without cleaning systolic peaks, specify if systolic peaks are cleaned 
    RR_intervals, systolic_peaks_secs = data_loader.calculate_RR_intervals(systolic_peaks_secs, Cleaned=False)
    # Calculate HR from RR intervals
    hrt = data_loader.calculate_heartrate(RR_intervals)
    # Smooth HR with savgol filter
    smooth_hrt = data_loader.smooth_heart_rate(hrt)
    
    # Output eda, skin temp, HR to a json file 
    data_loader.outout_data(metrics_data, systolic_peaks_secs, smooth_hrt, 'output_data_test_class.json')

    SDNN = data_loader.calculate_SDNN(RR_intervals)

    """Testing out moving average of HRV"""
    SDNN_cont = data_loader.moving_average_SDNN(RR_intervals)
    RMSSD = data_loader.calculate_RMSSD(RR_intervals)
    RMSSD_cont = data_loader.moving_average_RMSSD(RR_intervals)
   

    print("SDNN: ", SDNN * 1000)
    """with open('SDNN.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['SDNN'])
        for sdnn in SDNN_cont:
            writer.writerow([sdnn * 1000])

    with open('RMSSD.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['RMSSD'])
        for rmssd in RMSSD_cont:
            writer.writerow([rmssd * 1000])"""
    print("RMSSD: ", RMSSD * 1000)

    RR = np.array(RR_intervals)  # in seconds
    mean_HR_star = 60.0 / RR.mean()
    print("mean HR start: ", mean_HR_star)
    print("median HR: ", np.median(RR_intervals))
    
    # Plot metrics 
    plot_heart_rate_SDNN_dual_axis(smooth_hrt, SDNN_cont, RMSSD_cont)
    plot_RR_intervals(data_loader.metrics_data['rr_intervals']['timestamps'], data_loader.metrics_data['rr_intervals']['values'])
    plot_heart_rate(data_loader.metrics_data['rr_intervals']['timestamps'], hrt, smooth_hrt)


if __name__ == "__main__":
    main()