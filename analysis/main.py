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
    parser.add_argument('--data_path', type=str, required=True, help='Path to the raw data directory')
    parser.add_argument("--start_time", required=False, default=None, help="Start time in HH:MM or HH:MM:SS")
    parser.add_argument("--end_time", required=False, default=None, help="End time in HH:MM or HH:MM:SS")
    args = parser.parse_args()
    
    if args.start_time == None and args.end_time == None:
        start_time = None
        end_time = None
    else: 
        start_time = (datetime.strptime(args.start_time, '%H:%M:%S')).time()
        end_time = (datetime.strptime(args.end_time, '%H:%M:%S')).time()
            
    data_loader = DataLoader(args.data_path)

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
    #print("len of raw sys peaks: ", len(systolic_peaks_secs))
    raw_RRis = np.ediff1d(systolic_peaks_secs, to_begin=0)  
    
    raw_RRis[0] = np.mean(raw_RRis[1:])
    """with open('raw_rris_sitting.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                for rr in zip(raw_RRis):
                    writer.writerow(rr)"""
    
    # RR intervals with cleaned systolic peaks
    _, clean_peaks = data_loader.clean_systolic_peaks(systolic_peaks_secs)
    #print("post clean length of sys peaks: ", len(clean_peaks))

    RRis, sys_peaks = data_loader.calculate_RR_intervals(clean_peaks, Cleaned=True)
    #plot_RR_intervals(metrics_data['rr_intervals_clean']['timestamps'], RRis)

    #print("raw length of rris: ", len(raw_RRis))
    #print("post RRis calc length of rris: ", len(RRis))

    # RRis with auto beat correction (Good for testing threshold correction in Kubios)
    """with open('rr_nofilter_watch_data_sitting.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                for rr in zip(RRis):
                    writer.writerow(rr)
    """
    RRis, sys_peaks = data_loader.apply_threshold_correction(sys_peaks, Cleaned=True)    
    #plot_RR_intervals(metrics_data['rr_intervals_clean']['timestamps'], RRis)
    """with open('rr_nofilter_watch_data_2.csv', 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    for rr in zip(RRis):
                        writer.writerow(rr)"""
    #print("post threshold correction len of RRis: ", len(RRis))
    HR_nofilter = data_loader.calculate_heartrate(RRis)
    smooth_HR_nofilter = data_loader.smooth_heart_rate(HR_nofilter)
    """with open('smoothHR_sitting.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                for hr in zip(smooth_HR_nofilter):
                    writer.writerow(hr)"""
    SDNN_2 = data_loader.calculate_SDNN(RRis)
    SDNN = data_loader.calculate_segmented_sdnn(RRis, 100)
    print("SDNN not segmented: ", SDNN_2 *1000)
    print("SDNN test: ", SDNN * 1000)
    RMSSD = data_loader.calculate_RMSSD(RRis)
    print("RMSSD: ", RMSSD * 1000)

    RR = np.array(RRis)  # in seconds
    mean_HR_star = 60.0 / RR.mean()
    print("mean RR intervals: ", np.mean(RRis) * 1000)
    print("mean HR start: ", mean_HR_star)
    
    SDNN_cont = data_loader.smoothing_SDNN(RRis)
    """with open('contSDNN_sitting.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for sdnn in zip(SDNN_cont):
            writer.writerow(sdnn) """

    print("gaus kernel len: ", len(SDNN_cont))
    print("len of metrics_data SDNN: ", len(metrics_data['SDNN']['timestamps']), len(metrics_data['SDNN']['values']))
    print("gaus kernel SDNN values: ", np.array(SDNN_cont[:20]) *1000)

    RMSSD_cont = data_loader.smoothing_RMSSD(RRis)
    """with open('contRMSSD_sitting.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for rmssd in zip(RMSSD_cont):
            writer.writerow(rmssd)"""
    
    NN50, pNN50 = data_loader.calculate_pNN50(RRis)
    print("NN50: ", NN50)
    print("pNN50: ", pNN50)

    #data_loader.output_data(metrics_data, sys_peaks, smooth_HR_nofilter, 'output_data_test_class.json')

    print("SDNN kernel avg: ", np.mean(SDNN_cont) * 1000)
    print("RMSSD kernel avg: ", np.mean(RMSSD_cont) * 1000)
    
    # Plot important metrics 
    plot_RR_distribution(RRis)
    plot_RR_intervals(metrics_data['rr_intervals_clean']['timestamps'], RRis)
    plot_heart_rate(metrics_data['rr_intervals_clean']['timestamps'], HR_nofilter, smooth_HR_nofilter)
    plot_heart_rate_SDNN_dual_axis(smooth_HR_nofilter, SDNN_cont, RMSSD_cont)

 
if __name__ == "__main__":
    main()