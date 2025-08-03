import os
import re
from fastavro import reader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.interpolate import CubicSpline, interp1d
from scipy.signal import savgol_filter
from datetime import datetime, timezone, time
from zoneinfo import ZoneInfo
import neurokit2 as nk

class DataLoader:
    def __init__(self, data_path):
        self.data_path = data_path
        self.metrics_data = {
        'accelerometer': {'x': [], 'y': [], 'z': []},
        'gyroscope': {'x': [], 'y': [], 'z': []},
        'eda': {'timestamps': [], 'values': []},
        'temperature': {'timestamps': [], 'values': []},
        'tags': [],
        'bvp': [],
        'systolicPeaks': [],
        'steps': [], 
        'rr_intervals': {'timestamps': [], 'values': []},
        'rr_intervals_clean': {'timestamps': [], 'values': []},
        'SDNN': {'timestamps': [], 'values': []},
        'RMSSD': {'timestamps': [], 'values': []}
        }

    
    def get_timestamp_from_filename(self, filename):
        match = re.search(r'BETATEST_(\d+)', filename)
        return int(match.group(1)) if match else 0


    def load_files_in_time_order(self, directory_path):    
        # Get all .avro files
        avro_files = [f for f in os.listdir(directory_path) if f.endswith('.avro')]
        
        # Sort by timestamp extracted from filename
        avro_files.sort(key=self.get_timestamp_from_filename)
        
        return avro_files
    
    def process_avro_files_test(self, start_time, end_time):
            # Ensure the files are loaded in time order
            files_time_order = self.load_files_in_time_order(self.data_path)
            metrics_data = self.metrics_data
            
            # Loop through each avro file in the raw data path
            for i in range(len(files_time_order)):
                if files_time_order[i].endswith(".avro"):
                    file_path = os.path.join(self.data_path, files_time_order[i])
                    with open(file_path, 'rb') as f:
                        avro_reader = reader(f)

                        next_file_start_time_eda = None
                        next_file_start_time_temp = None 
                        if i + 1 < len(files_time_order):
                            next_file = files_time_order[i + 1]
                            if next_file.endswith(".avro"):
                                next_file_path = os.path.join(self.data_path, next_file)
                                with open(next_file_path, 'rb') as f_next:
                                    try:
                                        next_records = list(reader(f_next))
                                        next_file_start_time_eda = next_records[0]['rawData']['eda']['timestampStart']
                                        next_file_start_time_temp = next_records[0]['rawData']['temperature']['timestampStart']
                                    except Exception as e:
                                        print(f"Warning: Could not read next file {next_file}: {e}")
                        
                        for record in avro_reader:
                            current_timestamp = record['rawData']['accelerometer']['timestampStart']
                            current_us_time = datetime.fromtimestamp(current_timestamp / 1e6, tz=timezone.utc).astimezone(ZoneInfo("US/Pacific")).isoformat()
                            
                            dt = datetime.fromisoformat(current_us_time)
                            us_time = dt.timetz()
                            # Make time offset-naive because we control the date we load with the file path
                            us_time = dt.replace(tzinfo=None).time()

                            # Only do the time comparison if start_time and end_time are not None
                            if start_time != None and end_time != None:
                                # Don't load the record if it's not within the time bounds 
                                if not (start_time <= us_time <= end_time):
                                    continue
                            
                            for key in record['rawData'].keys(): 
                                    if key == 'accelerometer':
                                        metrics_data[key]['x'].extend(record['rawData'][key]['x'])
                                        metrics_data[key]['y'].extend(record['rawData'][key]['y'])
                                        metrics_data[key]['z'].extend(record['rawData'][key]['z'])
                                    elif key == 'gyroscope':
                                        metrics_data[key]['x'].extend(record['rawData'][key]['x'])
                                        metrics_data[key]['y'].extend(record['rawData'][key]['y'])
                                        metrics_data[key]['z'].extend(record['rawData'][key]['z'])
                                    elif key == 'eda': 
                                        # Handle the case of loading the last file
                                        if next_file_start_time_eda == None:    
                                            for i in range(len(record['rawData']['eda']['values'])): #record 2 has 484 values and an actual frequency of 3.903 hz (so approximately 4 samples per second 
                                                metrics_data[key]['timestamps'].append(np.float64((record['rawData'][key]['timestampStart'] + 250000 * i)))                                      
                                        else: 
                                            metrics_data[key]['timestamps'].extend(np.linspace(record['rawData'][key]['timestampStart'], next_file_start_time_eda - 250000, len(record['rawData'][key]['values'])))
                                        metrics_data[key]['values'].extend(record['rawData'][key]['values'])
                                    elif key == 'temperature':
                                        # Handle the case of loading last file
                                        if next_file_start_time_temp == None:
                                            for i in range(len(record['rawData']['temperature']['values'])): 
                                                metrics_data[key]['timestamps'].append(np.float64((record['rawData'][key]['timestampStart'] + 1000000 * i)))                                               
                                        else: 
                                            metrics_data[key]['timestamps'].extend(np.linspace(record['rawData'][key]['timestampStart'], next_file_start_time_temp - 1000000, len(record['rawData'][key]['values'])))
                                        metrics_data[key]['values'].extend(record['rawData'][key]['values'])
                                        
                                    elif key == 'bvp':
                                        metrics_data[key].extend(record['rawData'][key]['values'])
                                    elif key == 'systolicPeaks':
                                        metrics_data[key].extend(record['rawData'][key]['peaksTimeNanos'])                 

    """
    Computing the threshold for identifying artifacts in the RR intervals
    - alpha is a hyperparameter
    - window_size is the number of samples to look at in the given quartile (paper uses 91)
    """
    def compute_threshold(self, signal, alpha, window_size):
        data = pd.DataFrame({"signal": np.abs(signal)})
        
        # Calculate rolling statistics
        rolling_mean = data.rolling(window_size, center=True, min_periods=1).mean().signal.to_numpy()
        rolling_std = data.rolling(window_size, center=True, min_periods=1).std().signal.to_numpy()
        
        q1 = data.rolling(window_size, center=True, min_periods=1).quantile(0.25).signal.to_numpy()
        q3 = data.rolling(window_size, center=True, min_periods=1).quantile(0.75).signal.to_numpy()
        
        # Adaptive alpha based on signal variability
        signal_variability = rolling_std / (rolling_mean + 1e-8)
        adaptive_alpha = alpha * (1 + signal_variability)
        
        quartile_deviation = (q3 - q1) / 2.0
        thr = adaptive_alpha * quartile_deviation
    
        return thr

    def identify_artifacts(self, systolic_peaks_secs, alpha, window_size):
        # RR is difference between systolic peaks
        RRs = np.ediff1d(systolic_peaks_secs, to_begin=0)  # Get difference between consecutive systolic peaks
        RRs[0] = np.mean(RRs[1:])

        # Step 1: Get difference between consecutive RR intervals  
        dRRs = np.ediff1d(RRs, to_begin=0)  
        dRRs[0] = np.mean(dRRs[1:]) 

        # Step 2: Compute threshold 1: alpha times quartile deviation of qRRs of the surronding 91 samples
        th1 = self.compute_threshold(dRRs, alpha, window_size)
        # Normalize DRRs by the threshold 
        dRRs /= th1

        # Step 3: Compute mRRs
        RR_data = pd.DataFrame({"signal": np.abs(RRs)})
        medRR = RR_data.rolling(11, center=True, min_periods=1).median().signal.to_numpy()
        mRRs = RRs - medRR
        mRRs[mRRs < 0] = mRRs[mRRs < 0] * 2

        # Step 4: Compute threshold 2 
        th2 = self.compute_threshold(mRRs, alpha, window_size)
        mRRs /= th2
        
        return RRs, dRRs, mRRs, medRR, th2


    """
    Function to classify artifacts in the systolic peak data
    """
    def classify_artifact(self, RRs, dRRs, mRRs, medRR, th2, c1, c2):
        s12 = np.zeros(dRRs.size)
        padding = 2
        dRRs_pad = np.pad(dRRs, padding, 'reflect')
        
        # Loop from padding to padding + length of dRRs
        for d in np.arange(padding, padding + dRRs.size):
            if dRRs_pad[d] > 0:
                s12[d - padding] = np.max([dRRs_pad[d - 1], dRRs_pad[d + 1]])
            elif dRRs_pad[d] < 0:
                s12[d - padding] = np.min([dRRs_pad[d - 1], dRRs_pad[d + 1]])

        s22 = np.zeros(dRRs.size)
        # Loop from padding to padding + length of dRRs
        for d in np.arange(padding, padding + dRRs.size):
            if dRRs_pad[d] >= 0:
                s22[d - padding] = np.min([dRRs_pad[d + 1], dRRs_pad[d + 2]])
            elif dRRs_pad[d] < 0:
                s22[d - padding] = np.max([dRRs_pad[d + 1], dRRs_pad[d + 2]])

        ectopic_idx = []
        long_or_short_idx = []
        missed_idx = []
        extra_idx = []

        # Loop through data and identify the type of artifact
        i = 0
        while i < RRs.size - 2:
            if np.abs(dRRs[i]) <= 1:
                i += 1
                continue
            eq1 = np.logical_and(dRRs[i] > 1, (s12[i] < -c1 * dRRs[i] - c2))
            eq2 = np.logical_and(dRRs[i] < -1, (s12[i] > -c1 * dRRs[i] + c2))
            if np.any([eq1, eq2]):
                ectopic_idx.append(i)
                i += 1
                continue
            
            # If both of these are true, it's a normal beat continue to next index
            if (np.abs(dRRs[i]) <= 1) and (np.abs(mRRs[i]) <= 3):
                i += 1
                continue
            long_short_candidates = [i]
            
            # Check if next beat is also a candidate 
            if np.abs(dRRs[i + 1]) < np.abs(dRRs[i + 2]):
                long_short_candidates.append(i + 1)
            
            for j in long_short_candidates:
                # Long beat followed by short beat as compensation -> long beat
                eq3 = np.logical_and(dRRs[j] > 1, s22[j] < -1)
                # long or short beat
                eq4 = np.abs(mRRs[j]) > 3
                
                # short beat followed by long beat as compensation -> short beat 
                eq5 = np.logical_and(dRRs[i] < -1, s22[j] > 1)
                
                # if eq3, eq4, and eq5 are all false, it's a normal beat
                # if (not eq3) and (not eq4) and (not eq5):
                if ~np.any([eq3, eq4, eq5]):
                    i += 1
                    continue

                eq6 = np.abs(RRs[j] / 2 - medRR[j]) < th2[j]

                eq7 = np.abs(RRs[j] + RRs[j + 1] - medRR[j]) < th2[j]

                if np.all([eq5, eq7]):
                    extra_idx.append(j)
                    i += 1 
                    continue
                if np.all([eq3, eq6]):
                    missed_idx.append(j)
                    i += 1
                    continue

                long_or_short_idx.append(j)
                i += 1
        artifacts = {"ectopic": ectopic_idx, "missed": missed_idx, "extra": extra_idx, "longshort": long_or_short_idx}
        return artifacts


    """
    Function to correct artifacts in the systolic peak data
    """
    def correct_artifacts(self, artifacts, systolic_peaks_secs):
        # if extra beat, remove the peak index and update surrounding indices
        ectopic_idx = artifacts['ectopic']
        missed_idx = artifacts['missed']
        extra_idx = artifacts['extra']
        long_or_short_idx = artifacts['longshort']
        cleaned_peaks = systolic_peaks_secs.copy()
        #print("start of correct artifact sys peaks len: ", len(cleaned_peaks))
        if extra_idx:
            cleaned_peaks = np.delete(cleaned_peaks, extra_idx)
            # update other indices 
            ectopic_idx = self.update_indices(extra_idx, ectopic_idx, -1)
            long_or_short_idx = self.update_indices(extra_idx, long_or_short_idx, -1)
            missed_idx = self.update_indices(extra_idx, missed_idx, -1)
        #print("After extra idx len of syspeaks: ", len(cleaned_peaks))
        if missed_idx:
            cleaned_peaks = self.correct_missed(missed_idx, cleaned_peaks)
            ectopic_idx = self.update_indices(missed_idx, ectopic_idx, 1)
            long_or_short_idx = self.update_indices(missed_idx, long_or_short_idx, 1)
        #print("After missed idx len of syspeaks: ", len(cleaned_peaks))

        if ectopic_idx:
            cleaned_peaks = self.correct_ectopic_or_longshort(ectopic_idx, cleaned_peaks)
        #print("After ectopic idx len of syspeaks: ", len(cleaned_peaks))

        if long_or_short_idx:
            cleaned_peaks = self.correct_ectopic_or_longshort(long_or_short_idx, cleaned_peaks)
        #print("After long/short idx len of syspeaks: ", len(cleaned_peaks))
        
        return cleaned_peaks


    def update_indices(self, source, to_update, update):
        if not to_update:
            return to_update
        
        for s in source:
            to_update = [u + update if u > s else u for u in to_update]
        return to_update


    def correct_missed(self, missed_idx, systolic_peaks_secs):
        fixed_peaks = systolic_peaks_secs.copy()
        missed_idx = np.array(missed_idx)

        valid_idx = np.logical_and(missed_idx > 1, missed_idx < len(fixed_peaks))
        missed_idx = missed_idx[valid_idx]
        prev_peaks = fixed_peaks[[i - 1 for i in missed_idx]]
        next_peaks = fixed_peaks[missed_idx]
        assert prev_peaks.size == next_peaks.size, "Prev peaks and Next peaks must be same size"
        added_peaks = prev_peaks + (next_peaks - prev_peaks) / 2
       
        fixed_peaks = np.insert(fixed_peaks, missed_idx, added_peaks)
        return fixed_peaks


    def correct_ectopic_or_longshort(self, ectopic_or_longshort_idx, peaks):
        fixed_peaks = peaks.copy()
        ectopic_or_longshort_idx = np.array(ectopic_or_longshort_idx)
        
        valid_idx = np.logical_and(
            ectopic_or_longshort_idx > 1, 
            ectopic_or_longshort_idx < len(peaks) - 1)
        
        ectopic_or_longshort_idx = ectopic_or_longshort_idx[valid_idx]
        prev_peaks = fixed_peaks[[i - 1 for i in ectopic_or_longshort_idx]]
        next_peaks = fixed_peaks[[i + 1 for i in ectopic_or_longshort_idx]]

        interpolate_vals = prev_peaks + (next_peaks - prev_peaks) / 2

        fixed_peaks = np.delete(fixed_peaks, ectopic_or_longshort_idx)
        fixed_peaks = np.concatenate((fixed_peaks, interpolate_vals))
        # Sort the systolic peaks so they are in increasing time order 
        fixed_peaks.sort(kind="mergesort")

        return fixed_peaks


    def clean_systolic_peaks(self, systolic_peaks_secs):
        # Identify artifacts in systolic peaks 
        alpha = 5.2
        window_size = 91
        c1 = 0.13
        c2 = 0.17
        RRs, dRRs, mRRs, medRR, th2 = self.identify_artifacts(systolic_peaks_secs, alpha, window_size)
        print("in clean len of RRis: ", len(RRs))
        # Classify type of artifact 
        artifacts = self.classify_artifact(RRs, dRRs, mRRs, medRR, th2, c1, c2)  
        clean_peaks = self.correct_artifacts(artifacts, systolic_peaks_secs)

        n_artifacts_previous = np.inf
        n_artifacts_current = sum([len(i) for i in artifacts.values()])
    
        previous_diff = 0
        while n_artifacts_current - n_artifacts_previous != previous_diff:

            previous_diff = n_artifacts_previous - n_artifacts_current
            RRs, dRRs, mRRs, medRR, th2 = self.identify_artifacts(systolic_peaks_secs, 5.2, 91)
            artifacts = self.classify_artifact(RRs, dRRs, mRRs, medRR, th2, 0.13, 0.17)
            clean_peaks = self.correct_artifacts(artifacts, clean_peaks)

            n_artifacts_previous = n_artifacts_current
            n_artifacts_current = sum([len(i) for i in artifacts.values()])
        print("final cleaned len of peaks: ", len(clean_peaks))
        print("final cleaned len of RRis: ", len(RRs))
        print("ectopic idx: ", artifacts['ectopic'])
        print("long/short idx: ", artifacts['longshort'])
        print("missed idx: ", artifacts['missed'])
        print("extra idx: ", artifacts['extra'])
        
        print("size of ectopic: ", len(artifacts['ectopic']))
        print("size of long/short: ", len(artifacts['longshort']))
        print("size of missed: ", len(artifacts['missed']))
        print("size of extra: ", len(artifacts['extra']))
        
        assert np.all(np.diff(clean_peaks) > 0), "Non-monotonic peaks found"
        return artifacts, clean_peaks

    def cumlative_sum(self, RR_intervals):
        cum_sum = np.cumsum(RR_intervals)
        return cum_sum - RR_intervals[0]

    def parse_timestamps(self, RR_timestamps):
        fmt = "%Y-%m-%dT%H:%M:%S.%f%z"
        dts = [datetime.strptime(ts, fmt) for ts in RR_timestamps]

        # Make them relative to the first beat (in seconds)
        t0 = dts[0]
        rri_time = [(dt - t0).total_seconds() for dt in dts]
        return rri_time


    def threshold_correction(self, metrics_data, Cleaned, threshold=0.25, local_median_size=91):
        # Loop through RR intervals looking at local_median_size intervals
        RRs_to_remove = []
        sys_peaks_to_remove = []
        
        if Cleaned == False: 
            RR_intervals = metrics_data['rr_intervals']['values']
            RR_timestamps = metrics_data['rr_intervals']['timestamps']
        if Cleaned == True: 
            RR_intervals = metrics_data['rr_intervals_clean']['values']
            RR_timestamps = metrics_data['rr_intervals_clean']['timestamps']

        # Convert RR timestamps into a format that can be passed into Cubic Spline 
        rri_time = self.parse_timestamps(RR_timestamps)
    
        for i in range(local_median_size, len(RR_intervals)):
            cur_window = RR_intervals[i - local_median_size : i]
            if abs(RR_intervals[i] - np.median(cur_window)) > threshold:
                RRs_to_remove.append(i)
                sys_peaks_to_remove.append(i)

        first_idx = list(range(local_median_size + 1))
        for j in range(local_median_size):
            slice = [f for f in first_idx if not f == j]
            if abs(RR_intervals[j] - np.median(RR_intervals[slice])) > threshold:
                RRs_to_remove.append(j)
        
        list_of_cur_window_times = [RR_timestamps[i] for i in RRs_to_remove]
        # Print the number of beats to correct
        print(f"len of cur window: {Cleaned} ", len(list_of_cur_window_times))

        RR_goodvals = [RR_intervals[i] for i in range(len(RR_intervals)) if i not in RRs_to_remove]
        RR_goodtimes = [rri_time[i] for i in range(len(rri_time)) if i not in RRs_to_remove]
        
        # Linear interpolation to fill in bad vals
        linear = interp1d(RR_goodtimes, RR_goodvals, 'linear', bounds_error=False, fill_value='extrapolate')
        
        cubic_spline = CubicSpline(RR_goodtimes, RR_goodvals)
        RR_intervals_new = cubic_spline(rri_time)
        
        RR_intervals_new_linear = linear(rri_time)
        
        return RR_intervals_new_linear


    def calculate_RR_intervals(self, systolic_peaks_secs, Cleaned):
        # Get difference between consecutive systolic peaks
        us_date_times = [datetime.fromtimestamp(peak, tz=timezone.utc).astimezone(ZoneInfo("US/Pacific")).isoformat() for peak in systolic_peaks_secs]
        RR_intervals = np.ediff1d(systolic_peaks_secs, to_begin=0)  
        
        print("original length of rris: ", len(RR_intervals))
        RR_intervals[0] = np.mean(RR_intervals[1:])

        # Convert systolic peaks to UTC time
        us_date_times = [datetime.fromtimestamp(peak, tz=timezone.utc).astimezone(ZoneInfo("US/Pacific")).isoformat() for peak in systolic_peaks_secs]

        if Cleaned == False: 
            self.metrics_data['rr_intervals']['timestamps'] = us_date_times
            self.metrics_data['rr_intervals']['values'] = RR_intervals
        if Cleaned == True:
            self.metrics_data['rr_intervals_clean']['timestamps'] = us_date_times
            self.metrics_data['rr_intervals_clean']['values'] = RR_intervals
        return RR_intervals, systolic_peaks_secs

    def apply_threshold_correction(self, systolic_peaks_secs, Cleaned):
        RR_intervals_new = self.threshold_correction(self.metrics_data, Cleaned)
        
        # Filter our unrealistic interval values (HR < 40 bpm or HR > 200 bpm)
        invalid_idx = []
        HR_toohigh = []
        HR_toolow = []
        for i in range(len(RR_intervals_new)):
            if RR_intervals_new[i] <= 0.3 or RR_intervals_new[i] >= 1.5:
                invalid_idx.append(i)

            if RR_intervals_new[i] <= 0.3:
                HR_toohigh.append(i)

            if RR_intervals_new[i] >= 1.5:
                HR_toolow.append(i)
        print("length of HR too low from interpolation: ", len(HR_toolow))
        print("length of HR too high from interpolation: ", len(HR_toohigh))
        
        # For anomaly RRis, find the average between neighboring RR intervals 
        RR_new = RR_intervals_new.copy()
        sys_peak_new = systolic_peaks_secs.copy()
        for i in invalid_idx:
            avg_ls = []
            avg_ls.append(RR_intervals_new[i - 1])
            avg_ls.append(RR_intervals_new[i + 1])
            avg = np.mean(avg_ls)
            RR_new[i] = avg 
            print(avg)
       
        us_date_times = [datetime.fromtimestamp(peak, tz=timezone.utc).astimezone(ZoneInfo("US/Pacific")).isoformat() for peak in sys_peak_new]

        if Cleaned == False: 
            self.metrics_data['rr_intervals']['timestamps'] = us_date_times
            self.metrics_data['rr_intervals']['values'] = RR_new
        if Cleaned == True:
            self.metrics_data['rr_intervals_clean']['timestamps'] = us_date_times
            self.metrics_data['rr_intervals_clean']['values'] = RR_new
        print("final len of RRis: ", len(RR_new))
        print("final len of sys peaks: ", len(sys_peak_new))
        return RR_new, sys_peak_new


    def calculate_heartrate(self, RR_intervals):
        heart_rate = 60 / RR_intervals 
        return heart_rate.tolist()
    

    def smooth_heart_rate(self, heart_rate):
        # Use savgolay filter to smooth the heart rate data
        smooth_hr = savgol_filter(heart_rate, 61, 3)
        return smooth_hr.tolist()
    

    def calculate_SDNN(self, RR_intervals):
        # Calculate SDNN contiuously over the trial not just on stand alone measurement
        if len(RR_intervals) < 2:
            return 0.0
        mean_RR = np.mean(RR_intervals)
        squared_diffs = [(x - mean_RR) ** 2 for x in RR_intervals]
        squared_diffs_mean = sum(squared_diffs) / (len(squared_diffs) - 1)
        SDNN = np.sqrt(squared_diffs_mean)
        return SDNN


    def calculate_segmented_sdnn(self, RRis_filter, segment_duration_sec):
        cumulative_time = np.cumsum(RRis_filter)
        #print(cumulative_time)

        segments = []
        start_idx = 0
        while start_idx < len(cumulative_time):
            end_time = cumulative_time[start_idx] + segment_duration_sec
            end_idx = np.searchsorted(cumulative_time, end_time)
            segment = RRis_filter[start_idx:end_idx]
            if len(segment) > 1:
                segments.append(np.std(segment, ddof=1))
            start_idx = end_idx

        return np.mean(segments) if segments else 0.0

    """
    Use a gaussian kernel for moving average SDNN
    """
    def smoothing_SDNN(self, RRis, sigma=3.0):
        SDNN_values = []
        weights = []
        # Loop through time stamps 
        #RRis = self.metrics_data['rr_intervals_clean']['values']
        timestamps = self.metrics_data['rr_intervals_clean']['timestamps']
        rri_time = np.array(self.parse_timestamps(timestamps))

        for i, t_0 in enumerate(rri_time): 
            # 
            diffs = rri_time - t_0
            # Get the gaussian kernel value 
            weights = np.exp(-0.5 * (diffs / sigma) ** 2)       
            # normalize weights 
            weights /= weights.sum()     
            
            weighted_mean = np.sum(weights * RRis)
            weighted_variance = np.sum(weights * (RRis - weighted_mean) ** 2)
            weighted_sdnn = np.sqrt(weighted_variance)

            SDNN_values.append(weighted_sdnn)
        self.metrics_data['SDNN']['timestamps'] = timestamps
        self.metrics_data['SDNN']['values'] = np.array(SDNN_values) * 1000
        return SDNN_values
    
    def smoothing_RMSSD(self, RRis, sigma=3.0):
        RMSSD_values = []
        weights = []
        # Loop through time stamps 
        #RRis = self.metrics_data['rr_intervals_clean']['values']
        timestamps = self.metrics_data['rr_intervals_clean']['timestamps']
        rri_time = np.array(self.parse_timestamps(timestamps))

        for i, t_0 in enumerate(rri_time):  
            diffs = rri_time - t_0
            # Get the gaussian kernel value 
            weights = np.exp(-0.5 * (diffs / sigma) ** 2)       
            # normalize weights 
            weights /= weights.sum()     

            # Compute successive RR differences and pairwise weights
            rri_diffs = np.diff(RRis)
            pairwise_weights = 0.5 * (weights[:-1] + weights[1:])
            pairwise_weights /= pairwise_weights.sum()  # normalize to avoid scaling

            # Weighted RMSSD calculation
            squared_diffs = rri_diffs ** 2
            weighted_rmssd = np.sqrt(np.sum(pairwise_weights * squared_diffs))
            RMSSD_values.append(weighted_rmssd)
        self.metrics_data['RMSSD']['timestamps'] = timestamps
        self.metrics_data['RMSSD']['values'] = np.array(RMSSD_values) * 1000
        return RMSSD_values

    """
    Need to work on this: need a continuous HRV calculation (chunk up the intervals)
    Also, use some type of weighted moving average
    """

    def calculate_RMSSD(self, RR_intervals):
        if len(RR_intervals) < 2:
            return 0.0
        diffs = np.diff(RR_intervals)
        squared_diffs = diffs ** 2
        RMSSD = np.sqrt(np.mean(squared_diffs))
        return RMSSD
    
    def calculate_pNN50(self, RR_intervals):
        NN50 = 0
        RR_diff = np.ediff1d(RR_intervals)
        for rr in RR_diff:
            if rr >= 0.05:
                NN50 += 1
        pNN50 = (NN50 / len(RR_intervals)) * 100
        return NN50, pNN50

    def output_data(self, metrics_data, sys_peaks, hrt, output_path): 
        # Create the output directory if it doesn't exist
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Save the eda data into a json file
        with open(os.path.join(output_path, 'eda_07_15.json'), 'w') as f:
            for i in range(len(metrics_data['eda']['timestamps'])):
                timestamp = datetime.fromtimestamp(metrics_data['eda']['timestamps'][i] / 1e6, tz=timezone.utc).astimezone(ZoneInfo("US/Pacific")).isoformat()
                f.write(f'{{"source": "watch", "signal_id": "eda", "timestamp": {timestamp}, "value": {metrics_data["eda"]["values"][i]}, "unit": "microSiemens"}}\n')

        # Save the temperature data into a json file
        with open(os.path.join(output_path, 'temperature_07_15.json'), 'w') as f:
            for i in range(len(metrics_data['temperature']['timestamps'])):
                timestamp = datetime.fromtimestamp(metrics_data['temperature']['timestamps'][i] / 1e6, tz=timezone.utc).astimezone(ZoneInfo("US/Pacific")).isoformat()
                f.write(f'{{"source": "watch", "signal_id": "temperature", "timestamp": {timestamp}, "value": {metrics_data["temperature"]["values"][i]}, "unit": "C"}}\n')

        # Save the SDNN data into a json file
        with open(os.path.join(output_path, 'SDNN_07_15.json'), 'w') as f:
            for i in range(len(metrics_data['SDNN']['timestamps'])):
                #timestamp = datetime.fromtimestamp(metrics_data['SDNN']['timestamps'][i], tz=timezone.utc).astimezone(ZoneInfo("US/Pacific")).isoformat()
                f.write(f'{{"source": "watch", "signal_id": "SDNN", "timestamp": {metrics_data['SDNN']['timestamps'][i]}, "value": {metrics_data["SDNN"]["values"][i]}, "unit": "ms"}}\n')

        with open(os.path.join(output_path, 'RMSSD_07_15.json'), 'w') as f:
            for i in range(len(metrics_data['RMSSD']['timestamps'])):
                f.write(f'{{"source": "watch", "signal_id": "RMSSD", "timestamp": {metrics_data['RMSSD']['timestamps'][i]}, "value": {metrics_data["RMSSD"]["values"][i]}, "unit": "ms"}}\n')

        # Each line of the json file will be in this format: {"source":"watch","signal_id":"heart_rate","timestamp":"2025-06-24T18:24:11.640171+00:00","value":"66","unit":"bpm"}
        with open(os.path.join(output_path, 'heart_rate_07_15.json'), 'w') as f:
            for i in range(len(hrt)):
                corresponding_sys = sys_peaks[i]
                corresponding_rr_time = datetime.fromtimestamp(corresponding_sys, tz=timezone.utc).astimezone(ZoneInfo("US/Pacific")).isoformat()
                f.write(f'{{"source": "watch", "signal_id": "heart_rate", "timestamp": {corresponding_rr_time}, "value": {hrt[i]}, "unit": "bpm"}}\n')
        
