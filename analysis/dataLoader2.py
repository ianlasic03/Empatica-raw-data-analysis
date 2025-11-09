import os
import re
import json
from fastavro import reader
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

class DataLoader2:
    
    @staticmethod
    def parse_metadata(metadata_path):
        """
        Parse pilot metadata JSON file to extract relevant information.
        
        Args:
            metadata_path: Path to the metadata JSON file
            
        Returns:
            dict: Contains pilot_id, flight_date, scenarios with timing and condition info
        """
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Parse scenarios with their conditions
        scenarios = []
        for i, scenario_letter in enumerate(metadata['scenario_order'], start=1):
            scenario_key = f"scenario_{scenario_letter}"
            scenario_info = metadata[scenario_key]
            
            # Determine condition (with_clippy or without_clippy)
            condition = scenario_info.get('condition', '')
            clippy_suffix = "_Clippy" if "with_clippy" in condition else ""
            
            # Parse start and end times
            start_time = scenario_info.get('start_time')
            end_time = scenario_info.get('end_time')
            
            scenarios.append({
                'sequence': f"Seq{i}_{scenario_letter}",
                'condition': condition,
                'clippy_suffix': clippy_suffix,
                'start_time': start_time,
                'end_time': end_time
            })
        
        return {
            'pilot_id': metadata['pilot_id'],
            'flight_date': metadata['flight_date'],
            'scenarios': scenarios
        }
    
    @staticmethod
    def get_raw_data_path(flight_date, base_path="/Users/ianlasic/Empatica-raw-data-analysis/1/1/participant_data"):
        """
        Construct the path to the raw data v6 directory based on flight date.
        
        Args:
            flight_date: Date in format YYYY-MM-DD
            base_path: Base path to participant data
            
        Returns:
            str: Path to raw_data/v6 directory, or None if not found
        """
        # Construct path to the date folder
        date_folder = os.path.join(base_path, flight_date)
        
        if not os.path.exists(date_folder):
            print(f"Warning: Date folder not found: {date_folder}")
            return None
        
        # Look for BETATEST folder (should be only one)
        betatest_folders = [f for f in os.listdir(date_folder) if f.startswith('BETATEST-')]
        
        if not betatest_folders:
            print(f"Warning: No BETATEST folder found in {date_folder}")
            return None
        
        if len(betatest_folders) > 1:
            print(f"Warning: Multiple BETATEST folders found in {date_folder}, using first one: {betatest_folders[0]}")
        
        # Construct full path to raw_data/v6
        raw_data_path = os.path.join(
            date_folder,
            betatest_folders[0],
            "raw_data",
            "v6"
        )
        
        if not os.path.exists(raw_data_path):
            print(f"Warning: raw_data/v6 not found: {raw_data_path}")
            return None
        
        return raw_data_path
    
    @staticmethod
    def get_output_path(flight_date, pilot_id, base_path="/Users/ianlasic/Empatica-raw-data-analysis/Flight_test_data"):
        """
        Construct the output path based on flight date and pilot ID.
        
        Args:
            flight_date: Date in format YYYY-MM-DD
            pilot_id: Pilot identifier
            base_path: Base path to flight test data
            
        Returns:
            str: Path to output directory
        """
        # Convert flight_date from YYYY-MM-DD to MM_DD format
        date_parts = flight_date.split('-')
        month_day = f"{date_parts[1]}_{date_parts[2]}"
        
        # Construct path: base_path/output_data_MM_DD/pilot_id_data
        output_path = os.path.join(
            base_path,
            f"output_data_{month_day}",
            f"{pilot_id}_data"
        )
        
        return output_path
    
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
    

    def _filter_timestamped_data(self, timestamps, values, start_time, end_time):
        """Filter timestamped data based on start and end times"""
        if start_time is None or end_time is None:
            return {'timestamps': timestamps, 'values': values}
        
        filtered_timestamps = []
        filtered_values = []
        
        for i, timestamp in enumerate(timestamps):
            timestamp_dt = datetime.fromtimestamp(timestamp / 1e6, tz=timezone.utc)
            timestamp_dt = timestamp_dt.time()
            if start_time <= timestamp_dt <= end_time:
                filtered_timestamps.append(timestamp)
                filtered_values.append(values[i])
        
        return {'timestamps': filtered_timestamps, 'values': filtered_values}
    
    def process_avro_files_test(self, start_time, end_time):
            # Ensure the files are loaded in time order
            files_time_order = self.load_files_in_time_order(self.data_path)
            metrics_data = self.metrics_data
            
            # Loop through each avro file in the raw data path
            for i in range(len(files_time_order)):
                if files_time_order[i].endswith(".avro"):
                    file_path = os.path.join(self.data_path, files_time_order[i])
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
                            
                            # Check if this record overlaps with our time window
                            if start_time is not None and end_time is not None:
                                record_start_time = datetime.fromtimestamp(current_timestamp / 1e6, tz=timezone.utc)
                                
                                # Estimate record end time (you'll need to calculate this based on your data)
                                # This is a rough estimate - adjust based on your actual record duration
                                estimated_record_duration = timedelta(minutes=30, seconds=3)  # Adjust as needed
                                record_end_time = (record_start_time + estimated_record_duration)
                                
                                # Convert to datetime.time() for easy comparsion to input times
                                record_start_time = record_start_time.time()
                                record_end_time = record_end_time.time()
                                # Skip if record doesn't overlap with our time window
                                if record_end_time < start_time or record_start_time > end_time:
                                    continue
                            
                            # Now filter individual data points within the record
                            for key in record['rawData'].keys():
                                """if key == 'accelerometer':
                                    # Filter accelerometer data based on calculated timestamps
                                    filtered_indices = self._get_filtered_indices_for_sensor(
                                        record['rawData'][key], start_time, end_time, 'accelerometer'
                                    )
                                    if filtered_indices:
                                        metrics_data[key]['x'].extend([record['rawData'][key]['x'][i] for i in filtered_indices])
                                        metrics_data[key]['y'].extend([record['rawData'][key]['y'][i] for i in filtered_indices])
                                        metrics_data[key]['z'].extend([record['rawData'][key]['z'][i] for i in filtered_indices])
                                """
                                if key == 'eda':
                                    # Generate timestamps and filter
                                    if next_file_start_time_eda is None:
                                        timestamps = [record['rawData'][key]['timestampStart'] + 250000 * i 
                                                    for i in range(len(record['rawData']['eda']['values']))]
                                    else:
                                        timestamps = np.linspace(record['rawData'][key]['timestampStart'], 
                                                            next_file_start_time_eda - 250000, 
                                                            len(record['rawData'][key]['values']))
                                    
                                    filtered_data = self._filter_timestamped_data(timestamps, 
                                                                                record['rawData'][key]['values'], 
                                                                                start_time, end_time)
                                    metrics_data[key]['timestamps'].extend(filtered_data['timestamps'])
                                    metrics_data[key]['values'].extend(filtered_data['values'])

                                elif key == 'temperature':
                                     # Generate timestamps and filter
                                    if next_file_start_time_temp is None:
                                        timestamps = [record['rawData'][key]['timestampStart'] + 1000000 * i 
                                                    for i in range(len(record['rawData']['eda']['values']))]
                                    else:
                                        timestamps = np.linspace(record['rawData'][key]['timestampStart'], 
                                                            next_file_start_time_eda - 1000000, 
                                                            len(record['rawData'][key]['values']))
                                    
                                    filtered_data = self._filter_timestamped_data(timestamps, 
                                                                                record['rawData'][key]['values'], 
                                                                                start_time, end_time)
                                    metrics_data[key]['timestamps'].extend(filtered_data['timestamps'])
                                    metrics_data[key]['values'].extend(filtered_data['values'])
                                
                                elif key == 'systolicPeaks':
                                    # Filter peaks based on their actual timestamps
                                    peak_timestamps = record['rawData'][key]['peaksTimeNanos']
                                    if start_time is not None and end_time is not None:
                                        filtered_peaks = []
                                        for peak_time in peak_timestamps:
                                            peak_datetime = datetime.fromtimestamp(peak_time / 1e9, tz=timezone.utc).time()
                                            if start_time <= peak_datetime <= end_time:
                                                filtered_peaks.append(peak_time)
                                        metrics_data[key].extend(filtered_peaks)
                                    else:
                                        metrics_data[key].extend(peak_timestamps)


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

        if extra_idx:
            cleaned_peaks = np.delete(cleaned_peaks, extra_idx)
            # update other indices 
            ectopic_idx = self.update_indices(extra_idx, ectopic_idx, -1)
            long_or_short_idx = self.update_indices(extra_idx, long_or_short_idx, -1)
            missed_idx = self.update_indices(extra_idx, missed_idx, -1)
        
        if missed_idx:
            cleaned_peaks = self.correct_missed(missed_idx, cleaned_peaks)
            ectopic_idx = self.update_indices(missed_idx, ectopic_idx, 1)
            long_or_short_idx = self.update_indices(missed_idx, long_or_short_idx, 1)

        if ectopic_idx:
            cleaned_peaks = self.correct_ectopic_or_longshort(ectopic_idx, cleaned_peaks)

        if long_or_short_idx:
            cleaned_peaks = self.correct_ectopic_or_longshort(long_or_short_idx, cleaned_peaks)
        
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
    
    def parse_timestamps(self, RR_timestamps):
        dts = []
        for ts in RR_timestamps:
            dt = datetime.fromisoformat(ts)
            dts.append(dt)
        
        # Make them relative to the first beat (in seconds)
        if dts:
            t0 = dts[0]
            rri_time = [(dt - t0).total_seconds() for dt in dts]
            return rri_time
        else:
            raise ValueError("No valid timestamps could be parsed")

    """def parse_timestamps(self, RR_timestamps):
        print(f"Number of RR_timestamps: {len(RR_timestamps)}")
        if len(RR_timestamps) > 0:
            print(f"First few timestamps: {RR_timestamps[:3]}")
        
        fmt = "%Y-%m-%dT%H:%M:%S.%f%z"
        dts = []
        
        for i, ts in enumerate(RR_timestamps):
            try:
                dt = datetime.strptime(ts, fmt)
                dts.append(dt)
            except ValueError as e:
                print(f"Failed to parse timestamp {i}: '{ts}' - Error: {e}")
        
        if len(dts) == 0:
            raise ValueError("No valid timestamps could be parsed")
        
        # Make them relative to the first beat (in seconds)
        t0 = dts[0]
        rri_time = [(dt - t0).total_seconds() for dt in dts]
        return rri_time
    
     def parse_timestamps(self, RR_timestamps):
        fmt = "%Y-%m-%dT%H:%M:%S.%f%z"
        dts = [datetime.strptime(ts, fmt) for ts in RR_timestamps]

        # Make them relative to the first beat (in seconds)
        t0 = dts[0]
        rri_time = [(dt - t0).total_seconds() for dt in dts]
        return rri_time"""

    def threshold_correction(self, metrics_data, threshold=0.25, local_median_size=91):
        # Loop through RR intervals looking at local_median_size intervals
        RRs_to_remove = []
        sys_peaks_to_remove = []
    
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
        print(f"len of cur window: ", len(list_of_cur_window_times))

        RR_goodvals = [RR_intervals[i] for i in range(len(RR_intervals)) if i not in RRs_to_remove]
        RR_goodtimes = [rri_time[i] for i in range(len(rri_time)) if i not in RRs_to_remove]
        
        # Linear interpolation to fill in bad vals
        linear = interp1d(RR_goodtimes, RR_goodvals, 'linear', bounds_error=False, fill_value='extrapolate')
        RR_intervals_new_linear = linear(rri_time)
        
        return RR_intervals_new_linear


    def calculate_RR_intervals(self, systolic_peaks_secs):
        # Get difference between consecutive systolic peaks
        us_date_times = [datetime.fromtimestamp(peak, tz=timezone.utc).astimezone(ZoneInfo("US/Pacific")).isoformat() for peak in systolic_peaks_secs]
        RR_intervals = np.ediff1d(systolic_peaks_secs, to_begin=0)  
        
        RR_intervals[0] = np.mean(RR_intervals[1:])

        # Convert systolic peaks to UTC time
        us_date_times = [datetime.fromtimestamp(peak, tz=timezone.utc).isoformat() for peak in systolic_peaks_secs]

        self.metrics_data['rr_intervals_clean']['timestamps'] = us_date_times
        self.metrics_data['rr_intervals_clean']['values'] = RR_intervals
        return RR_intervals, systolic_peaks_secs


    def apply_threshold_correction(self, systolic_peaks_secs):
        RR_intervals_new = self.threshold_correction(self.metrics_data)
        
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
    
            # Handle boundary cases
            if i == 0:
                # First element: use next two values
                if len(RR_intervals_new) > 1:
                    avg_ls.append(RR_intervals_new[i + 1])
                    if len(RR_intervals_new) > 2:
                        avg_ls.append(RR_intervals_new[i + 2])
            elif i == len(RR_intervals_new) - 1:
                # Last element: use previous two values
                avg_ls.append(RR_intervals_new[i - 1])
                if i > 1:
                    avg_ls.append(RR_intervals_new[i - 2])
            else:
                # Middle elements: use neighboring values
                avg_ls.append(RR_intervals_new[i - 1])
                avg_ls.append(RR_intervals_new[i + 1])
            
            if avg_ls:  # Only calculate if we have valid neighbors
                avg = np.mean(avg_ls)
                RR_new[i] = avg
            
        us_date_times = [datetime.fromtimestamp(peak, tz=timezone.utc).isoformat() for peak in sys_peak_new]

        self.metrics_data['rr_intervals_clean']['timestamps'] = us_date_times
        self.metrics_data['rr_intervals_clean']['values'] = RR_new
        return RR_new, sys_peak_new


    def calculate_heartrate(self, RR_intervals):
        heart_rate = 60 / RR_intervals 
        return heart_rate.tolist()
    

    def smooth_heart_rate(self, heart_rate):
        # Use savgolay filter to smooth the heart rate data
        smooth_hr = savgol_filter(heart_rate, 61, 3)
        return smooth_hr.tolist()
    
    """
    Calculate one shot SDNN over experiment
    """
    def calculate_SDNN(self, RR_intervals):
        if len(RR_intervals) < 2:
            return 0.0
        mean_RR = np.mean(RR_intervals)
        squared_diffs = [(x - mean_RR) ** 2 for x in RR_intervals]
        squared_diffs_mean = sum(squared_diffs) / (len(squared_diffs) - 1)
        SDNN = np.sqrt(squared_diffs_mean)
        return SDNN
    

    """
    Calculate one shot RMSSD over experiment
    """
    def calculate_RMSSD(self, RR_intervals):
        if len(RR_intervals) < 2:
            return 0.0
        diffs = np.diff(RR_intervals)
        squared_diffs = diffs ** 2
        RMSSD = np.sqrt(np.mean(squared_diffs))
        return RMSSD


    """
    Guassian kernel for moving average/continuous SDNN measurments
    """
    def smoothing_SDNN(self, RRis, sigma=3.0):
        SDNN_values = []
        weights = []
        # Loop through time stamps 
        timestamps = self.metrics_data['rr_intervals_clean']['timestamps']
        rri_time = np.array(self.parse_timestamps(timestamps))

        for t_0 in rri_time: 
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
    
    """
    Guassian kernel for moving average/continuous RMSSD measurments
    """
    def smoothing_RMSSD(self, RRis, sigma=3.0):
        RMSSD_values = []
        weights = []
        # Loop through time stamps 
        timestamps = self.metrics_data['rr_intervals_clean']['timestamps']
        rri_time = np.array(self.parse_timestamps(timestamps))

        for t_0 in rri_time:  
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
    

    def calculate_pNN50(self, RR_intervals):
        NN50 = 0
        RR_diff = np.ediff1d(RR_intervals)
        for rr in RR_diff:
            if rr >= 0.05:
                NN50 += 1
        pNN50 = (NN50 / len(RR_intervals)) * 100
        return NN50, pNN50


    def output_data(self, end_time, metrics_data, sys_peaks, RRis, hrt, output_path, pilot_id, scenario_sequence, clippy_suffix, flight_date): 
        """
        Save processed data to JSON files with dynamic naming.
        
        Args:
            end_time: End time of the scenario
            metrics_data: Dictionary containing all metrics
            sys_peaks: Systolic peaks array
            RRis: RR intervals array
            hrt: Heart rate array
            output_path: Base output directory path
            pilot_id: Pilot identifier (e.g., 'aldi', 'japh')
            scenario_sequence: Scenario sequence (e.g., 'Seq1_A', 'Seq2_B')
            clippy_suffix: Clippy suffix ('_Clippy' or '')
            flight_date: Flight date in YYYY-MM-DD format
        """
        # Create the output directory if it doesn't exist
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        # Convert flight_date to MM_DD format for filenames
        date_parts = flight_date.split('-')
        date_str = f"{date_parts[1]}_{date_parts[2]}"
        
        # Construct base filename: pilot_id_scenario_clippy
        base_filename = f"{pilot_id}_{scenario_sequence}{clippy_suffix}"
        
        # Save systolic peaks into a json file
        sys_peaks_file = f'physio_{base_filename}_sys_peaks_{date_str}.json'
        with open(os.path.join(output_path, sys_peaks_file), 'w') as f:
            for i in range(len(sys_peaks)):
                timestamp = datetime.fromtimestamp(sys_peaks[i], tz=timezone.utc).isoformat()
                f.write(f'{{"source": "watch", "signal_id": "systolic_peaks", "timestamp": "{timestamp}", "value": "{sys_peaks[i]}", "unit": "Seconds"}}\n')

        # Save the one shot RMSSD into a json file
        rmssd_oneshot_file = f'physio_{base_filename}_RMSSD(1_shot)_{date_str}.json'
        with open(os.path.join(output_path, rmssd_oneshot_file), 'w') as f:
            # Calculate RMSSD value, multiply by 1000 to get in ms
            RMSSD_value = self.calculate_RMSSD(RRis) * 1000
            f.write(f'{{"source": "watch", "signal_id": "RMSSD", "timestamp": "{end_time}", "value": "{RMSSD_value}", "unit": "ms"}}\n')

        # Save the one shot SDNN into a json file
        sdnn_oneshot_file = f'physio_{base_filename}_SDNN(1_shot)_{date_str}.json'
        with open(os.path.join(output_path, sdnn_oneshot_file), 'w') as f:
            # Calculate SDNN value, multiply by 1000 to get in ms
            SDNN_value = self.calculate_SDNN(RRis) * 1000
            f.write(f'{{"source": "watch", "signal_id": "SDNN", "timestamp": "{end_time}", "value": "{SDNN_value}", "unit": "ms"}}\n')

        # Save the eda data into a json file
        eda_file = f'physio_{base_filename}_eda_{date_str}.json'
        with open(os.path.join(output_path, eda_file), 'w') as f:
            for i in range(len(metrics_data['eda']['timestamps'])):
                timestamp = datetime.fromtimestamp(metrics_data['eda']['timestamps'][i] / 1e6, tz=timezone.utc).isoformat()
                f.write(f'{{"source": "watch", "signal_id": "eda", "timestamp": "{timestamp}", "value": "{metrics_data["eda"]["values"][i]}", "unit": "microSiemens"}}\n')

        # Save the temperature data into a json file
        temp_file = f'physio_{base_filename}_temperature_{date_str}.json'
        with open(os.path.join(output_path, temp_file), 'w') as f:
            for i in range(len(metrics_data['temperature']['timestamps'])):
                timestamp = datetime.fromtimestamp(metrics_data['temperature']['timestamps'][i] / 1e6, tz=timezone.utc).isoformat()
                f.write(f'{{"source": "watch", "signal_id": "temperature", "timestamp": "{timestamp}", "value": "{metrics_data["temperature"]["values"][i]}", "unit": "C"}}\n')

        # Save the continuous SDNN data into a json file
        sdnn_file = f'physio_{base_filename}_SDNN_{date_str}.json'
        with open(os.path.join(output_path, sdnn_file), 'w') as f:
            for i in range(len(metrics_data['SDNN']['timestamps'])):
                timestamp = datetime.fromisoformat(metrics_data['SDNN']['timestamps'][i]).isoformat()
                f.write(f'{{"source": "watch", "signal_id": "SDNN", "timestamp": "{timestamp}", "value": "{metrics_data["SDNN"]["values"][i]}", "unit": "ms"}}\n')

        # Save the continuous RMSSD data into a json file
        rmssd_file = f'physio_{base_filename}_RMSSD_{date_str}.json'
        with open(os.path.join(output_path, rmssd_file), 'w') as f:
            for i in range(len(metrics_data['RMSSD']['timestamps'])):
                timestamp = datetime.fromisoformat(metrics_data['RMSSD']['timestamps'][i]).isoformat()
                f.write(f'{{"source": "watch", "signal_id": "RMSSD", "timestamp": "{timestamp}", "value": "{metrics_data["RMSSD"]["values"][i]}", "unit": "ms"}}\n')

        # Save the heart rate data into a json file
        hr_file = f'physio_{base_filename}_heart_rate_{date_str}.json'
        with open(os.path.join(output_path, hr_file), 'w') as f:
            for i in range(len(hrt)):
                corresponding_sys = sys_peaks[i]
                corresponding_rr_time = datetime.fromtimestamp(corresponding_sys, tz=timezone.utc).isoformat()
                f.write(f'{{"source": "watch", "signal_id": "heart_rate", "timestamp": "{corresponding_rr_time}", "value": "{hrt[i]}", "unit": "bpm"}}\n')
        
        print(f"\n✓ Data saved to: {output_path}")
        print(f"  - {sys_peaks_file}")
        print(f"  - {rmssd_oneshot_file}")
        print(f"  - {sdnn_oneshot_file}")
        print(f"  - {eda_file}")
        print(f"  - {temp_file}")
        print(f"  - {sdnn_file}")
        print(f"  - {rmssd_file}")
        print(f"  - {hr_file}")