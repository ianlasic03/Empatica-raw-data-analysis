import pandas as pd
from statsmodels.formula.api import ols
import os

"""
Access all the flight files
- Calculate average for each metric and save as a dataframe
"""
def access_files(path):
    files = os.listdir(path)
    records = []
    
    for file in files:
        if file.endswith('.json'):  # Only process JSON files
            file_name = os.path.basename(file)
            file_path = os.path.join(path, file_name)
            
            # Create a DataFrame directly using pandas read_json with lines=True
            try:
                df = pd.read_json(file_path, lines=True)
                
                # Extract metadata from filename
                sequence = '1' if 'seq1' in file_name.lower() else '2'
                has_clippy = 1 if 'clippy' in file_name.lower() else 0
                metric_type = next((m for m in ['eda', 'heart_rate', 'RMSSD', 'SDNN', 'temperature'] 
                                 if m in file_name), None)
                
                if metric_type:
                    # Calculate average value for this metric
                    avg_value = df['value'].astype(float).mean()
                    records.append({
                        'file': file_name,
                        'sequence': sequence,
                        'has_clippy': has_clippy,
                        'metric': metric_type,
                        'average_value': avg_value
                    })


            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
    
    # Convert records to DataFrame for analysis
    if records:
        results_df = pd.DataFrame(records)
        return results_df
    return None


"""
Run a ols model on all participants using average value for each biometric
"""
def process_all_participants(base_path):
    all_records = []
    
    # Get all date folders
    date_folders = sorted([f for f in os.listdir(base_path) if f.startswith('output_data')])
    
    for date_folder in date_folders:
        date_path = os.path.join(base_path, date_folder)
        if not os.path.isdir(date_path):
            continue
            
        # Get all participant folders for this date
        participant_folders = [f for f in os.listdir(date_path) if os.path.isdir(os.path.join(date_path, f))]
        
        for participant_folder in participant_folders:
            folder_path = os.path.join(date_path, participant_folder)

            # Get participant ID from folder name
            participant_id = participant_folder.split('_')[0]
            
            # Process this participant's data
            results = access_files(folder_path)
            if results is not None:
                # Add participant ID to results
                results['participant_id'] = participant_id

                # --- START: Z-SCORE CALCULATION ---
                # This is the new block. We group by metric to calculate the z-score
                # for each metric type (eda, heart_rate, etc.) separately.
                # .transform() applies the function to each group and returns a series
                # with the same index as the original dataframe.
                
                # Define a small z-score function to handle cases with zero standard deviation
                def z_score_transform(series):
                    mean = series.mean()
                    std = series.std()
                    if std == 0:
                        return 0 # Or handle as you see fit (e.g., return np.nan)
                    return (series - mean) / std

                results['z_score_value'] = results.groupby('metric')['average_value'].transform(z_score_transform)
                # --- END: Z-SCORE CALCULATION ---
                
                all_records.append(results)

    # Combine all results
    if all_records:
        combined_results = pd.concat(all_records, ignore_index=True)
    print("Linear Regression Analysis:")
    print(combined_results)
    # Loop through each metric and run a separate regression model
    for metric in combined_results['metric'].unique():
        # Get the data for the current metric
        metric_data = combined_results[combined_results['metric'] == metric].copy()
        print('Metric z-score: ', metric_data['z_score_value'])

        # Ensure data types are correct for the model
        metric_data['sequence'] = metric_data['sequence'].astype('category')
        metric_data['has_clippy'] = metric_data['has_clippy'].astype('category')
        
        # Define interaction model
        formula = "z_score_value ~ C(sequence) * C(has_clippy)"
        
        try:
            model = ols(formula, data=metric_data).fit()
            print("\n" + "="*70)
            print(f"Analyzing Metric: {metric}")
            #print(model.summary())
            print("\n Interpretation of Coefficients: \n")
            print(f"Analyzing Metric: {metric}")

            # Extract Coefficients and P-values 
            params = model.params
            pvalues = model.pvalues
            
            intercept = params['Intercept']
            seq_coef = params.get('C(sequence)[T.2]', 0) 
            clippy_coef = params.get('C(has_clippy)[T.1]', 0)
            interaction_coef = params.get('C(sequence)[T.2]:C(has_clippy)[T.1]', 0)

            seq_pvalue = pvalues.get('C(sequence)[T.2]', 99)
            clippy_pvalue = pvalues.get('C(has_clippy)[T.1]', 99)
            interaction_pvalue = pvalues.get('C(sequence)[T.2]:C(has_clippy)[T.1]', 99)

            # Print Conditional Effects and Interaction
            print(f"* Baseline (Seq 1, No-Clippy): The predicted average_value is {intercept:.4f}.")
            print(f"* Effect of Sequence 2 (for No-Clippy group): Being in Seq 2 changes the value by {seq_coef:.4f}. (p-value: {seq_pvalue:.4f})")
            print(f"* Effect of Clippy (for Seq 1 group): Having Clippy changes the value by {clippy_coef:.4f}. (p-value: {clippy_pvalue:.4f})")
            print(f"* Interaction Effect: {interaction_coef:.4f} (p-value: {interaction_pvalue:.4f})")
            if interaction_pvalue < 0.05:
                print("The interaction is statistically significant. The effect of sequence depends on whether the user has clippy.")
            else:
                print("The interaction is NOT statistically significant. The effect of sequence is roughly the same regardless of clippy.")

            # Summary of effects
            print("Summary of Effects: ")
            effect_seq2_no_clippy = seq_coef
            effect_seq2_has_clippy = seq_coef + interaction_coef
            print(f"For users WITHOUT Clippy, the effect of moving from Seq 1 to Seq 2 is: {effect_seq2_no_clippy:+.4f}")
            print(f"For users WITH Clippy, the effect of moving from Seq 1 to Seq 2 is: {effect_seq2_has_clippy:+.4f}")
            print("\n" + "="*70)

        except Exception as e:
            print(f"Could not fit model for {metric}. Error: {e}")


def main():
    base_path = 'Flight_test_participant_data'
    process_all_participants(base_path)

if __name__ == '__main__':
    main()