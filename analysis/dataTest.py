import os
import json
from multiprocessing import Pool

import json
import re
import shutil

#input_file = "/Users/ianlasic/Empatica-raw-data-analysis/output_data_09_18/hbxm_data/hbxm_Seq1_A_eda_09_18.json"
#output_file = "/Users/ianlasic/Empatica-raw-data-analysis/output_data_09_18/hbxm_data/hbxm_Seq1_A_eda_09_18_strings.json"

def convert_to_strings_only(input_file, output_file):
    successful_count = 0
    error_count = 0
    fixed_count = 0
    
    try:
        with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
            for line_num, line in enumerate(infile, 1):
                original_line = line.strip()
                if not original_line:
                    outfile.write(line)  # Preserve empty lines
                    continue
                
                # Try to fix unquoted timestamps first
                fixed_line = re.sub(
                    r'"timestamp":\s*([0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]+\+[0-9]{2}:[0-9]{2})',
                    r'"timestamp": "\1"',
                    original_line
                )
                
                if fixed_line != original_line:
                    fixed_count += 1
                
                try:
                    # Parse the JSON
                    data = json.loads(fixed_line)
                    
                    # Convert only timestamp and value to strings
                    if 'timestamp' in data:
                        data['timestamp'] = str(data['timestamp'])
                    if 'value' in data:
                        data['value'] = str(data['value'])
                    
                    # Write back as single line JSON (preserving JSONL format)
                    outfile.write(json.dumps(data) + '\n')
                    successful_count += 1
                    
                except json.JSONDecodeError as e:
                    print(f"Error parsing line {line_num}: {e}")
                    print(f"Line content: {original_line[:100]}...")
                    # Write the original line if we can't parse it
                    outfile.write(line)
                    error_count += 1
        
        print(f"Summary:")
        print(f"Successfully processed: {successful_count} lines")
        print(f"Fixed unquoted timestamps: {fixed_count} lines")
        print(f"Errors encountered: {error_count} lines")
        print(f"Output saved to: {output_file}")
        
    except Exception as e:
        print(f"Error processing file: {e}")

# Run the conversion
directory = 'output_data_09_08/xrst_data'
for filename in os.listdir(directory):
    if filename.endswith('.json') and not filename.endswith('_processed.json'):
        print(f"Processing: {filename}")
        
        input_path = os.path.join(directory, filename)
        temp_path = os.path.join(directory, filename + '.temp')
        
        # Process to temporary file
        convert_to_strings_only(input_path, temp_path)
        
        # Replace original with processed version
        shutil.move(temp_path, input_path)
        print(f"Updated: {filename}")