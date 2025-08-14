import re
import os
import pandas as pd

def parse_log_file_entries(text):
    pattern = r"(?P<filename>[\w\-.]+\.log\.part\d+_sorted_processed\.csv),\s*(?P<unixtime>\d+\.\d+),\s*(?P<id>\d+)"
    
    match = re.match(pattern, text.strip())
    if not match:
        raise ValueError(f"Input text does not match the expected format: {text}")
    
    return match.groupdict()
    
t_zero = {}
with open("D:\\SWAT\\network\\processed\\files_and_ts.txt", 'r') as f:
    for line in f.readlines():
        parsed_dict = parse_log_file_entries(line)
        t_zero[parsed_dict['filename']] = float(parsed_dict['unixtime'])

files_with_offset_time = "D:\\SWAT\\network\\processed\\by_file"
days_to_process = ["2015-12-28", "2015-12-29", "2015-12-30", "2015-12-31", "2016-01-01", "2016-01-02", "2016-01-03"]
output_path = "D:\\SWAT\\network\\processed\\by_file_time_group"

part_pattern = r"\.part(?P<part>\d+)_"

files_done = 198
for file in sorted(os.listdir(files_with_offset_time))[files_done:]:
    if file.endswith(".csv") and any(day in file for day in days_to_process):
        print(f"Processing file: {file}. Files done: {files_done}.")
        subgroup = file[:17]
        match = re.search(part_pattern, file)
        part = int(match.group("part"))
        if part < 6:
            part_addition = "_part1"
        elif part >= 6 and part < 11:
            part_addition = "_part2"
        elif part >= 11:
            part_addition = "_part3"
        else:
            part_addition = "_err"

        df = pd.read_csv(os.path.join(files_with_offset_time, file), header=None)
        df.iloc[:,3] = df.iloc[:,3] + t_zero[file]
        print(f"All timestamps ascending: {df.iloc[:,3].is_monotonic_increasing}")
        output_file = os.path.join(output_path, subgroup + part_addition + ".csv")
        df.to_csv(output_file, mode='a', header=False, index=False)
        files_done += 1
        print(f"Processed {file} with {len(df)} rows. Output to {output_file}. Files done: {files_done}.")
        
