import pandas as pd
import numpy as np
import os

folder = "D:\\SWAT\\network\\processed\\by_file"
file_prefix = ["2015-12-23_003316"]#, "2015-12-24", "2015-12-25"]

output_file = "D:\\SWAT\\network\\processed\\samples\\SWaT_500k_processed_random_1.csv"

sample_rate = 0.075
total_samples = 0
unique_sampled_nodes = set()

np.random.seed(2020)

# Get list of files that match our criteria
matching_files = []
for file in os.listdir(folder):
    if file.endswith(".csv") and any(file.startswith(prefix) for prefix in file_prefix):
        matching_files.append(file)
#print(f"Found {len(matching_files)} files to process.")
#print("Files to process:", matching_files)

# Sort files to ensure consistent processing order
matching_files.sort()

# Remove output file if it exists to start fresh
if os.path.exists(output_file):
    os.remove(output_file)

for file in matching_files:
    file_path = os.path.join(folder, file)
    df = pd.read_csv(file_path)
    #print(df.head())
    
    #remove nan
    df = df.dropna()
    
    # Calculate number of rows to sample (at least 1 row)
    n_rows = len(df)
    print(f"Processing file: {file} with {n_rows} rows.")
    n_sample = max(1, int(n_rows * sample_rate))
    list_unique_src = df["src"].unique()
    list_unique_dst = df["dst"].unique()
    unique_set = set(list_unique_src).union(set(list_unique_dst))
    
    # Generate evenly spaced indices to maintain temporal order
    # This ensures we sample across the entire time range of the file
    indices = np.sort(np.random.choice(n_rows, size=n_sample, replace=False))
    
    # Sample the dataframe using the calculated indices
    sampled_df = df.iloc[indices].copy()
    
    # Append to output file
    if not os.path.exists(output_file):
        sampled_df.to_csv(output_file, index=False)
    else:
        sampled_df.to_csv(output_file, mode='a', header=False, index=False)
    total_samples += n_sample
    unique_sampled_nodes.update(unique_set)
    print(f"Processed {file} and sample now has {total_samples} rows. Unique nodes: {list(unique_sampled_nodes)}")