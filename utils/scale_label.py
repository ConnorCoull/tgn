import pandas as pd
import numpy as np
import os

folder = "D:\\SWAT\\network\\processed\\by_file"
days_to_process = ["2016-01-01", "2016-01-02", "2016-01-03"]
files_and_ts = "D:\\SWAT\\network\\processed\\files_and_ts.txt"
warning_files = "D:\\SWAT\\network\\processed\\warning_files.txt"

max_vals = np.array([1000.0, 800.0, 250.0, 36.0, 1014.724])

def normalise_last_5_columns(df, max_vals, clip=False):
    df_scaled = df.copy()
    
    cols_to_normalise = df.columns[-5:]
    
    df_scaled[cols_to_normalise] = df_scaled[cols_to_normalise].astype(float).values / max_vals

    if clip:
        df_scaled[cols_to_normalise] = np.clip(df_scaled[cols_to_normalise], 0, 1)

    return df_scaled

def insert_label_column(df, label_value=0, insert_index=4):
    df_with_label = df.copy()
    df_with_label.insert(insert_index, 'label', label_value)
    return df_with_label

def adjust_time_column(df, time_col_index=5):
    df_adj = df.copy()
    time_col_name = df_adj.columns[time_col_index]
    min_time = df_adj.iloc[0, time_col_index]
    df_adj[time_col_name] = df_adj[time_col_name] - min_time
    return df_adj

def reset_first_column_index(df):
    df_updated = df.copy()
    first_col = df_updated.columns[0]
    df_updated[first_col] = np.arange(1, len(df_updated) + 1)
    return df_updated


#file = "SWaT_500k_processed_random_1.csv"

# before [4, 6, 26, 172000, ..., 1000.0, 800.0, 250.0, 36.0, 1014.724]
# after [1, 6, 26, 0, 0, ..., 1, 1, 1, 1, 1]

total_done = 0

for file in sorted(os.listdir(folder))[total_done:]:
    if file.endswith(".csv") and any(day in file for day in days_to_process):
        df = pd.read_csv(os.path.join(folder, file))
        print(f"Processing file: {file}")

        print(df.head())

        # check if file has a 'label' column
        if 'label' not in df.columns:
            with open(warning_files, 'a') as wf:
                wf.write(f"File {file} does not have a 'label' column.\n")

        with open(files_and_ts, 'a') as fts:
            fts.write(f"{file}, {df['ts'].iloc[0]}, {df['num'].iloc[0]}\n")


        df_scaled = normalise_last_5_columns(df, max_vals)
        #df_with_label = insert_label_column(df_scaled, label_value=0, insert_index=4)
        #print(df_with_label.iloc[:, :5].head())
        df_adjusted_time = adjust_time_column(df_scaled, time_col_index=3)
        df_final = reset_first_column_index(df_adjusted_time)

        # print head of first 5 columns and last 5 columns
        print(df_final.iloc[:, :5].head())
        print(df_final.iloc[:, -5:].head())

        # write to original file
        output_file = os.path.join(folder, file)
        df_final.to_csv(output_file, index=False, header=False)
        print(f"Total done: {total_done } files.")
        total_done += 1