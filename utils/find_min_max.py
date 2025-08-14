folder = "D:\\SWAT\\network\\processed\\by_file\\"
extra_file = "D:\\SWAT\\network\\processed\\extra_file.csv"

import pandas as pd
import numpy as np
import os



# for file in os.listdir(folder):
#     if file.endswith(".csv"):
#         df = pd.read_csv(os.path.join(folder, file), header=None)
        
#         # get unique values of columns 1 & 2
#         unique_src = df[1].unique()
#         unique_dst = df[2].unique()

#         print(f"File: {file}")
#         print(f"Unique source nodes: {len(unique_src)}")
#         print(f"Unique destination nodes: {len(unique_dst)}")

for file in os.listdir(folder):
    if file.endswith(".csv"):
        df = pd.read_csv(os.path.join(folder, file), header=None)
        
        for col in df.columns:
            min = df[col].min()
            max = df[col].max()
            print(f"File: {file}, Column: {col}, Min: {min}, Max: {max}")