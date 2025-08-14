import os
import pandas as df
import numpy as np

# paths
files_to_process_folder = "d:\\SWAT\\network\\processed\\by_file"
attacj_times_folder = "d:\\SWAT\\attack_times"

files_with_attacks = []

# Iterate through files in the folder
for file in os.listdir(files_to_process_folder):
    # normalise
    # remember to consider adding label means diff feat len, consider for data_preprocessing
    pass