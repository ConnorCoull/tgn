folder = "d:\\SWAT\\physical\\physical\\ByDay"
output_folder = "d:\\SWAT\\attack_times"

import os
import pandas as pd
import numpy as np

to_drop = ['FIT101', 'LIT101', 'MV101', 'P101', 'P102',
           'AIT201', 'AIT202', 'AIT203', 'FIT201', 'MV201', 'P201', 'P202',
           'P203', 'P204', 'P205', 'P206', 'DPIT301', 'FIT301', 'LIT301',
           'MV301', 'MV302', 'MV303', 'MV304', 'P301', 'P302', 'AIT401',
           'AIT402', 'FIT401', 'LIT401', 'P401', 'P402', 'P403', 'P404',
           'UV401', 'AIT501', 'AIT502', 'AIT503', 'AIT504', 'FIT501', 'FIT502',
           'FIT503', 'FIT504', 'P501', 'P502', 'PIT501', 'PIT502', 'PIT503',
           'FIT601', 'P601', 'P602', 'P603', ' MV101',
           ' AIT201', ' MV201', ' P201', ' P202', ' P204', ' MV303']
# For file in folder:
for file in os.listdir(folder):
    if file.endswith(".csv"):
        df = pd.read_csv(os.path.join(folder, file))
        # Process the dataframe (df) as needed
        df.drop(columns=to_drop, inplace=True)
        # is_attack = 1 if df['Normal/Attack'] is "Attack" else 0
        df["is_attck"] = df['Normal/Attack'].apply(lambda x: 1 if x == "Attack" else 0)
        df.drop(columns=['Normal/Attack'], inplace=True)
        df['Timestamp'] = pd.to_datetime(df[' Timestamp'])  # Ensure it's a datetime object
        # Convert to seconds since epoch
        df['timestep'] = df['Timestamp'].astype('int64') // 10**9
        df.drop(columns=[' Timestamp', 'Timestamp'], inplace=True)


        
        base_name = os.path.splitext(file)[0]
        new_name = f"{base_name}-attacks.csv"

        print(df.head())  # Example operation

        for col in df.columns:
            print(f"Column: {col}, Unique values: {df[col].nunique()}")

        #df.to_csv(os.path.join(output_folder, new_name), index=False)