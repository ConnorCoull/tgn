days_to_label = [
    "2015-12-28",
    "2015-12-29",
    "2015-12-30",
    "2015-12-31",
    "2016-01-01",
    "2016-01-02",
    "2016-01-03",
]

days_with_attacks = [
    "2015-12-28",
    "2015-12-29",
    "2015-12-30",
    "2015-12-31",
    "2016-01-01",
    "2016-01-02"
]

data_folder = "D:\\SWAT\\network\\processed\\by_file"
attack_times_folder = "D:\\SWAT\\attack_times"
files_for_file_first_ts = "D:\\SWAT\\network\\processed\\by_file_first_ts.csv"
timestamps = "D:\\SWAT\\network\\processed\\by_file_first_ts.csv"

# files_to_do = filenames = [
#     "2016-01-02_121729_121.log.part13_sorted_processed.csv",
#     "2016-01-02_121729_121.log.part14_sorted_processed.csv",
#     "2016-01-02_172943_122.log.part01_sorted_processed.csv",
#     "2016-01-02_172943_122.log.part02_sorted_processed.csv",
#     "2016-01-02_172943_122.log.part03_sorted_processed.csv",
#     "2016-01-02_172943_122.log.part04_sorted_processed.csv",
#     "2016-01-02_172943_122.log.part05_sorted_processed.csv",
#     "2016-01-02_172943_122.log.part06_sorted_processed.csv",
#     "2016-01-02_172943_122.log.part07_sorted_processed.csv",
#     "2016-01-02_172943_122.log.part08_sorted_processed.csv",
#     "2016-01-02_172943_122.log.part09_sorted_processed.csv",
#     "2016-01-02_172943_122.log.part10_sorted_processed.csv",
#     "2016-01-02_172943_122.log.part11_sorted_processed.csv",
#     "2016-01-02_172943_122.log.part12_sorted_processed.csv",
#     "2016-01-02_172943_122.log.part13_sorted_processed.csv",
#     "2016-01-02_172943_122.log.part14_sorted_processed.csv",
#     "2016-01-02_223411_123.log.part01_sorted_processed.csv",
#     "2016-01-02_223411_123.log.part02_sorted_processed.csv",
#     "2016-01-02_223411_123.log.part03_sorted_processed.csv",
#     "2016-01-02_223411_123.log.part04_sorted_processed.csv",
#     "2016-01-02_223411_123.log.part05_sorted_processed.csv",
#     "2016-01-02_223411_123.log.part06_sorted_processed.csv",
#     "2016-01-02_223411_123.log.part07_sorted_processed.csv",
#     "2016-01-02_223411_123.log.part08_sorted_processed.csv",
#     "2016-01-02_223411_123.log.part09_sorted_processed.csv",
#     "2016-01-02_223411_123.log.part10_sorted_processed.csv",
#     "2016-01-02_223411_123.log.part11_sorted_processed.csv",
#     "2016-01-02_223411_123.log.part12_sorted_processed.csv",
#     "2016-01-02_223411_123.log.part13_sorted_processed.csv",
#     "2016-01-02_223411_123.log.part14_sorted_processed.csv",
#     "2016-01-03_033842_124.log.part01_sorted_processed.csv",
#     "2016-01-03_033842_124.log.part02_sorted_processed.csv",
#     "2016-01-03_033842_124.log.part03_sorted_processed.csv",
#     "2016-01-03_033842_124.log.part04_sorted_processed.csv",
#     "2016-01-03_033842_124.log.part05_sorted_processed.csv",
#     "2016-01-03_033842_124.log.part06_sorted_processed.csv",
#     "2016-01-03_033842_124.log.part07_sorted_processed.csv",
#     "2016-01-03_033842_124.log.part08_sorted_processed.csv",
#     "2016-01-03_033842_124.log.part09_sorted_processed.csv",
#     "2016-01-03_033842_124.log.part10_sorted_processed.csv",
#     "2016-01-03_033842_124.log.part11_sorted_processed.csv",
#     "2016-01-03_033842_124.log.part12_sorted_processed.csv",
#     "2016-01-03_033842_124.log.part13_sorted_processed.csv",
#     "2016-01-03_033842_124.log.part14_sorted_processed.csv"
# ]

# time_add_per_file = {
#     "2016-01-02_121729_121.log.part13_sorted_processed.csv": 1451733919.0,
#     "2016-01-02_121729_121.log.part14_sorted_processed.csv": 1451735285.0,
#     "2016-01-02_172943_122.log.part01_sorted_processed.csv": 1451736484.0,
#     "2016-01-02_172943_122.log.part02_sorted_processed.csv": 1451737848.0,
#     "2016-01-02_172943_122.log.part03_sorted_processed.csv": 1451739212.0,
#     "2016-01-02_172943_122.log.part04_sorted_processed.csv": 1451740577.0,
#     "2016-01-02_172943_122.log.part05_sorted_processed.csv": 1451741943.0,
#     "2016-01-02_172943_122.log.part06_sorted_processed.csv": 1451743308.0,
#     "2016-01-02_172943_122.log.part07_sorted_processed.csv": 1451744672.0,
#     "2016-01-02_172943_122.log.part08_sorted_processed.csv": 1451746036.0,
#     "2016-01-02_172943_122.log.part09_sorted_processed.csv": 1451747399.0,
#     "2016-01-02_172943_122.log.part10_sorted_processed.csv": 1451748763.0,
#     "2016-01-02_172943_122.log.part11_sorted_processed.csv": 1451750106.0,
#     "2016-01-02_172943_122.log.part12_sorted_processed.csv": 1451751422.0,
#     "2016-01-02_172943_122.log.part13_sorted_processed.csv": 1451752738.0,
#     "2016-01-02_172943_122.log.part14_sorted_processed.csv": 1451754053.0,
#     "2016-01-02_223411_123.log.part01_sorted_processed.csv": 1451755216.0,
#     "2016-01-02_223411_123.log.part02_sorted_processed.csv": 1451756533.0,
#     "2016-01-02_223411_123.log.part03_sorted_processed.csv": 1451757849.0,
#     "2016-01-02_223411_123.log.part04_sorted_processed.csv": 1451759166.0,
#     "2016-01-02_223411_123.log.part05_sorted_processed.csv": 1451760482.0,
#     "2016-01-02_223411_123.log.part06_sorted_processed.csv": 1451761798.0,
#     "2016-01-02_223411_123.log.part07_sorted_processed.csv": 1451763113.0,
#     "2016-01-02_223411_123.log.part08_sorted_processed.csv": 1451764429.0,
#     "2016-01-02_223411_123.log.part09_sorted_processed.csv": 1451765743.0,
#     "2016-01-02_223411_123.log.part10_sorted_processed.csv": 1451767058.0,
#     "2016-01-02_223411_123.log.part11_sorted_processed.csv": 1451768372.0,
#     "2016-01-02_223411_123.log.part12_sorted_processed.csv": 1451769688.0,
#     "2016-01-02_223411_123.log.part13_sorted_processed.csv": 1451771001.0,
#     "2016-01-02_223411_123.log.part14_sorted_processed.csv": 1451772316.0,    
#     "2016-01-03_033842_124.log.part01_sorted_processed.csv": 1451773481.0,
#     "2016-01-03_033842_124.log.part02_sorted_processed.csv": 1451774796.0,
#     "2016-01-03_033842_124.log.part03_sorted_processed.csv": 1451776111.0,
#     "2016-01-03_033842_124.log.part04_sorted_processed.csv": 1451777427.0,
#     "2016-01-03_033842_124.log.part05_sorted_processed.csv": 1451778743.0,
#     "2016-01-03_033842_124.log.part06_sorted_processed.csv": 1451780059.0,
#     "2016-01-03_033842_124.log.part07_sorted_processed.csv": 1451781374.0,
#     "2016-01-03_033842_124.log.part08_sorted_processed.csv": 1451782689.0,
#     "2016-01-03_033842_124.log.part09_sorted_processed.csv": 1451784004.0,
#     "2016-01-03_033842_124.log.part10_sorted_processed.csv": 1451785319.0,
#     "2016-01-03_033842_124.log.part11_sorted_processed.csv": 1451786636.0,
#     "2016-01-03_033842_124.log.part12_sorted_processed.csv": 1451787951.0,
#     "2016-01-03_033842_124.log.part13_sorted_processed.csv": 1451789269.0,
#     "2016-01-03_033842_124.log.part14_sorted_processed.csv": 1451790585.0
# }


import pandas as pd
import numpy as np
import os

# Iterate through files in the folder


attacks = pd.DataFrame()
for day in days_with_attacks:
    attack_times_path = os.path.join(attack_times_folder, f"{day}-attacks.csv")
    day_attacks = pd.read_csv(attack_times_path)
    attacks = pd.concat([attacks, day_attacks], ignore_index=True)

ts_to_attack = dict(zip(attacks["timestep"], attacks["is_attck"]))




files_done = 0

for day in days_to_label:
    for file in sorted(os.listdir(data_folder))[files_done:]:
        print(f"Starting: {file}, files_done: {files_done}")
        file_path = os.path.join(data_folder, file)
        df = pd.read_csv(file_path)
        df["label"] = df["ts"].map(ts_to_attack).fillna(0).astype(int)

        s = pd.to_datetime(df["ts"].iloc[0], unit='s')
        with open(files_for_file_first_ts, "a") as f:
            f.write(f"{file},{s}\n")
            # move this to add third column which is True if any label = 1 for a file
            f.write(f"{file},{s},{df['label'].max()==1}\n")

        # Number of row with label 1
        num_attacks = df["label"].sum()

        df = df[
            [
                "num",
                "src",
                "dst",
                "ts",
                "label",
                "is_response",
                "subnet1",
                "subnet2",
                "subnet3",
                "subnet4",
                "type_is_log",
                "type_is_control",
                "type_is_alert",
                "type_is_other",
                "proto",
                "appi_name_0",
                "appi_name_1",
                "appi_name_2",
                "appi_name_3",
                "appi_name_4",
                "appi_name_5",
                "Modbus_Function_Code_0",
                "Modbus_Function_Code_1",
                "Modbus_Function_Code_2",
                "Modbus_Function_Code_3",
                "Modbus_Function_Code_4",
                "Modbus_Function_Code_5",
                "Modbus_Function_Code_6",
                "Modbus_Function_Code_7",
                "Modbus_Transaction_ID_0",
                "Modbus_Transaction_ID_1",
                "Modbus_Transaction_ID_2",
                "Modbus_Transaction_ID_3",
                "Modbus_Transaction_ID_4",
                "Modbus_Transaction_ID_5",
                "Modbus_Transaction_ID_6",
                "Modbus_Transaction_ID_7",
                "Modbus_Transaction_ID_8",
                "Modbus_Transaction_ID_9",
                "Modbus_Transaction_ID_10",
                "Modbus_Transaction_ID_11",
                "Modbus_Transaction_ID_12",
                "Modbus_Transaction_ID_13",
                "Modbus_Transaction_ID_14",
                "Modbus_Transaction_ID_15",
                "service_0",
                "service_1",
                "service_2",
                "service_3",
                "service_4",
                "service_5",
                "service_6",
                "service_7",
                "service_8",
                "service_9",
                "service_10",
                "service_11",
                "service_12",
                "service_13",
                "service_14",
                "service_15",
                "s_port_0",
                "s_port_1",
                "s_port_2",
                "s_port_3",
                "s_port_4",
                "s_port_5",
                "s_port_6",
                "s_port_7",
                "s_port_8",
                "s_port_9",
                "s_port_10",
                "s_port_11",
                "s_port_12",
                "s_port_13",
                "s_port_14",
                "s_port_15",
                "Modbus_Value_0",
                "Modbus_Value_1",
                "Modbus_Value_2",
                "Modbus_Value_3",
                "Modbus_Value_4",
            ]
        ]

        print(df.head())

        # Save the updated dataframe to original file
        df.to_csv(file_path, index=False)
        files_done += 1
        print(f"Processed file: {file_path} with {len(df)} rows and {num_attacks} attack rows.")

# for day in files_to_do:
#     file_path = os.path.join(data_folder, day)
#     if os.path.exists(file_path):
#         print(f"Starting: {day}, files_done: {files_done}")
#         df = pd.read_csv(file_path, header=None)
#         df.columns = [
#                     "num",
#                     "src",
#                     "dst",
#                     "ts",
#                     "is_response",
#                     "subnet1",
#                     "subnet2",
#                     "subnet3",
#                     "subnet4",
#                     "type_is_log",
#                     "type_is_control",
#                     "type_is_alert",
#                     "type_is_other",
#                     "proto",
#                     "appi_name_0",
#                     "appi_name_1",
#                     "appi_name_2",
#                     "appi_name_3",
#                     "appi_name_4",
#                     "appi_name_5",
#                     "Modbus_Function_Code_0",
#                     "Modbus_Function_Code_1",
#                     "Modbus_Function_Code_2",
#                     "Modbus_Function_Code_3",
#                     "Modbus_Function_Code_4",
#                     "Modbus_Function_Code_5",
#                     "Modbus_Function_Code_6",
#                     "Modbus_Function_Code_7",
#                     "Modbus_Transaction_ID_0",
#                     "Modbus_Transaction_ID_1",
#                     "Modbus_Transaction_ID_2",
#                     "Modbus_Transaction_ID_3",
#                     "Modbus_Transaction_ID_4",
#                     "Modbus_Transaction_ID_5",
#                     "Modbus_Transaction_ID_6",
#                     "Modbus_Transaction_ID_7",
#                     "Modbus_Transaction_ID_8",
#                     "Modbus_Transaction_ID_9",
#                     "Modbus_Transaction_ID_10",
#                     "Modbus_Transaction_ID_11",
#                     "Modbus_Transaction_ID_12",
#                     "Modbus_Transaction_ID_13",
#                     "Modbus_Transaction_ID_14",
#                     "Modbus_Transaction_ID_15",
#                     "service_0",
#                     "service_1",
#                     "service_2",
#                     "service_3",
#                     "service_4",
#                     "service_5",
#                     "service_6",
#                     "service_7",
#                     "service_8",
#                     "service_9",
#                     "service_10",
#                     "service_11",
#                     "service_12",
#                     "service_13",
#                     "service_14",
#                     "service_15",
#                     "s_port_0",
#                     "s_port_1",
#                     "s_port_2",
#                     "s_port_3",
#                     "s_port_4",
#                     "s_port_5",
#                     "s_port_6",
#                     "s_port_7",
#                     "s_port_8",
#                     "s_port_9",
#                     "s_port_10",
#                     "s_port_11",
#                     "s_port_12",
#                     "s_port_13",
#                     "s_port_14",
#                     "s_port_15",
#                     "Modbus_Value_0",
#                     "Modbus_Value_1",
#                     "Modbus_Value_2",
#                     "Modbus_Value_3",
#                     "Modbus_Value_4",
#                 ]

        
#         df["temp_ts"] = df["ts"].astype(float) + time_add_per_file.get(day)
#         df["label"] = df["temp_ts"].map(ts_to_attack).fillna(0).astype(int)

#         s = pd.to_datetime(df["temp_ts"].iloc[0], unit='s')
#         df.drop(columns=["temp_ts"], inplace=True)


#         with open(files_for_file_first_ts, "a") as f:
#             f.write(f"{day},{s}\n")
#             # move this to add third column which is True if any label = 1 for a file
#             f.write(f"{day},{s},{df['label'].max()==1}\n")

#         # Number of row with label 1
#         num_attacks = df["label"].sum()

#         df = df[
#                 [
#                     "num",
#                     "src",
#                     "dst",
#                     "ts",
#                     "label",
#                     "is_response",
#                     "subnet1",
#                     "subnet2",
#                     "subnet3",
#                     "subnet4",
#                     "type_is_log",
#                     "type_is_control",
#                     "type_is_alert",
#                     "type_is_other",
#                     "proto",
#                     "appi_name_0",
#                     "appi_name_1",
#                     "appi_name_2",
#                     "appi_name_3",
#                     "appi_name_4",
#                     "appi_name_5",
#                     "Modbus_Function_Code_0",
#                     "Modbus_Function_Code_1",
#                     "Modbus_Function_Code_2",
#                     "Modbus_Function_Code_3",
#                     "Modbus_Function_Code_4",
#                     "Modbus_Function_Code_5",
#                     "Modbus_Function_Code_6",
#                     "Modbus_Function_Code_7",
#                     "Modbus_Transaction_ID_0",
#                     "Modbus_Transaction_ID_1",
#                     "Modbus_Transaction_ID_2",
#                     "Modbus_Transaction_ID_3",
#                     "Modbus_Transaction_ID_4",
#                     "Modbus_Transaction_ID_5",
#                     "Modbus_Transaction_ID_6",
#                     "Modbus_Transaction_ID_7",
#                     "Modbus_Transaction_ID_8",
#                     "Modbus_Transaction_ID_9",
#                     "Modbus_Transaction_ID_10",
#                     "Modbus_Transaction_ID_11",
#                     "Modbus_Transaction_ID_12",
#                     "Modbus_Transaction_ID_13",
#                     "Modbus_Transaction_ID_14",
#                     "Modbus_Transaction_ID_15",
#                     "service_0",
#                     "service_1",
#                     "service_2",
#                     "service_3",
#                     "service_4",
#                     "service_5",
#                     "service_6",
#                     "service_7",
#                     "service_8",
#                     "service_9",
#                     "service_10",
#                     "service_11",
#                     "service_12",
#                     "service_13",
#                     "service_14",
#                     "service_15",
#                     "s_port_0",
#                     "s_port_1",
#                     "s_port_2",
#                     "s_port_3",
#                     "s_port_4",
#                     "s_port_5",
#                     "s_port_6",
#                     "s_port_7",
#                     "s_port_8",
#                     "s_port_9",
#                     "s_port_10",
#                     "s_port_11",
#                     "s_port_12",
#                     "s_port_13",
#                     "s_port_14",
#                     "s_port_15",
#                     "Modbus_Value_0",
#                     "Modbus_Value_1",
#                     "Modbus_Value_2",
#                     "Modbus_Value_3",
#                     "Modbus_Value_4",
#                 ]
#             ]
#         # Save the updated dataframe to original file
#         df.to_csv(file_path, header=False, index=False)
#         files_done += 1
#         print(f"Processed file: {file_path} with {len(df)} rows and {num_attacks} attack rows.")