import pandas as pd
import numpy as np
import struct
import os
import time
import argparse


def extract_last5_floats(val):
    # Case: no real values
    if isinstance(val, str) and val.startswith("Number of Elements"):
        return [0.0] * 5

    # Split on ';' and strip whitespace
    parts = [p.strip() for p in str(val).split(";") if p.strip()]
    floats = []
    for part in parts:
        # hex‑byte sequence?
        if part.startswith("0x"):
            # parse each byte, pack into bytes, unpack as little‑endian float
            try:
                byte_vals = [int(b, 16) for b in part.split()]
                b = bytes(byte_vals)
                floats.append(struct.unpack("<f", b)[0])
            except:
                # skip malformed
                continue
        else:
            # standalone decimal
            try:
                floats.append(float(part))
            except:
                continue

    # take last 5, pad left with zeros if needed
    last5 = floats[-5:]
    if len(last5) < 5:
        last5 = [0.0] * (5 - len(last5)) + last5
    return last5


# 'num', 'date', 'time', 'orig', 'type', 'i/f_name', 'i/f_dir', 'src', 'dst', 'proto', 'appi_name', 'proxy_src_ip', 'Modbus_Function_Code', 'Modbus_Function_Description', 'Modbus_Transaction_ID', 'SCADA_Tag', 'Modbus_Value', 'service', 's_port', 'Tag'
columns_to_remove = [3, 5, 6, 11, 15, 19]

ip_to_id = {
    "10.15.51.65": 0,
    "10.19.1.211": 1,
    "10.19.255.255": 2,
    "169.254.245.229": 3,
    "169.254.255.255": 4,
    "192.148.1.10": 5,
    "192.168.1.10": 6,
    "192.168.1.100": 7,
    "192.168.1.103": 8,
    "192.168.1.107": 9,
    "192.168.1.11": 10,
    "192.168.1.134": 11,
    "192.168.1.136": 12,
    "192.168.1.157": 13,
    "192.168.1.19": 14,
    "192.168.1.199": 15,
    "192.168.1.20": 16,
    "192.168.1.200": 17,
    "192.168.1.201": 18,
    "192.168.1.202": 19,
    "192.168.1.249": 20,
    "192.168.1.255": 21,
    "192.168.1.30": 22,
    "192.168.1.40": 23,
    "192.168.1.49": 24,
    "192.168.1.50": 25,
    "192.168.1.60": 26,
    "192.168.1.70": 27,
    "192.168.1.87": 28,
    "192.168.1.88": 29,
    "224.0.0.251": 30,
    "224.0.0.252": 31,
    "239.192.2.63": 32,
    "239.192.4.191": 33,
    "239.255.255.250": 34,
    "255.255.255.255": 35,
}

appi_to_id = {
    "CIP_func75": 0,
    "CIP_func79": 1,
    "CIP_func92": 2,
    "CIP_gen": 3,
    "CIP_read_tag_service": 4,
    "CIP_set_attributes_list": 5,
    "CIP_write_tag_service": 6,
    "Common Industrial Protocol": 7,
    "Common Industrial Protocol - RMW (Read/Modify/Write)": 8,
    "Common Industrial Protocol - execute PCCC": 9,
    "Common Industrial Protocol - get attribute all": 10,
    "Common Industrial Protocol - get attribute single": 11,
    "Common Industrial Protocol - multiple service packet": 12,
    "Common Industrial Protocol - read data fragmented": 13,
    "Common Industrial Protocol - success": 14,
    "Common Industrial Protocol - unconnected send-get attribute all": 15,
    "DCE-RPC Protocol": 16,
    "DHCP Protocol": 17,
    "DNP3 Protocol - initialize data to defaults": 18,
    "DNS Protocol": 19,
    "EtherNet/IP": 20,
    "Google Chrome": 21,
    "Kaspersky Lab-update": 22,
    "LLMNR Protocol": 23,
    "Multicast DNS Protocol (mDNS)": 24,
    "NetBIOS Datagram Service": 25,
    "NetBIOS Name Service": 26,
    "NetBIOS Session Service": 27,
    "OSIsoft PI": 28,
    "Remote Desktop Protocol": 29,
    "SIC Protocol": 30,
    "SSDP": 31,
    "Server Message Block (SMB)": 32,
    "Simple Object Access Protocol": 33,
    "Unknown Traffic": 34,
    "VNC": 35,
    "Web Browsing": 36,
}


def process(filename, true_label_start=0):

    df = pd.read_csv(filename)
    df.drop(df.columns[columns_to_remove], axis=1, inplace=True)
    df.dropna(inplace=True)

    # Merge the date an time columns into a ts column that measure seconds (going from 1Jan1970 00:00:00 to 0)
    # Try both date formats
    df["ts"] = (
        pd.to_datetime(df["date"] + " " + df["time"], format="mixed").astype("int64")
        // 10**9
    )

        

    df.drop(["date", "time"], axis=1, inplace=True)

    # subtract the value of the first timestamp from all timestamps to start at 0
    df["ts"] -= df["ts"].min()

    # If "Modbus_Function_Description" contains " - resposne", the is_response field should be 1
    df["is_response"] = (
        df["Modbus_Function_Description"].str.contains(" - response").astype(int)
    )
    df.drop("Modbus_Function_Description", axis=1, inplace=True)

    src_split = df["src"].str.split(".")
    dst_split = df["dst"].str.split(".")

    df["subnet1"] = (src_split.str[0] == dst_split.str[0]).astype(int)
    df["subnet2"] = (src_split.str[1] == dst_split.str[1]).astype(int)
    df["subnet3"] = (src_split.str[2] == dst_split.str[2]).astype(int)
    df["subnet4"] = (src_split.str[3] == dst_split.str[3]).astype(int)

    type_values = ["log", "control", "alert", "other"]
    df["type_is_log"] = (df["type"] == type_values[0]).astype(int)
    df["type_is_control"] = (df["type"] == type_values[1]).astype(int)
    df["type_is_alert"] = (df["type"] == type_values[2]).astype(int)
    df["type_is_other"] = (df["type"] == type_values[3]).astype(int)

    # proto is 1 if tcp, otherwise 0
    df["proto"] = (df["proto"] == "tcp").astype(int)

    df.drop(["type"], axis=1, inplace=True)

    df["src"] = df["src"].map(ip_to_id)
    df["dst"] = df["dst"].map(ip_to_id)

    df["Modbus_Transaction_ID"] = df["Modbus_Transaction_ID"].astype(int)

    last_request_ts = {}
    response_times = []

    for idx, row in df.iterrows():
        trans_id = row["Modbus_Transaction_ID"]
        ts = row["ts"]

        if row["is_response"] == 0:
            last_request_ts[trans_id] = ts
            response_times.append(0)  # No response time for request
        else:
            if trans_id in last_request_ts:
                response_time = ts - last_request_ts[trans_id]
                response_times.append(response_time)
            else:
                response_times.append(0)

    df["modbus_response_time"] = response_times

    # Convert 'Modbus_Transaction_ID', 'service' and 's_port' to binary strings of length 16
    df["service"] = df["service"].astype(int).apply(lambda x: f"{x:016b}")
    df["s_port"] = df["s_port"].astype(int).apply(lambda x: f"{x:016b}")
    df["Modbus_Transaction_ID"] = (
        df["Modbus_Transaction_ID"].astype(int).apply(lambda x: f"{x:016b}")
    )

    service_array = np.array([list(s) for s in df["service"]], dtype=int)
    s_port_array = np.array([list(s) for s in df["s_port"]], dtype=int)
    modbus_transaction_id_array = np.array(
        [list(s) for s in df["Modbus_Transaction_ID"]], dtype=int
    )
    df_service = pd.DataFrame(
        service_array, columns=[f"service_{i}" for i in range(16)]
    )
    df_s_port = pd.DataFrame(s_port_array, columns=[f"s_port_{i}" for i in range(16)])
    df_modbus_transaction_id = pd.DataFrame(
        modbus_transaction_id_array,
        columns=[f"Modbus_Transaction_ID_{i}" for i in range(16)],
    )

    df = pd.concat([df, df_modbus_transaction_id, df_service, df_s_port], axis=1)
    df.drop(["Modbus_Transaction_ID", "service", "s_port"], axis=1, inplace=True)

    # Convert 'appi_name' to binary strings of length 6
    df["appi_name"] = df["appi_name"].map(appi_to_id)
    df["appi_name"] = df["appi_name"].fillna(0).astype(int)
    df["appi_name"] = df["appi_name"].apply(lambda x: f"{x:06b}")
    df_appi_name_array = np.array([list(s) for s in df["appi_name"]], dtype=int)
    df_appi_name = pd.DataFrame(
        df_appi_name_array, columns=[f"appi_name_{i}" for i in range(6)]
    )
    df = pd.concat([df, df_appi_name], axis=1)
    df.drop("appi_name", axis=1, inplace=True)

    # Convert 'Modbus_Function_Code' to binary strings of length 8
    df["Modbus_Function_Code"] = df["Modbus_Function_Code"].fillna(0).astype(int)
    df["Modbus_Function_Code"] = df["Modbus_Function_Code"].apply(lambda x: f"{x:08b}")
    df_modbus_code_array = np.array(
        [list(s) for s in df["Modbus_Function_Code"]], dtype=int
    )
    df_modbus_code = pd.DataFrame(
        df_modbus_code_array, columns=[f"Modbus_Function_Code_{i}" for i in range(8)]
    )
    df = pd.concat([df, df_modbus_code], axis=1)
    df.drop("Modbus_Function_Code", axis=1, inplace=True)

    float_cols = df["Modbus_Value"].apply(extract_last5_floats).tolist()
    df_float = pd.DataFrame(float_cols, columns=[f"Modbus_Value_{i}" for i in range(5)])

    df = pd.concat([df, df_float], axis=1)
    df.drop("Modbus_Value", axis=1, inplace=True)

    # Change num to a sequential index starting from 0, then add true_label_start to new value
    df["num"] = range(true_label_start, true_label_start + len(df))
    new_true_label_start = df["num"].max() + 1

    # reshape the dataframe to the following format
    # num, src, dst, ts, is_response, subnet1, subnet2, subnet3, subnet4, type_is_log, type_is_control, type_is_alert, type_is_other, proto, appi_name_0, ..., appi_name_5, Mobus_Function_Code_0, ..., Modbus_Function_Code_7, Modbus_Transaction_ID_0, ..., Modbus_Transaction_ID_15, response_time, service_0, ..., service_15, s_port_0, ..., s_port_15, Modbus_Value_0, ..., Modbus_Value_4
    df = df[
        [
            "num",
            "src",
            "dst",
            "ts",
            "is_response",
            "modbus_response_time",
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

    return df, new_true_label_start


# Unique source IPs: 22
# {'192.168.1.19', '10.19.1.211', '192.168.1.202', '192.168.1.49', '192.168.1.134', '192.168.1.20', '192.168.1.157', '192.168.1.200', '192.168.1.60', '192.168.1.107', '192.168.1.100', '192.168.1.88', '192.168.1.40', '192.168.1.201', '169.254.245.229', '192.168.1.10', '10.15.51.65', '192.168.1.199', '192.168.1.30', '192.168.1.136', nan, '192.168.1.249'}
# Unique destination IPs: 27
# {'192.168.1.19', '192.168.1.103', '224.0.0.251', '239.192.4.191', '239.192.2.63', '239.255.255.250', '224.0.0.252', '192.168.1.49', '255.255.255.255', '192.168.1.157', '192.168.1.20', '192.168.1.200', '192.168.1.60', '192.168.1.100', '192.168.1.88', '192.168.1.50', '192.168.1.40', '192.168.1.201', '192.168.1.10', '192.168.1.11', '192.168.1.255', '192.168.1.30', '10.19.255.255', '169.254.255.255', '192.168.1.87', nan, '192.148.1.10'}
# Unique application names: 36
# {'SIC Protocol', 'Google Chrome', 'Multicast DNS Protocol (mDNS)', 'Kaspersky Lab-update', 'Common Industrial Protocol - execute PCCC', 'Web Browsing', 'Simple Object Access Protocol', 'Common Industrial Protocol - get attribute single', 'LLMNR Protocol', 'NetBIOS Datagram Service', 'CIP_gen', 'Common Industrial Protocol - read data fragmented', 'NetBIOS Name Service', 'CIP_func79', 'DNS Protocol', 'Common Industrial Protocol', 'EtherNet/IP', 'CIP_func75', 'CIP_func92', nan, 'Unknown Traffic', 'CIP_read_tag_service', 'Common Industrial Protocol - multiple service packet', 'Common Industrial Protocol - unconnected send-get attribute all', 'VNC', 'OSIsoft PI', 'DHCP Protocol', 'SSDP', 'NetBIOS Session Service', 'DCE-RPC Protocol', 'Remote Desktop Protocol', 'Server Message Block (SMB)', 'CIP_set_attributes_list', 'Common Industrial Protocol - success', 'Common Industrial Protocol - RMW (Read/Modify/Write)', 'Common Industrial Protocol - get attribute all'}
# Unique types: 4
# {'alert', 'control', 'loe', 'log'}
# Unique protocols: 3
# {'udp', nan, 'tcp'}

folder = "D:\\SWAT\\network"
folders = [f for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f))]
folders.remove("processed")
folders_done = 0
files_done = 0
true_label_start = 0

# folders.remove('processed')
print(folders)

for subfolder in folders[folders_done:]:
    print(f"Processing folder: {subfolder}")
    if subfolder == "2015-12-28":
        for file in sorted(os.listdir(os.path.join(folder, subfolder)))[files_done:]:
            if file.endswith(".csv"):
                print(
                    f"In case of failure - Files Done: {files_done}, True Label Start: {true_label_start}"
                )
                current_time = pd.Timestamp.now()
                date = file[0:4] + "-" + file[5:7] + "-" + file[8:10]
                df, true_label_start = process(
                    os.path.join(folder, subfolder, file), true_label_start
                )
                output_file = os.path.join(folder, "processed", f"{date}_processed.csv")
                with open(output_file, "a") as f:
                    df.to_csv(output_file, mode="a", header=False, index=False)
                time_taken = (pd.Timestamp.now() - current_time).total_seconds()
                print(
                    f"Processed {file} in {time_taken:.4f} seconds, true_label_start: {true_label_start}"
                )
                # I think my bug is caused by not waiting enough for the output file to no longer be in use
                # I will wait 5 seconds before processing the next file
                time.sleep(5)
                files_done += 1

    true_label_start = 0
    files_done = 0
    folders_done += 1

# parser = argparse.ArgumentParser('Interface for TGN data preprocessing')
# parser.add_argument('--d', type=str, help='Path to the input dataset file')
# parser.add_argument('--o', type=str, default='output.csv', help='Path to the output file')
# parser.add_argument('--ts', type=int, default=0, help='True label start value')
# args = parser.parse_args()

# file_path = args.d
# if not os.path.exists(file_path):
#     raise FileNotFoundError(f"The file {file_path} does not exist.")

# df, new_true_label_start = process(file_path, args.ts)

# with open(args.o, 'a') as f:
#     df.to_csv(args.o, mode='a', header=False, index=False)

# print(f"Processed {file_path} with true label start: {new_true_label_start}")
