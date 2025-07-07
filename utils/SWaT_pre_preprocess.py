"""
num    date    time    orig    type    i/f_name    i/f_dir    src    dst    proto    appi_name    
proxy_src_ip    Modbus_Function_Code    Modbus_Function_Description    Modbus_Transaction_ID    SCADA_Tag    
Modbus_Value    service    s_port    Tag    
"""
"""
Features
binary flag indicating request/response (taken from Modbus_Function_Description having "- Response") - 1
4 binary fields indicating if src and dst share same IP (i.e. 192.225.225.224 and 192.225.225.1 = [1, 1, 1, 0]) - 4
type one-hot encoded [log, control, alert, other] - 4
protocol one-hot encoded [tcp, udp, other] - 3
appi_name binary encoded with value representing appi_name (0 - CIP_read_tag_service, 1 - NetBIOS Datagram Service) - 5
SCADA_Tag - one-hot encoded - 6 (To be deleted)
Modbus_Function_Code - binary encoding representing the function code - 8
Modbus_Transaction_ID - Binary encoding representing the transaction ID - 16
if response, time in s from when request with same Modbus_Transaction_ID was sent - 1
service - binary encoded representing the port - 16
s_port - binary encoded representing the port - 16
Modbus_Value - turned into 38 features, each feature representing a bit in the value - 5
total = 1 + 4 + 4 + 3 + 5 + 6 + 8 + 16 + 1 + 16 + 16 + 5 = 85 features
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import struct
import re
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import os
import sys


def datetime_to_s(date_str, time_str):
    datetime_str = f"{date_str} {time_str}"    
    dt_format = "%d%b%Y %H:%M:%S"
    dt = datetime.strptime(datetime_str, dt_format)    
    s = int(dt.timestamp())
    return s


def convert_to_binary_list(value, length):
    """
    Converts an integer value to a binary list of a specified length.
    """
    binary_str = format(value, f'0{length}b')
    return [int(bit) for bit in binary_str]


def convert_hex_to_decimal(hex_string):
    result = []
    for line in hex_string.strip().splitlines():
        segments = [seg.strip() for seg in line.strip().split(';') if seg.strip()]
        for seg in segments:
            hex_bytes = re.findall(r'0x[0-9a-fA-F]{2}', seg)
            if len(hex_bytes) == 4:
                byte_values = bytes(int(h, 16) for h in hex_bytes)
                float_val = struct.unpack('<f', byte_values)[0]
                result.append(float_val)
            elif re.fullmatch(r'\d+', seg):
                result.append(int(seg))
            else:
                continue
    return result[-5:]


def preprocess(folder_name):
    first_time_s = None
    appi_name_dict = {}
    modbus_transaction_id_ts_dict = {}
    ip_to_id = {}
    counter = 1
    for filename in os.listdir(folder_name):
        print(f"Processing file: {filename}")
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_name, filename)
            with open(file_path) as f:
                s = next(f)
                for _, line in enumerate(f, start=1):
                    e = line.strip().split(',')

                    # if any of the fields are '', skip the line
                    if any(field == '' for field in e):
                        continue

                    # Get Source and Destination IPs
                    src, dst = str(e[7]), str(e[8])

                    # Convert date and time to a timestamp in milliseconds after the first event
                    ts = datetime_to_s(e[1], e[2])

                    if first_time_s is None:
                        first_time_s = ts
                    rel_ts = ts - first_time_s

                    # Checks if " - Resposne" appears in Modbus_Function_Description
                    is_response = 1 if " - Response" in e[13] else 0

                    is_same_mask = [0, 0, 0, 0]
                    src_split = src.split('.')
                    dst_split = dst.split('.')
                    # Check if src and dst share the same IP address - remove?
                    if src_split[0] == dst_split[0]:
                        is_same_mask[0] = 1
                    if src_split[1] == dst_split[1]:
                        is_same_mask[1] = 1
                    if src_split[2] == dst_split[2]:
                        is_same_mask[2] = 1
                    if src_split[3] == dst_split[3]:
                        is_same_mask[3] = 1

                    # Type one-hot encoded
                    type_one_hot = [0, 0, 0, 0]
                    if e[4] == 'log':
                        type_one_hot[0] = 1
                    elif e[4] == 'control':
                        type_one_hot[1] = 1
                    elif e[4] == 'alert':
                        type_one_hot[2] = 1
                    else:
                        type_one_hot[3] = 1

                    # Protocol one-hot encoded
                    proto_one_hot = [0, 0, 0]
                    if e[10] == 'tcp':
                        proto_one_hot[0] = 1
                    elif e[10] == 'udp':
                        proto_one_hot[1] = 1
                    else:
                        proto_one_hot[2] = 1

                    # Appi_name binary encoded
                    appi_name = e[10]
                    if appi_name not in appi_name_dict:
                        appi_name_dict[appi_name] = len(appi_name_dict)
                    # convert appi_name to binary encoding
                    appi_name_bin = convert_to_binary_list(appi_name_dict[appi_name], 5)

                    # SCADA_Tag one-hot encoded - DROP
                    scada_tag_one_hot = [0] * 6
                    scada_tag = e[15]
                    if scada_tag == 'HMI_LIT101':
                        scada_tag_one_hot[0] = 1
                    elif scada_tag == 'HMI_FIT201':
                        scada_tag_one_hot[1] = 1
                    elif scada_tag == 'HMI_AIT202':
                        scada_tag_one_hot[2] = 1
                    elif scada_tag == 'HMI_LIT301':
                        scada_tag_one_hot[3] = 1
                    elif scada_tag == 'HMI_LIT401':
                        scada_tag_one_hot[4] = 1
                    else:
                        scada_tag_one_hot[5] = 1

                    # Modbus_Function_Code binary encoding
                    if e[12] == '':
                        modbus_function_code = 0
                    modbus_function_code = int(e[12])
                    modbus_function_code_bin = convert_to_binary_list(modbus_function_code, 8)

                    # Modbus_Transaction_ID binary encoding - remove ID (perhaps, keep ts)
                    if e[13] == '':
                        modbus_transaction_id = 0
                    else:
                        modbus_transaction_id = int(e[14])
                    modbus_transaction_id_bin = convert_to_binary_list(modbus_transaction_id, 16)

                    # If response, time in s from when request with same Modbus_Transaction_ID was sent
                    if is_response:
                        if modbus_transaction_id in modbus_transaction_id_ts_dict:
                            request_ts = modbus_transaction_id_ts_dict[modbus_transaction_id]
                            response_time_s = ts - request_ts
                        else:
                            response_time_s = 0
                    else:
                        response_time_s = 0
                        # Store the request timestamp for this transaction ID
                        modbus_transaction_id_ts_dict[modbus_transaction_id] = ts

                    # Service binary encoding
                    service = e[17]
                    service_bin = convert_to_binary_list(int(service), 16)

                    # s_port binary encoding
                    s_port = e[18]
                    s_port_bin = convert_to_binary_list(int(s_port), 16)

                    # Modbus_Value turned into 38 features, each feature representing a bit in the value
                    modbus_value = e[16]
                    if modbus_value == "Number of Elements: 1" or modbus_value == '':
                        list_of_modbus_value = [0, 0, 0, 0, 0]
                    else:
                        list_of_modbus_value = convert_hex_to_decimal(modbus_value)

                    # Write to file in format idx, src, dst, ts, features
                    features = (is_response, *is_same_mask, *type_one_hot, *proto_one_hot,
                                *appi_name_bin, *scada_tag_one_hot, *modbus_function_code_bin,
                                *modbus_transaction_id_bin, response_time_s, *service_bin,
                                *s_port_bin, *list_of_modbus_value)
                    feature_str = ','.join(map(str, features))
                    if src not in ip_to_id:
                        ip_to_id[src] = len(ip_to_id)
                    if dst not in ip_to_id:
                        ip_to_id[dst] = len(ip_to_id)
                    src = ip_to_id[src]
                    dst = ip_to_id[dst]

                    with open(f"data/SWaT_processed.csv", 'a') as out_f:
                        out_f.write(f"{counter},{src},{dst},{rel_ts},{feature_str}\n")
                    
                    counter += 1


def normalise():
    df = pd.read_csv(f"data/SWaT_processed.csv", header=None)
    scaler = MinMaxScaler()
    df.iloc[:,4:] = scaler.fit_transform(df.iloc[:,4:])
    return df


def run(folder_name, to_normalise):
    print("Starting Preprocessing")
    preprocess(folder_name)
    if to_normalise:
        df = normalise()
        #remove .csv from filename
        filename = filename[:-4]
        df.to_csv(f"data/{filename}_normalised.csv", index=False, header=False)
        # remove the original file
        os.remove(f"data/{filename}.csv")


parser = argparse.ArgumentParser('Interface for TGN data preprocessing')
parser.add_argument('data', type=str, help='Path to the input dataset file')
parser.add_argument('--normalise', action='store_true', help='Whether or not to apply normalisation (MinMax) after data is processed')

args = parser.parse_args()

folder_path = sys.argv[1]
if not os.path.isdir(args.data):
    print(f"The path {folder_path} is not a valid directory.")
    sys.exit(1)

run(args.data, args.normalise)

print("PROCESS COMPLETE")
