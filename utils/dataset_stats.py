"""
sAddress	rAddress	sMACs	rMACs	sIPs	rIPs	protocol	
startDate	endDate	start	end	startOffset	endOffset	duration	
sPackets	rPackets	sBytesSum	rBytesSum	sBytesMax	rBytesMax	
sBytesMin	rBytesMin	sBytesAvg	rBytesAvg	sLoad	rLoad	
sPayloadSum	rPayloadSum	sPayloadMax	rPayloadMax	sPayloadMin	rPayloadMin	
sPayloadAvg	rPayloadAvg	sInterPacketAvg	rInterPacketAvg	sttl	rttl	
sAckRate	rAckRate	sUrgRate	rUrgRate	sFinRate	rFinRate	
sPshRate	rPshRate	sSynRate	rSynRate	sRstRate	rRstRate	
sWinTCP	rWinTCP	sFragmentRate	rFragmentRate	sAckDelayMax	
rAckDelayMax	sAckDelayMin	rAckDelayMin	sAckDelayAvg	
rAckDelayAvg	IT_B_Label	IT_M_Label	NST_B_Label	NST_M_Label


MVP - Aim is currently just to have binary "is attack" label, 
as a result only IT_B_Label is used as ground truth.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse


def run(data_path):
    unique_src_ips = set()
    unique_dst_ips = set()
    unique_dual_ips = set()
    total_ips = set()
    unique_ip_1s = set()
    unique_ip_2s = set()
    unique_ip_3s = set()
    unique_ip_4s = set()

    protocols = set()

    unique_src_macs = set()
    unique_dst_macs = set()
    unique_dual_macs = set()
    total_macs = set()

    unique_sbytes_max = set()
    unique_rbytes_max = set()
    unique_sbyte_rbyte_pairs = set()

    mismatched_ip_pairs = set()

    total_entries = 0
    attack_entries = 0


    with open(data_path) as f:
        next(f)  # skip headers
        for _, line in enumerate(f):
            e = line.strip().split(',')
            #TODO: src and dst are IP addresses, convert them to IPv4Address
            src = str(e[0])
            dst = str(e[1])
            src_mac = str(e[2])
            dst_mac = str(e[3])
            sip = str(e[4])
            rip = str(e[5])
            protocol = str(e[6])

            ip_sections = src.split('.')
            if len(ip_sections) == 4:
                unique_ip_1s.add(ip_sections[0])
                unique_ip_2s.add(ip_sections[1])
                unique_ip_3s.add(ip_sections[2])
                unique_ip_4s.add(ip_sections[3])

            # if sip == '':
            #     print("WE GOT ONE BOYS! - " + src)
            
            # If ip are in IPv4 format, add them to the set
            if src.count('.') == 3 and dst.count('.') == 3:
                unique_src_ips.add(src)
                unique_dst_ips.add(dst)
                if src != sip:
                    mismatched_ip_pairs.add((src, sip))
                if dst != rip:
                    mismatched_ip_pairs.add((dst, rip))

            # If mac are in MAC format, add them to the set
            if src_mac.count(':') == 5 and dst_mac.count(':') == 5:
                unique_src_macs.add(src_mac)
                unique_dst_macs.add(dst_mac)

            protocols.add(protocol)
            unique_sbytes_max.add(e[18])
            unique_rbytes_max.add(e[19])
            unique_sbyte_rbyte_pairs.add((e[18], e[19]))

            attack_label = e[60]

            if attack_label == '1':
                attack_entries += 1

            total_entries += 1

    # Print out the results
    print("Of " + str(total_entries) + " total entries, " + str(attack_entries) + " are attacks (" + str(round(attack_entries / total_entries * 100, 2)) + "%)")


        # Read the CSV file into a DataFrame
        #df = pd.read_csv(data_path, na_values=[''])
        #cols_to_norm = df.columns[13:60]

        #import matplotlib.pyplot as plt

        # import tkinter
        # tkinter._test()  # should pop up a small window  



        # Get the 60th column (Python is 0-indexed, so column 59)
        # attack_col = df.iloc[:, 59]

        # # Plot the distribution of 0s and 1s
        # plt.figure(figsize=(6, 4))
        # attack_col.value_counts().sort_index().plot(kind='bar', color=['blue', 'red'])
        # plt.title('Distribution of Attack Labels (0 = Normal, 1 = Attack)')
        # plt.xlabel('Label')
        # plt.ylabel('Count')
        # plt.xticks([0, 1], ['Normal', 'Attack'], rotation=0)
        # plt.grid(axis='y')
        # plt.tight_layout()
        # plt.show()

        # Print out min/max values for each column
        #print("_____________________________________")
        #print("Min/Max values by column:\n", df.min(), df.max())



        #Compute skewness
        # skewness = df[cols_to_norm].skew()
        # print("_____________________________________")

        # print("Skewness by column:\n", skewness)

        # threshold = 1.0
        # high_skew = skewness[skewness.abs() > threshold].sort_values()
        # print("Highly skewed columns:\n", high_skew)
        # Extract the relevant columns
        # feat_block = df.iloc[:, 13:60]

        # print("_____________________________________")
        # print(feat_block.head())

        # print(feat_block.describe())

        # print(feat_block.info())



    # total_ips = total_ips.union(unique_src_ips, unique_dst_ips)
    # unique_dual_ips = unique_src_ips.intersection(unique_dst_ips)

    # total_macs = total_macs.union(unique_src_macs, unique_dst_macs)
    # unique_dual_macs = unique_src_macs.intersection(unique_dst_macs)

    # print("All " + str(len(total_ips)) + " unique IPs")
    # print(total_ips)
    # print("\n")
    # print("Appeared in src and dst: " + str(len(unique_dual_ips)) + " IPs")
    # print(unique_dual_ips)
    # print("\n")
    # print("Source IPs: " + str(len(unique_src_ips)))
    # print(unique_src_ips)
    # print("\n")
    # print("Destination IPs: " + str(len(unique_dst_ips)))
    # print(unique_dst_ips)
    # print("\n")
    # print("\n")
    # print("Mismatched IP pairs: " + str(len(mismatched_ip_pairs)))
    # print(mismatched_ip_pairs)
    # print("\n")
    # print("\n")
    # print("Unique IP sections: " + str(len(unique_ip_1s)) + " " + str(len(unique_ip_2s)) + " " + str(len(unique_ip_3s)) + " " + str(len(unique_ip_4s)))
    # print("Unique IP sections: " + str(unique_ip_1s) + " " + str(unique_ip_2s) + " " + str(unique_ip_3s) + " " + str(unique_ip_4s))
    # print("\n")
    # print("\n")
    # print("All " + str(len(total_macs)) + " unique MACs")
    # print(total_macs)
    # print("\n")
    # print("Appeared in src and dst: " + str(len(unique_dual_macs)) + " MACs")
    # print(unique_dual_macs)
    # print("\n")
    # print("Source MACs: " + str(len(unique_src_macs)))
    # print(unique_src_macs)
    # print("\n")
    # print("Destination MACs: " + str(len(unique_dst_macs)))
    # print(unique_dst_macs)
    # print("\n")
    # print("\n")
    # print("Protocols: " + str(len(protocols)))
    # print(protocols)
    # print("\n")
    # print("\n")
    # print("Max sBytesMax: " + str(len(unique_sbytes_max)))
    # print(unique_sbytes_max)
    # print("\n")
    # print("Max rBytesMax: " + str(len(unique_rbytes_max)))
    # print(unique_rbytes_max)
    # print("\n")
    # print("sBytesMax and rBytesMax pairs: " + str(len(unique_sbyte_rbyte_pairs)))
    # print(unique_sbyte_rbyte_pairs)
    # print("\n")
    # print("\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Count unique IPs in dataset')
    parser.add_argument('data', type=str, help='Path to dataset file (CSV)')

    args = parser.parse_args()

    run(args.data)
