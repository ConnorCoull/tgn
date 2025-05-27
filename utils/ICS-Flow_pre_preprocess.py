"""
Transform ICS-Flow data into a csv of:
- source_id: source node
- destination_id: destination node
- timestamp: time of the flow
- is_attack: 1 if the flow is an attack, 0 otherwise
- comma_separated_list_of_features: list of features for the flow

Implementation details
sip and rip are not used as they are always identical to sAddress and rAddress
IP addresses are split into 4 columns, each containing a number between 0 and 255, then normalised to [0, 1]
MAC addresses are split into 6 columns, each containing a number between 0 and 255, then normalised to [0, 1]
Protocol is one-hot encoded

The features are normalised from [0, inf] to [0, 1] using the inverse logit after log

Original Feature List
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

New Feature List
sAddress    rAddress	timestamp    is_attack
sAddress    rAddress    protocol    duration
sPackets    rPackets    sBytesSum    rBytesSum
sBytesMax    rBytesMax    sBytesMin    rBytesMin
sBytesAvg    rBytesAvg    sLoad    rLoad
sPayloadSum    rPayloadSum    sPayloadMax    rPayloadMax
sPayloadMin    rPayloadMin    sPayloadAvg    rPayloadAvg
sInterPacketAvg    rInterPacketAvg    sttl    rttl
sAckRate    rAckRate    sUrgRate    rUrgRate
sFinRate    rFinRate    sPshRate    rPshRate
sSynRate    rSynRate    sRstRate    rRstRate
sWinTCP    rWinTCP    sFragmentRate    rFragmentRate
sAckDelayMax    rAckDelayMax    sAckDelayMin    rAckDelayMin
sAckDelayAvg    rAckDelayAvg

startduration may be of interest to detect replay attacks, dos, etc but left out for now
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import csv
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler


def run(data_name):
    Path("data/").mkdir(parents=True, exist_ok=True)
    PATH = './data/{}.csv'.format(data_name)
    OUT_PATH = './data/processed_ics-flow.csv'.format(data_name)

    with open(PATH, newline='') as infile, open(OUT_PATH, 'w', newline='') as outfile:

        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        ip_mac_table = {}
        counter = 1

        # read & transform column headers

        """[src_node, dst_node, ts_rel, label] +
        src_ip_parts + dst_ip_parts + src_mac_parts +
        dst_mac_parts + protocol_one_hot + row[8:60]"""

        old_header = next(reader)
        new_header = ['src_node','dst_node','ts','is_attack'] +\
            ['src_ip_section_1', 'src_ip_section_2', 'src_ip_section_3', 'src_ip_section_4'] + \
            ['dst_ip_section_1', 'dst_ip_section_2', 'dst_ip_section_3', 'dst_ip_section_4'] + \
            ['src_mac_section_1', 'src_mac_section_2', 'src_mac_section_3', 'src_mac_section_4', 'src_mac_section_5', 'src_mac_section_6'] + \
            ['dst_mac_section_1', 'dst_mac_section_2', 'dst_mac_section_3', 'dst_mac_section_4', 'dst_mac_section_5', 'dst_mac_section_6'] + \
            ['protocol_icmp', 'protocol_arp', 'protocol_ipv6', 'protocol_tcp', 'protocol_udp'] + \
            old_header[13:60]
        writer.writerow(new_header)

        min_time = None
        for idx, row in enumerate(reader):
            src   = str(row[0])
            dst   = str(row[1])
            ts    = float(row[9])
            label = int(row[60])
            src_mac = str(row[2])
            dst_mac = str(row[3])

            if min_time is None:
                min_time = ts
            ts_rel = ts - min_time

            # Convert IP address to node ID, keep a table for reference, remember each node has an IP and MAC
            if src not in ip_mac_table:
                ip_mac_table[src] = counter
                ip_mac_table[src_mac] = counter
                src_node = ip_mac_table[src]
                counter += 1
            else:
                src_node = ip_mac_table[src]

            if dst not in ip_mac_table:
                ip_mac_table[dst] = len(ip_mac_table)
                ip_mac_table[dst_mac] = len(ip_mac_table)
                dst_node = ip_mac_table[dst]
                counter += 1
            else:
                dst_node = ip_mac_table[dst]

            # Authors sin: not sure why I'm multiplying by 1000, I just wanted the ts units similar to wikipedia
            ts_rel = ts_rel * 1000
            ts_rel = round(ts_rel, 2)

            # Split IP address into 4 columns, each containing a number between 0 and 255
            # If IP is not in correct format, replace with 0.0.0.0
            try:
                src_ip_parts = [int(x) for x in src.split('.')]
                dst_ip_parts = [int(x) for x in dst.split('.')]
            except ValueError:
                src_ip_parts = [0, 0, 0, 0]
                dst_ip_parts = [0, 0, 0, 0]

            src_ip_parts = [x / 255.0 for x in src_ip_parts]
            dst_ip_parts = [x / 255.0 for x in dst_ip_parts]

            # Split MAC address into 6 columns, each containing a number between 0 and 255
            # If MAC does not exist, replace with 0,0,0,0,0,0
            try:
                src_mac_parts = [int(x, 16) for x in src_mac.split(':')]
                dst_mac_parts = [int(x, 16) for x in dst_mac.split(':')]
            except ValueError:
                src_mac_parts = [0, 0, 0, 0, 0, 0]
                dst_mac_parts = [0, 0, 0, 0, 0, 0]

            src_mac_parts = [x / 255.0 for x in src_mac_parts]
            dst_mac_parts = [x / 255.0 for x in dst_mac_parts]

            # One-hot encode protocol (5 protocols)
            # 0: ICMP, 1: ARP, 2: IPV6, 3: TCP, 4: UDP
            protocol = str(row[6])
            protocol_one_hot = [0] * 5
            if protocol == 'IPV4-ICMP':
                protocol_one_hot[0] = 1
            elif protocol == 'ARP':
                protocol_one_hot[1] = 1
            elif protocol == 'IPV6':
                protocol_one_hot[2] = 1
            elif protocol == 'IPV4-TCP':
                protocol_one_hot[3] = 1
            elif protocol == 'IPV4-UDP':
                protocol_one_hot[4] = 1

            # TODO: features 13-59 need to be normalised, as many are skewed, both left and right
            # If I don't split the features beforehand, then I leak information
                        
            writer.writerow(
                [src_node, dst_node, ts_rel, label] + src_ip_parts + dst_ip_parts + src_mac_parts + dst_mac_parts + protocol_one_hot + row[13:60]
            )
    print("ICS-Flow dataset processed and saved to {}".format(OUT_PATH))
    print("IP-MAC table: " + str(ip_mac_table))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Count unique IPs in dataset')
    parser.add_argument('data', type=str, help='Path to dataset file (CSV)')

    args = parser.parse_args()

    run(args.data)