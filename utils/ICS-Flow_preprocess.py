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

import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse

def preprocess(data_name):
    src_list, dst_list, ts_list, label_list = [], [], [], []
    feat_l = []
    idx_list = []

    with open(data_name) as f:
        s = next(f) # skip headers
        for idx, line in enumerate(f):
            e = line.strip().split(',')
            # TODO: src and dst are IP addresses, do I convert them to int?
            src = int(e[0])
            dst = int(e[1])

            ts = float(e[9])
            label = float(e[60])

            feat = np.array([float(x) for x in e[0:60]]) # not including label

            src_list.append(src)
            dst_list.append(dst)
            ts_list.append(ts)
            label_list.append(label)
            idx_list.append(idx)

            feat_l.append(feat)

        return pd.DataFrame({'src': src_list,
                       'dst': dst_list,
                       'ts': ts_list,
                       'label': label_list,
                       'idx': idx_list}), np.array(feat_l)
    
# Because src and dst are IP addresses, we may not need this function
# In fact we may need it to build an alias table of:
# | id | ip | mac |
def reindex(df, bipartite=True):
    pass


# ICS data is not being treated as bipartite, so we set it to False as default
def run(data_name, bipartite=False):
    Path("data/").mkdir(parents=True, exist_ok=True)
    PATH = './data/{}.csv'.format(data_name)
    OUT_DF = './data/ml_{}.csv'.format(data_name)
    OUT_FEAT = './data/ml_feat_{}.npy'.format(data_name)
    OUT_NODE_FEAT = './data/ml_node_feat_{}.npy'.format(data_name)

    df, feat = preprocess(PATH)
    #new_df = reindex(df, bipartite)

    empty = np.zeros(feat.shape[1])[np.newaxis, :]
    feat = np.vstack([empty, feat])

    max_idx = max(df.u.max(), df.i.max())
    rand_feat = np.zeros((max_idx + 1, 172))

    df.to_csv(OUT_DF)
    np.save(OUT_FEAT, feat)
    np.save(OUT_NODE_FEAT, rand_feat)

parser = argparse.ArgumentParser('Interface for TGN data preprocessing')
parser.add_argument('--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='wikipedia')
# Really legacy here, I don't believe ICS data will be treated as bipartite
parser.add_argument('--bipartite', action='store_true', help='Whether the graph is bipartite')

# Note to reader: Bipartite is when the graph can be divided into two disjoint sets such that every edge connects a node in one set to a node in the other set.
# For example in Rossi et al. the Reddit dataset is bipartite, as it has two types of nodes: users and posts.

# In ICS we have source and destination IP addresses, but they are not disjoint sets.

args = parser.parse_args()

run(args.data, bipartite=args.bipartite)