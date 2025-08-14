from sklearn.metrics import roc_curve
import pandas as pd
import numpy as np
import os

value = 7

options_old = ["TGN-Vanilla-Mean-Sigmoid", "TGN-Vanilla-Mean-ReLU", "TGN-Sparse-Mean", "TGN-Variational-Mean",
           "Base", "TGN-Sparse-Mean-200", "TGN-Vanilla-Mean-16mem", "TGN-Vanilla-Mean-64mem", "TGN-Vanilla8-Mean",
           "TGN-Sparse16-Mean", "TGN-Sparse8-Mean", "TGN-20-GAT-MEAN-Sparse4", "TGN-20-GAT-MEAN-Sparse8",
           "TGN-20-GAT-MEAN-Sparse16"
           ]

options = ["TGN20-Vanilla16",
           "TGN20-Vanilla8",
           "TGN20-Vanilla8-Sigmoid",
           "TGN20-Vanilla8-Tanh",
           "TGN20-Vanilla4",
           "TGN20-Sparse16",
           "TGN20-Sparse8",
           "TGN20-Sparse4",
           "TGN20-Variational8",
           "TGN50-Vanilla16",
           "TGN50-Vanilla8",
           "TGN50-Vanilla4",
           "TGN50-Sparse16",
           "TGN50-Sparse8",
           "TGN50-Sparse4",
           "TGN100-Vanilla16",
           "TGN100-Vanilla8",
           "TGN100-Vanilla4",
           "TGN100-Sparse16",
           "TGN100-Sparse8",
           "TGN100-Sparse4"
           ]

file_options = {
    "TGN20-Vanilla16": "C:\\Users\\conno\\Downloads\\RECONSTRUCTION_LOSS_icsflow-eval_graph_attention-mean-20-vanilla16-autoencoder.csv",
    "TGN20-Vanilla8": "C:\\Users\\conno\\Downloads\\RECONSTRUCTION_LOSS_icsflow-eval_graph_attention-mean-20-vanilla8-autoencoder_2.csv",
    "TGN20-Vanilla8-Sigmoid": "C:\\Users\\conno\\Downloads\\RECONSTRUCTION_LOSS_icsflow-eval_graph_attention-mean-20-vanilla8-autoencoder-sigmoid.csv",
    "TGN20-Vanilla8-Tanh": "C:\\Users\\conno\\Downloads\\RECONSTRUCTION_LOSS_icsflow-eval_graph_attention-mean-20-vanilla8-autoencoder-tanh.csv",
    "TGN20-Vanilla4": "C:\\Users\\conno\\Downloads\\RECONSTRUCTION_LOSS_icsflow-eval_graph_attention-mean-20-vanilla4-autoencoder.csv",
    "TGN20-Sparse16": "C:\\Users\\conno\\Downloads\\RECONSTRUCTION_LOSS_icsflow-eval_graph_attention-mean-20-sparse16-autoencoder (1).csv",
    "TGN20-Sparse8": "C:\\Users\\conno\\Downloads\\RECONSTRUCTION_LOSS_icsflow-eval_graph_attention-mean-20-sparse8-autoencoder (1).csv",
    "TGN20-Sparse4": "C:\\Users\\conno\\Downloads\\RECONSTRUCTION_LOSS_icsflow-eval_graph_attention-mean-20-sparse4-autoencoder.csv",
    "TGN20-Variational8": "C:\\Users\\conno\\Downloads\\RECONSTRUCTION_LOSS_icsflow-eval_graph_attention-mean-20-variational8-autoencoder.csv",
    "TGN50-Vanilla16": "C:\\Users\\conno\\Downloads\\RECONSTRUCTION_LOSS_icsflow-eval_graph_attention-mean-50-vanilla16-autoencoder.csv",
    "TGN50-Vanilla8": "C:\\Users\\conno\\Downloads\\RECONSTRUCTION_LOSS_icsflow-eval_graph_attention-mean-50-vanilla8-autoencoder.csv",
    "TGN50-Vanilla4": "C:\\Users\\conno\\Downloads\\RECONSTRUCTION_LOSS_icsflow-eval_graph_attention-mean-50-vanilla4-autoencoder.csv",
    "TGN50-Sparse16": "C:\\Users\\conno\\Downloads\\RECONSTRUCTION_LOSS_icsflow-eval_graph_attention-mean-50-sparse16-autoencoder.csv",
    "TGN50-Sparse8": "C:\\Users\\conno\\Downloads\\RECONSTRUCTION_LOSS_icsflow-eval_graph_attention-mean-50-sparse8-autoencoder.csv",
    "TGN50-Sparse4": "C:\\Users\\conno\\Downloads\\RECONSTRUCTION_LOSS_icsflow-eval_graph_attention-mean-50-sparse4-autoencoder.csv",
    "TGN100-Vanilla16": "C:\\Users\\conno\\Downloads\\RECONSTRUCTION_LOSS_icsflow-eval_graph_attention-mean-100-vanilla16-autoencoder.csv",
    "TGN100-Vanilla8": "C:\\Users\\conno\\Downloads\\RECONSTRUCTION_LOSS_icsflow-eval_graph_attention-mean-100-vanilla8-autoencoder.csv",
    "TGN100-Vanilla4": "C:\\Users\\conno\\Downloads\\RECONSTRUCTION_LOSS_icsflow-eval_graph_attention-mean-100-vanilla4-autoencoder.csv",
    "TGN100-Sparse16": "C:\\Users\\conno\\Downloads\\RECONSTRUCTION_LOSS_icsflow-eval_graph_attention-mean-100-sparse16-autoencoder.csv",
    "TGN100-Sparse8": "C:\\Users\\conno\\Downloads\\RECONSTRUCTION_LOSS_icsflow-eval_graph_attention-mean-100-sparse8-autoencoder.csv",
    "TGN100-Sparse4": "C:\\Users\\conno\\Downloads\\RECONSTRUCTION_LOSS_icsflow-eval_graph_attention-mean-100-sparse4-autoencoder.csv"
}

file_options_old = {
    "TGN-Vanilla32-Mean-Sigmoid": "C:\\Users\\conno\\Downloads\\icsflow-eval_reconstruction_losses_vanilla_ae_sigmoid.csv",
    "TGN-Vanilla32-Mean-ReLU": "C:\\Users\\conno\\Downloads\\icsflow-eval_reconstruction_losses_vanilla_ae_relu.csv",
    "TGN-Sparse32-Mean": "C:\\Users\\conno\\Downloads\\icsflow-eval_sparse_reconstruction_losses_sparse_ae.csv",
    "TGN-Variational32-Mean": "C:\\Users\\conno\\Downloads\\icsflow-eval_variational_reconstruction_losses.csv",
    "Base": "C:\\Users\\conno\\Downloads\\ics_flow_training_loss_sparse.csv",
    "TGN-Sparse32-Mean-200": "C:\\Users\\conno\\Downloads\\icsflow-eval_sparse_200epoch_reconstruction_losses.csv",
    "TGN-Vanilla16-Mean": "C:\\Users\\conno\\Downloads\\icsflow-eval_vanilla_reconstruction_losses_vanilla16.csv",
    "TGN-Vanilla64-Mean": "C:\\Users\\conno\\Downloads\\icsflow-eval_vanilla_reconstruction_losses_vanilla64.csv",
    "TGN-Vanilla8-Mean": "C:\\Users\\conno\\Downloads\\icsflow-eval_vanilla_reconstruction_losses_vanilla8.csv",
    "TGN-Sparse16-Mean": "C:\\Users\\conno\\Downloads\\icsflow-eval_sparse_reconstruction_losses_sparse16.csv",
    "TGN-Sparse8-Mean": "C:\\Users\\conno\\Downloads\\icsflow-eval_sparse_reconstruction_losses_sparse8.csv",
    "TGN-20-GAT-MEAN-Sparse4": "C:\\Users\\conno\\Downloads\\RECONSTRUCTION_LOSS_icsflow-eval_graph_attention-mean-20-sparse4-autoencoder.csv",
    "TGN-20-GAT-MEAN-Sparse8": "C:\\Users\\conno\\Downloads\\RECONSTRUCTION_LOSS_icsflow-eval_graph_attention-mean-20-sparse8-autoencoder.csv",
    "TGN-20-GAT-MEAN-Sparse16": "C:\\Users\\conno\\Downloads\\RECONSTRUCTION_LOSS_icsflow-eval_graph_attention-mean-20-sparse16-autoencoder.csv"
}

file = file_options[options[value]]

# Define a color for each attack type
attack_colors = {
    'port-scan': 'red',
    'ip-scan': 'orange',
    'mitm': 'purple',
    'replay': 'green',
    'ddos': 'brown',
    'DDoS': 'brown',
}


df = pd.read_csv(file, header=None)
df.columns = ["id", "ts", "label", "reconstruction_loss"]
df["id"] = df["id"] + 16935

attacks_start_time = 1663323340.022929
adjusted_attack_start_time = attacks_start_time - df[df["label"] == 1]["ts"].iloc[0]
df["ts"] = df["ts"] + adjusted_attack_start_time
df["ts"] = pd.to_datetime(df["ts"], unit='s')
df["Date and Time"] = df["ts"].dt.strftime('%Y-%m-%d %H:%M:%S')

print(options[value])

#Calculate average reconstruction loss for the whole dataset
print("Average reconstruction loss:", df["reconstruction_loss"].mean())

# Calculate average reconstruction loss per label
print(f"Average reconstruction loss per label: {df.groupby('label')['reconstruction_loss'].mean()}")

# Print mean and std dev of label 0


fpr, tpr, thresholds = roc_curve(df["label"], df["reconstruction_loss"])
j_scores = tpr - fpr
best_threshold = thresholds[np.argmax(j_scores)] # maximise tp
best_threshold = 440

#best_threshold = 1
print("Best threshold for reconstruction loss:", best_threshold)

df["predicted_label"] = (df["reconstruction_loss"] >= best_threshold).astype(int)

# TP
tp = ((df["label"] == 1) & (df["predicted_label"] == 1)).sum()
# TN
tn = ((df["label"] == 0) & (df["predicted_label"] == 0)).sum()
# FP
fp = ((df["label"] == 0) & (df["predicted_label"] == 1)).sum()
# FN
fn = ((df["label"] == 1) & (df["predicted_label"] == 0)).sum()

print("True Positives (TP):", tp)
print("True Negatives (TN):", tn)
print("False Positives (FP):", fp)
print("False Negatives (FN):", fn)

# Graph reconstruction loss against time, with added times where attacks stop and start

#cutoff = pd.to_datetime("16/09/2022 16:11:26", format="%d/%m/%Y %H:%M:%S")
cutoff = pd.to_datetime(1663327581.195362, unit='s')

df = df[df["ts"] < cutoff]


import matplotlib.pyplot as plt

attacks_summary_df = pd.read_csv("C:\\Users\\conno\\OneDrive\\Documents\\Level_5_UofG\\Project\\archive\\attacker_machine_summary.csv")
attacks_summary_df["startStamp"] = pd.to_datetime(attacks_summary_df["startStamp"], unit='s')
attacks_summary_df["endStamp"] = pd.to_datetime(attacks_summary_df["endStamp"], unit='s')

attacks_summary_df = attacks_summary_df[attacks_summary_df["startStamp"] < cutoff]
attacks_summary_df["endStamp"] = attacks_summary_df["endStamp"].apply(lambda x: min(x, cutoff))

# plt.figure(figsize=(14, 6))
# plt.plot(df['ts'], df['reconstruction_loss'], label='Anomaly Score', color='blue')
# plt.yscale('log')
# #plt.axhline(y=628, color='red', linestyle='--', label='Best Threshold')
# #plt.axhline(y=21312, color='red', linestyle='--', label='Another Threshold')


# for _, row in attacks_summary_df.iterrows():
#    attack_type = row['attack']
#    color = attack_colors.get(attack_type, 'grey')  # fallback color
#    plt.axvspan(row['startStamp'], row['endStamp'], color=color, alpha=0.3, label=attack_type)


# # Optional: Deduplicate labels
# handles, labels = plt.gca().get_legend_handles_labels()
# by_label = dict(zip(labels, handles))
# # Optional: Deduplicate labels
# handles, labels = plt.gca().get_legend_handles_labels()
# by_label = dict(zip(labels, handles))
# plt.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0.)


# plt.title(f"{options[value]} Reconstruction Loss with Attacks")
# #plt.title("Reconstruction Loss on Benign Data using TGN-Sparse-Mean")
# plt.xlabel("Time")
# plt.ylabel("Anomaly Score (Reconstruction Loss)")
# plt.tight_layout()


# # # Save image
# #plt.savefig(f"C:\\Users\\conno\\OneDrive\\Documents\\Level_5_UofG\\Project\\ics_flow_graphs\\ics_flow_autoencoder_reconstruction_loss_{options[value]}.png", dpi=300, bbox_inches='tight')

# #plt.show()
# plt.close()

# best_threshold = 1
# print("\nAttack Detection by Reconstruction Loss:")
# for i, row in attacks_summary_df.iterrows():
#     attack_type = row['attack']
#     start = row['startStamp']
#     end = row['endStamp']
    
#     # Filter df for the time window of the attack
#     during_attack = df[(df["ts"] >= start) & (df["ts"] <= end)]
    
#     # Check if any reconstruction loss during this period exceeds the best threshold
#     detected = (during_attack["reconstruction_loss"] >= best_threshold).any()
    
#     print(f"{attack_type} !{i}! from {start} to {end} detected: {detected}")

# Print interquartile range for benign and anomalous data
print("\nInterquartile Range (IQR) for Benign and Anomalous Data:")

benign_iqr = df[df["label"] == 0]["reconstruction_loss"].quantile(0.75) - df[df["label"] == 0]["reconstruction_loss"].quantile(0.25)
anomalous_iqr = df[df["label"] == 1]["reconstruction_loss"].quantile(0.75) - df[df["label"] == 1]["reconstruction_loss"].quantile(0.25)

print("Benign IQR:", benign_iqr)
print("Anomalous IQR:", anomalous_iqr)
