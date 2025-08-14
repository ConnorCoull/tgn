from sklearn.metrics import roc_curve
import pandas as pd
import numpy as np
import os

options = ["TGN20-Vanilla16",
           "TGN20-Vanilla8",
           "TGN20-Sparse16",
           "TGN20-Sparse8",
           "TGN50-Vanilla16",
           "TGN50-Vanilla8",
           "TGN50-Sparse16",
           "TGN50-Sparse8"]

options = ["TGN20-Sparse8"]

file_options = {
    "TGN20-Vanilla16": "C:\\Users\\conno\\Downloads\\RECONSTRUCTION_LOSS_icsflow-eval_graph_attention-mean-20-vanilla16-autoencoder.csv",
    "TGN20-Vanilla8": "C:\\Users\\conno\\Downloads\\RECONSTRUCTION_LOSS_icsflow-eval_graph_attention-mean-20-vanilla8-autoencoder.csv",
    "TGN20-Sparse16": "C:\\Users\\conno\\Downloads\\RECONSTRUCTION_LOSS_icsflow-eval_graph_attention-mean-20-sparse16-autoencoder (1).csv",
    "TGN20-Sparse8": "C:\\Users\\conno\\Downloads\\RECONSTRUCTION_LOSS_icsflow-eval_graph_attention-mean-20-sparse8-autoencoder (1).csv",
    "TGN50-Vanilla16": "C:\\Users\\conno\\Downloads\\RECONSTRUCTION_LOSS_icsflow-eval_graph_attention-mean-50-vanilla16-autoencoder.csv",
    "TGN50-Vanilla8": "C:\\Users\\conno\\Downloads\\RECONSTRUCTION_LOSS_icsflow-eval_graph_attention-mean-50-vanilla8-autoencoder.csv",
    "TGN50-Sparse16": "C:\\Users\\conno\\Downloads\\RECONSTRUCTION_LOSS_icsflow-eval_graph_attention-mean-50-sparse16-autoencoder.csv",
    "TGN50-Sparse8": "C:\\Users\\conno\\Downloads\\RECONSTRUCTION_LOSS_icsflow-eval_graph_attention-mean-50-sparse8-autoencoder.csv"
}

attacks_summary_df = pd.read_csv("C:\\Users\\conno\\OneDrive\\Documents\\Level_5_UofG\\Project\\archive\\attacker_machine_summary.csv")
attacks_summary_df["startStamp"] = pd.to_datetime(attacks_summary_df["startStamp"], unit='s')
attacks_summary_df["endStamp"] = pd.to_datetime(attacks_summary_df["endStamp"], unit='s')
cutoff = pd.to_datetime(1663327581.195362, unit='s')
attacks_summary_df = attacks_summary_df[attacks_summary_df["startStamp"] < cutoff]
attacks_summary_df["endStamp"] = attacks_summary_df["endStamp"].apply(lambda x: min(x, cutoff))

for value in range(len(options)):
    print(f"Processing option: {options[value]}")
    file = file_options[options[value]]

    df = pd.read_csv(file, header=None)
    df.columns = ["id", "ts", "label", "reconstruction_loss"]
    df["id"] = df["id"] + 16935

    attacks_start_time = 1663323340.022929
    adjusted_attack_start_time = attacks_start_time - df[df["label"] == 1]["ts"].iloc[0]
    df["ts"] = df["ts"] + adjusted_attack_start_time
    df["ts"] = pd.to_datetime(df["ts"], unit='s')
    df["Date and Time"] = df["ts"].dt.strftime('%Y-%m-%d %H:%M:%S')

    #Calculate average reconstruction loss for the whole dataset
    print("Average reconstruction loss:", df["reconstruction_loss"].mean())

    # Calculate average reconstruction loss per label
    print(f"Average reconstruction loss per label: {df.groupby('label')['reconstruction_loss'].mean()}")

    thresholds = [440, 618, 868, 1220, 1714, 2408, 3383, 4753, 6678, 9382, 13181, 18518]
    for threshold in thresholds:
        print(f"Testing threshold: {threshold}")
        detected_attacks = 0
        undetected_attacks = 0
        total_attacks = 0
        for i, row in attacks_summary_df.iterrows():
            attack_type = row['attack']
            start = row['startStamp']
            end = row['endStamp']

            # Filter df for the time window of the attack
            during_attack = df[(df["ts"] >= start) & (df["ts"] <= end)]

            # Check if any reconstruction loss during this period exceeds the best threshold
            detected = (during_attack["reconstruction_loss"] >= threshold).any()

            if detected:
                detected_attacks += 1
            else:
                undetected_attacks += 1
            total_attacks += 1

        df["predicted_label"] = (df["reconstruction_loss"] >= threshold).astype(int)

        # TP
        tp = ((df["label"] == 1) & (df["predicted_label"] == 1)).sum()
        # TN
        tn = ((df["label"] == 0) & (df["predicted_label"] == 0)).sum()
        # FP
        fp = ((df["label"] == 0) & (df["predicted_label"] == 1)).sum()
        # FN
        fn = ((df["label"] == 1) & (df["predicted_label"] == 0)).sum()

        print(f"Detected attacks: {detected_attacks}")
        print(f"Undetected attacks: {undetected_attacks}")
        print(f"Total attacks: {total_attacks}")
        print(f"False Positives (FP): {fp}")
        print(f"Detection rate: {detected_attacks / total_attacks * 100:.2f}%\n")