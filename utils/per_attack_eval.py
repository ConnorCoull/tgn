import pandas as pd
import matplotlib.pyplot as plt

data = "C:\\Users\\conno\\Downloads\\RECONSTRUCTION_LOSS_icsflow-eval_graph_attention-mean-20-sparse8-autoencoder (1).csv"
attack_data = "C:\\Users\\conno\\OneDrive\\Documents\\Level_5_UofG\\Project\\archive\\attacker_machine_summary.csv"


# Load model results (no header in first CSV)
df = pd.read_csv(data, header=None, names=["id", "timestamp", "label", "reconstruction_loss"])

df["id"] = df["id"] + 16935

attacks_start_time = 1663323340.022929
adjusted_attack_start_time = attacks_start_time - df[df["label"] == 1]["timestamp"].iloc[0]
df["timestamp"] = df["timestamp"] + adjusted_attack_start_time
df["timestamp"] = pd.to_datetime(df["timestamp"], unit='s')
df["Date and Time"] = df["timestamp"].dt.strftime('%Y-%m-%d %H:%M:%S')

# Load attack events table
df_attacks = pd.read_csv(attack_data)
df_attacks["startStamp"] = pd.to_datetime(df_attacks["startStamp"], unit='s')
df_attacks["endStamp"] = pd.to_datetime(df_attacks["endStamp"], unit='s')

# Tag each score row with its attack type
df["attack"] = None
for i, row in df_attacks.iterrows():
    mask = (df["timestamp"] >= row["startStamp"]) & (df["timestamp"] <= row["endStamp"])
    df.loc[mask, "attack"] = row["attack"]

# Drop rows without a matching attack (optional)
df = df.dropna(subset=["attack"])

summary = df.groupby("attack")["reconstruction_loss"].agg(
    min_loss="min",
    max_loss="max",
    mean_loss="mean",
    median_loss="median",
    std_loss="std",
    q25=lambda x: x.quantile(0.25),
    q75=lambda x: x.quantile(0.75)
).reset_index()

print(summary)

# ---- BOX PLOT ----
#plt.figure(figsize=(10, 6))
df.boxplot(column="reconstruction_loss", by="attack", showfliers=True)
plt.yscale("log")  # log scale handles extreme range
plt.xlabel("Attack Type")
plt.ylabel("Reconstruction Loss")
plt.title("Reconstruction Loss Distribution by Attack Type for TGN20-Sparse8 on ICS-Flow")
plt.suptitle("")  # Remove default 'by attack' title
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.savefig(f"C:\\Users\\conno\\OneDrive\\Documents\\Level_5_UofG\\Project\\ics_flow_attack_boxplot_TGN20-Sparse8.png", dpi=300, bbox_inches='tight')
plt.show()