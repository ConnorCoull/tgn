from sklearn.metrics import roc_curve
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

value = 2

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
           "TGN100-Vanilla4-Sigmoid",
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
    "TGN100-Vanilla4-Sigmoid": "C:\\Users\\conno\\Downloads\\RECONSTRUCTION_LOSS_icsflow-eval_graph_attention-mean-100-vanilla4-autoencoder-sigmoid.csv",
    "TGN100-Sparse16": "C:\\Users\\conno\\Downloads\\RECONSTRUCTION_LOSS_icsflow-eval_graph_attention-mean-100-sparse16-autoencoder.csv",
    "TGN100-Sparse8": "C:\\Users\\conno\\Downloads\\RECONSTRUCTION_LOSS_icsflow-eval_graph_attention-mean-100-sparse8-autoencoder.csv",
    "TGN100-Sparse4": "C:\\Users\\conno\\Downloads\\RECONSTRUCTION_LOSS_icsflow-eval_graph_attention-mean-100-sparse4-autoencoder.csv"
}

# file = file_options[options[value]]

# # Load data
# df = pd.read_csv(file, header=None)
# df.columns = ["id", "ts", "label", "reconstruction_loss"]
# df["id"] += 16935

# # Adjust timestamps
# attacks_start_time = 1663323340.022929
# adjusted_attack_start_time = attacks_start_time - df[df["label"] == 1]["ts"].iloc[0]
# df["ts"] += adjusted_attack_start_time
# df["ts"] = pd.to_datetime(df["ts"], unit='s')

# # Threshold calculation
# fpr, tpr, thresholds = roc_curve(df["label"], df["reconstruction_loss"])
# best_threshold = thresholds[np.argmax(tpr - fpr)]
# df["predicted_label"] = (df["reconstruction_loss"] >= best_threshold).astype(int)

# # Scatter plot only
# plt.figure(figsize=(14, 6))
# colors = df["label"].map({0: 'blue', 1: 'red'})
# plt.scatter(df["ts"], df["reconstruction_loss"], c=colors, s=10, alpha=0.7)

# plt.yscale('log')
# plt.title(f"{options[value]} Reconstruction Loss by Label (Scatter)")
# plt.xlabel("Time")
# # Hide x-axis
# #plt.xticks([])

# plt.ylabel("Anomaly Score (Reconstruction Loss)")

# # Custom legend
# from matplotlib.lines import Line2D
# legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Benign (0)'),
#                    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Attack (1)')]
# plt.legend(handles=legend_elements)

# plt.tight_layout()
# plt.savefig(f"C:\\Users\\conno\\OneDrive\\Documents\\Level_5_UofG\\Project\\ics_flow_graphs\\ics_flow_autoencoder_scatter_{options[value]}.png", dpi=300)
# plt.show()
# plt.close()

file = file_options[options[value]]
df = pd.read_csv(file, header=None)
df.columns = ["id", "ts", "label", "reconstruction_loss"]

print(df['ts'].head(100))

# Adjust ID and timestamps
#df["id"] = df["id"] + 16935
#attacks_start_time = 1663323340.022929
#adjusted_attack_start_time = attacks_start_time - df[df["label"] == 1]["ts"].iloc[0]
#df["ts"] = df["ts"] + adjusted_attack_start_time
#df["ts"] = pd.to_datetime(df["ts"], unit='s')
#df["Date and Time"] = df["ts"].dt.strftime('%Y-%m-%d %H:%M:%S')

# Print timestamp information
#print(df["ts"].min(), df["ts"].max())
#print(df["ts"].max() - df["ts"].min())
#print(options[value])

# Calculate average reconstruction loss
#print("Average reconstruction loss:", df["reconstruction_loss"].mean())
#print(f"Average reconstruction loss per label: {df.groupby('label')['reconstruction_loss'].mean()}")

# ROC curve and best threshold
fpr, tpr, thresholds = roc_curve(df["label"], df["reconstruction_loss"])
j_scores = tpr - fpr
best_threshold = thresholds[np.argmax(j_scores)]
print("Best threshold for reconstruction loss:", best_threshold)

# Assign predicted labels
df["predicted_label"] = (df["reconstruction_loss"] >= best_threshold).astype(int)

# TP, TN, FP, FN calculations
tp = ((df["label"] == 1) & (df["predicted_label"] == 1)).sum()
tn = ((df["label"] == 0) & (df["predicted_label"] == 0)).sum()
fp = ((df["label"] == 0) & (df["predicted_label"] == 1)).sum()
fn = ((df["label"] == 1) & (df["predicted_label"] == 0)).sum()

print("True Positives (TP):", tp)
print("True Negatives (TN):", tn)
print("False Positives (FP):", fp)
print("False Negatives (FN):", fn)

# Optional cutoff (commented out)
# cutoff = pd.to_datetime(1663327591.195362, unit='s')
# df = df[df["ts"] < cutoff]

# Create the plot
plt.figure(figsize=(14, 6))

#df['ts'] = df['ts'] - df['ts'].min()

#convert ts to float
#df['ts'] = df['ts'].view("float64")

# Plot continuous line with color changes based on label
for i in range(len(df) - 1):
    print(i, end="\r")
    x_vals = [df["ts"].iloc[i], df["ts"].iloc[i + 1]]
    y_vals = [df['reconstruction_loss'].iloc[i], df['reconstruction_loss'].iloc[i + 1]]
    color = 'red' if df['label'].iloc[i] == 1 else 'blue'
    plt.plot(x_vals, y_vals, color=color, linewidth=1)

#plt.gcf().autofmt_xdate()

# Create legend manually
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='blue', label='Benign (Label 0)'),
    Line2D([0], [0], color='red', label='Attack (Label 1)')
]
plt.legend(handles=legend_elements)

plt.yscale('log')
plt.title(f"{options[value]} Reconstruction Loss by Label")
plt.xlabel("Time (seconds since first event)")
plt.ylabel("Anomaly Score (Reconstruction Loss)")
plt.tight_layout()

# Save image
plt.savefig(
    f"C:\\Users\\conno\\OneDrive\\Documents\\Level_5_UofG\\Project\\ics_flow_graphs\\"
    f"ics_flow_autoencoder_reconstruction_loss_{options[value]}_rb_by_label.png",
    dpi=300,
    bbox_inches='tight'
)

# Close plot
plt.close()
