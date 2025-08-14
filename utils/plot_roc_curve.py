data = "C:\\Users\\conno\\Downloads\\RECONSTRUCTION_LOSS_icsflow-eval_graph_attention-mean-20-sparse8-autoencoder (1).csv"

import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Load CSV (no header, so add names manually)
df = pd.read_csv(data, header=None, names=["id", "timestamp", "label", "reconstruction_loss"])

# Extract true labels and scores
y_true = df["label"]
y_scores = df["reconstruction_loss"]

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

# Plot
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--")  # Random guess line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for TGN20-Sparse8 on ICS-Flow")
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig(f"C:\\Users\\conno\\OneDrive\\Documents\\Level_5_UofG\\Project\\ics_flow_TGN20-Sparse8_roc_curve.png", dpi=300, bbox_inches='tight')
plt.show()
