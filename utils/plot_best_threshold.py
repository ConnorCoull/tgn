import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

# Configuration for TGN20-Sparse8 model
model_name = "TGN20-Sparse8"
file_path = "C:\\Users\\conno\\Downloads\\RECONSTRUCTION_LOSS_icsflow-eval_graph_attention-mean-20-sparse8-autoencoder.csv"

# Load and process the main dataset
print(f"Processing {model_name}...")
df = pd.read_csv(file_path, header=None)
df.columns = ["id", "ts", "label", "reconstruction_loss"]
df["id"] = df["id"] + 16935

# Load attacks summary data
attacks_summary_df = pd.read_csv("C:\\Users\\conno\\OneDrive\\Documents\\Level_5_UofG\\Project\\archive\\attacker_machine_summary.csv")
attacks_summary_df["startStamp"] = pd.to_datetime(attacks_summary_df["startStamp"], unit='s')
attacks_summary_df["endStamp"] = pd.to_datetime(attacks_summary_df["endStamp"], unit='s')
cutoff = pd.to_datetime(1663327581.195362, unit='s')
attacks_summary_df = attacks_summary_df[attacks_summary_df["startStamp"] < cutoff]
attacks_summary_df["endStamp"] = attacks_summary_df["endStamp"].apply(lambda x: min(x, cutoff))

# Time alignment
attacks_start_time = 1663323340.022929
adjusted_attack_start_time = attacks_start_time - df[df["label"] == 1]["ts"].iloc[0]
df["ts"] = df["ts"] + adjusted_attack_start_time
df["ts"] = pd.to_datetime(df["ts"], unit='s')
df["Date and Time"] = df["ts"].dt.strftime('%Y-%m-%d %H:%M:%S')

print("Average reconstruction loss:", df["reconstruction_loss"].mean())
print(f"Average reconstruction loss per label: {df.groupby('label')['reconstruction_loss'].mean()}")

# Define threshold range
thresholds = np.arange(0, 20001, 1)
detection_rates = []
false_positive_rates = []

print("Evaluating thresholds...")
for threshold in thresholds:
    #print(f"Testing threshold: {threshold}")
    
    # Attack-level detection rate
    detected_attacks = 0
    total_attacks = 0
    
    for i, row in attacks_summary_df.iterrows():
        start = row['startStamp']
        end = row['endStamp']
        
        # Filter df for the time window of the attack
        during_attack = df[(df["ts"] >= start) & (df["ts"] <= end)]
        
        # Check if any reconstruction loss during this period exceeds the threshold
        detected = (during_attack["reconstruction_loss"] >= threshold).any()
        
        if detected:
            detected_attacks += 1
        total_attacks += 1
    
    detection_rate = detected_attacks / total_attacks if total_attacks > 0 else 0
    
    # Point-level false positive rate
    df["predicted_label"] = (df["reconstruction_loss"] >= threshold).astype(int)
    
    # Calculate confusion matrix components
    tp = ((df["label"] == 1) & (df["predicted_label"] == 1)).sum()
    tn = ((df["label"] == 0) & (df["predicted_label"] == 0)).sum()
    fp = ((df["label"] == 0) & (df["predicted_label"] == 1)).sum()
    fn = ((df["label"] == 1) & (df["predicted_label"] == 0)).sum()
    
    # False positive rate = FP / (FP + TN)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    detection_rates.append(detection_rate * 100)  # Convert to percentage
    false_positive_rates.append(fpr * 100)  # Convert to percentage
    
    #print(f"Detection rate: {detection_rate * 100:.2f}%, FPR: {fpr * 100:.4f}%")

# Create the plot
fig, ax1 = plt.subplots(figsize=(12, 8))

# Plot detection rate
# color1 = 'tab:blue'
# ax1.set_xlabel('Threshold', fontsize=12)
# ax1.set_ylabel('Detection Rate (%)', color=color1, fontsize=12)
# line1 = ax1.plot(thresholds, detection_rates, 'o-', color=color1, linewidth=2, 
#                 markersize=6, label='Detection Rate')
# ax1.tick_params(axis='y', labelcolor=color1)
# ax1.grid(True, alpha=0.3)

# # Create second y-axis for false positive rate
# ax2 = ax1.twinx()
# color2 = 'tab:red'
# ax2.set_ylabel('False Positive Rate (%)', color=color2, fontsize=12)
# line2 = ax2.plot(thresholds, false_positive_rates, 's-', color=color2, linewidth=2, 
#                 markersize=6, label='False Positive Rate')
# ax2.tick_params(axis='y', labelcolor=color2)

color1 = 'tab:blue'
ax1.set_xlabel('Threshold', fontsize=12)
ax1.set_ylabel('Detection Rate (%)', color=color1, fontsize=12)
line1 = ax1.plot(thresholds, detection_rates, 'o-', color=color1, linewidth=2, 
                markersize=6, label='Detection Rate')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 100)  # Set detection rate axis from 0-100

# Create second y-axis for false positive rate
ax2 = ax1.twinx()
color2 = 'tab:red'
ax2.set_ylabel('False Positive Rate (%)', color=color2, fontsize=12)
line2 = ax2.plot(thresholds, false_positive_rates, 's-', color=color2, linewidth=2, 
                markersize=6, label='False Positive Rate')
ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_ylim(0, 10)  

# Add title and legend
plt.title(f'{model_name} Model Performance\nDetection Rate vs False Positive Rate', 
          fontsize=14, fontweight='bold', pad=20)

# Combine legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=11)

# Improve layout
plt.tight_layout()

# Add some annotations for key points
# optimal_idx = np.argmax(np.array(detection_rates) - np.array(false_positive_rates))
# optimal_threshold = thresholds[optimal_idx]
# optimal_detection = detection_rates[optimal_idx]
# optimal_fpr = false_positive_rates[optimal_idx]

# ax1.annotate(f'Optimal Point\nThreshold: {optimal_threshold}\nDR: {optimal_detection:.1f}%\nFPR: {optimal_fpr:.3f}%',
#             xy=(optimal_threshold, optimal_detection), 
#             xytext=(optimal_threshold + 20, optimal_detection + 10),
#             arrowprops=dict(arrowstyle='->', color='black', alpha=0.7),
#             bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
#             fontsize=10)

plt.savefig(f"C:\\Users\\conno\\OneDrive\\Documents\\Level_5_UofG\\Project\\ics_flow_graphs\\threshold_to_zero_{model_name}.png", dpi=300, bbox_inches='tight')
#plt.show()

# Print summary statistics
print(f"\n=== {model_name} Performance Summary ===")
print(f"Threshold range: {min(thresholds)} - {max(thresholds)}")
print(f"Best detection rate: {max(detection_rates):.2f}% at threshold {thresholds[np.argmax(detection_rates)]}")
print(f"Lowest FPR: {min(false_positive_rates):.4f}% at threshold {thresholds[np.argmin(false_positive_rates)]}")
#print(f"Optimal trade-off point: threshold {optimal_threshold}, DR: {optimal_detection:.2f}%, FPR: {optimal_fpr:.4f}%")

# Create a summary dataframe
results_df = pd.DataFrame({
    'Threshold': thresholds,
    'Detection_Rate_%': detection_rates,
    'False_Positive_Rate_%': false_positive_rates
})

print(f"\nDetailed Results:")
#print(results_df.to_string(index=False, float_format='%.3f'))