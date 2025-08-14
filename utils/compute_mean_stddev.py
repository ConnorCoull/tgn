import pandas as pd

file = "C:\\Users\\conno\\Downloads\\ics_flow_training_loss_sparse.csv"

df = pd.read_csv(file, header=None)
df.columns = ["id", "ts", "label", "reconstruction_loss"]

mean = df['reconstruction_loss'].mean()
std = df['reconstruction_loss'].std()

print(f"Mean: {mean}")
print(f"Standard Deviation: {std}")