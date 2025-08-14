import pandas as pd
import numpy as np

file = "C:\\Users\\conno\\Downloads\\CIC-IDS-Results\\tgn-pretrain-cicids-mean-Friday-WorkingHours-Afternoon-DDos_Normalised_reconstruction_losses.csv"
attack_file = "C:\\Users\\conno\\Downloads\\CIC-IDS-Results\\CIC-IDS_attack_friday_afternoon_DoS.csv"

attack_df = pd.read_csv(attack_file)

attack_df['startTime'] = pd.to_datetime(attack_df['startTime'], format='mixed')
attack_df['endTime'] = pd.to_datetime(attack_df['endTime'], format='mixed')

#convert to seconds
attack_df['startTime'] = attack_df['startTime'].astype(np.int64) // 10**9
attack_df['endTime'] = attack_df['endTime'].astype(np.int64) // 10**9

df = pd.read_csv(file, header=None)
df.columns = ["id", "ts", "label", "reconstruction_loss"]
#07/07/2017  03:30:00

df['ts'] = pd.to_datetime(df['ts'], unit='s')
df['ts'] = df['ts'] + pd.to_datetime('07/07/2017 03:30:00', format='%d/%m/%Y %H:%M:%S').astype(np.int64) // 10**9

#print(attack_df.head())
print(df['ts'].unique())

