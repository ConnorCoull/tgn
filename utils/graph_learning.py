import re
import pandas as pd
import matplotlib.pyplot as plt

# --- Configuration ---
# Set this to the path of your log file
filename = "C:\\Users\\conno\\OneDrive\\Documents\\Level_5_UofG\\Project\\training_raw_data\\TGN-AE-ICSFlow.txt"
training = "TGN50-Vanilla16"
model = "TGN50-Vanilla16"

# --- Regex patterns ---
EPOCH_START = re.compile(r"INFO:root:start (\d+) epoch")
TRAINING_LOSS = re.compile(r"INFO:root:Train reconstruction loss: ([0-9\.eE+-]+)")
VAL_LOSS = re.compile(r"INFO:root:Val reconstruction loss: ([0-9\.eE+-]+)")
#VAL_AP = re.compile(r"INFO:root:val ap: ([0-9\.eE+-]+)")


def parse_log(path):
    """
    Parse a single training log file and extract epoch, loss, val_auc, val_ap.
    Returns:
        pandas.DataFrame with columns ['epoch','loss','val_auc','val_ap']
    """
    records = []
    with open(path, 'r') as f:
        epoch = loss = auc = ap = None
        for line in f:
            m = EPOCH_START.search(line)
            if m:
                if epoch is not None and loss is not None:
                    records.append((epoch, loss, auc, ap))
                epoch = int(m.group(1))
                loss = auc = ap = None
                continue
            m = TRAINING_LOSS.search(line)
            if m:
                loss = float(m.group(1)); continue
            m = VAL_LOSS.search(line)
            if m:
                auc = float(m.group(1)); continue
            # m = VAL_AP.search(line)
            # if m:
            #     ap = float(m.group(1)); continue

    # append last epoch if present
    if epoch is not None and loss is not None:
        records.append((epoch, loss, auc, ap))

    return pd.DataFrame(records, columns=['epoch','loss','val_auc','val_ap'])


def plot_loss_and_ap(df):
    """
    Plot training loss (blue) and validation AP (yellow) on the same axes over epochs.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(df['epoch'], df['loss'], marker='o', color='blue', label='Training Loss')
    #plt.plot(df['epoch'], df['val_ap'], marker='s', color='green', label='Validation AP')
    plt.plot(df['epoch'], df['val_auc'], marker='^', color='orange', label='Validation Loss')
    plt.title(f"TGN50-Vanilla16 Training and Validation on ICS-Flow.")
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(f"C:\\Users\\conno\\OneDrive\\Documents\\Level_5_UofG\\Project\\ics_flow_graphs\\{training}.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


if __name__ == '__main__':
    # parse the single log file
    df = parse_log(filename)
    if df.empty:
        print(f"No epochs parsed from {filename}. Check the file and patterns.")
    else:
        plot_loss_and_ap(df)
        print("Done.")
