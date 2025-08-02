import math
import pandas as pd
import os
import logging
import time
import sys
import argparse
import torch
import numpy as np
import pickle
from pathlib import Path
import torch.nn.functional as F
import time

from model.tgn import TGN
from utils.utils import get_neighbor_finder, Autoencoder, SparseAutoencoder, VariationalAutoencoder
from utils.data_processing import get_data, compute_time_statistics

# For consistency, same as other training files
torch.manual_seed(0)
np.random.seed(0)

### Parser stuff kept mostly same as training files bc I'm not removing any functionality, added hidden dim flag
parser = argparse.ArgumentParser('TGN autoencoder training')
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='wikipedia')
parser.add_argument('--edge_features', type=int, help='Path to edge features file',default=85)
parser.add_argument('--bs', type=int, default=50, help='Batch_size')
parser.add_argument('--prefix', type=str, default='', help='Prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=1, help='Number of epochs')
parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')
parser.add_argument('--lr', type=float, default=0.0, help='Learning rate')
parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')
parser.add_argument('--drop_out', type=float, default=0, help='Dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=100, help='Dimensions of the node embedding')
parser.add_argument('--time_dim', type=int, default=100, help='Dimensions of the time embedding')
parser.add_argument('--backprop_every', type=int, default=1, help='Every how many batches to backprop')
parser.add_argument('--use_memory', action='store_true', default=True,
                    help='Whether to augment the model with a node memory')
parser.add_argument('--embedding_module', type=str, default="graph_attention", choices=[
    "graph_attention", "graph_sum", "identity", "time"], help='Type of embedding module')
parser.add_argument('--message_function', type=str, default="identity", choices=[
    "mlp", "identity"], help='Type of message function')
parser.add_argument('--memory_updater', type=str, default="gru", choices=[
    "gru", "rnn"], help='Type of memory updater')
parser.add_argument('--aggregator', type=str, default="mean", choices=["last", "mean", "attention"], help='Type of message aggregator')
parser.add_argument('--memory_update_at_end', action='store_true',
                    help='Whether to update memory at the end or at the start of the batch')
parser.add_argument('--message_dim', type=int, default=100, help='Dimensions of the messages')
parser.add_argument('--memory_dim', type=int, default=172, help='Dimensions of the memory for each user')
parser.add_argument('--different_new_nodes', action='store_true',
                    help='Whether to use disjoint set of new nodes for train and val')
parser.add_argument('--uniform', action='store_true',
                    help='take uniform sampling from temporal neighbors')
parser.add_argument('--randomize_features', action='store_true',
                    help='Whether to randomize node features')
parser.add_argument('--use_destination_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the destination node as part of the message')
parser.add_argument('--use_source_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the source node as part of the message')
parser.add_argument('--dyrep', action='store_true',
                    help='Whether to run the dyrep model')
parser.add_argument('--hidden_dim', type=int, default=16, help='Hidden dimension for autoencoder')
parser.add_argument('--learnable', action="store_true",
                    help="Whether Message Aggregator is learnable module")
parser.add_argument('--add_cls_token', action="store_true",
                    help="Apend cls token like BERT to represent the final message")
parser.add_argument('--autoencoder', type=str, default="vanilla", choices=["vanilla", "variational", "sparse"],
                    help="Type of autoencoder to use: vanilla, variational, or sparse")
parser.add_argument('--latent_dim', type=int, default=64, help='Latent dimension for VAE')
parser.add_argument('--beta', type=float, default=1.0, help='Beta parameter for KL divergence weight (beta-VAE)')
parser.add_argument('--sparsity_weight', type=float, default=0.01, help='Weight for sparsity loss')
parser.add_argument('--sparsity_target', type=float, default=0.05, help='Target sparsity level')

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim
USE_MEMORY = args.use_memory
MESSAGE_DIM = args.message_dim
MEMORY_DIM = args.memory_dim
HIDDEN_DIM = args.hidden_dim
EDGE_FEAT = args.edge_features

Path("./saved_models/").mkdir(parents=True, exist_ok=True)
Path("./saved_checkpoints/").mkdir(parents=True, exist_ok=True)

if args.autoencoder == "vanilla":
    MODEL_SAVE_PATH = f'./saved_models/{args.prefix}_{args.embedding_module}-{args.aggregator}-{args.memory_dim}-vanilla{args.hidden_dim}-autoencoder.pth'
    get_checkpoint_path = lambda epoch: f'./saved_checkpoints/{args.prefix}_{args.embedding_module}-{args.aggregator}-{args.memory_dim}-vanilla{args.hidden_dim}-autoencoder-{epoch}.pth'
elif args.autoencoder == "variational":
    MODEL_SAVE_PATH = f'./saved_models/{args.prefix}_{args.embedding_module}-{args.aggregator}-{args.memory_dim}-var{args.hidden_dim}-autoencoder.pth'
    get_checkpoint_path = lambda epoch: f'./saved_checkpoints/{args.prefix}_{args.embedding_module}-{args.aggregator}-{args.memory_dim}-var{args.hidden_dim}-autoencoder-{epoch}.pth'
elif args.autoencoder == "sparse":
    MODEL_SAVE_PATH = f'./saved_models/{args.prefix}_{args.embedding_module}-{args.aggregator}-{args.memory_dim}-sparse{args.hidden_dim}-autoencoder.pth'
    get_checkpoint_path = lambda epoch: f'./saved_checkpoints/{args.prefix}_{args.embedding_module}-{args.aggregator}-{args.memory_dim}-sparse{args.hidden_dim}-autoencoder-{epoch}.pth'

### set up logger (same as other training files)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
Path("log/").mkdir(parents=True, exist_ok=True)
fh = logging.FileHandler('log/{}.log'.format(str(time.time())))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)

### Extract data for training, validation and testing
node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, \
new_node_test_data = get_data(DATA,
                              different_new_nodes_between_val_and_test=args.different_new_nodes, 
                              randomize_features=args.randomize_features)

full_ngh_finder = get_neighbor_finder(full_data, args.uniform)

device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
print(device_string)
device = torch.device(device_string)

# Compute time statistics
mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
    compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)

if args.autoencoder == "vanilla":
    results_path = "results/{}_evaluation_autoencoder.pkl".format(args.prefix)
    Path("results/").mkdir(parents=True, exist_ok=True)
elif args.autoencoder == "variational":
    results_path = "results/{}_evaluation_variational_autoencoder.pkl".format(args.prefix)
    Path("results/").mkdir(parents=True, exist_ok=True)
elif args.autoencoder == "sparse":
    results_path = "results/{}_evaluation_sparse_autoencoder.pkl".format(args.prefix)
    Path("results/").mkdir(parents=True, exist_ok=True)


# Initialize TGN Model
tgn = TGN(neighbor_finder=full_ngh_finder, node_features=node_features,
            edge_features=edge_features, device=device,
            n_layers=NUM_LAYER,
            n_heads=NUM_HEADS, dropout=DROP_OUT, use_memory=USE_MEMORY,
            message_dimension=MESSAGE_DIM, memory_dimension=MEMORY_DIM,
            memory_update_at_start=not args.memory_update_at_end,
            embedding_module_type=args.embedding_module,
            message_function=args.message_function,
            aggregator_type=args.aggregator,
            memory_updater_type=args.memory_updater,
            n_neighbors=NUM_NEIGHBORS,
            mean_time_shift_src=mean_time_shift_src, std_time_shift_src=std_time_shift_src,
            mean_time_shift_dst=mean_time_shift_dst, std_time_shift_dst=std_time_shift_dst,
            use_destination_embedding_in_message=args.use_destination_embedding_in_message,
            use_source_embedding_in_message=args.use_source_embedding_in_message,
            dyrep=args.dyrep,
            learnable=args.learnable,
            add_cls_token=args.add_cls_token)
# No criterion or optimiser here, this is done below as AE needs one too
tgn = tgn.to(device)


#input_dim = NODE_DIM + NODE_DIM + edge_features.shape[1]
input_dim = 50 + 50 + EDGE_FEAT # 185

num_instance = len(full_data.sources)
num_batch = math.ceil(num_instance / BATCH_SIZE)

logger.info('num of instances: {}'.format(num_instance))
logger.info('num of batches per epoch: {}'.format(num_batch))    
idx_list = np.arange(num_instance)

logger.info('Loading saved TGN model')
model_path = f'./saved_models/{args.prefix}_{args.embedding_module}-{args.aggregator}-{args.memory_dim}.pth'
tgn.load_state_dict(torch.load(model_path))
tgn.eval()
logger.info('TGN models loaded')
logger.info('Start evaluation')


if args.autoencoder == "vanilla":
    autoencoder = Autoencoder(input_dim, args.hidden_dim, DROP_OUT)
    autoencoder = autoencoder.to(device)
    reconstruction_criterion = torch.nn.MSELoss(reduction='none')
elif args.autoencoder == "variational":
    autoencoder = VariationalAutoencoder(input_dim, args.hidden_dim, args.latent_dim, DROP_OUT)
    autoencoder = autoencoder.to(device)
    reconstruction_criterion = torch.nn.MSELoss(reduction='none')
elif args.autoencoder == "sparse":
    autoencoder = SparseAutoencoder(input_dim, args.hidden_dim, DROP_OUT, args.sparsity_weight, args.sparsity_target)
    autoencoder = autoencoder.to(device)
    reconstruction_criterion = torch.nn.MSELoss(reduction='none')
else:
    raise ValueError("Unknown autoencoder type: {}".format(args.autoencoder))
# Load the autoencoder model if it exists
if os.path.exists(MODEL_SAVE_PATH):
    checkpoint = torch.load(MODEL_SAVE_PATH, weights_only=False)
    logger.info('Loading saved autoencoder model')
    autoencoder.load_state_dict(checkpoint['autoencoder_state_dict'])
    autoencoder.eval()


reconstruction_losses = []
epoch_times = []
total_epoch_times = []
test_kl_loss_vals = []

start_epoch = time.time()

if USE_MEMORY:
    tgn.memory.__init_memory__()

tgn.eval()
autoencoder.eval()

#load orginal edge feature
df = pd.read_csv(f"data/{DATA}.csv", header=None)

for k in range(0, num_batch):
    if k >= num_batch:
        continue

    start_idx = k * BATCH_SIZE
    end_idx = min(num_instance, start_idx + BATCH_SIZE)
    sources_batch = full_data.sources[start_idx:end_idx]
    destinations_batch = full_data.destinations[start_idx:end_idx]
    edge_idxs_batch = full_data.edge_idxs[start_idx:end_idx]
    timestamps_batch = full_data.timestamps[start_idx:end_idx]

    size = len(sources_batch)

    source_embeddings, destination_embeddings, _ = tgn.compute_temporal_embeddings(
                    sources_batch, destinations_batch, destinations_batch,
                    timestamps_batch, edge_idxs_batch, NUM_NEIGHBORS)
    
    # print(edge_idxs_batch)
    # print(df.shape)
    # print(df.iloc[0].head(10))

    edge_features_batch_df = df.iloc[edge_idxs_batch - 1]
    edge_features_batch = torch.from_numpy(edge_features[edge_idxs_batch]).float().to(device)

    # Concatenate source embeddings, dest embeddings, and edge features into input
    input_representation = torch.cat([source_embeddings, destination_embeddings, edge_features_batch], dim=1)

    input_representation = input_representation.to(device)

    # reconstruction_output = autoencoder(input_representation)

    # reconstruction_loss = reconstruction_criterion(reconstruction_output, input_representation)

    start_time = time.time()

    if args.autoencoder == "vanilla":
        reconstruction_output = autoencoder(input_representation)
        # Compute per-sample reconstruction loss
        per_sample_losses = F.mse_loss(reconstruction_output, input_representation, reduction='none').sum(dim=1)

    elif args.autoencoder == "variational":
        reconstruction_output, mu, logvar = autoencoder(input_representation)
        # Per-sample reconstruction loss
        recon_loss_per_sample = F.mse_loss(reconstruction_output, input_representation, reduction='none').sum(dim=1)
        # Per-sample KL divergence
        kl_loss_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        per_sample_losses = recon_loss_per_sample + args.beta * kl_loss_per_sample

    elif args.autoencoder == "sparse":
        reconstruction_output, encoded = autoencoder(input_representation)
        # Per-sample reconstruction loss
        recon_loss_per_sample = F.mse_loss(reconstruction_output, input_representation, reduction='none').sum(dim=1)
        # Per-sample sparsity violation (L1 norm of activations)
        sparsity_violation_per_sample = torch.norm(encoded, p=1, dim=1)
        # Combined loss: reconstruction + weighted sparsity violation
        per_sample_losses = recon_loss_per_sample + args.sparsity_weight * sparsity_violation_per_sample

    final_time = time.time()

    logger.info(f"Batch {k+1}/{num_batch} processed in {final_time - start_time:.4f} seconds")

    with open("results/{}_{}_reconstruction_losses.csv".format(args.prefix, args.autoencoder), "ab") as f:
        # output: id, ts, label, reconstruction_loss
        for i in range(size):
            row = edge_features_batch_df.iloc[i]
            f.write(f"{row.iloc[0]},{row.iloc[3]},{row.iloc[4]},{per_sample_losses[i].item()}\n".encode())
