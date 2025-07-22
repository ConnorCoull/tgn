import math
import logging
import time
import sys
import argparse
import torch
import numpy as np
import pickle
from pathlib import Path

from model.tgn import TGN
from utils.utils import EarlyStopMonitor, get_neighbor_finder, Autoencoder
from utils.data_processing import get_data, compute_time_statistics

# For consistency, same as other training files
torch.manual_seed(0)
np.random.seed(0)

### Parser stuff kept mostly same as training files bc I'm not removing any functionality, added hidden dim flag
parser = argparse.ArgumentParser('TGN autoencoder training')
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='wikipedia')
parser.add_argument('--bs', type=int, default=200, help='Batch_size')
parser.add_argument('--prefix', type=str, default='', help='Prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=50, help='Number of epochs')
parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')
parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=100, help='Dimensions of the node embedding')
parser.add_argument('--time_dim', type=int, default=100, help='Dimensions of the time embedding')
parser.add_argument('--backprop_every', type=int, default=1, help='Every how many batches to backprop')
parser.add_argument('--use_memory', action='store_true',
                    help='Whether to augment the model with a node memory')
parser.add_argument('--embedding_module', type=str, default="graph_attention", choices=[
    "graph_attention", "graph_sum", "identity", "time"], help='Type of embedding module')
parser.add_argument('--message_function', type=str, default="identity", choices=[
    "mlp", "identity"], help='Type of message function')
parser.add_argument('--memory_updater', type=str, default="gru", choices=[
    "gru", "rnn"], help='Type of memory updater')
parser.add_argument('--aggregator', type=str, default="attention", choices=["last", "mean", "attention"], help='Type of message aggregator')
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
parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension for autoencoder')
parser.add_argument('--learnable', action="store_true",
                    help="Whether Message Aggregator is learnable module")
parser.add_argument('--add_cls_token', action="store_true",
                    help="Apend cls token like BERT to represent the final message")

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

Path("./saved_models/").mkdir(parents=True, exist_ok=True)
Path("./saved_checkpoints/").mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.data}-autoencoder.pth'
get_checkpoint_path = lambda epoch: f'./saved_checkpoints/{args.prefix}-{args.data}-autoencoder-{epoch}.pth'

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
# this is copied from other training files, relies on default so I don't think I need to change anything
# Why would I need to randomise node features though?

# Initialize training neighbor finder to retrieve temporal graph
train_ngh_finder = get_neighbor_finder(train_data, args.uniform)

# Initialize validation and test neighbor finder to retrieve temporal graph
full_ngh_finder = get_neighbor_finder(full_data, args.uniform)

# No negative samplers - not needed for autoencoder

# Set device
device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
print(device_string)
device = torch.device(device_string)

# Compute time statistics
mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
    compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)

for i in range(args.n_runs):
    results_path = "results/{}_{}_autoencoder.pkl".format(args.prefix, i) if i > 0 else "results/{}_autoencoder.pkl".format(args.prefix)
    Path("results/").mkdir(parents=True, exist_ok=True)

    # Initialize TGN Model
    tgn = TGN(neighbor_finder=train_ngh_finder, node_features=node_features,
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
    input_dim = 50 + 50 + 85 # 185

    num_instance = len(train_data.sources)
    num_batch = math.ceil(num_instance / BATCH_SIZE)

    logger.info('num of training instances: {}'.format(num_instance))
    logger.info('num of batches per epoch: {}'.format(num_batch))    
    idx_list = np.arange(num_instance)

    logger.info('Loading saved TGN model')
    model_path = f'./saved_models/{args.prefix}-{DATA}.pth'
    tgn.load_state_dict(torch.load(model_path))
    tgn.eval()
    logger.info('TGN models loaded')
    logger.info('Start training node classification task')

    autoencoder = Autoencoder(input_dim, args.hidden_dim, DROP_OUT)
    autoencoder = autoencoder.to(device)
    autoencoder_optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LEARNING_RATE)
    reconstruction_criterion = torch.nn.MSELoss()

    val_reconstruction_losses = []
    train_reconstruction_losses = []
    epoch_times = []
    total_epoch_times = []

    early_stopper = EarlyStopMonitor(max_round=args.patience)
    
    for epoch in range(NUM_EPOCH):
        start_epoch = time.time()
        ### Training

        # Reinitialize memory of the model at the start of each epoch
        if USE_MEMORY:
            tgn.memory.__init_memory__()

        # Train using only training graph
        tgn.eval()
        autoencoder.train()
        train_losses = []

        logger.info('start {} epoch'.format(epoch))
        
        for k in range(0, num_batch, args.backprop_every):
            loss = 0
            autoencoder_optimizer.zero_grad()

            # Custom loop to allow to perform backpropagation only every a certain number of batches
            for j in range(args.backprop_every):
                batch_idx = k + j

                if batch_idx >= num_batch:
                    continue

                start_idx = batch_idx * BATCH_SIZE
                end_idx = min(num_instance, start_idx + BATCH_SIZE)
                sources_batch = train_data.sources[start_idx:end_idx]
                destinations_batch = train_data.destinations[start_idx:end_idx]
                edge_idxs_batch = train_data.edge_idxs[start_idx:end_idx]
                timestamps_batch = train_data.timestamps[start_idx:end_idx]

                size = len(sources_batch)
                # Again most of this is the same, removed negs

                # Removed the pos/neg label stuff too

                # Set models to training mode
                #autoencoder_optimizer.zero_grad()

                ### THE ACTUAL TRAINING SECTION OF THE TRAINING CODE!!!!!!!

                # Get embeddings from TGN
                source_embeddings, destination_embeddings, _ = tgn.compute_temporal_embeddings(
                    sources_batch, destinations_batch, destinations_batch,
                    timestamps_batch, edge_idxs_batch, NUM_NEIGHBORS)

                # Get edge features for this batch
                edge_features_batch = torch.from_numpy(edge_features[edge_idxs_batch]).float().to(device)

                # Concatenate source embeddings, dest embeddings, and edge features into input
                input_representation = torch.cat([source_embeddings, destination_embeddings, edge_features_batch], dim=1)

                # Check input sizes here
                # create a tensor of shape (1, len(input_representation)) where each value is the length of the input_representation[x]
                #print(len(input_representation[1]), input_representation.shape)

                # Autoencoder reconstruction
                reconstructed = autoencoder(input_representation)

                # Calc reconstruction loss
                reconstruction_loss = reconstruction_criterion(reconstructed, input_representation)
                loss += reconstruction_loss

            loss /= args.backprop_every

            loss.backward()
            autoencoder_optimizer.step()
            train_losses.append(loss.item())

            # Detach memory after 'args.backprop_every' number of batches so we don't backpropagate to
            # the start of time
            if USE_MEMORY:
                tgn.memory.detach_memory()

        epoch_time = time.time() - start_epoch
        epoch_times.append(epoch_time)

        ### Validation
        # Validation uses the full graph
        tgn.set_neighbor_finder(full_ngh_finder)

        # if USE_MEMORY:
        #     # Backup memory at the end of training, so later we can restore it and use it for the
        #     # validation on unseen nodes
        #     train_memory_backup = tgn.memory.backup_memory()

        # Validation reconstruction loss
        autoencoder.eval()
        val_losses = []

        # Do I need some eval_autoencoder_reconstruction(...) function here?
        
        with torch.no_grad():
            val_num_batch = math.ceil(len(val_data.sources) / BATCH_SIZE)
            for k in range(val_num_batch):
                start_idx = k * BATCH_SIZE
                end_idx = min(len(val_data.sources), start_idx + BATCH_SIZE)
                
                val_sources = val_data.sources[start_idx:end_idx]
                val_destinations = val_data.destinations[start_idx:end_idx]
                val_edge_idxs = val_data.edge_idxs[start_idx:end_idx]
                val_timestamps = val_data.timestamps[start_idx:end_idx]

                # Get embeddings
                val_source_embeddings, val_destination_embeddings, _ = tgn.compute_temporal_embeddings(
                    val_sources, val_destinations, val_destinations,
                    val_timestamps, val_edge_idxs, NUM_NEIGHBORS)

                # Get edge features
                val_edge_features_batch = torch.from_numpy(edge_features[val_edge_idxs]).float().to(device)

                # Concatenate
                val_input_representation = torch.cat([val_source_embeddings, val_destination_embeddings, val_edge_features_batch], dim=1)

                # Reconstruct
                val_reconstructed = autoencoder(val_input_representation)

                # Loss
                val_reconstruction_loss = reconstruction_criterion(val_reconstructed, val_input_representation)
                val_losses.append(val_reconstruction_loss.item())

        avg_val_loss = np.mean(val_losses)
        val_reconstruction_losses.append(avg_val_loss)
        train_reconstruction_losses.append(np.mean(train_losses))

        # Save temporary results to disk
        pickle.dump({
            "train_reconstruction_losses": train_reconstruction_losses,
            "val_reconstruction_losses": val_reconstruction_losses,
            "epoch_times": epoch_times,
            "total_epoch_times": total_epoch_times
        }, open(results_path, "wb"))

        total_epoch_time = time.time() - start_epoch
        total_epoch_times.append(total_epoch_time)

        logger.info('epoch: {} took {:.2f}s'.format(epoch, total_epoch_time))
        logger.info('Train reconstruction loss: {:.6f}'.format(np.mean(train_losses)))
        logger.info('Val reconstruction loss: {:.6f}'.format(avg_val_loss))

        # Early stopping based on validation reconstruction loss (lower is better)
        if early_stopper.early_stop_check(-avg_val_loss):  # Negative because EarlyStopMonitor expects higher=better
            logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
            logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
            best_model_path = get_checkpoint_path(early_stopper.best_epoch)
            checkpoint = torch.load(best_model_path)
            tgn.load_state_dict(checkpoint['tgn_state_dict'])
            autoencoder.load_state_dict(checkpoint['autoencoder_state_dict'])
            logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
            tgn.eval()
            autoencoder.eval()
            break
        else:
            torch.save({
                'tgn_state_dict': tgn.state_dict(),
                'autoencoder_state_dict': autoencoder.state_dict(),
                'epoch': epoch
            }, get_checkpoint_path(epoch))

    # Training has finished, backup memory for testing
    if USE_MEMORY:
        val_memory_backup = tgn.memory.backup_memory()

    ### Test
    autoencoder.eval()
    
    test_losses = []
    with torch.no_grad():
        test_num_batch = math.ceil(len(test_data.sources) / BATCH_SIZE)
        for k in range(test_num_batch):
            start_idx = k * BATCH_SIZE
            end_idx = min(len(test_data.sources), start_idx + BATCH_SIZE)
            
            test_sources = test_data.sources[start_idx:end_idx]
            test_destinations = test_data.destinations[start_idx:end_idx]
            test_edge_idxs = test_data.edge_idxs[start_idx:end_idx]
            test_timestamps = test_data.timestamps[start_idx:end_idx]

            # Get embeddings
            test_source_embeddings, test_destination_embeddings, _ = tgn.compute_temporal_embeddings(
                test_sources, test_destinations, test_destinations,
                test_timestamps, test_edge_idxs, NUM_NEIGHBORS)

            # Get edge features
            test_edge_features_batch = torch.from_numpy(edge_features[test_edge_idxs]).float().to(device)

            # Concatenate
            test_input_representation = torch.cat([test_source_embeddings, test_destination_embeddings, test_edge_features_batch], dim=1)

            # Reconstruct
            test_reconstructed = autoencoder(test_input_representation)

            # Loss
            test_reconstruction_loss = reconstruction_criterion(test_reconstructed, test_input_representation)
            test_losses.append(test_reconstruction_loss.item())

    avg_test_loss = np.mean(test_losses)

    logger.info('Test reconstruction loss: {:.6f}'.format(avg_test_loss))
    
    # Save final results for this run
    pickle.dump({
        "train_reconstruction_losses": train_reconstruction_losses,
        "val_reconstruction_losses": val_reconstruction_losses,
        "test_reconstruction_loss": avg_test_loss,
        "epoch_times": epoch_times,
        "total_epoch_times": total_epoch_times
    }, open(results_path, "wb"))

    logger.info('Saving TGN and Autoencoder models')
    if USE_MEMORY:
        # Restore memory at the end of validation
        tgn.memory.restore_memory(val_memory_backup)
    
    torch.save({
        'tgn_state_dict': tgn.state_dict(),
        'autoencoder_state_dict': autoencoder.state_dict(),
        'args': args,
        'input_dim': input_dim
    }, MODEL_SAVE_PATH)
    logger.info('Models saved')
