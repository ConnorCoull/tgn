import numpy as np
import torch
from collections import defaultdict

class MergeLayer(torch.nn.Module):
  def __init__(self, dim1, dim2, dim3, dim4):
    super().__init__()
    self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
    self.fc2 = torch.nn.Linear(dim3, dim4)
    self.act = torch.nn.ReLU()

    torch.nn.init.xavier_normal_(self.fc1.weight)
    torch.nn.init.xavier_normal_(self.fc2.weight)

  def forward(self, x1, x2):
    x = torch.cat([x1, x2], dim=1)
    h = self.act(self.fc1(x))
    return self.fc2(h)

"""
This class takes in 2 node embeddings and edge features and 
calculates a score between 0 and 1 for how likely the edge is to be malicious.
""" 
class MaliciousClassifier(torch.nn.Module):
    def __init__(self, total_dim, hidden_dim=128, dropout=0.1):
        super().__init__()
        #total_dim = src_dim + dst_dim + edge_feat_dim
        
        # Two-layer MLP with dropout
        # Two layers bc I feel there is a lot of features here
        self.fc1 = torch.nn.Linear(total_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.act = torch.nn.ReLU()
        
        # Initialize weights - copied from MergeLayer
        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)
    
    def forward(self, src_emb, dst_emb, edge_feat):
        x = torch.cat([src_emb, dst_emb, edge_feat], dim=1)
        
        h = self.act(self.fc1(x))
        h = self.dropout(h)
        return self.fc2(h).squeeze(dim=1)


class MLP(torch.nn.Module):
  def __init__(self, dim, drop=0.3):
    super().__init__()
    self.fc_1 = torch.nn.Linear(dim, 80)
    self.fc_2 = torch.nn.Linear(80, 10)
    self.fc_3 = torch.nn.Linear(10, 1)
    self.act = torch.nn.ReLU()
    self.dropout = torch.nn.Dropout(p=drop, inplace=False)

  def forward(self, x):
    x = self.act(self.fc_1(x))
    x = self.dropout(x)
    x = self.act(self.fc_2(x))
    x = self.dropout(x)
    return self.fc_3(x).squeeze(dim=1)


class EarlyStopMonitor(object):
  def __init__(self, max_round=3, higher_better=True, tolerance=1e-10):
    self.max_round = max_round
    self.num_round = 0

    self.epoch_count = 0
    self.best_epoch = 0

    self.last_best = None
    self.higher_better = higher_better
    self.tolerance = tolerance

  def early_stop_check(self, curr_val):
    if not self.higher_better:
      curr_val *= -1
    if self.last_best is None:
      self.last_best = curr_val
    elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
      self.last_best = curr_val
      self.num_round = 0
      self.best_epoch = self.epoch_count
    else:
      self.num_round += 1

    self.epoch_count += 1

    return self.num_round >= self.max_round


class RandEdgeSampler(object):
  def __init__(self, src_list, dst_list, seed=None):
    self.seed = None
    self.src_list = np.unique(src_list)
    self.dst_list = np.unique(dst_list)

    if seed is not None:
      self.seed = seed
      self.random_state = np.random.RandomState(self.seed)

  def sample(self, size):
    if self.seed is None:
      src_index = np.random.randint(0, len(self.src_list), size)
      dst_index = np.random.randint(0, len(self.dst_list), size)
    else:

      src_index = self.random_state.randint(0, len(self.src_list), size)
      dst_index = self.random_state.randint(0, len(self.dst_list), size)
    return self.src_list[src_index], self.dst_list[dst_index]

  def reset_random_state(self):
    self.random_state = np.random.RandomState(self.seed)


def get_neighbor_finder(data, uniform, max_node_idx=None):
  max_node_idx = max(data.sources.max(), data.destinations.max()) if max_node_idx is None else max_node_idx
  adj_list = [[] for _ in range(max_node_idx + 1)]
  for source, destination, edge_idx, timestamp in zip(data.sources, data.destinations,
                                                      data.edge_idxs,
                                                      data.timestamps):
    adj_list[source].append((destination, edge_idx, timestamp))
    adj_list[destination].append((source, edge_idx, timestamp))

  return NeighborFinder(adj_list, uniform=uniform)


class NeighborFinder:
  def __init__(self, adj_list, uniform=False, seed=None):
    self.node_to_neighbors = []
    self.node_to_edge_idxs = []
    self.node_to_edge_timestamps = []

    for neighbors in adj_list:
      # Neighbors is a list of tuples (neighbor, edge_idx, timestamp)
      # We sort the list based on timestamp
      sorted_neighhbors = sorted(neighbors, key=lambda x: x[2])
      self.node_to_neighbors.append(np.array([x[0] for x in sorted_neighhbors]))
      self.node_to_edge_idxs.append(np.array([x[1] for x in sorted_neighhbors]))
      self.node_to_edge_timestamps.append(np.array([x[2] for x in sorted_neighhbors]))

    self.uniform = uniform

    if seed is not None:
      self.seed = seed
      self.random_state = np.random.RandomState(self.seed)

  def find_before(self, src_idx, cut_time):
    """
    Extracts all the interactions happening before cut_time for user src_idx in the overall interaction graph. The returned interactions are sorted by time.

    Returns 3 lists: neighbors, edge_idxs, timestamps

    """
    i = np.searchsorted(self.node_to_edge_timestamps[src_idx], cut_time)

    return self.node_to_neighbors[src_idx][:i], self.node_to_edge_idxs[src_idx][:i], self.node_to_edge_timestamps[src_idx][:i]

  def get_temporal_neighbor(self, source_nodes, timestamps, n_neighbors=20):
    """
    Given a list of users ids and relative cut times, extracts a sampled temporal neighborhood of each user in the list.

    Params
    ------
    src_idx_l: List[int]
    cut_time_l: List[float],
    num_neighbors: int
    """
    assert (len(source_nodes) == len(timestamps))

    tmp_n_neighbors = n_neighbors if n_neighbors > 0 else 1
    # NB! All interactions described in these matrices are sorted in each row by time
    neighbors = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.int32)  # each entry in position (i,j) represent the id of the item targeted by user src_idx_l[i] with an interaction happening before cut_time_l[i]
    edge_times = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.float32)  # each entry in position (i,j) represent the timestamp of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]
    edge_idxs = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.int32)  # each entry in position (i,j) represent the interaction index of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]

    for i, (source_node, timestamp) in enumerate(zip(source_nodes, timestamps)):
      source_neighbors, source_edge_idxs, source_edge_times = self.find_before(source_node,
                                                   timestamp)  # extracts all neighbors, interactions indexes and timestamps of all interactions of user source_node happening before cut_time

      if len(source_neighbors) > 0 and n_neighbors > 0:
        if self.uniform:  # if we are applying uniform sampling, shuffles the data above before sampling
          sampled_idx = np.random.randint(0, len(source_neighbors), n_neighbors)

          neighbors[i, :] = source_neighbors[sampled_idx]
          edge_times[i, :] = source_edge_times[sampled_idx]
          edge_idxs[i, :] = source_edge_idxs[sampled_idx]

          # re-sort based on time
          pos = edge_times[i, :].argsort()
          neighbors[i, :] = neighbors[i, :][pos]
          edge_times[i, :] = edge_times[i, :][pos]
          edge_idxs[i, :] = edge_idxs[i, :][pos]
        else:
          # Take most recent interactions
          source_edge_times = source_edge_times[-n_neighbors:]
          source_neighbors = source_neighbors[-n_neighbors:]
          source_edge_idxs = source_edge_idxs[-n_neighbors:]

          assert (len(source_neighbors) <= n_neighbors)
          assert (len(source_edge_times) <= n_neighbors)
          assert (len(source_edge_idxs) <= n_neighbors)

          neighbors[i, n_neighbors - len(source_neighbors):] = source_neighbors
          edge_times[i, n_neighbors - len(source_edge_times):] = source_edge_times
          edge_idxs[i, n_neighbors - len(source_edge_idxs):] = source_edge_idxs

    return neighbors, edge_idxs, edge_times
  
class Autoencoder(torch.nn.Module):
  def __init__(self, input_dim, hidden_dim=128, dropout=0.1):
    super(Autoencoder, self).__init__()
    self.encoder = torch.nn.Sequential(
      torch.nn.Linear(input_dim, hidden_dim),
      torch.nn.ReLU(),
      torch.nn.Dropout(p=dropout),
      torch.nn.Linear(hidden_dim, hidden_dim // 2),
      torch.nn.ReLU(),
      torch.nn.Dropout(p=dropout)
    )
    self.decoder = torch.nn.Sequential(
      torch.nn.Linear(hidden_dim // 2, hidden_dim),
      torch.nn.ReLU(),
      torch.nn.Dropout(p=dropout),
      torch.nn.Linear(hidden_dim, input_dim)
    )

  def forward(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded
  
# def get_negative_edges(sources_batch, destinations_batch, timestamps_batch):
#   sources_batch = np.array(sources_batch)
#   destinations_batch = np.array(destinations_batch)
#   timestamps_batch = np.array(timestamps_batch)

#   np.random.seed(2025)

#   # Set of all unique node IDs
#   all_nodes = np.unique(np.concatenate((sources_batch, destinations_batch)))
#   num_nodes = all_nodes.max() + 1 # +1 as first node ID is 0

#   fake_srcs = []
#   fake_dsts = []

#   unique_ts = np.unique(timestamps_batch)

#   for ts in unique_ts:
#     # Filter for timestamp
#     mask = timestamps_batch == ts
#     srcs = sources_batch[mask]
#     dsts = destinations_batch[mask]

#     # Positive edges
#     pos_edges = set(zip(srcs, dsts))

#     # Generate all possible (src, dst) pairs (excluding self-loops) as an IP won't talk to itself
#     grid_src, grid_dst = np.meshgrid(all_nodes, all_nodes)
#     all_possible_edges = np.vstack([grid_src.ravel(), grid_dst.ravel()]).T
#     all_possible_edges = all_possible_edges[grid_src.ravel() != grid_dst.ravel()]

#     all_possible_edges_set = set(map(tuple, all_possible_edges))
#     negative_edges = list(all_possible_edges_set - pos_edges)

#     # Sample same number of negatives as positives (or fewer if limited)
#     sample_size = len(sources_batch[mask])
#     sampled_negatives = np.random.choice(len(negative_edges), size=sample_size, replace=True)
#     sampled_edges = [negative_edges[i] for i in sampled_negatives]

#     # Append to results
#     fake_srcs.extend([src for src, _ in sampled_edges])
#     fake_dsts.extend([dst for _, dst in sampled_edges])

#   return fake_srcs, fake_dsts

def get_negative_edges(sources_batch, destinations_batch, timestamps_batch):
  np.random.seed(2025)

  sources = np.array(sources_batch)
  dests   = np.array(destinations_batch)
  times   = np.array(timestamps_batch)

  # 1) Precompute the global list of nodes
  all_nodes = np.unique(np.concatenate((sources, dests)))
  num_nodes = all_nodes.max() + 1

  # 2) For each (src, ts), record the set of forbiddens (i.e. observed dests)
  forbidden = defaultdict(set)
  for src, dst, ts in zip(sources, dests, times):
      forbidden[(src, ts)].add(dst)

  fake_srcs = []
  fake_dsts = []

  # 3) Now, for *each* original positive edge (in order!), sample a negative
  for src, ts in zip(sources, times):
      # the candidates are all_nodes except src itself and except forbidden[src,ts]
      forb = forbidden[(src, ts)]
      # note: self‐loop is forbidden too
      forb_with_self = forb | {src}

      # fast way to pick one at random:
      #   keep drawing until you hit a node not in the forbid set
      while True:
          neg_dst = np.random.randint(0, num_nodes)
          if neg_dst not in forb_with_self:
              break

      fake_srcs.append(src)
      fake_dsts.append(neg_dst)

  return np.array(fake_srcs), np.array(fake_dsts)
