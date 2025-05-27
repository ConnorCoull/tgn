class TGN(torch.nn.Module):

    """
    Coull, 2025
    Using TGNs to detect malious flows of data in an ICS network.
    Takes in the node and a message and predicts if the message is 0, benign, or 1, malicious.
    """
    def compute_edge_probabilities_malicious(self, source_nodes, destination_nodes, edge_times,
                                             edge_idxs, n_neighbors=None): # n_neighbors can be passed or use self.n_neighbors
        """
        Computes probabilities of edges between sources and destination on whether or not they are malicious.

        Nodes represent networked devices in an ICS network.
        Edges represent messages sent between devices.

        An edge can be either 0 (benign), or 1 (malicious).
        This means that the output of the model can represent the probability of the edge being malicious.

        :param source_nodes [batch_size]: Source device IDs.
        :param destination_nodes [batch_size]: Destination device IDs.
        :param edge_times [batch_size]: Timestamp of the message/flow.
        :param edge_idxs [batch_size]: Index of the edge features for the message/flow.
        :param n_neighbors [scalar, optional]: Number of temporal neighbors to consider. Uses default if None.
        :return: Probabilities [batch_size] for each edge being malicious.
        """
        n_samples = len(source_nodes)
  
        # Get temporal embeddings for source and destination nodes
        source_node_embedding, destination_node_embedding, _ = self.compute_temporal_embeddings(
            source_nodes, destination_nodes, destination_nodes, edge_times, edge_idxs, n_neighbors)
        
        # Get edge features
        edge_features = self.edge_raw_features[edge_idxs]
        
        # Combine node embeddings with edge features
        combined_features = torch.cat([source_node_embedding, destination_node_embedding, edge_features], dim=1)

        score = self.malicious_classifier(combined_features, torch.zeros(combined_features.shape[0], 0).to(self.device)).squeeze(dim=1)

        return score.sigmoid()

