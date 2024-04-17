import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

similarity_transcriptome = cosine_similarity(alligned_single_transcriptome)
similarity_metagenome = cosine_similarity(alligned_single_metagenome)

label_to_index = pd.read_csv(r"label_to_index.csv", header = None)

similarity_df1 = pd.DataFrame(similarity_transcriptome, index=label_to_index.values, columns=label_to_index.values)
similarity_df2 = pd.DataFrame(similarity_metagenome, index=label_to_index.values, columns=label_to_index.values)

import numpy as np

def normalize_matrix(matrix):
    """ Normalize the matrix so that the sum of each row is 1. """
    return matrix / matrix.sum(axis=1, keepdims=True)

def update_similarity(W, W_other, K=20):
    """
    Update similarity matrix W based on another similarity matrix W_other.
    Uses the adaptive neighbor approach, selecting the top K neighbors.
    """
    n = W.shape[0]
    W_new = np.zeros_like(W)
    for i in range(n):
        # Get indices of the top K neighbors from W_other
        neighbors_idx = np.argsort(W_other[i])[::-1][:K + 1]
        
        # Calculate new weights for W using only these top K neighbors
        W_new[i, neighbors_idx] = W[i, neighbors_idx]
        
    # Normalize the new matrix
    W_new = normalize_matrix(W_new)
    return W_new

def similarity_network_fusion(W1, W2, K=20, t=20):
    """
    Fuse two similarity matrices using SNF.
    
    :param W1, W2: The similarity matrices to fuse.
    :param K: Number of neighbors to consider.
    :param t: Number of iterations.
    :return: Fused similarity matrix.
    """
    # Normalize initial matrices
    W1 = normalize_matrix(W1)
    W2 = normalize_matrix(W2)
    
    for _ in range(t):
        W1_old = W1
        W2_old = W2
        
        W1 = update_similarity(W1_old, W2_old, K)
        W2 = update_similarity(W2_old, W1_old, K)
    
    # Final fusion
    return (W1 + W2) / 2


W_fused = similarity_network_fusion(similarity_df1.values, similarity_df2.values, K=20, t=20)

import networkx as nx

def create_network_from_similarity_matrix(fused_similarity_matrix, labels):
    G = nx.Graph()
    n = fused_similarity_matrix.shape[0]  # Number of nodes

    # Add nodes with labels
    for i in range(n):
        G.add_node(i, label=labels[i])

    # Add edges between all pairs of nodes
    for i in range(n):
        for j in range(i + 1, n):  # Only consider upper triangle for undirected graph
            if fused_similarity_matrix[i, j] > 0:  # Optionally filter by a threshold
                G.add_edge(i, j, weight=fused_similarity_matrix[i, j])

    return G

G = create_network_from_similarity_matrix(W_fused, label_to_index.values)

def visualise_network(G):
  pos = nx.spring_layout(G, seed=42)  # for consistent layout
  nx.draw(G, pos, node_size=50, with_labels=False, node_color='blue', edge_color='gray')
  plt.title('Fused Omics Similarity Network')
  plt.show()

def calculate_network_measures(G):
  degree_centrality = nx.degree_centrality(G)
  betweenness_centrality = nx.betweenness_centrality(G)
  closeness_centrality = nx.closeness_centrality(G)

top_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
print("Top 5 nodes by degree centrality:", top_degree)

top_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
print("Top 5 nodes by betweenness centrality:", top_betweenness)

top_closeness = sorted(closeness_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
print("Top 5 nodes by closeness centrality:", top_closeness)