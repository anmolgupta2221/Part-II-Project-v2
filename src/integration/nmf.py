import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
import seaborn as sns

def complete_nmf(combined_data):
    model = NMF(n_components=10, init='random', random_state=0)
    W = model.fit_transform(combined_data)
    H = model.components_
    return W,H

W,H = complete_nmf(alligned_combined_df)

def heatmap_basis_vectors(W):
    plt.figure(figsize=(10, 8))
    sns.heatmap(W, annot=False)
    plt.title("Heatmap of the NMF Basis Matrix (W)")
    plt.xlabel("Components")
    plt.ylabel("Samples")
    plt.show()

label_to_index = pd.read_csv(r"label_to_index.csv", header = None)

def dendo_plotter(cluster_type):
    Z = linkage(W, cluster_type)
    plt.figure(figsize=(10, 7))
    dendrogram(Z, labels = label_to_index.values, leaf_rotation=90, color_threshold=0)
    plt.title("Dendrogram of NMF Components")
    plt.xlabel("Sample Type")
    plt.ylabel("Distance")
    plt.show()

dendo_plotter('ward')
dendo_plotter('single')
dendo_plotter('complete')
dendo_plotter('average')