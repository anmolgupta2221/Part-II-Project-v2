import torch
import numpy as np
from pyswarm import pso
from deap import creator, base, tools, algorithms
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.utils import resample
import pandas as pd
from UnionCom import UnionCom

# Function to load and filter data
def load_and_filter_data(x_filepath, y_filepath, metadata_filepath):
    x_data = pd.read_csv(x_filepath)
    y_data = pd.read_csv(y_filepath)
    metadata = pd.read_csv(metadata_filepath)

    # Merge metadata with datasets
    x_data = pd.merge(x_data, metadata, on='SampleID')
    y_data = pd.merge(y_data, metadata, on='SampleID')

    return x_data, y_data

# Function to run MMD-MA on filtered datasets
def run_mmd_ma_on_filtered_data(x_data, y_data):
    results = {}
    # Group by genotype, sex, and diet
    grouped_x = x_data.groupby(['Genotype', 'Sex', 'Diet'])
    grouped_y = y_data.groupby(['Genotype', 'Sex', 'Diet'])

    for (key, x_group) in grouped_x:
        if key in grouped_y.groups:  # Check if there is a matching y group
            y_group = grouped_y.get_group(key)
            # Convert to tensors
            x_tensor = torch.tensor(x_group.drop(['Genotype', 'Sex', 'Diet', 'SampleID'], axis=1).values, dtype=torch.float)
            y_tensor = torch.tensor(y_group.drop(['Genotype', 'Sex', 'Diet', 'SampleID'], axis=1).values, dtype=torch.float)

# Define various kernel functions
def linear_kernel(x, y):
    return x @ y.T

def polynomial_kernel(x, y, c=1, d=3):
    return (x @ y.T + c) ** d

def gaussian_kernel(x, y, sigma=1.0):
    beta = 1.0 / (2.0 * sigma ** 2)
    dist_sq = torch.cdist(x, y, p=2)**2
    return torch.exp(-beta * dist_sq)

def sigmoid_kernel(x, y, alpha=0.01, c=1):
    return torch.tanh(alpha * (x @ y.T) + c)

def laplacian_kernel(x, y, sigma=1.0):
    dist = torch.cdist(x, y, p=1)  # Using L1 norm
    return torch.exp(-dist / sigma)

# Compute the MMD statistic using a specified kernel
def compute_mmd(x, y, kernel_func, **kernel_params):
    x_kernel = kernel_func(x, x, **kernel_params)
    y_kernel = kernel_func(y, y, **kernel_params)
    xy_kernel = kernel_func(x, y, **kernel_params)
    return x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()

# Function to perform sample matching with optimization method choice
def optimize_samples(x_data, y_data, kernel_type='gaussian', method='stochastic', **kernel_args):
    idx = np.arange(y_data.size(0))

    # Mapping kernel types to kernel functions
    kernel_map = {
        'linear': linear_kernel,
        'polynomial': polynomial_kernel,
        'gaussian': gaussian_kernel,
        'sigmoid': sigmoid_kernel,
        'laplacian': laplacian_kernel
    }
    kernel_func = kernel_map[kernel_type]

    def objective(indices):
        indices = indices.astype(int)
        shuffled_y = y_data[indices]
        return compute_mmd(x_data, torch.tensor(shuffled_y), kernel_func, **kernel_args).item()

    if method == 'pso':
        lb = [0] * len(idx)
        ub = [y_data.size(0) - 1] * len(idx)
        xopt, _ = pso(objective, lb, ub, swarmsize=100, maxiter=100)
        idx = xopt.astype(int)

    elif method == 'ga':
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()
        toolbox.register("indices", np.random.permutation, len(idx))
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("evaluate", objective)
        toolbox.register("mate", tools.cxPartialyMatched)
        toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
        toolbox.register("select", tools.selTournament, tournsize=3)

        pop = toolbox.population(n=50)
        algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=40, verbose=False)

        best_ind = tools.selBest(pop, 1)[0]
        idx = np.array(best_ind)

    elif method == 'stochastic':
        for i in range(1000):
        # Calculate current MMD
            current_mmd =  objective(idx)

            # Simplified optimization: randomly swap two elements and keep change if MMD improves
            with torch.no_grad():
                idx2 = idx.clone()
                swap_indices = torch.randperm(y_data.size(0))[:2]
                idx2[swap_indices[0]], idx2[swap_indices[1]] = idx2[swap_indices[1]], idx2[swap_indices[0]]
                new_mmd = compute_mmd(x_data, y_data[idx2], kernel_func, **kernel_args)
                if new_mmd < current_mmd:
                    idx = idx2
                    y_data_shuffled = y_data[idx]
                    current_mmd = new_mmd
        
        return idx

    elif method == 'random':
        np.random.shuffle(idx)  # Only shuffle once for random assignment

    return torch.tensor(idx, dtype=torch.long)

# Example data
x_samples = torch.randn(100, 50)  # 100 samples, 50 features each
y_samples = torch.randn(100, 50)  # 100 samples, 50 features each

# Example usage
kernel_choice = 'gaussian'
optimization_method = 'ga'
best_matching = optimize_samples(x_samples, y_samples, kernel_type=kernel_choice, method=optimization_method, sigma=0.5)
print(f"Best matching indices using {optimization_method} and {kernel_choice} kernel:", best_matching)

# Calculate final MMD
final_mmd = compute_mmd(x_samples, y_samples[best_matching], gaussian_kernel, sigma=0.5)
print("Final MMD after matching:", final_mmd.item())

# Example using KMeans clustering
def cluster_and_evaluate(data, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters).fit(data)
    labels = kmeans.labels_
    
    silhouette = silhouette_score(data, labels)
    db_index = davies_bouldin_score(data, labels)
    ch_index = calinski_harabasz_score(data, labels)
    
    return silhouette, db_index, ch_index

# Assuming `integrated_data` is the combined dataset from both sources
silhouette, db_index, ch_index = cluster_and_evaluate(integrated_data)
print(f"Silhouette Score: {silhouette}")
print(f"Davies-Bouldin Index: {db_index}")
print(f"Calinski-Harabasz Index: {ch_index}")

def plot_tsne(data):
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(data)

    plt.figure(figsize=(10, 6))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], alpha=0.5)
    plt.title('t-SNE visualization of the Integrated Data')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()

plot_tsne(integrated_data)

def bootstrap_stability(data, n_clusters=3, n_iterations=100):
    # Perform initial clustering
    initial_kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(data)
    initial_labels = initial_kmeans.labels_
    
    # List to store ARI scores
    ari_scores = []
    
    # Bootstrap resampling and clustering
    for _ in range(n_iterations):
        # Resample data with replacement
        bootstrap_sample = resample(data)
        
        # Clustering on the bootstrap sample
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(bootstrap_sample)
        bootstrap_labels = kmeans.labels_
        
        # Compare bootstrap clustering to the initial clustering using ARI
        ari = adjusted_rand_score(initial_labels, bootstrap_labels)
        ari_scores.append(ari)
    
    # Calculate the average ARI score
    average_ari = np.mean(ari_scores)
    return average_ari

# Assuming 'integrated_data' is your dataset
average_ari = bootstrap_stability(integrated_data)
print(f"Average Adjusted Rand Index over {n_iterations} iterations: {average_ari}")


from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt

def plot_cluster_distances(data, labels):
    # Calculate pairwise distances
    distances = pdist(data)
    distance_matrix = squareform(distances)
    
    intra_cluster_distances = []
    inter_cluster_distances = []
    
    # Separate intra-cluster and inter-cluster distances
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if labels[i] == labels[j]:
                intra_cluster_distances.append(distance_matrix[i][j])
            else:
                inter_cluster_distances.append(distance_matrix[i][j])
    
    # Plot distributions
    plt.figure(figsize=(12, 6))
    plt.hist(intra_cluster_distances, bins=50, alpha=0.5, label='Intra-cluster Distances')
    plt.hist(inter_cluster_distances, bins=50, alpha=0.5, label='Inter-cluster Distances')
    plt.title('Distribution of Pairwise Distances')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

# Assuming `integrated_data` is your dataset and `labels` are the cluster labels
kmeans = KMeans(n_clusters=3, random_state=42).fit(integrated_data)
plot_cluster_distances(integrated_data, kmeans.labels_)

# Union-Com implementation

import pandas as pd

conditions = {
    'group1': {'genotype_nb': 1, 'sex': 'M', 'food': 'CD'},
    'group2': {'genotype_nb': 0, 'sex': 'M', 'food': 'CD'},
    'group3': {'genotype_nb': 1, 'sex': 'F', 'food': 'CD'},
    'group4': {'genotype_nb': 0, 'sex': 'F', 'food': 'CD'},
    'group5': {'genotype_nb': 1, 'sex': 'F', 'food': 'HFD'},
    'group6': {'genotype_nb': 1, 'sex': 'M', 'food': 'HFD'},
    'group7': {'genotype_nb': 0, 'sex': 'F', 'food': 'HFD'},
    'group8': {'genotype_nb': 0, 'sex': 'M', 'food': 'HFD'}
}

def split_data(df):
  grouped_dataframes = {}
  for group_name, filters in conditions.items():
      query = ' & '.join([f"{col} == '{val}'" if isinstance(val, str) else f"{col} == {val}"
                          for col, val in filters.items()])
      grouped_dataframes[group_name] = df.query(query)
  return grouped_dataframes

grouped_df_metagenome = split_data(metagenomic_data)

transcriptomic_data_filter = transcriptomic_data.reset_index()
if 'index' in transcriptomic_data_filter and 'SampleID' in transcriptome_metadata:
    valid_values = set(transcriptomic_data_filter['index'])
    transcriptome_metadata_filtered = transcriptome_metadata[transcriptome_metadata['SampleID'].isin(valid_values)]
transcriptomic_data = transcriptomic_data_filter.set_index('index')

# Check if both DataFrames have the same number of rows
if len(transcriptomic_data) == len(transcriptome_metadata_filtered):
    # Resetting indices to ensure rows align perfectly
    transcriptomic_data.reset_index(drop=True, inplace=True)
    transcriptome_metadata_filtered.reset_index(drop=True, inplace=True)

    # Concatenate horizontally
    combined_df = pd.concat([transcriptomic_data, transcriptome_metadata_filtered], axis=1)

import pandas as pd

conditions = {
    'group1': {'GT': 'DP1Y', 'Sex': 'M', 'Diet': 'CD'},
    'group2': {'GT': 'WT', 'Sex': 'M', 'Diet': 'CD'},
    'group3': {'GT': 'DP1Y', 'Sex': 'F', 'Diet': 'CD'},
    'group4': {'GT': 'WT', 'Sex': 'F', 'Diet': 'CD'},
    'group5': {'GT': 'DP1Y', 'Sex': 'F', 'Diet': 'HFD'},
    'group6': {'GT': 'DP1Y', 'Sex': 'M', 'Diet': 'HFD'},
    'group7': {'GT': 'WT', 'Sex': 'F', 'Diet': 'HFD'},
    'group8': {'GT': 'WT', 'Sex': 'M', 'Diet': 'HFD'}
}

def split_data(df):
  grouped_dataframes = {}
  for group_name, filters in conditions.items():
      query = ' & '.join([f"{col} == '{val}'" if isinstance(val, str) else f"{col} == {val}"
                          for col, val in filters.items()])
      grouped_dataframes[group_name] = df.query(query)
  return grouped_dataframes

grouped_df = split_data(combined_df)

def filter_columns_transcriptome(df):
    df = df.loc[:, df.columns.str.startswith('ENSM') | df.columns.str.startswith('SampleID')]
    df = df.set_index('SampleID')
    return df

def filter_group_df(group_df, filter_func):
    filtered_grouped_dataframes = {key: filter_func(df) for key, df in group_df.items()}
    return filtered_grouped_dataframes

filtered_grouped_dataframes_transcriptome = filter_group_df(grouped_df, filter_columns_transcriptome)

def filter_columns_metagenome(df):
    columns_to_remove = ['TubeID', 'miceID', 'IP', 'batch', 'sex', 'Sampletype', 'cage', 'genotype', 'genotype_nb', 'age_harvested', 'food', 'Selected']
    df = df.drop(columns=columns_to_remove)
    df = df.set_index('index')
    return df

filtered_grouped_dataframes_metagenome = filter_group_df(grouped_df_metagenome, filter_columns_metagenome)

def define_allignment_mappings_union_com(filtered_grouped_dataframes_transcriptome, filtered_grouped_dataframes_metagenome):
  
  transcriptome_mapping = []
  metagenome_mapping = []

  for key in filtered_grouped_dataframes_transcriptome:
    uc = UnionCom()
    integrated_data, corr_pairs, pairs_x, pairs_y = uc.fit_transform(dataset=[filtered_grouped_dataframes_transcriptome[key],filtered_grouped_dataframes_metagenome[key]])
    transcriptome_mapping.append(pairs_x)
    metagenome_mapping.append(pairs_y)
    uc.Visualize([filtered_grouped_dataframes_transcriptome[key],filtered_grouped_dataframes_metagenome[key]], integrated_data, mode='UMAP')

  return (transcriptome_mapping, metagenome_mapping)

transcriptome_mapping, metagenome_mapping = define_allignment_mappings_union_com(filtered_grouped_dataframes_transcriptome, filtered_grouped_dataframes_metagenome)
  
keys = filtered_grouped_dataframes_transcriptome.keys()
new_alligned_transcriptome = {}
keys_array = list(keys)
print(keys_array)

for i, array in enumerate(transcriptome_mapping):
  print(array)
  group = filtered_grouped_dataframes_transcriptome[keys_array[i]]
  filtered_group = group.iloc[array[0]]
  new_alligned_transcriptome[keys_array[i]] = filtered_group

new_alligned_transcriptome

keys = filtered_grouped_dataframes_metagenome.keys()
new_alligned_metagenome = {}
keys_array = list(keys)

for i, array in enumerate(metagenome_mapping):
  group = filtered_grouped_dataframes_metagenome[keys_array[i]]
  filtered_group = group.iloc[array[0]]
  new_alligned_metagenome[keys_array[i]] = filtered_group

new_alligned_metagenome

alligned_combined_df = pd.DataFrame()
sample_id = []
for key in keys_array:
  new_alligned_transcriptome[key].reset_index(drop=False, inplace=True)
  sample_id.append(new_alligned_transcriptome[key]['SampleID'])
  new_alligned_transcriptome[key].drop('SampleID', axis = 1, inplace=True)
  new_alligned_metagenome[key].reset_index(drop=True, inplace=True)
  intermediate_df = pd.concat([new_alligned_transcriptome[key], new_alligned_metagenome[key]], axis=1)
  alligned_combined_df = pd.concat([alligned_combined_df, intermediate_df], axis = 0)

alligned_combined_df.drop('index', axis = 1, inplace=True)
alligned_combined_df

#middle integration use
alligned_single_transcriptome = pd.DataFrame()
for key in keys_array:
  alligned_single_transcriptome = pd.concat([alligned_single_transcriptome, new_alligned_transcriptome[key]], axis = 0)

alligned_single_transcriptome.drop('index', axis = 1, inplace = True)


alligned_single_metagenome = pd.DataFrame()
for key in keys_array:
  alligned_single_metagenome = pd.concat([alligned_single_metagenome, new_alligned_metagenome[key]], axis = 0)