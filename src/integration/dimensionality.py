from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
from scipy.stats import chi2
from sklearn.cross_decomposition import CCA
from sklearn.metrics.pairwise import cosine_similarity

def standardize_data(data):
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(data)
    return standardized_data

def perform_pca(data, n_components=2):
    pca = PCA(n_components=n_components)
    pca_results = pca.fit_transform(data)
    return pca_results

def plot_pca(pca_results, metadata, group_column, title='2D Principal Component Analysis (PCA)'):
    plt.figure(figsize=(12, 10))
    unique_groups = metadata[group_column].unique() 
    palette = sns.color_palette("Set2", len(unique_groups))

    for group, colour in zip(unique_groups, palette): 
        idx = metadata[group_column] == group
        plt.scatter(pca_results[idx, 0], 
                    pca_results[idx, 1], 
                    alpha=0.7, 
                    edgecolors='w', 
                    label=group, 
                    s=100, 
                    color=colour) 
    
    plt.xlabel('Principal Component 1', fontsize=14) 
    plt.ylabel('Principal Component 2', fontsize=14)
    plt.title(title, fontsize=16) 
    plt.legend(title= group_column, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11, title_fontsize=12)
    plt.grid(True)
    sns.despine()  
    plt.tight_layout() 
    plt.show()


def perform_pls_da(x, y, n=2):
    plsda = PLSRegression(n_components=n)
    plsda_results = plsda.fit_transform(x, y)[0] 
    return plsda_results


def plot_pls_da(plsda_results, labels, legend_title, title='PLS-DA Results'):
    plt.figure(figsize=(12, 10))
    unique_labels = np.unique(labels)
    colours = sns.color_palette('viridis', n_colors=len(unique_labels))
    cmap = ListedColormap(colours)
    for i, label in enumerate(unique_labels):
        idx = labels == label
        plt.scatter(plsda_results[idx, 0], 
                    plsda_results[idx, 1], 
                    color=colours[i], 
                    label=f'Class {label}', 
                    alpha=0.7)
    plt.xlabel('PLS Component 1', fontsize=14)
    plt.ylabel('PLS Component 2', fontsize=14)
    plt.title(title)
    plt.legend(title= legend_title, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11, title_fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def perform_cca(X, Y, n_components=2):
    cca = CCA(n_components=n_components)
    cca_results = cca.fit(X, Y).transform(X, Y)
    return cca_results

def plot_cca(cca_results, title='CCA Results'):
    plt.figure(figsize=(8, 6))
    plt.scatter(cca_results[0][:, 0], cca_results[0][:, 1], alpha=0.5, color='red', label='Transcriptomic Data')
    plt.scatter(cca_results[1][:, 0], cca_results[1][:, 1], alpha=0.5, color='blue', label='Metagenomic Data')
    plt.xlabel('Canonical Variable 1')
    plt.ylabel('Canonical Variable 2')
    plt.title(title)
    plt.legend()
    plt.show()

cca_results = perform_cca(alligned_single_transcriptome, alligned_single_metagenome)


def perform_and_plot_pca_3d(data, metadata, feature_column, n_components=3, title='PCA 3D Plot'):
    pca = PCA(n_components=n_components)
    pca_results = pca.fit_transform(data)

    unique_categories = metadata[feature_column].unique()
    color_palette = sns.color_palette("Set2", len(unique_categories))
    colors = metadata[feature_column].map({cat: color for cat, color in zip(unique_categories, color_palette)})

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(pca_results[:, 0], pca_results[:, 1], pca_results[:, 2], 
                         c=colors, marker='o', alpha=0.5)

    if metadata[feature_column].dtype == 'object':
        legend_labels = [plt.Line2D([0], [0], marker='o', color=color_palette[i], label=category, markersize=10, linestyle='') 
                         for i, category in enumerate(unique_categories)]
        ax.legend(handles=legend_labels, title=feature_column)

    else:
        cbar = plt.colorbar(scatter)
        cbar.set_label(feature_column)

    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    plt.title(title)
    plt.show()

def perform_and_plot_pls_da_3d(X, Y, metadata, group_column, n=3, title='PLS-DA 3D Plot'):
    plsda = PLSRegression(n_components=n_components)
    plsda_results = plsda.fit_transform(X, Y)[0]

    groups = np.unique(metadata[group_column])
    colors = sns.color_palette("viridis", len(groups))
    group_colors = dict(zip(groups, colors))
    colors_array = metadata[group_column].map(group_colors)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    for group, color in group_colors.items():
        idx = metadata[group_column] == group
        ax.scatter(plsda_results[idx, 0], plsda_results[idx, 1], plsda_results[idx, 2], 
                   color=color, label=group, alpha=0.6)

    ax.set_xlabel('PLS Component 1')
    ax.set_ylabel('PLS Component 2')
    ax.set_zlabel('PLS Component 3')
    plt.title(title)
    ax.legend(title=group_column)
    ax.grid(True)
    plt.tight_layout()
    plt.show()

def plot_cca_biplot(X, Y, n=2):
    cca = CCA(n_components=n)
    X_c, Y_c = cca.fit_transform(X, Y)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X_c[:, 0], X_c[:, 1], color='red', label='Transcriptome')
    ax.scatter(Y_c[:, 0], Y_c[:, 1], color='blue', label='Metagenome')
    
    for i, txt in enumerate(X.columns):
        ax.arrow(0, 0, cca.x_weights_[i, 0], cca.x_weights_[i, 1], color='red', alpha=0.5)
        ax.text(cca.x_weights_[i, 0]*1.15, cca.x_weights_[i, 1]*1.15, txt, color='red')
    for i, txt in enumerate(Y.columns):
        ax.arrow(0, 0, cca.y_weights_[i, 0], cca.y_weights_[i, 1], color='blue', alpha=0.5)
        ax.text(cca.y_weights_[i, 0]*1.15, cca.y_weights_[i, 1]*1.15, txt, color='blue')
    
    ax.set_xlabel('Canonical Variable 1')
    ax.set_ylabel('Canonical Variable 2')
    ax.axhline(0, color='grey', lw=1)
    ax.axvline(0, color='grey', lw=1)
    plt.title('CCA Biplot')
    plt.legend()
    plt.show()

def plot_correlation_circle(X, Y, n=2):
    cca = CCA(n_components=n)
    X_c, Y_c = cca.fit_transform(X, Y)
    fig, ax = plt.subplots(figsize=(8, 8))
    max_radius = np.sqrt(cca.x_weights_[:, 0]**2 + cca.x_weights_[:, 1]**2).max()
    circle = plt.Circle((0, 0), max_radius, color='gray', fill=False)
    ax.add_artist(circle)

    for i in range(X.shape[1]):
        ax.arrow(0, 0, cca.x_weights_[i, 0], cca.x_weights_[i, 1], color='red', alpha=0.5)
        ax.text(cca.x_weights_[i, 0], cca.x_weights_[i, 1], X.columns[i], color='red')

    for i in range(Y.shape[1]):
        ax.arrow(0, 0, cca.y_weights_[i, 0], cca.y_weights_[i, 1], color='blue', alpha=0.5)
        ax.text(cca.y_weights_[i, 0], cca.y_weights_[i, 1], Y.columns[i], color='blue')
    
    ax.set_xlim(-1.1*max_radius, 1.1*max_radius)
    ax.set_ylim(-1.1*max_radius, 1.1*max_radius)
    ax.set_xlabel('Canonical Variable 1')
    ax.set_ylabel('Canonical Variable 2')
    plt.title('Correlation Circle')
    plt.grid(True)
    plt.show()

plot_correlation_circle(alligned_single_transcriptome, alligned_single_metagenome)

def plot_cosine_similarity(X, Y, n=2):
    cca = CCA(n_components=n)
    X_c, Y_c = cca.fit_transform(X, Y)
    cos_sim = cosine_similarity(np.hstack((X_c, Y_c)))
    sns.heatmap(cos_sim, annot=True, cmap='coolwarm')
    plt.title('Cosine Similarity Heatmap of Canonical Variables')
    plt.show()

plot_cosine_similarity(alligned_single_transcriptome, alligned_single_metagenome)


def calculate_vip_scores(x, y):
    """
    Calculate the Variable Importance in Projection (VIP) scores for a fitted PLS model.

    Args:
    pls_model (PLSRegression): A fitted PLSRegression model.
    X (numpy.ndarray): The predictor data used to fit the model; shape (n_samples, n_features).

    Returns:
    numpy.ndarray: VIP scores for each feature in X.
    """

    plsda = PLSRegression(n_components=2)
    plsda.fit(transcriptomic_data, filtered_series)
    T = plsda.x_scores_  # Scores: shape (n_samples, n_components)
    W = plsda.x_weights_  # Weights: shape (n_features, n_components)
    P = plsda.x_loadings_  # Loadings: shape (n_features, n_components)
    W_squared = np.square(W)  # Element-wise square of the weights
    T_squared = np.sum(np.square(T), axis=0)  # shape (n_components,)
    vip_numerator = np.sum(W_squared * T_squared, axis=1)  # shape (n_features,)
    vip_denominator = np.sum(T_squared)  # A scalar value
    vip_scores = np.sqrt(x.shape[1] * vip_numerator / vip_denominator)

    return vip_scores

def plot_vip_score(vip_scores):
  plt.bar(range(len(vip_scores)), vip_scores, color='dodgerblue')
  plt.xlabel('Variables')
  plt.title('Variable Importance in the Projection (VIP)')
  plt.ylabel('VIP Score')
  plt.show()

vip_scores = calculate_vip_scores(transcriptomic_data, filtered_series)
plot_vip_score(vip_scores)


def plot_loadings(x, n_components=2):
    plsda = PLSRegression(n_components)
    plsda.fit(transcriptomic_data, filtered_series)
    plt.figure(figsize=(10, 8))
    for i in range(n_components):
        loadings = plsda.x_loadings_[:, i]
        plt.subplot(1, n_components, i+1)
        plt.bar(range(x.shape[1]), loadings, color='dark green')
        plt.xlabel('Variables')
        plt.ylabel('Loadings for Component {}'.format(i + 1))
        plt.title(f'Loadings Plot for PLS Component {i+1}')
    plt.tight_layout()
    plt.show()

plot_loadings(transcriptomic_data)

def plot_scores_with_confidence(x, y, confidence_level=0.95):
    plsda = PLSRegression(n_components=2)
    plsda.fit(x, y)
    x_scores = plsda.x_scores_
    categories = np.unique(y)
    fig, ax = plt.subplots()

    for category in categories:
        subset = x_scores[y == category]

        mean_x, mean_y = np.mean(subset, axis=0)
        cov = np.cov(subset, rowvar=False)

        lambda_, v = np.linalg.eig(cov)
        lambda_ = np.sqrt(lambda_)
        ell = mpatches.Ellipse(xy=(mean_x, mean_y),
                               width=lambda_[0]*np.sqrt(chi2.ppf(confidence_level, 2)),
                               height=lambda_[1]*np.sqrt(chi2.ppf(confidence_level, 2)),
                               angle=np.rad2deg(np.arccos(v[0, 0])),
                               edgecolor='black',
                               facecolor='none')
        ax.add_artist(ell)
        ax.scatter(subset[:, 0], subset[:, 1], s=30, label=f'Class {category}')
    
    ax.set_xlabel('PLS Component 1')
    ax.set_ylabel('PLS Component 2')
    plt.title('Score Plot with Confidence Ellipses')
    plt.legend()
    plt.show()

plot_scores_with_confidence(transcriptomic_data, filtered_series, 0.999)

def plot_scree(data, n):
    pca = PCA(n_components=n)
    pca.fit(data)
    explained_variance = pca.explained_variance_ratio_ * 100

    plt.figure(figsize=(8, 4))
    plt.bar(range(1, len(explained_variance) + 1), explained_variance, color="blue", alpha=0.7)
    plt.xlabel('Principal Components')
    plt.ylabel('Percentage of Variance Explained')
    plt.title('Scree Plot')
    plt.xticks(range(1, len(explained_variance) + 1))
    plt.show()

plot_scree(transcriptomic_data,5)

def plot_cumulative_variance(data, n):
    pca = PCA(n_components=n)
    pca.fit(data)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_ * 100)

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='-', color="blue")
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Variance Plot')
    plt.axhline(y=95, color='r', linestyle='-')  # 95% variance line
    plt.text(x=10, y=95, s='95% variance', color='red', va='bottom', ha='right')
    plt.show()

plot_cumulative_variance(transcriptomic_data, 10)

def plot_loading_heatmap(data, n):
    pca = PCA(n_components=n)
    pca.fit(data)
    loadings = pca.components_

    sns.heatmap(loadings, cmap="coolwarm", annot=True)
    plt.xlabel("Principal Components")
    plt.ylabel("Features")
    plt.title("Heatmap of PCA Loadings")
    plt.show()

plot_loading_heatmap(transcriptomic_data, 10)

from sklearn.decomposition import SparsePCA
import matplotlib.pyplot as plt
import numpy as np

def perform_sparse_pca(data, n_components=2, alpha=1.0):
    """
    Perform Sparse PCA on the provided dataset.

    Parameters:
    data (array-like): The data to perform Sparse PCA on, where rows are samples and columns are features.
    n_components (int): The number of sparse components to extract.
    alpha (float): Sparsity controlling parameter. Higher values lead to sparser components.

    Returns:
    numpy.ndarray: The transformed data (component scores).
    """
    sparse_pca = SparsePCA(n_components=n_components, alpha=alpha, random_state=42)
    sparse_pca.fit(data)
    components = sparse_pca.transform(data)
    return components, sparse_pca.components_

import matplotlib.pyplot as plt
import seaborn as sns

def plot_sparse_pca(components, metadata, feature_column, n_components=2):
    """
    Plot the results of Sparse PCA, including component scores colored by a specified metadata feature.
    
    Parameters:
    components (numpy.ndarray): The scores from the Sparse PCA transformation.
    metadata (pandas.DataFrame): DataFrame containing the metadata with a column for the feature to color by.
    feature_column (str): Column name in metadata DataFrame that contains the feature information for coloring.
    n_components (int): The number of components.
    """
    plt.figure(figsize=(12, 6))

    # Create a color palette based on unique feature values
    unique_values = metadata[feature_column].unique()
    palette = sns.color_palette("hsv", len(unique_values))
    feature_colors = dict(zip(unique_values, palette))

    # Plot component scores
    plt.subplot(1, 2, 1)
    for value in unique_values:
        idx = metadata[feature_column] == value
        plt.scatter(components[idx, 0], components[idx, 1], alpha=0.7, color=feature_colors[value], label=value)
    
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title(f'Sparse PCA - Component Scores by {feature_column}')
    plt.legend(title=feature_column)

    # Adjust plot settings
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.show()

components, _ = perform_sparse_pca(transcriptomic_data)
plot_sparse_pca(components, transcriptome_metadata_filtered, 'GT')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import scale

def fit_sparse_pls_da(x, y, n=5, threshold=0.1):
    """
    Fit a PLS-DA model and apply a threshold to create sparsity in the coefficients.

    Parameters:
    X (numpy.ndarray): Predictor data.
    Y (numpy.ndarray): Response data.
    n_components (int): Number of components to use in the model.
    threshold (float): Threshold for determining sparsity in the coefficients.

    Returns:
    tuple: A tuple containing the model, scores, and sparse coefficients.
    """
    # Scale the data
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    # Fit the PLS model
    pls = PLSRegression(n_components=n)
    pls.fit(x_scaled, y)
    scores = pls.transform(x_scaled)

    # Apply sparsity threshold to coefficients
    coefficients = pls.coef_.ravel()
    sparse_coefficients = (np.abs(coefficients) > threshold) * coefficients

    return pls, scores, sparse_coefficients


def plot_pls_da_scores(scores, metadata, category_column):
    """
    Plot the scores from PLS-DA and color points based on metadata.

    Parameters:
    scores (numpy.ndarray): Scores from the PLS model.
    metadata (pandas.DataFrame): DataFrame containing metadata for coloring points.
    category_column (str): Column in metadata to use for point colors.
    """
    unique_categories = metadata[category_column].unique()
    colors = plt.cm.get_cmap('viridis', len(unique_categories))

    # Create the plot
    plt.figure(figsize=(8, 6))
    for i, category in enumerate(unique_categories):
        idx = metadata[category_column] == category
        plt.scatter(scores[idx, 0], scores[idx, 1], color=colors(i), label=category)

    plt.xlabel('PLS Component 1')
    plt.ylabel('PLS Component 2')
    plt.title('PLS-DA Scores Plot')
    plt.legend(title=category_column)
    plt.grid(True)
    plt.show()

pls, scores, sparse_coefficients = fit_sparse_pls_da(transcriptomic_data, filtered_series, n=5, threshold=0.1)
plot_pls_da_scores(scores, transcriptome_metadata_filtered, 'GT')