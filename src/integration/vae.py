import tensorflow as tf
from tensorflow.keras import layers, models, losses, optimizers
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tensorflow.keras.callbacks import Callback
import numpy as np

class Sampling(layers.Layer):
    """Sampling z given (z_mean, z_log_var)."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def build_encoder(latent_dim, input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Dense(512, activation='relu')(inputs)
    x = layers.Dense(256, activation='relu')(x)
    z_mean = layers.Dense(latent_dim)(x)
    z_log_var = layers.Dense(latent_dim)(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = models.Model(inputs, [z_mean, z_log_var, z], name='encoder')
    encoder.summary()
    return encoder

def build_decoder(latent_dim, output_shape):
    latent_inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(256, activation='relu')(latent_inputs)
    x = layers.Dense(512, activation='relu')(x)
    outputs = layers.Dense(output_shape, activation='sigmoid')(x)
    decoder = models.Model(latent_inputs, outputs, name='decoder')
    decoder.summary()
    return decoder

def build_vae(encoder, decoder, input_shape):
    inputs = layers.Input(shape=input_shape)
    z_mean, z_log_var, z = encoder(inputs)
    reconstructed = decoder(z)
    vae = models.Model(inputs, reconstructed, name='vae')
    
    # Calculate VAE loss (KL divergence + reconstruction loss)
    kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
    reconstruction_loss = losses.MeanSquaredError()(inputs, reconstructed)
    vae_loss = reconstruction_loss + kl_loss
    vae.add_loss(vae_loss)
    return vae

# Example using Pandas for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Assuming genomic_data, transcriptomic_data, and proteomic_data are your datasets
data_frames = [genomic_data, transcriptomic_data, proteomic_data]
scaled_data_frames = []

for df in data_frames:
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    scaled_data_frames.append(scaled_data)

# Concatenate all scaled data horizontally (axis=1)
combined_data = np.hstack(scaled_data_frames)

from sklearn.model_selection import train_test_split

x_train, x_test = train_test_split(combined_data, test_size=0.2, random_state=42)

input_shape = x_train.shape[1:]
latent_dim = 50  # Define based on the desired complexity of the latent space

encoder = build_encoder(latent_dim, input_shape)
decoder = build_decoder(latent_dim, input_shape[0])
vae = build_vae(encoder, decoder, input_shape)

vae.compile(optimizer=optimizers.Adam(), loss=None)  # Loss is already added in build_vae
vae.fit(x_train, x_train, epochs=100, batch_size=32, validation_data=(x_test, x_test))

z_mean, _, _ = encoder.predict(combined_data)
# z_mean now represents your reduced-dimensionality data

# Assuming `z_mean` is the output of your encoder
tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
tsne_results = tsne.fit_transform(z_mean)

vae_metadata = pd.read_csv(r'label_to_index_genotype_final.csv')
# Assuming 'metadata' is your DataFrame and it has a 'GT' column with the same order as 'z_mean'
gt_labels = list((vae_metadata['GT']).values)  # This extracts the GT column
colours = ['red' if value == 1 else 'blue' for value in gt_labels]


# t-SNE plot function
def plot_tsne(z_mean, colours):
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    tsne_results = tsne.fit_transform(z_mean)
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], alpha=0.5, c = colours)
    plt.title('t-SNE visualization of the autoencoder latent space')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.show()

# UMAP plot function
def plot_umap(z_mean, colours):
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(z_mean)
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], alpha=0.5, c = colours)
    plt.title('UMAP visualization of the autoencoder latent space')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.show()

# Now you can call these functions with your encoded data 'z_mean' and the 'colors' array
plot_tsne(z_mean, colours)
plot_umap(z_mean, colours)

