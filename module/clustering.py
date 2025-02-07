import os
import glob
import shutil

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import librosa

from sklearn.cluster import DBSCAN

def compute_embeddings_and_distance(
    directory,
    max_audio_length=4.0,
    use_half=False,
    chunk_size=512,
    min_duration=0.5,
    max_duration=10.0,
    cache_dir="./model_cache"  # New parameter: Model cache directory
):
    """
    Load WAV files, compute embeddings using ReDimNet, and return a cosine distance matrix.

    Parameters:
      directory (str): Path to the folder containing WAV files.
      max_audio_length (float): Maximum audio length (in seconds) to process.
      use_half (bool): Whether to use half precision for model inference.
      chunk_size (int): Chunk size for computing the cosine similarity matrix.
      min_duration (float): Minimum duration (in seconds) for a file to be considered valid.
      max_duration (float): Maximum duration (in seconds) for a file to be considered valid.
      cache_dir (str): Directory to cache the model for faster subsequent loads.

    Returns:
      distance_matrix (np.ndarray): Cosine distance matrix of shape (N, N).
      valid_wav_files (list): List of valid WAV file paths.
      noise_files (list): List of file paths classified as noise.
      embeddings (torch.Tensor): Embedding tensor of shape (N, D).
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)

    torch.hub.set_dir(cache_dir)
    # Load ReDimNet model with cache directory
    model = torch.hub.load(
        "IDRnD/ReDimNet",   # Repository
        "ReDimNet",         # Function name
        model_name="b6",    # Model version (e.g., b6)
        train_type="ft_lm", # Fine-tuned large margin loss version
        dataset="vox2",     # Pretrained dataset: VOXCELEB2
        source="github",    # Explicitly specify GitHub as the source
        force_reload=False, # Avoid redundant downloads
        skip_validation=True  # Skip unnecessary validation
    )
    model.to(device)
    model.eval()

    # Find all WAV files in the directory
    all_wav_files = sorted(glob.glob(os.path.join(directory, "*.wav")))
    if len(all_wav_files) == 0:
        print("No valid WAV files found.")
        return None, [], [], None

    def extract_embedding(filepath):
        # Load audio with librosa to determine duration
        audio_librosa, sr_librosa = librosa.load(filepath, sr=16000)
        duration_sec = len(audio_librosa) / sr_librosa

        # Load audio with torchaudio for processing
        signal, sr = torchaudio.load(filepath)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            signal = resampler(signal)

        max_samples = int(max_audio_length * 16000)
        if signal.shape[1] > max_samples:
            signal = signal[:, :max_samples]

        if use_half:
            model.half()

        signal = signal.to(device)
        with torch.no_grad():
            emb = model(signal)  # Shape: (1, emb_dim)
        emb = emb.squeeze(0).cpu()  # Shape: (emb_dim,)
        return emb, duration_sec

    valid_embeddings = []
    valid_wav_files = []
    noise_files = []

    # Process files and compute embeddings
    for path in all_wav_files:
        emb, duration_sec = extract_embedding(path)
        if duration_sec < min_duration or duration_sec > max_duration:
            noise_files.append(path)
            continue
        valid_embeddings.append(emb)
        valid_wav_files.append(path)

    # Exit if no valid embeddings were found
    if len(valid_embeddings) == 0:
        print("All files were out of the specified duration range.")
        return None, [], noise_files, None

    embeddings = torch.stack(valid_embeddings, dim=0)  # Shape: (N, D), float32
    # Compute cosine similarity matrix
    sim_mat = chunked_cosine_similarity(embeddings, device, chunk_size, use_half)
    sim_mat = torch.clamp(sim_mat, -1.0, 1.0)  # Clamp values to valid range
    distance_matrix = ((1.0 - sim_mat.numpy()) / 2.0).astype(np.float64)

    return distance_matrix, valid_wav_files, noise_files, embeddings


def chunked_cosine_similarity(embeddings, device, chunk_size=512, use_half=False):
    """
    Compute the cosine similarity matrix (N, N) for an embedding tensor (N, D) using chunking.

    Parameters:
      embeddings (torch.Tensor): Input embedding tensor of shape (N, D).
      device (torch.device): Device for computation.
      chunk_size (int): Size of chunks to process.
      use_half (bool): Whether to use half precision.

    Returns:
      sim_mat (torch.Tensor): Cosine similarity matrix of shape (N, N) with float32 values.
    """
    N, D = embeddings.shape
    norms = torch.norm(embeddings, p=2, dim=1, keepdim=True)
    sim_mat = torch.zeros(N, N, dtype=torch.float32)

    for start_i in range(0, N, chunk_size):
        end_i = min(start_i + chunk_size, N)
        row_emb = embeddings[start_i:end_i].to(device)
        row_norm = norms[start_i:end_i].to(device)
        if use_half:
            row_emb = row_emb.half()
            row_norm = row_norm.half()

        for start_j in range(0, N, chunk_size):
            end_j = min(start_j + chunk_size, N)
            col_emb = embeddings[start_j:end_j].to(device)
            col_norm = norms[start_j:end_j].to(device)
            if use_half:
                col_emb = col_emb.half()
                col_norm = col_norm.half()

            dot_chunk = torch.matmul(row_emb, col_emb.transpose(0, 1))
            denom = torch.matmul(row_norm, col_norm.transpose(0, 1))
            cos_chunk = dot_chunk / (denom + 1e-9)
            sim_mat[start_i:end_i, start_j:end_j] = cos_chunk.float().cpu()

    return sim_mat


def dbscan_kmeans_clustering(
    distance_matrix,
    embeddings,
    valid_wav_files,
    noise_files,
    destination_folder,
    eps=0.05,
    min_samples=2,
    max_distance=0.2,
    max_iters=100,
):
    """
    Cluster the data using DBSCAN followed by a variant of K-Means with noise handling.

    Steps:
      1. Apply DBSCAN (with metric='precomputed') to form initial clusters.
         Use the first core sample of each cluster as the initial centroid for K-Means.
      2. Run a K-Means variant where points with cosine distance greater than max_distance
         to any centroid are labeled as noise.
      3. Create cluster folders (and a noise folder) and copy the corresponding WAV files.

    Parameters:
      distance_matrix (np.ndarray): Precomputed cosine distance matrix (N, N).
      embeddings (torch.Tensor): Embedding tensor of shape (N, D) on CPU.
      valid_wav_files (list): List of valid WAV file paths.
      noise_files (list): List of WAV files previously classified as noise.
      destination_folder (str): Folder to store clustering results.
      eps (float): DBSCAN epsilon parameter.
      min_samples (int): DBSCAN min_samples parameter.
      max_distance (float): Maximum allowed cosine distance for cluster assignment.
      max_iters (int): Maximum iterations for the K-Means variant.

    Returns:
      labels_total (list): Final labels for all files (valid and noise); -1 indicates noise.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Apply DBSCAN clustering
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed").fit(distance_matrix)
    labels = db.labels_  # DBSCAN labels; -1 indicates noise
    core_sample_mask = np.zeros_like(labels, dtype=bool)
    core_sample_mask[db.core_sample_indices_] = True

    # Collect core and overall sample indices for each cluster (excluding noise)
    clusters_core_indices = {}
    clusters_all_indices = {}
    unique_labels = set(labels)
    for label in unique_labels:
        if label == -1:
            continue
        core_indices = np.where((labels == label) & core_sample_mask)[0]
        all_indices = np.where(labels == label)[0]
        clusters_core_indices[label] = core_indices
        clusters_all_indices[label] = all_indices

    def cos_distance(a, b):
        """
        Compute the cosine distance between two 1-D tensors.

        Parameters:
          a, b (torch.Tensor): 1-D tensors.
        Returns:
          dist (float): Cosine distance computed as (1 - cosine_similarity) / 2.
        """
        a_ = a.unsqueeze(0).to(device)
        b_ = b.unsqueeze(0).to(device)
        sim = F.cosine_similarity(a_, b_, dim=-1).clamp(-1.0, 1.0)
        dist = (1.0 - sim) / 2.0
        return dist.cpu().item()

    # Set initial centroids: use the first core sample from each cluster (or first sample if core not available)
    k = len(clusters_core_indices.keys())
    if k == 0:
        print("No clusters found from DBSCAN; all files classified as noise.")
        return finalize_clustering(labels, valid_wav_files, noise_files, destination_folder)

    initial_centroid_indices = []
    for lbl, core_inds in clusters_core_indices.items():
        if len(core_inds) > 0:
            initial_centroid_indices.append(core_inds[0])
        else:
            initial_centroid_indices.append(clusters_all_indices[lbl][0])

    X_start_cpu = embeddings[initial_centroid_indices]  # Shape: (k, D)

    # Run K-Means variant with noise handling
    centroids_result, clusters_result, noise_points_result = kmeans_with_noise(
        embeddings, X_start_cpu, k, cos_distance, max_distance=max_distance, max_iters=max_iters
    )

    # Construct final labels: default label is -1 (noise)
    N = embeddings.shape[0]
    labels_kmeans = np.full(N, -1, dtype=int)

    for cluster_idx, idx_list in enumerate(clusters_result):
        for idx in idx_list:
            labels_kmeans[idx] = cluster_idx

    # Mark points flagged as noise by the K-Means variant
    for idx, is_noise in enumerate(noise_points_result):
        if is_noise:
            labels_kmeans[idx] = -1

    # Enforce DBSCAN noise labels
    for i, lbl in enumerate(labels):
        if lbl == -1:
            labels_kmeans[i] = -1

    return finalize_clustering(labels_kmeans, valid_wav_files, noise_files, destination_folder)


def kmeans_with_noise(
    embeddings_cpu,
    centroids_cpu,
    k,
    distance_func,
    max_distance=0.2,
    max_iters=100,
):
    """
    Perform a variant of K-Means clustering with noise detection.

    Parameters:
      embeddings_cpu (torch.Tensor): Embedding tensor of shape (N, D) on CPU.
      centroids_cpu (torch.Tensor): Initial centroids of shape (k, D) on CPU.
      k (int): Number of clusters.
      distance_func (function): Function to compute cosine distance between two 1-D tensors.
      max_distance (float): Maximum allowed cosine distance for cluster assignment.
      max_iters (int): Maximum number of iterations.

    Returns:
      centroids_curr (torch.Tensor): Final centroids of shape (k, D).
      idx_clusters (list): List of length k, each containing indices of points in the cluster.
      noise_points (np.ndarray): Boolean array of length N indicating noise points.
    """
    N = embeddings_cpu.shape[0]
    # Pair each index with its corresponding embedding
    X_pairs = [[idx, embeddings_cpu[idx]] for idx in range(N)]

    centroids_curr = centroids_cpu.clone()  # Shape: (k, D)
    for _ in range(max_iters):
        clusters = [[] for _ in range(k)]
        noise_points = np.full(N, False)

        # Assign each point to the nearest centroid or mark it as noise
        for pair in X_pairs:
            idx_data, x_cpu = pair
            distances = [distance_func(x_cpu, centroids_curr[c_idx]) for c_idx in range(k)]
            best_cluster = np.argmin(distances)
            if distances[best_cluster] > max_distance:
                noise_points[idx_data] = True
            else:
                clusters[best_cluster].append([idx_data, x_cpu])

        # Update centroids as the mean of points in each cluster
        new_centroids = []
        for cluster_idx in range(k):
            if len(clusters[cluster_idx]) > 0:
                emb_list = [x for (_, x) in clusters[cluster_idx]]
                emb_stack = torch.stack(emb_list, dim=0).float()
                centroid = emb_stack.mean(dim=0)
                new_centroids.append(centroid)
            else:
                new_centroids.append(centroids_curr[cluster_idx])

        new_centroids_cpu = torch.stack(new_centroids, dim=0)
        if torch.allclose(centroids_curr, new_centroids_cpu, atol=1e-3):
            centroids_curr = new_centroids_cpu
            break
        centroids_curr = new_centroids_cpu

    # Prepare final clusters based on point indices
    idx_clusters = []
    for cluster_idx in range(k):
        idx_in_cluster = [pair[0] for pair in clusters[cluster_idx]]
        idx_clusters.append(idx_in_cluster)

    return centroids_curr, idx_clusters, noise_points


def finalize_clustering(labels, valid_wav_files, noise_files, destination_folder):
    """
    Create cluster folders and copy WAV files based on the clustering labels.

    Parameters:
      labels (array-like): Labels for valid WAV files; -1 indicates noise.
      valid_wav_files (list): List of valid WAV file paths.
      noise_files (list): List of noise file paths.
      destination_folder (str): Root folder to store clustering results.

    Returns:
      labels_total (list): Combined labels for all files (valid and noise).
    """
    labels_total = list(labels) + ([-1] * len(noise_files))
    all_files = valid_wav_files + noise_files

    os.makedirs(destination_folder, exist_ok=True)

    unique_labels = set(labels_total)
    for lbl in unique_labels:
        if lbl == -1:
            cluster_folder = os.path.join(destination_folder, "noise")
        else:
            cluster_folder = os.path.join(destination_folder, f"clustering_{lbl}")
        os.makedirs(cluster_folder, exist_ok=True)

    for path, lbl in zip(all_files, labels_total):
        if lbl == -1:
            cluster_folder = os.path.join(destination_folder, "noise")
        else:
            cluster_folder = os.path.join(destination_folder, f"clustering_{lbl}")
        file_name = os.path.basename(path)
        dest_path = os.path.join(cluster_folder, file_name)
        shutil.copy(path, dest_path)
        print(f"Copied to: {dest_path}")

    print("Clustering complete.")
    return labels_total


def remove_small_clusters(destination_folder, min_files=10):
    """
    Remove cluster folders that contain min_files or fewer WAV files.

    Parameters:
      destination_folder (str): Root folder containing clustering results.
      min_files (int): Minimum number of WAV files required to keep a folder.
    """
    cluster_folders = [
        os.path.join(destination_folder, f)
        for f in os.listdir(destination_folder)
        if os.path.isdir(os.path.join(destination_folder, f))
    ]

    for folder in cluster_folders:
        wav_files = [f for f in os.listdir(folder) if f.endswith(".wav")]
        num_files = len(wav_files)
        if num_files <= min_files:
            print(f"Deleting folder: {folder} (contains {num_files} files)")
            shutil.rmtree(folder)
        else:
            print(f"Keeping folder: {folder} (contains {num_files} files)")

    print("Cleanup complete.")

def clustering_for_main(wav_folder, output_folder, cache_dir):
    # Step 1: Compute embeddings and distance matrix
    distance_matrix, valid_wav_files, noise_files, embeddings = compute_embeddings_and_distance(
        directory=wav_folder,
        max_audio_length=10.0,
        use_half=False,
        chunk_size=512,
        min_duration=0.5,
        max_duration=10.0,
        cache_dir=cache_dir
    )

    # Check if there are valid files to process
    if distance_matrix is None or embeddings is None or len(valid_wav_files) == 0:
        print("No valid audio files found for clustering.")
    else:
        # Step 2: Perform clustering
        labels = dbscan_kmeans_clustering(
            distance_matrix=distance_matrix,
            embeddings=embeddings,
            valid_wav_files=valid_wav_files,
            noise_files=noise_files,
            destination_folder=output_folder,
            eps=0.2,  # DBSCAN hyperparameter
            min_samples=2,  # DBSCAN hyperparameter
            max_distance=0.2,  # K-Means noise threshold
            max_iters=100  # Maximum iterations for K-Means
        )

        # Step 3: Remove small clusters
        remove_small_clusters(destination_folder=output_folder, min_files=10)