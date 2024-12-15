#!/usr/bin/env python
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image


class ImageClusterer:
    def __init__(self, directory):
        """
        Initialize the image clusterer with a directory of images

        :param directory: Path to the directory containing images
        """
        self.directory = directory
        self.image_paths = []
        self.features = []

        # Load pre-trained ResNet50 model for feature extraction
        self.model = ResNet50(weights="imagenet", include_top=False, pooling="avg")

    def extract_features(self):
        """
        Extract deep learning features from images using ResNet50
        """
        self.image_paths = []
        self.features = []

        # Supported image extensions
        valid_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif"]

        for filename in os.listdir(self.directory):
            # Check if file is an image
            if any(filename.lower().endswith(ext) for ext in valid_extensions):
                img_path = os.path.join(self.directory, filename)
                try:
                    # Load and preprocess image
                    img = image.load_img(img_path, target_size=(224, 224))
                    x = image.img_to_array(img)
                    x = np.expand_dims(x, axis=0)
                    x = preprocess_input(x)

                    # Extract features
                    features = self.model.predict(x).flatten()

                    self.image_paths.append(img_path)
                    self.features.append(features)
                except Exception as e:
                    print(f"Error processing {filename}: {e}")

        # Convert to numpy array
        self.features = np.array(self.features)

        return self

    def save_features(self, output_path="image_features.pkl"):
        """
        Save extracted features and image paths to a pickle file

        :param output_path: Path to save the features
        :return: self
        """
        if len(self.features) == 0:
            raise ValueError("No features extracted. Call extract_features() first.")

        # Create directory if it doesn't exist
        outdir = os.path.dirname(output_path)
        if outdir != "":
            os.makedirs(outdir, exist_ok=True)

        # Save features and image paths
        with open(output_path, "wb") as f:
            pickle.dump({"features": self.features, "image_paths": self.image_paths}, f)

        print(f"Features saved to {output_path}")
        return self

    def load_features(self, input_path="image_features.pkl"):
        """
        Load previously extracted features from a pickle file

        :param input_path: Path to load the features from
        :return: self
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Feature file not found: {input_path}")

        # Load features and image paths
        with open(input_path, "rb") as f:
            data = pickle.load(f)

        self.features = data["features"]
        self.image_paths = data["image_paths"]

        print(f"Loaded {len(self.image_paths)} image features")
        return self

    def cluster_images(self, method="dbscan", n_clusters=5):
        """
        Cluster images using either DBSCAN or KMeans

        :param method: Clustering method ('dbscan' or 'kmeans')
        :param n_clusters: Number of clusters for KMeans (ignored for DBSCAN)
        :return: Array of cluster labels
        """
        if len(self.features) == 0:
            raise ValueError("No features available. Extract or load features first.")

        # Normalize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(self.features)

        # Dimensionality reduction for visualization
        reducer = PCA(n_components=2)
        reduced_features = reducer.fit_transform(scaled_features)

        # Clustering
        if method == "dbscan":
            clusterer = DBSCAN(eps=0.5, min_samples=3)
        else:
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)

        self.cluster_labels = clusterer.fit_predict(scaled_features)
        self.reduced_features = reduced_features

        return self

    def visualize_clusters(self, save_path=None):
        """
        Visualize image clusters using scatter plot

        :param save_path: Path to save the visualization
        """
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(
            self.reduced_features[:, 0],
            self.reduced_features[:, 1],
            c=self.cluster_labels,
            cmap="tab20",  # "viridis",
        )
        plt.colorbar(scatter)
        plt.title("Image Clusters")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

        return self

    def create_cluster_montage(self, output_dir=None):
        """
        Create a montage of images for each cluster

        :param output_dir: Directory to save cluster montages
        """
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Number of unique clusters
        unique_clusters = np.unique(self.cluster_labels)

        for cluster in unique_clusters:
            # Get indices of images in this cluster
            cluster_indices = np.where(self.cluster_labels == cluster)[0]

            # Select up to 16 images from the cluster
            selected_indices = cluster_indices[:16]

            # Create subplot grid
            fig, axs = plt.subplots(4, 4, figsize=(12, 12))
            fig.suptitle(f"Cluster {cluster}")

            for i, idx in enumerate(selected_indices):
                row = i // 4
                col = i % 4

                img = image.load_img(self.image_paths[idx])
                axs[row, col].imshow(img)
                axs[row, col].axis("off")

            # Hide any unused subplots
            for i in range(len(selected_indices), 16):
                row = i // 4
                col = i % 4
                axs[row, col].axis("off")

            plt.tight_layout()

            if output_dir:
                plt.savefig(os.path.join(output_dir, f"cluster_{cluster}_montage.png"))
            else:
                plt.show()

            plt.close()

        return self

    def validate_clusters(self, max_clusters=10):
        """
        Validate clustering using multiple methods

        :param max_clusters: Maximum number of clusters to test
        :return: self
        """
        if len(self.features) == 0:
            raise ValueError("No features available. Extract features first.")

        # Run validation methods
        print("Elbow Method Analysis:")
        elbow_results = ClusterValidator.elbow_method(self.features, max_clusters)

        print("\nSilhouette Score Analysis:")
        silhouette_results = ClusterValidator.silhouette_analysis(
            self.features, max_clusters
        )

        print("\nGap Statistic Analysis:")
        gap_results = ClusterValidator.gap_statistic(self.features, max_clusters)

        return self


class ClusterValidator:
    @staticmethod
    def elbow_method(features, max_clusters=10):
        """
        Perform elbow method to find optimal number of clusters

        :param features: Extracted image features
        :param max_clusters: Maximum number of clusters to test
        :return: Dictionary with distortion and inertia values
        """
        # Normalize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        # Calculate distortion for different numbers of clusters
        distortions = []
        inertias = []
        k_values = range(1, max_clusters + 1)

        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(scaled_features)

            # Distortion is the average of the squared distances from each point to its assigned center
            distortions.append(
                np.mean(
                    np.min(
                        cdist(scaled_features, kmeans.cluster_centers_, "euclidean"),
                        axis=1,
                    )
                )
            )

            # Inertia is the sum of squared distances of samples to their closest cluster center
            inertias.append(kmeans.inertia_)

        # Visualize results
        plt.figure(figsize=(12, 5))

        # Distortion subplot
        plt.subplot(1, 2, 1)
        plt.plot(k_values, distortions, "bx-")
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("Distortion")
        plt.title("Elbow Method - Distortion")

        # Inertia subplot
        plt.subplot(1, 2, 2)
        plt.plot(k_values, inertias, "rx-")
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("Inertia")
        plt.title("Elbow Method - Inertia")

        plt.tight_layout()
        plt.show()

        return {"k_values": k_values, "distortions": distortions, "inertias": inertias}

    @staticmethod
    def silhouette_analysis(features, max_clusters=10):
        """
        Compute silhouette scores for different numbers of clusters

        :param features: Extracted image features
        :param max_clusters: Maximum number of clusters to test
        :return: Dictionary with silhouette scores
        """
        # Normalize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        # Reduce dimensionality for visualization
        reducer = PCA(n_components=2)
        reduced_features = reducer.fit_transform(scaled_features)

        # Calculate silhouette scores
        silhouette_scores = []
        k_values = range(2, max_clusters + 1)

        for k in k_values:
            # Perform clustering
            kmeans = KMeans(n_clusters=k, random_state=42)
            cluster_labels = kmeans.fit_predict(scaled_features)

            # Compute silhouette score
            score = silhouette_score(scaled_features, cluster_labels)
            silhouette_scores.append(score)

        # Visualize results
        plt.figure(figsize=(10, 5))
        plt.plot(k_values, silhouette_scores, "bo-")
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("Silhouette Score")
        plt.title("Silhouette Analysis")
        plt.show()

        return {"k_values": k_values, "silhouette_scores": silhouette_scores}

    @staticmethod
    def gap_statistic(features, max_clusters=10, n_references=20):
        """
        Compute Gap Statistic for determining optimal number of clusters

        :param features: Extracted image features
        :param max_clusters: Maximum number of clusters to test
        :param n_references: Number of reference datasets to generate
        :return: Dictionary with gap statistic results
        """
        # Normalize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        # Generate reference datasets
        def generate_reference_data(features):
            return np.random.random_sample(size=features.shape)

        # Compute reference dispersion
        reference_dispersions = []
        gap_values = []

        for k in range(1, max_clusters + 1):
            # Compute dispersion for actual data
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(scaled_features)
            actual_dispersion = np.log(kmeans.inertia_)

            # Compute reference dispersions
            reference_disps = []
            for _ in range(n_references):
                ref_data = generate_reference_data(scaled_features)
                ref_kmeans = KMeans(n_clusters=k, random_state=42)
                ref_kmeans.fit(ref_data)
                reference_disps.append(np.log(ref_kmeans.inertia_))

            # Compute gap statistic
            ref_dispersion = np.mean(reference_disps)
            gap = ref_dispersion - actual_dispersion

            reference_dispersions.append(ref_dispersion)
            gap_values.append(gap)

        # Visualize results
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, max_clusters + 1), gap_values, "go-")
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("Gap Statistic")
        plt.title("Gap Statistic Analysis")
        plt.show()

        return {
            "k_values": range(1, max_clusters + 1),
            "gap_values": gap_values,
            "reference_dispersions": reference_dispersions,
        }


# Example usage
def main():
    # Specify your image directory
    image_directory = "./images"

    # Create clusterer
    clusterer = ImageClusterer(image_directory)

    # Choose one...
    if False:
        # Option 1: Extract and save features
        clusterer.extract_features().save_features("my_image_features.pkl")
    else:
        # Option 2: Load previously extracted features
        clusterer.load_features("my_image_features.pkl")

    # Choose one...
    if False:
        # Validate clustering and find optimal number of clusters
        clusterer.validate_clusters(max_clusters=10)
        # Looking at the code, they want to do this interactively...
    else:
        # cluster, visualize, create montage
        (
            clusterer.cluster_images(method="KMeans", n_clusters=20)
            .visualize_clusters(save_path="cluster_visualization.png")
            .create_cluster_montage(output_dir="cluster_montages")
        )


if __name__ == "__main__":
    main()
