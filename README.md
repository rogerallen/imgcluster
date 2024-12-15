# Image Cluster

An experiment in using Claude.ai to code something up.

## prompts

### prompt 1

I would like to take a directory of images and "score" the images in some manner in order to find and display the images in a way to show clusters of similar images.

Afterwards, I also asked to load & save the model.

### claude 3.5 Haiku response

see imgcluster.py

I'll help you create a Python script that can analyze and cluster images from a directory. We'll break this down into several steps:

Image Feature Extraction
Clustering
Visualization

Here's a comprehensive solution using popular Python libraries:
Image Clustering and Visualization ScriptClick to open code
This script provides a comprehensive solution for image clustering. Here's a breakdown of its key features:

Feature Extraction:

Uses ResNet50 pre-trained neural network to extract deep features from images
Supports multiple image formats (jpg, png, bmp, gif)
Handles image preprocessing and feature normalization


Clustering Methods:

Supports two clustering algorithms:

DBSCAN (density-based spatial clustering)
KMeans (centroid-based clustering)


Allows flexible clustering based on your specific needs


Visualization:

Creates a 2D scatter plot of image clusters using PCA dimensionality reduction
Generates montages of images within each cluster



Prerequisites:

Python 3.7+
Libraries:

numpy
matplotlib
scikit-learn
tensorflow
umap-learn (optional, for alternative dimensionality reduction)



Recommended Setup:
bashCopypip install numpy matplotlib scikit-learn tensorflow umap-learn
How to Use:

Replace 'path/to/your/image/directory' with the actual path to your image directory
Run the script
Check the output:

Cluster visualization plot
Cluster montage images



Customization Options:

Change clustering method ('dbscan' or 'kmeans')
Adjust clustering parameters
Modify visualization techniques

Would you like me to elaborate on any part of the script or explain how to adapt it to your specific use case?

### prompt 2

when doing kmeans clustering, how can I decide n_clusters?  Can I try many clusters and "score" the clustering?

### response

Absolutely! I'll enhance the script to include several methods for determining the optimal number of clusters, including:

1. Elbow Method
2. Silhouette Score
3. Gap Statistic



```python
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

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
                np.mean(np.min(cdist(scaled_features, kmeans.cluster_centers_, 'euclidean'), axis=1))
            )
            
            # Inertia is the sum of squared distances of samples to their closest cluster center
            inertias.append(kmeans.inertia_)
        
        # Visualize results
        plt.figure(figsize=(12, 5))
        
        # Distortion subplot
        plt.subplot(1, 2, 1)
        plt.plot(k_values, distortions, 'bx-')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Distortion')
        plt.title('Elbow Method - Distortion')
        
        # Inertia subplot
        plt.subplot(1, 2, 2)
        plt.plot(k_values, inertias, 'rx-')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia')
        plt.title('Elbow Method - Inertia')
        
        plt.tight_layout()
        plt.show()
        
        return {
            'k_values': k_values,
            'distortions': distortions,
            'inertias': inertias
        }
    
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
        plt.plot(k_values, silhouette_scores, 'bo-')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Analysis')
        plt.show()
        
        return {
            'k_values': k_values,
            'silhouette_scores': silhouette_scores
        }
    
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
        plt.plot(range(1, max_clusters + 1), gap_values, 'go-')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Gap Statistic')
        plt.title('Gap Statistic Analysis')
        plt.show()
        
        return {
            'k_values': range(1, max_clusters + 1),
            'gap_values': gap_values,
            'reference_dispersions': reference_dispersions
        }

# Example usage in the ImageClusterer class
class ImageClusterer:
    # ... (previous methods remain the same)
    
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
        silhouette_results = ClusterValidator.silhouette_analysis(self.features, max_clusters)
        
        print("\nGap Statistic Analysis:")
        gap_results = ClusterValidator.gap_statistic(self.features, max_clusters)
        
        return self

# Main function to demonstrate usage
def main():
    # Specify your image directory
    image_directory = 'path/to/your/image/directory'
    
    # Create clusterer
    clusterer = ImageClusterer(image_directory)
    
    # Extract features
    clusterer.extract_features()
    
    # Validate clustering and find optimal number of clusters
    clusterer.validate_clusters(max_clusters=10)
    
    # Optionally, cluster with a specific number of clusters based on the analysis
    clusterer.cluster_images(method='kmeans', n_clusters=5)  # Choose based on validation results
    clusterer.visualize_clusters()

```

I've added three primary methods for determining the optimal number of clusters:

1. **Elbow Method**:
   - Looks at the "elbow" point in the distortion or inertia curve
   - Helps identify where adding more clusters provides diminishing returns
   - Plots both distortion and inertia
   - Lower values indicate better clustering

2. **Silhouette Analysis**:
   - Measures how similar an object is to its own cluster compared to other clusters
   - Ranges from -1 to 1
   - Higher score indicates better-defined clusters
   - Best number of clusters is where the score peaks

3. **Gap Statistic**:
   - Compares the total within intra-cluster variation for different values of k
   - Compares clustering to a reference distribution
   - Helps identify the optimal number of clusters statistically

**How to Use**:
```python
# Extract features first
clusterer.extract_features()

# Validate clustering
clusterer.validate_clusters(max_clusters=10)

# Cluster based on the analysis
clusterer.cluster_images(method='kmeans', n_clusters=5)
```

**Interpretation Tips**:
- **Elbow Method**: Look for the "elbow" where the curve starts to flatten
- **Silhouette Score**: Highest point indicates the best number of clusters
- **Gap Statistic**: Look for the peak or where the gap is maximized

**Recommended Workflow**:
1. Run all validation methods
2. Compare the results
3. Choose the number of clusters that makes the most sense for your data
4. Potentially combine insights from multiple methods

Would you like me to explain how to interpret these results or help you determine the best number of clusters for your specific image set?

## setup

```
python3 -m venv venv
source venv/bin/activate
python -m pip install numpy matplotlib scikit-learn tensorflow umap-learn

mkdir images
# here is an archive of 500 tweegeemee images from the end of 2022 
cp /mnt/prev/home/rallen/Documents/tweegeemee_image_archive_2408/2022/*png images
$ ls images/ |wc -l
536

```

## results

DBScan not as good as KMeans.

Works in about 2m37s !  

Increasing the number of clusters works better for me.

### clustervalidator.ipynb

Going to eavaluate what this shows.  For now I'm not really sure...