# Image Cluster

The www.tweegeemee.com bot has created a ton of images and I would like to do some exploration of the images to find similar images.  Instead of diving into coding right off, I decided to ask an LLM for help.  This repo is an experiment in using Claude.ai 3.5 Haiku to code something up quickly.

## setup

I did the following without Claude's help to get a python environment setup and some images to play with.

```
python3 -m venv venv
source venv/bin/activate
# (okay the list of modules was from claude)
python -m pip install numpy matplotlib scikit-learn tensorflow umap-learn

mkdir images
# here is an archive of 500 tweegeemee images from the end of 2022 
cp /mnt/prev/home/rallen/Documents/tweegeemee_image_archive_2408/2022/*png images
$ ls images/ |wc -l
536

```

## TLDR results

I found this experience indistinguishable from working with a knowledgable expert.   There were only a very minor number of issues with the code.  Crazily bug free.  Not that I will use this code as-is, but I got something working and useful without taxing my brain at all.  I'm quite happy with how this experiment went.  I think I'll be able to step off to make something of my own pretty easily.

* __imgcluster.py__ - The main python code went into this file.

* __clustervalidator.ipynb__ - When I asked about clusters, it created functions that plotted data so I put it in a Jupyter notebook.  Looking at what this shows...it just always seems to get better wtih more clusters.  I didn't see much of an "elbow" or "knee" in the curve.

* __similar_images.ipynb__ - Again, the similar images code was plotting interactive graphs, so I made another notebook.  This looks really useful.  

## prompts

Not a full transcript, but most of the non-code output.  For the code see the code.

### prompt 1

_I would like to take a directory of images and "score" the images in some manner in order to find and display the images in a way to show clusters of similar images._

_Afterwards, I also asked to load & save the model._

### response

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

_when doing kmeans clustering, how can I decide n_clusters?  Can I try many clusters and "score" the clustering?_

### response

Absolutely! I'll enhance the script to include several methods for determining the optimal number of clusters, including:

1. Elbow Method
2. Silhouette Score
3. Gap Statistic

[code deleted]

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

_I asked for more information about Distortion and Inertia_

Distortion:

Measures the average distance between each point and its assigned cluster center
Calculated by finding the minimum distance from each point to all cluster centers, then taking the mean
Lower distortion means points are closer to their assigned cluster centers
As you increase the number of clusters, distortion typically decreases
Represents how well the clusters represent the underlying data points

Inertia:

Measures the sum of squared distances of samples to their closest cluster center
Essentially calculates the compactness of the clusters
Lower inertia indicates more compact, tightly grouped clusters
Calculated by summing the squared distances between each point and its cluster center
Directly related to how "tight" or "spread out" your clusters are

Practical Analogy:
Imagine you're grouping a set of books into shelves:

Distortion is like measuring the average distance each book is from the center of its shelf
Inertia is like measuring how tightly packed the books are on each shelf

Mathematical Representation:

Distortion: Average(min(distance(point, all cluster centers)))
Inertia: Sum((distance(point, cluster_center))²)

Key Differences:

Distortion gives an average view of cluster spread
Inertia provides a total measure of cluster compactness
Both help in determining the optimal number of clusters

### prompt 3 

_I'd like to ask the model to answer a different question.  Given a self.features index (i), find the closest N images or perhaps the closest images that are within some metric (R?) distance from self.features[i]_

### response

I'll modify the ImageClusterer class to add methods for finding the closest images:

[removed code]

I've added two new methods to the ImageClusterer class:

find_similar_images(index, n_images=5, max_distance=None):

Finds the most similar images to a reference image
Two ways to control similarity:
a. n_images: Number of similar images to return
b. max_distance: Maximum distance for considering an image similar


visualize_similar_images(index, n_images=5, max_distance=None):

Creates a visual grid of the reference image and its most similar images
Shows the distance to each similar image



Usage Examples:
```
# Find 5 most similar images to the image at index 0
similar_images = clusterer.find_similar_images(index=0, n_images=5)

# Find images within a distance of 2.0 from the image at index 0
similar_within_distance = clusterer.find_similar_images(index=0, max_distance=2.0)

# Visualize similar images
clusterer.visualize_similar_images(index=0, n_images=5)
```
Key Features:

Uses Euclidean distance to measure image similarity
Flexible filtering by number of images or maximum distance
Provides both programmatic and visual ways to explore image similarities

What's Happening Under the Hood:

Calculates distances between feature vectors
Sorts images by their distance to the reference image
Can return either a fixed number of closest images or all images within a distance threshold

Would you like me to elaborate on how the similarity is calculated or show you how to use these methods with your specific image set?



