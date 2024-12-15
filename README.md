# Image Cluster

An experiment in using Claude.ai to code something up.

## prompt

I would like to take a directory of images and "score" the images in some manner in order to find and display the images in a way to show clusters of similar images.

Afterwards, I also asked to load & save the model.

## claude 3.5 Haiku response

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

