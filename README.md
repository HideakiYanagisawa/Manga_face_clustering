# Manga_face_clustering
* Clustering character facial images in manga using DBSCAN.
* I obtain the image features from CNN middle layer, using fine-tuning and DeepClustering.
## Enviroment
 - Python2
 - [Pytorch](http://pytorch.org/)

## Create manga face dataset
* Download images from http://www.manga109.org/ja/index.html
* Extract characters' face images using extract_faces.py

## Fine-Tuning
* Fine tune VGG16 model for manga facial images using finetune/finetune_vgg.py
* Fine-tuned weight is saved as checkpoint.tar

## References

DeepCluster
https://github.com/facebookresearch/deepcluster

feature_extraction
https://github.com/achalddave/pytorch-extract-features
