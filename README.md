# Manga_face_clustering
* Clustering character facial images in manga using DBSCAN.
* I obtain the image features from CNN middle layer, using fine-tuning and DeepClustering.
## Enviroment
 - Python2
 - [Pytorch](http://pytorch.org/)

## 1. Create manga face dataset
* Download images from http://www.manga109.org/ja/index.html
* Extract characters' face images using extract_faces.py

## 2. Fine-Tuning
* Fine tune VGG16 model for manga facial images using finetune/finetune_vgg.py
* Fine-tuned weight is saved as checkpoint.tar

## 3. Train DeepCluster using fine-tuned weight
* Use deepcluster_finetuned_model.py
* example: python deepcluster_finetuned_model.py {dataset derectory} --exp exp_vgg_ft --arch vgg16 --lr 0.05 --wd -5 --k 3000 --sobel --verbose --workers 0 --batch 32 --epochs 200

## 4. Clustering evaluation
* Calcurate the average value of 10 clustering results for character face images of each title in the test set.
* example: python dbscan_vgg_average_new_auto.py --model {directory deepcluster model} --layer fc --layer_num 1 --dim 256

## References

DeepCluster
https://github.com/facebookresearch/deepcluster

feature_extraction
https://github.com/achalddave/pytorch-extract-features
