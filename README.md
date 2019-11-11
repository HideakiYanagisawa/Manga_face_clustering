# Manga_face_clustering
* Classify character face images using density-based clustering.
* Obtain the image features from fine-tuned CNN middle layer.

## Enviroment
 - Python3
 - [Pytorch](http://pytorch.org/)

## 1. Create manga face dataset
* Download images from http://www.manga109.org/ja/index.html
* Write manga titles using trainig and evaluation and save as 'finetune.txt' and 'eval.txt'
* Extract characters' face images for training using train_extract.py
* Split trainig images into 'train_images' and 'test_images' using setup.py
* Extract characters' face images for evaluation using eval_extract.py

## 2. Fine-Tuning
* Fine tune VGG16 or ResNet50 or ResNet101 for manga face images using finetune/finetune_vgg.py or finetune/finetune_resnet.py
* Fine-tuned weight is saved as checkpoint.tar

## 3. visualization
* Visualize image features for character face images in 1 manga book.
* example: python viualize.py --dataset {manga title in evaluation files} --model {fine-tuned model} --layer fc

## 4. Clustering evaluation
* Calcurate the average value of 10 clustering results for character face images of each title in the test set.
* example: python clustering_dbscan.py --model {directory deepcluster model} --layer fc --layer_num 1 --dim 256

## References

UMAP
https://github.com/lmcinnes/umap

DBSCAN
https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html

OPTICS
https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html

HDBSCAN
https://github.com/scikit-learn-contrib/hdbscan
