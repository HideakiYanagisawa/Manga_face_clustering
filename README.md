# Manga_face_clustering
* Classify character face images using density-based clustering.
* Obtain the image features from fine-tuned CNN middle layer.

* 密度ベースアルゴリズムを用いた漫画キャラクター顔画像の分類。
* ファインチューニングしたCNNの中間出力を画像特徴量として使用する。

## Enviroment
 - Python3
 - [Pytorch](http://pytorch.org/)
 - sklearn
 - [umap](https://github.com/lmcinnes/umap)
 - [hdbscan](https://github.com/scikit-learn-contrib/hdbscan)

## 1. Create manga face dataset
* Download images from http://www.manga109.org/ja/index.html
* Write manga titles using trainig and evaluation and save as 'dataset/finetune.txt' and 'dataset/eval.txt'
* Edit fine_path in 'dataset/train_extract.py' and 'dataset/eval_extract.py'
* Extract characters' face images for training using dataset/train_extract.py
* Split trainig images into 'train_images' and 'test_images' using dataset/setup.py
* Extract characters' face images for evaluation using dataset/eval_extract.py

* [Manga109](http://www.manga109.org/ja/index.html)より漫画画像をダウンロード。
* CNNの学習とクラスタリング評価に使用したい漫画のファイル名をそれぞれ「dataset/finetune.txt」と「dataset/eval.txt」に記載。
* 「dataset/train_extract.py」と「dataset/eval_extract.py」のfile_pathを正しいアドレスに変更。
* Manga109のアノテーションデータに従い、dataset/train_extract.pyで学習用の顔画像データを抽出。
* dataset/eval_extract.pyで評価用の顔画像データを抽出。

## 2. Fine-Tuning
* Fine tune VGG16 or ResNet50 or ResNet101 for manga face images using finetune/finetune_vgg.py or finetune/finetune_resnet.py
* Fine-tuned weight is saved as tar file

* VGG16を使用する場合はfinetune/finetune_vgg.pyで、ResNet50またはResNet101を使用する場合はfinetune/finetune_vgg.pyでCNNを学習。
* 学習データはtarファイルで保存される。

## 3. visualization
* Visualize image features for character face images in 1 manga book.
* python viualize.py --dataset {manga title in evaluation files} --cnn {vgg16 | resnet50 | resnet101} --model {fine-tuned model} --layer {fc | conv (only vgg16)} --layer_num 2 --clustering {dbscan | optics | hdbscan}

* 以下のコードで評価データセットの1作品について、可視化した登場キャラクターの特徴空間を表示する。
* python viualize.py --dataset {manga title in evaluation files} --cnn {vgg16 | resnet50 | resnet101} --model {fine-tuned model} --layer {fc | conv (only vgg16)} --layer_num 2 --clustering {dbscan | optics | hdbscan}

## 4. Clustering evaluation
* Calcurate the average value of 10 clustering results for character face images of each title in the test set.
* python clustering_dbscan.py  --cnn {vgg16 | resnet50 | resnet101} --model {fine-tuned model} --layer {fc | conv (only vgg16)} --layer_num 2 --clustering {dbscan | optics | hdbscan} --dim 32

* 以下のコードで評価データセットの全作品についてクラスタリング精度を評価する。
* それぞれの作品について10回ずつクラスタリングを行い、平均値を求める。
* python clustering_dbscan.py  --cnn {vgg16 | resnet50 | resnet101} --model {fine-tuned model} --layer {fc | conv (only vgg16)} --layer_num 2 --clustering {dbscan | optics | hdbscan} --dim 32

## References

UMAP
https://github.com/lmcinnes/umap

DBSCAN
https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html

OPTICS
https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html

HDBSCAN
https://github.com/scikit-learn-contrib/hdbscan
