# Extract feature quantities and find clustering results that maximize AMI value.
# Use UMAP for dimensional reduction.

import argparse
import os
import logging
import random
import sys
from os import path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets.folder import default_loader
from tqdm import tqdm
from glob import glob
from sklearn.decomposition import PCA, KernelPCA
import umap
#from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import math
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn import metrics

class RegLog(nn.Module):
    """Creates logistic regression on top of frozen features"""
    def __init__(self, layer_num, num_labels):
        super(RegLog, self).__init__()
        if layer_num ==1:
            conv = 2
        elif layer_num ==2:
            conv = 4
        elif layer_num ==3:
            conv = 7
        elif layer_num ==4:
            conv = 10
        elif layer_num ==5:
            conv = 13

        self.conv = conv
        if conv==2:
            self.av_pool = nn.AvgPool2d(19, stride=19, padding=2)
            s = 9216
        elif conv==4:
            self.av_pool = nn.AvgPool2d(14, stride=14, padding=0)
            s = 8192
        elif conv==7:
            self.av_pool = nn.AvgPool2d(10, stride=10, padding=2)
            s = 9216
        elif conv==10:
            self.av_pool = nn.AvgPool2d(7, stride=7, padding=0)
            s = 8192
        elif conv==13:
            self.av_pool = nn.AvgPool2d(4, stride=4, padding=1)
            s = 8192
        self.linear = nn.Linear(s, num_labels)

    def forward(self, x):
        x = self.av_pool(x)
        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        return self.linear(x)


def forward(x, model, conv):
    if hasattr(model, 'sobel') and model.sobel is not None:
        x = model.sobel(x)
    count = 1
    for m in model.features.modules():
        if not isinstance(m, nn.Sequential):
            x = m(x)
        if isinstance(m, nn.ReLU):
            if count == conv:
                return x
            count = count + 1
    return x

class ListDataset(torch.utils.data.Dataset):
    def __init__(self,
                 images_list,
                 transform=None,
                 loader=default_loader):
        self.images_list = images_list
        self.loader = loader
        self.transform = transform

    def __getitem__(self, index):
        image_path = self.images_list[index]
        image = self.loader(image_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, image_path

    def __len__(self):
        return len(self.images_list)

def EpsValue(D, k):
    nn = NearestNeighbors(n_neighbors=k+1)
    nn.fit(D)
    distances, indices = nn.kneighbors(D)
    distances = np.delete(distances, 0, 1)
    Dist = distances.max(axis=1)
    AvgDist = distances.sum(axis=1)/k

    out = (max(Dist) - min(AvgDist))/100

    return min(AvgDist), out

def extract_features_to_disk(image_paths, model, batch_size, workers, reglog, layer, dim):
    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #print image_paths
    dataset = ListDataset(image_paths,
                          transforms.Compose([
                              transforms.Resize(256),
                              transforms.CenterCrop(224),
                              transforms.ToTensor(),
                              normalize,
                          ]))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True)

    features = {}
    for i, (input_data, paths) in enumerate(tqdm(loader)):
        input_var = torch.autograd.Variable(input_data, volatile=True).cuda()
        # compute conv features
        if layer=='conv':
            current_features = reglog(forward(input_var, model, reglog.conv)).data.cpu().numpy()
        # compute fc features
        elif layer=='fc':
            current_features = model(input_var).data.cpu().numpy()
            
        for j, image_path in enumerate(paths):
            features[image_path] = current_features[j]
    feature_shape = features[list(features.keys())[0]].shape
        
    paths = image_paths
    features_stacked = np.vstack([features[path] for path in paths])
    
    umap_model = umap.UMAP(n_components=dim)
    reduction_result = umap_model.fit_transform(features_stacked)

    return reduction_result

def load_model(path, cls_num):
    """Loads model and return it without DataParallel table."""
    if os.path.isfile(path):
        checkpoint = torch.load(path)

        # size of the top layer
        N = checkpoint['state_dict']['top_layer.bias'].size()

        # build skeleton of the model
        sob = 'sobel.0.weight' in checkpoint['state_dict'].keys()

        model = models.vgg16(pretrained=None)
        model.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
        )
        model.top_layer = nn.Linear(4096, cls_num)

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            if self.top_layer:
                x = self.top_layer(x)
            return x

        # deal with a dataparallel table
        def rename_key(key):
            if not 'module' in key:
                return key
            return ''.join(key.split('.module'))

        checkpoint['state_dict'] = {rename_key(key): val
                                    for key, val
                                    in checkpoint['state_dict'].items()}

        # load weights
        model.load_state_dict(checkpoint['state_dict'])
        #print("Loaded")
    else:
        model = None
        #print("=> no checkpoint found at '{}'".format(path))
    return model

def clustering_dbscan(result, MINPTS, labels_true):
    min_eps, e_value = EpsValue(result, MINPTS)

    best_eps = 0.0
    cluster_num = 0
    noise_num = 0
    best_homogeneity = 0.0
    best_completeness = 0.0
    best_v_measure = 0.0
    best_ari = 0.0
    best_nmi = 0.0
    best_ami = 0.0

    data_num = 0
    e = min_eps
    N = len(labels_true)

    while data_num < N:
        e = e + e_value
        db = DBSCAN(eps=e, min_samples=MINPTS).fit(result)
        data_num = len(db.labels_[db.labels_>=0])

        sum_max = 0.0
        sum_cluster = 0.0
        sum_inverse = 0.0
        sum_cluster_inverse = 0.0
        sum_cls_num = 0.0

        if (max(db.labels_) >= 0):
            homogeneity = metrics.homogeneity_score(labels_true, db.labels_)
            completeness = metrics.completeness_score(labels_true, db.labels_)
            v_measure = metrics.v_measure_score(labels_true, db.labels_)
            ari = metrics.adjusted_rand_score(labels_true, db.labels_)
            ami = metrics.adjusted_mutual_info_score(labels_true, db.labels_, average_method='arithmetic')
            nmi = metrics.normalized_mutual_info_score(labels_true, db.labels_)

            if ami > best_ami:
                best_ami = ami
                best_nmi = nmi
                best_homogeneity = homogeneity
                best_completeness = completeness
                best_v_measure = v_measure
                best_ari = ari
                cluster_num = max(db.labels_)+1
                noise_num = len(db.labels_[db.labels_ == -1])

    print 'Homogeneity | Completeness | V-measure | cluster num | noise num | ARI | NMI | AMI'
    print '{0:.3f}'.format(best_homogeneity), '|', '{0:.3f}'.format(best_completeness), '|', '{0:.3f}'.format(best_v_measure), '|', '{0}'.format(cluster_num), '|', '{0}'.format(noise_num), '|', '{0:.3f}'.format(best_ari), '|', '{0:.3f}'.format(best_nmi), '|', '{0:.3f}'.format(best_ami)

    return best_v_measure, best_ari, best_ami

def main():

    parser = argparse.ArgumentParser(description="""Extract image features of manga facial images.""")
    
    parser.add_argument('--model', type=str, help='path to model')
    parser.add_argument('--layer', type=str, help='layer type')
    parser.add_argument('--layer_num', default=2, type=int, help='layer number')
    parser.add_argument('--cls_num', default=3000, type=int, help='model class')
    parser.add_argument('--workers', default=0, type=int,
                        help='number of data loading workers (default: 0)')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='mini-batch size (default: 32)')
    parser.add_argument('--dim', default=2, type=int, help='feature dimension')

    global args

    args = parser.parse_args()

    layer = args.layer
    layer_num = args.layer_num

    # load model
    model = load_model(args.model, args.cls_num)

    if layer == 'fc':
        if layer_num == 1:
            new_classifier = nn.Sequential(*list(model.classifier.children())[:-2])
            model.classifier = new_classifier
        if layer_num == 2:
            new_classifier = nn.Sequential(*list(model.classifier.children())[:-1])
            model.classifier = new_classifier

    model.cuda()
    cudnn.benchmark = True
    model.eval()

    datasets = glob('/dataset/eval_images/*')

    V_MEASUREs = [0.0] * len(datasets)
    ARIs = [0.0] * len(datasets)
    AMIs = [0.0] * len(datasets)

    num = 0
    for filepath in datasets:
        filepath = filepath + '/'
        print filepath
        class_list = glob(filepath+'*')
        class_list = [os.path.basename(r) for r in class_list]
        class_num = len(class_list)

        ex_class_list = class_list
        ex_class_list.remove('other')

        len_images = []
        images = []
        labels = []
        n = 0
        for class_name in class_list:
            class_images = glob(filepath+class_name+'/*.jpg')
            images.extend(class_images)
            len_images.append(float(len(class_images)))
            label = [n] * len(class_images)
            labels.extend(label)
            n += 1

        ex_len_images = []
        for class_name in ex_class_list:
            class_images = glob(filepath+class_name+'/*.jpg')
            ex_len_images.append(float(len(class_images)))

        # logistic regression
        reglog = RegLog(layer_num, 10000).cuda()

        seed_v_measure = 0.0
        seed_ari = 0.0
        seed_ami = 0.0

        for seed in range(10):
            #fix random seeds
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)

            reduction_result=extract_features_to_disk(images, model, args.batch_size,
                                                      args.workers, reglog, layer, args.dim)

            v_measure, ari, ami = clustering_dbscan(reduction_result, 10, labels)
            seed_v_measure = seed_v_measure + v_measure
            seed_ari = seed_ari + ari
            seed_ami = seed_ami + ami

        seed_v_measure = seed_v_measure / 10
        seed_ari = seed_ari / 10
        seed_ami = seed_ami / 10
        V_MEASUREs[num] = seed_v_measure
        ARIs[num] = seed_ari
        AMIs[num] = seed_ami
        num += 1
        print 'class average V_measure | ARI | AMI'
        print '{0:.3f}'.format(seed_v_measure), '|', '{0:.3f}'.format(seed_ari), '|', '{0:.3f}'.format(seed_ami)
        print ''

    print 'average V_measure | ARI | AMI'
    print '{0:.3f}'.format(sum(V_MEASUREs)/len(V_MEASUREs)), '|', '{0:.3f}'.format(sum(ARIs)/len(ARIs)), '|', '{0:.3f}'.format(sum(AMIs)/len(AMIs))

if __name__ == '__main__':
    main()
