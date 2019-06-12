# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import os
import argparse
import logging
import random
import sys
from os import path

import h5py
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
import torchvision.models as models
import math
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn import metrics
import hdbscan
from sklearn.cluster import OPTICS

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

def EpsDBSCAN(D, k):
    nn = NearestNeighbors(n_neighbors=k+1)
    nn.fit(D)
    distances, indices = nn.kneighbors(D)
    distances = np.delete(distances, 0, 1)
    Dist = distances.max(axis=1)
    Array = sorted(Dist)
    AvgDist = distances.sum(axis=1)/k
    Avg_Array = sorted(AvgDist)
    #plt.plot(Avg_Array, 'b')

    num = len(Avg_Array)
    n_Array = [0 for i in range(num)]
    minArray = min(Avg_Array)
    maxArray = max(Avg_Array)

    for i in range(num):
        n_Array[i] = (Avg_Array[i]-minArray)/(maxArray-minArray)*(1.0-0.0)

    bins = np.linspace(0, 1, 10)
    bin_indice = np.digitize(n_Array, bins)
    Eps = []
    Avg_Array = np.array(Avg_Array)
    count_max = 0

    for i in range(10):
        count = len(np.where(bin_indice == i)[0])
        if count >= k:
            e = np.sum(Avg_Array[bin_indice == i], axis=0)/count
            #plt.hlines(e, xmin=0, xmax=len(Array), colors='r')
            Eps.append(e)

    N = len(Eps)
    Eps_index = []

    for i in range(N):
        for j in range(num):
            if Avg_Array[j] > Eps[i]:
                Eps_index.append(j)
                break

    ave_slope = (maxArray - minArray)/num
    
    #print('ave slope')
    #print(ave_slope)
    #print('')
    for i in range(N-1):
        slope = (Eps[i+1] - Eps[i]) / (Eps_index[i+1] - Eps_index[i])
        #print(slope)
        if slope > ave_slope * 2:
            out = Eps[i]
            break
        else:
            out = Eps[i+1]

    return out

def EpsValue(D, k):
    nn = NearestNeighbors(n_neighbors=k+1)
    nn.fit(D)
    distances, indices = nn.kneighbors(D)
    distances = np.delete(distances, 0, 1)
    Dist = distances.max(axis=1)
    AvgDist = distances.sum(axis=1)/k

    out = (max(Dist) - min(AvgDist))/100

    return min(AvgDist), out

def extract_features_to_disk(image_paths, model, batch_size, workers, reglog, layer):
    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #print(image_paths)
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
        #input_var = torch.autograd.Variable(input_data, volatile=True).cuda()
        with torch.no_grad():
            input_var = torch.autograd.Variable(input_data).cuda()
        # compute conv features
        if layer=='conv':
            current_features = reglog(forward(input_var, model, reglog.conv)).data.cpu().numpy()
        # compute fc features
        elif layer=='fc':
            current_features = model(input_var).data.cpu().numpy()
        for j, image_path in enumerate(paths):
            features[image_path] = current_features[j]
    feature_shape = features[list(features.keys())[0]].shape
    logging.info('Feature shape: %s' % (feature_shape, ))
    #logging.info('Outputting features')

    if sys.version_info >= (3, 0):
        string_type = h5py.special_dtype(vlen=str)
    else:
        string_type = h5py.special_dtype(vlen=unicode)  # noqa
    #paths = features.keys()
    paths = image_paths
    logging.info('Stacking features')
    features_stacked = np.vstack([features[path] for path in paths])
    ##
    logging.info('Output feature size: %s' % (features_stacked.shape, ))

    ##dim = 64
    ##pca = KernelPCA(n_components=dim, kernel='cosine')
    ##pca.fit(features_stacked)
    ##reduction_result = pca.transform(features_stacked)
    umap_model = umap.UMAP(n_components=2, metric='cosine')
    ##reduction_result = umap_model.fit_transform(reduction_result)
    reduction_result = umap_model.fit_transform(features_stacked)
    print(reduction_result.shape)

    return reduction_result

def load_model(path, cls_num):
    """Loads model and return it without DataParallel table."""
    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path)

        # size of the top layer
        N = checkpoint['state_dict']['top_layer.bias'].size()

        # build skeleton of the model
        sob = 'sobel.0.weight' in checkpoint['state_dict'].keys()
        #model = models.__dict__[checkpoint['arch']](sobel=sob, out=int(N[0]))

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
        print("Loaded")
    else:
        model = None
        print("=> no checkpoint found at '{}'".format(path))
    return model

def clustering_dbscan(result, MINPTS, labels):
    min_eps, e_value = EpsValue(result, MINPTS)
    best_eps = 0.0
    best_alpha = 0.0
    best_fmeasure = 0.0
    best_nmi = 0.0
    best_ami = 0.0

    #data_num = 0
    #e = min_eps
    #N = len(labels)

    e = EpsDBSCAN(result, MINPTS)
    #print e
    data_num = 0
    N = len(labels)

    db = DBSCAN(eps=e, min_samples=MINPTS).fit(result)

    #while data_num < N:
    #    e = e + e_value
    #    db = DBSCAN(eps=e, min_samples=MINPTS).fit(result)
    #    data_num = len(db.labels_[db.labels_>=0])

    #    if (max(db.labels_) >= 0):

    #        ami = metrics.adjusted_mutual_info_score(labels, db.labels_, average_method='arithmetic')
    #        nmi = metrics.normalized_mutual_info_score(labels, db.labels_, average_method='arithmetic')
 
            #if nmi > best_nmi:
    #        if ami > best_ami:
    #            best_e = e
    #            best_nmi = nmi
                #best_purity = purity
                #best_inverse = inverse
                #best_f_measure = f_measure
    #            best_class = float(max(db.labels_)+1)
                #best_precision = precision
                #best_recall = recall
    #            best_ami = ami

    #print (best_ami)
    #print ('purity | inverse purity | F-measure | class num | precision | recall | NMI')
    #print ('{0:.3f}'.format(best_purity), '|', '{0:.3f}'.format(best_inverse), '|', '{0:.3f}'.format(best_f_measure), '|', '{0:.3f}'.format(best_class), '|', '{0:.3f}'.format(best_precision), '|', '{0:.3f}'.format(best_recall), '|', '{0:.3f}'.format(best_nmi))

    #best_db = DBSCAN(eps=best_e, min_samples=MINPTS).fit(result)
    return db

def main():

    parser = argparse.ArgumentParser(description="""Train linear classifier on top
                                 of frozen convolutional layers of an AlexNet.""")

    parser.add_argument('--dataset', type=str, help='target dataset')
    parser.add_argument('--model', type=str, help='path to model')
    parser.add_argument('--layer', type=str, help='layer type')
    parser.add_argument('--layer_num', default=2, type=int, help='layer number')
    parser.add_argument('--cls_num', default=3000, type=int, help='model class')
    parser.add_argument('--workers', default=0, type=int,
                        help='number of data loading workers (default: 0)')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='mini-batch size (default: 64)')
    parser.add_argument('--seed', type=int, default=31, help='random seed')

    global args

    args = parser.parse_args()

    #print args.output_features
    #assert not path.exists(args.output_features)

    #if args.output_log is None:
    #    args.output_log = args.output_features + '.log'
    #_set_logging(args.output_log)
    #logging.info('Args: %s', args)

    #fix random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    layer = args.layer
    layer_num = args.layer_num

    # load model
    model = load_model(args.model, args.cls_num)

    #param = torch.load(args.model)
    #model.load_state_dict(param)
    #print(model)
    model.top_layer = None
    if layer == 'fc':
        if layer_num == 1:
            new_classifier = nn.Sequential(*list(model.classifier.children())[:-4])
            model.classifier = new_classifier
        if layer_num == 2:
            new_classifier = nn.Sequential(*list(model.classifier.children())[:-1])
            model.classifier = new_classifier

    model.cuda()
    cudnn.benchmark = True
    model.eval()

    filename = args.dataset
    filepath = '/faces_83/evaluation/' + filename + '/'
    #print(filepath)
    class_list = glob(filepath+'*')
    class_list = [os.path.basename(r) for r in class_list]
    class_num = len(class_list)

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

    # logistic regression
    reglog = RegLog(layer_num, 10000).cuda()

    reduction_result=extract_features_to_disk(images, model, args.batch_size,
                                              args.workers, reglog, layer)

    #db = clustering_dbscan(reduction_result, 10, labels)

    cluster = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=1)
    cluster_labels = cluster.fit_predict(reduction_result)

    #cluster = OPTICS(min_samples=20, min_cluster_size=10, xi=0.05)
    #cluster.fit(reduction_result)
    #cluster_labels = cluster.labels_
    #print cluster_labels

    ami = metrics.adjusted_mutual_info_score(labels, cluster_labels, average_method='arithmetic')
    print (ami)   


    colormap=['red', 'blue', 'green', 'yellow', 'orange', 'pink', 'brown', 'purple', 'aqua', 'lime', 'gray', 'aqua', 'black', 'peru', 'indigo', 'tomato', 'skyblue', 'darkgreen', 'gold', 'silver']
    c = 0
    for i in range(len(len_images)):
       cls = int(len_images[i])
       plt.plot(reduction_result[c:c+cls-1,0], reduction_result[c:c+cls-1,1], ".", color=colormap[i])
       c += cls
    plt.savefig(filename + '_ft_' + layer + str(layer_num) + '_gt' + '.png')

    #for i in range(-1, max(db.labels_)+1):
    #    label = reduction_result[db.labels_ == i]
    #    plt.plot(label[:,0], label[:,1], ".", color=colormap[i+1])

    #for i in range(-1, max(cluster_labels)+1):
    #    label = reduction_result[cluster_labels == i]
    #    plt.plot(label[:,0], label[:,1], ".", color=colormap[i+1])

    #plt.savefig(filename + '_ft_' + layer + str(layer_num) + '_db' + '.png')
    #plt.savefig(filename + '_ft_' + layer + str(layer_num) + '_db' + '_dbscan.png')
    plt.show()


if __name__ == '__main__':
    main()
