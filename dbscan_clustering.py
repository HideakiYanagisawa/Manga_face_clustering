import os
import random
import sys
from os import path
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets.folder import default_loader
from tqdm import tqdm
from glob import glob
import umap
from scipy.spatial import distance
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.decomposition import PCA, KernelPCA
from torch.nn import DataParallel
import hdbscan

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        x = nn.functional.normalize(x)
        return x

class RegLog(nn.Module):
    """Creates logistic regression on top of frozen features"""
    def __init__(self, layer_num, num_labels):
        super(RegLog, self).__init__()
        if layer_num ==1:
            # layer2
            conv = 2
            self.av_pool = nn.AvgPool2d(19, stride=19, padding=2)
            s = 9216
        elif layer_num ==2:
            # layer4
            conv = 4
            self.av_pool = nn.AvgPool2d(14, stride=14, padding=0)
            s = 8192
        elif layer_num ==3:
            # layer7
            conv = 7
            self.av_pool = nn.AvgPool2d(10, stride=10, padding=2)
            s = 9216
        elif layer_num ==4:
            # layer10
            conv = 10
            self.av_pool = nn.AvgPool2d(7, stride=7, padding=0)
            s = 8192
        elif layer_num ==5:
            # layer13
            conv = 13
            self.av_pool = nn.AvgPool2d(4, stride=4, padding=1)
            s = 8192
        self.conv = conv
        self.linear = nn.Linear(s, num_labels)

    def forward(self, x):
        x = self.av_pool(x)
        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        #return self.linear(x)

        x = self.linear(x)
        x = nn.functional.normalize(x)
        return x

def conv_forward(x, model, conv):
    #if hasattr(model, 'sobel') and model.sobel is not None:
    #    x = model.sobel(x)
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
    # find min nearest neighbor and interval
    nearest = NearestNeighbors(n_neighbors=k+1)
    nearest.fit(D)
    distances, indices = nearest.kneighbors(D)
    distances = np.delete(distances, 0, 1)
    Dist = distances.max(axis=1)
    AvgDist = distances.sum(axis=1)/k

    interval = (max(Dist) - min(AvgDist)) / 100

    return min(AvgDist), interval

def EpsDBSCAN(D, k):
    # Find the optimal eps based on the gradient of Nearest Neighbor
    nearest = NearestNeighbors(n_neighbors=k+1)
    nearest.fit(D)
    distances, indices = nearest.kneighbors(D)
    distances = np.delete(distances, 0, 1)
    Dist = distances.max(axis=1)
    #Array = sorted(Dist)
    AvgDist = distances.sum(axis=1)/k
    Avg_Array = sorted(AvgDist)

    N = len(Avg_Array)
    norm_Array = [0.0 for i in range(N)]
    minArray = min(Avg_Array)
    maxArray = max(Avg_Array)

    # normalization
    for i in range(N):
        norm_Array[i] = (Avg_Array[i]-minArray)/(maxArray-minArray)*(1.0-0.0)

    # binning
    bins = np.linspace(0, 1, 10)
    bin_indice = np.digitize(norm_Array, bins)
    Eps = []
    Avg_Array = np.array(Avg_Array)

    for i in range(10):
        count = len(np.where(bin_indice == i)[0])
        if count >= k:
            e = np.sum(Avg_Array[bin_indice == i], axis=0)/count
            Eps.append(e)

    N = len(Eps)
    num = len(Avg_Array)
    Eps_index = []

    # find Avg_Array index over each Eps value
    for i in range(N):
        for j in range(num):
            if Avg_Array[j] > Eps[i]:
                Eps_index.append(j)
                break

    ave_slope = (maxArray - minArray)/N
    Slopes = []
    
    # caluculate slope of Eps value
    for i in range(N-1):
        slope = (Eps[i+1] - Eps[i]) / (Eps_index[i+1] - Eps_index[i])
        Slopes.append(slope)

    ave_slope = sum(Slopes)/len(Slopes)

    # find the point over average slope
    for i in range(N-1):
        if i > 0 and Slopes[i] > ave_slope:
            out = Eps[i]
            break
        else:
            out = Eps[i+1]

    return out


def extract_features(image_paths, model, batch_size, workers, reglog, layer):
    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
            input_var = input_data.to('cuda')
        # compute conv features
        if layer=='conv':
            current_features = reglog(conv_forward(input_var, model, reglog.conv)).data.to('cpu').numpy()
        # compute fc features
        elif layer=='fc':
            current_features = model(input_var).data.to('cpu').numpy()
            
        for j, image_path in enumerate(paths):
            features[image_path] = current_features[j]
    ##feature_shape = features[list(features.keys())[0]].shape
        
    features_stacked = np.vstack([features[path] for path in image_paths])

    return features_stacked

def clustering_score(labels_true, cluster_labels):

    homogeneity = metrics.homogeneity_score(labels_true, cluster_labels)
    completeness = metrics.completeness_score(labels_true, cluster_labels)
    v_measure = metrics.v_measure_score(labels_true, cluster_labels)
    ari = metrics.adjusted_rand_score(labels_true, cluster_labels)
    ami = metrics.adjusted_mutual_info_score(labels_true, cluster_labels, average_method='arithmetic')
    nmi = metrics.normalized_mutual_info_score(labels_true, cluster_labels, average_method='arithmetic')

    cluster_num = max(cluster_labels)+1
    noise_num = len(cluster_labels[cluster_labels == -1])

    print('Homogeneity | Completeness | V-measure | cluster num | noise num | ARI | NMI | AMI')
    print('{0:.3f}'.format(homogeneity), '|', '{0:.3f}'.format(completeness), '|', '{0:.3f}'.format(v_measure), '|', '{0}'.format(cluster_num), '|', '{0}'.format(noise_num), '|', '{0:.3f}'.format(ari), '|', '{0:.3f}'.format(nmi), '|', '{0:.3f}'.format(ami))

    return v_measure, ari, ami

def clustering_dbscan_best(result, MINPTS, labels_true):
    min_eps, e_interval = EpsValue(result, MINPTS)

    best_ami = 0.0
    data_num = 0
    e = min_eps
    N = len(labels_true)

    while data_num < N:
        e += e_interval
        db = DBSCAN(eps=e, min_samples=MINPTS).fit(result)
        data_num = len(db.labels_[db.labels_>=0])

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

    print('Homogeneity | Completeness | V-measure | cluster num | noise num | ARI | NMI | AMI')
    print('{0:.3f}'.format(best_homogeneity), '|', '{0:.3f}'.format(best_completeness), '|', '{0:.3f}'.format(best_v_measure), '|', '{0}'.format(cluster_num), '|', '{0}'.format(noise_num), '|', '{0:.3f}'.format(best_ari), '|', '{0:.3f}'.format(best_nmi), '|', '{0:.3f}'.format(best_ami))

    return best_v_measure, best_ari, best_ami

def clustering_dbscan(result, MINPTS, labels_true):
    e = EpsDBSCAN(result, MINPTS)
    db = DBSCAN(eps=e, min_samples=MINPTS).fit(result)
    v_measure, ari, ami = clustering_score(labels_true, db.labels_)
    return v_measure, ari, ami

def clustering_hdbscan(result, MINPTS, labels_true):
    cluster = hdbscan.HDBSCAN(min_cluster_size=MINPTS, min_samples=1)
    cluster_labels = cluster.fit_predict(result)
    v_measure, ari, ami = clustering_score(labels_true, cluster_labels)
    return v_measure, ari, ami

def clustering_optics(result, MINPTS, labels_true):
    cluster = OPTICS(min_samples=MINPTS, min_cluster_size=MINPTS, xi=0.35)
    cluster.fit(result)
    cluster_labels = cluster.labels_
    v_measure, ari, ami = clustering_score(labels_true, cluster_labels)
    return v_measure, ari, ami

def main():

    parser = argparse.ArgumentParser(description="""Classtering images using fine-tuned CNN""")
    parser.add_argument('--model', type=str, help='path to model')
    parser.add_argument('--cnn', type=str, help='model archtecture')
    parser.add_argument('--layer', type=str, help='layer type')
    parser.add_argument('--layer_num', default=2, type=int, help='layer number')
    parser.add_argument('--workers', default=0, type=int,
                        help='number of data loading workers (default: 0)')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='mini-batch size (default: 64)')
    parser.add_argument('--dim', default=32, type=int, help='feature dimension')
    parser.add_argument('--clustering', default='hdbscan', type=str, help='clustering method')

    #global args

    args = parser.parse_args()

    # load model
    if args.cnn == 'vgg16':
        model = models.vgg16(pretrained=None)
        model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
        model.top_layer = nn.Linear(4096, 1222)
        param = torch.load(args.model)
        model.load_state_dict(param)
        
        if args.layer == 'fc':
            if args.layer_num == 1:
                new_classifier = nn.Sequential(*list(model.classifier.children())[:-5])
            if args.layer_num == 2:
                new_classifier = nn.Sequential(*list(model.classifier.children())[:-2])
            model.classifier = new_classifier

        model.top_layer =  Identity()

    if args.cnn == 'resnet50':
        model = models.resnet50(pretrained=None)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 1222)

        param = torch.load(args.model)
        model.load_state_dict(param)
        model.fc =  Identity()

    if args.cnn == 'resnet101':
        model = models.resnet101(pretrained=None)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 1222)

        param = torch.load(args.model)
        model.load_state_dict(param)
        model.fc =  Identity()

    #model = DataParallel(model)
    model.to('cuda')
    cudnn.benchmark = True
    model.eval()

    # logistic regression
    reglog = RegLog(args.layer_num, 10000).to('cuda')

    datasets = glob('/faces_83/evaluation/*')

    V_MEASUREs = []
    ARIs = []
    AMIs = []

    for filepath in datasets:
        filepath = filepath + '/'
        print(filepath)
        class_list = glob(filepath+'*')
        class_list = [os.path.basename(r) for r in class_list]
        class_num = len(class_list)

        len_images = []
        images = []
        labels = []
        for n, class_name in enumerate(class_list):
            class_images = glob(filepath+class_name+'/*.jpg')
            images.extend(class_images)
            len_images.append(float(len(class_images)))
            label = [n] * len(class_images)
            labels.extend(label)

        seed_v_measure = []
        seed_ari = []
        seed_ami = []

        for seed in range(10):
            #fix random seeds
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)

            features_stacked = extract_features(images, model, args.batch_size,
                                                args.workers, reglog, args.layer)
            # dimensional compression
            umap_model = umap.UMAP(n_components=args.dim, metric='cosine')
            reduction_result = umap_model.fit_transform(features_stacked)

            if args.clustering == 'dbscan':
                v_measure, ari, ami = clustering_dbscan(reduction_result, 10, labels)
            elif args.clustering == 'optics':
                v_measure, ari, ami = clustering_optics(reduction_result, 10, labels)
            else:
                v_measure, ari, ami = clustering_hdbscan(reduction_result, 10, labels)

            seed_v_measure.append(v_measure)
            seed_ari.append(ari)
            seed_ami.append(ami)

        ave_v_measure = sum(seed_v_measure) / len(seed_v_measure)
        ave_ari = sum(seed_ari) / len(seed_ari)
        ave_ami = sum(seed_ami) / len(seed_ami)
        V_MEASUREs.append(ave_v_measure)
        ARIs.append(ave_ari)
        AMIs.append(ave_ami)
        print('class average V_measure | ARI | AMI')
        print('{0:.3f}'.format(ave_v_measure), '|', '{0:.3f}'.format(ave_ari), '|', '{0:.3f}'.format(ave_ami))
        print('')

    print('average V_measure | ARI | AMI')
    print('{0:.3f}'.format(sum(V_MEASUREs)/len(V_MEASUREs)), '|', '{0:.3f}'.format(sum(ARIs)/len(ARIs)), '|', '{0:.3f}'.format(sum(AMIs)/len(AMIs)))

if __name__ == '__main__':
    main()
