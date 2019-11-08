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
#from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from torch.nn import DataParallel

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

def main():

    parser = argparse.ArgumentParser(description="""Classtering images using fine-tuned CNN model""")
    parser.add_argument('--dataset', type=str, help='target dataset')
    parser.add_argument('--model', type=str, help='path to model')
    parser.add_argument('--cnn', type=str, help='model archtecture')
    parser.add_argument('--layer', type=str, help='layer type')
    parser.add_argument('--layer_num', default=2, type=int, help='layer number')
    parser.add_argument('--workers', default=0, type=int,
                        help='number of data loading workers (default: 0)')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='mini-batch size (default: 64)')
    parser.add_argument('--seed', type=int, default=1, help='random seed')

    args = parser.parse_args()

    #fix random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # load model
    if args.cnn == 'vgg16':
        model = models.vgg16(pretrained=None)
        model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
        model.top_layer = nn.Linear(4096, 1222)
        param = torch.load(args.model)
        model = DataParallel(model) # for vgg_old
        model.load_state_dict(param)
        model = model.module # for vgg_old
        
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

    filename = args.dataset
    filepath = '/faces_83/evaluation/' + filename + '/'
    #print(filepath)
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

    features_stacked = extract_features(images, model, args.batch_size,
                                        args.workers, reglog, args.layer)

    map_model = umap.UMAP(n_components=2, metric='cosine')
    reduction_result = umap_model.fit_transform(features_stacked)

    c = 0
    for i in range(len(len_images)):
       cls = int(len_images[i])
       plt.plot(reduction_result[c:c+cls-1,0], reduction_result[c:c+cls-1,1], ".", color=colormap[i])
       c += cls

    plt.show()

if __name__ == '__main__':
    main()
