from __future__ import division, print_function, absolute_import
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.models as models
from torch.cuda.amp import autocast as autocast
import config as C
from collections import defaultdict
import argparse
import sys
sys.path.append('.')
from MvMHAT_SSPCC.data_loader import Loader


def read_loader(dataset_name):
    dataset = Loader(frames=1, views=3, mode='test', dataset=dataset_name, testbb=  True, bbox_size=(224,224))
    dataset_loader = DataLoader(dataset, num_workers=8)
    dataset_info = {
        'view': dataset.view_ls,
        'seq_len': len(dataset),
        'start': dataset.cut_dict[dataset_name][0],
        'end': dataset.cut_dict[dataset_name][1]
    }
    return dataset_info, dataset_loader

def search_bbox(index, id, det_list):
    for item in det_list:
        if index == item[0] and id == item[1]:
            return item[2:6]
    return []

def gather_seq_info_multi_view(dataset_name, dataset_info, dataset_test, model):
    coffidence = 1
    print('loading dataset...')
    feature_dict = defaultdict(list)

    image_filenames = defaultdict(list)
    detections = defaultdict(list)
    for data_i, data in tqdm(enumerate(dataset_test), total=len(dataset_test)):
        
        feature_ls = []
        box_ls = []
        lbl_ls = []
        scn_ls = []
        for view_i, view in enumerate(dataset_info['view']):
            data_pack = data[view_i][0] #Only one frame during testing.
            if data_pack == []:
                continue
            img, box, lbl, scn = data[view_i][0] #img has shape (numbboxes, 3, 224, 224)
            model.eval()

            with torch.no_grad():
                img = img.squeeze(0).cuda()
                with autocast():
                    feat = model(img) #img has shape (numbboxes, 1000)
            feature_ls.append(feat.float())
            box_ls.append(box)
            lbl_ls.append(lbl)
            scn_ls.append(scn)

        for view, feat, box, lbl, scn in zip(dataset_info['view'], feature_ls, box_ls, lbl_ls, scn_ls):
            #feat has dimensions [num_bboxes, 1000], box has dimensions [num_bboxes, 4], lbl has dimensions [num_bboxes], scn is a string (or scn[0] is a string)
            image_filenames[view].append(scn[0])
            for feature, bndbox, id in zip(feat, box, lbl):
                index = int(scn[0].split('/')[-1].split('_')[-1].split('.')[0]) #split the image path, get the last element (image name), use the number in this name -1.
                bndbox = [int(i) for i in bndbox]
                id = int(id[0])

                feature_list = feature.detach().cpu().numpy().tolist()
                if not isinstance(feature_list, list):  # If feature_list is a scalar (i.e., a float)
                    feature_list = [feature_list]  # Convert it to a list
                fea = [index] + [id] + bndbox + [coffidence] + [0, 0, 0] + feature_list
                feature_dict[view].append(fea)
                #fea has "dimensions" [frame_index, id, x, y, h, w, confidence, 0, 0, 0, 1000]
                #feature_dict[view] is a list of these det lists.
    return feature_dict



def run(dataset_name, display, dataset, model):
    dataset_info, dataset_loader = read_loader(dataset)
    feature_dict = gather_seq_info_multi_view(dataset_name, dataset_info, dataset_loader, model)
    return feature_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=C.INF_ID)
    args = parser.parse_args()

    checkpoint_path = "MvMHAT_SSPCC/models/" + args.model + '.pth'
    output_name = "./MVM_EVAL/test_features/" + args.model + '.npy'

    save_feature = defaultdict(list)
    print('model: ' + args.model)

    model = models.resnet50(weights=None)
    model = model.cuda()
    ckp = torch.load(checkpoint_path)
    if 'model' in ckp:
        model_ckp = ckp['model']
    else:
        model_ckp = ckp
    model.load_state_dict(model_ckp)

    for dataset_name in C.TEST_DATASET:
        save_feature[dataset_name] = run(dataset_name , display=C.DISPLAY, dataset=dataset_name, model=model)

    np.save(output_name, save_feature)

