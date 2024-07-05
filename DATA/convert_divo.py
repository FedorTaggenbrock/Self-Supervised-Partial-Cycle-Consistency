import os
import os.path as osp
import numpy as np
import cv2
from tqdm import tqdm

seqs_dict = {'circleRegion': 'Circle',
             'innerShop': 'Shop',
             'movingView': 'Moving',
             'park': 'Park',
             'playground': 'Ground',
             'shopFrontGate': 'Gate1',
             'shopSecondFloor': 'Floor',
             'shopSideGate': 'Side',
             'shopSideSquare': 'Square',
             'southGate': 'Gate2'}
view_dict= {"Drone": "View1", "View1": "View2", "View2": "View3"}
seqs_dict = {v: k for k, v in seqs_dict.items()}
view_dict = {v: k for k, v in view_dict.items()}

def mkdir(path):
    if not osp.exists(path):
        os.mkdir(path)

splits = ['train','test'] #Changed code for test split, different file structure. 
output_dir = './DIVO' 
if not osp.exists(output_dir):
    os.mkdir(output_dir)

root_dir = "./DIVO_unprocessed/images"
scene_ls = ['circleRegion', 'innerShop', 'movingView', 'park', 'playground', 'shopFrontGate', 'shopSecondFloor', 'shopSideGate', 'shopSideSquare', 'southGate']
mkdir(output_dir)
print("The dataset will be save to {}".format(output_dir))
for split in splits:
    print("Converting {}".format(split))
    mkdir(osp.join(output_dir, 'images' + "_" + split))
    mkdir(osp.join(output_dir, '{}_gt'.format(split)))
    split_dir = osp.join(root_dir, split)
    for scene in tqdm(sorted(os.listdir(split_dir))):
        print("scene_view= ", scene)
        new_scene = seqs_dict[scene.split('_')[0]]
        new_view = view_dict[scene.split('_')[1]]
        new_scene_view =  new_scene + "_" + new_view

        mkdir(osp.join(output_dir, 'images' + "_" + split, new_scene))
        mkdir(osp.join(output_dir, '{}_gt'.format(split), new_scene))
        img_list = sorted(os.listdir(osp.join(split_dir, scene, 'img1')))

        #gt_path = osp.join("/cross-view/DIVOTrack/datasets/DIVO/gt", new_scene, new_view+".txt")
        if split == 'test':
            gt_path = "./DIVO_unprocessed/self/" +new_scene_view+ '/gt/gt.txt'

        elif split == 'train':
            gt_path = "./DIVO_unprocessed/images/train/"+ scene + '/gt/gt.txt'
            
        gt_file = sorted(np.loadtxt(gt_path, delimiter=',').tolist(), key=lambda x:x[0])
        gt_file = np.array(gt_file)
        gt_file[:,4] += gt_file[:,2]
        gt_file[:,5] += gt_file[:,3]
        gt_file = np.delete(gt_file, -1, 1)
        gt_file = np.delete(gt_file, -1, 1)
        gt_file = np.delete(gt_file, -1, 1)
        np.savetxt(osp.join(output_dir, '{}_gt'.format(split), new_scene, "{}.txt".format(new_view)), gt_file, fmt="%d", delimiter=' ')

        if split == 'test':
            #Also store test data in different format in the folder called "gt".
            gt_file2 = sorted(np.loadtxt(gt_path, delimiter=',').tolist(), key=lambda x:x[1])
            gt_file2 = np.array(gt_file2)
            #change the last 0 0 0 on the line to 1 -1 -1 -1:
            gt_file2[:,6] = 1
            gt_file2[:,7] = -1
            gt_file2[:,8] = -1
            # Create a new column filled with -1. The new column should have the same number of rows as gt_file2.
            new_column = np.full((gt_file2.shape[0], 1), -1)
            gt_file2 = np.hstack((gt_file2, new_column))

            save_file = output_dir + '/gt/' + new_scene + '/' + new_view + '.txt'
            if not osp.exists(output_dir + '/gt/' + new_scene):
                os.makedirs(output_dir + '/gt/' + new_scene)
            np.savetxt(save_file, gt_file2, fmt="%d", delimiter=',')

        for img in tqdm(img_list):
            img_path = osp.join(split_dir, scene, 'img1', img)

            img = cv2.imread(img_path)
            if img is None:
                print("img with path {} is None".format(img_path))
                continue
            img_name = img_path.split("/")[-1]
            new_img_name = new_scene_view + "_" + img_name.split("_")[-1]
            write_path = osp.join(output_dir, 'images' + "_" + split, new_scene, new_img_name)

            cv2.imwrite(write_path, img)

        
