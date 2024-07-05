from torch.utils.data import Dataset
from collections import defaultdict
import os
import numpy as np
import cv2
import random
import re
import MvMHAT_SSPCC.config as C
from glob import glob
from torch.utils.data import Sampler
from torch.utils.data import DataLoader, ConcatDataset

class Loader(Dataset):
    def __init__(self, views=3, frames=2, timestep_range = 1, mode='train', dataset='1', bbox_size = (224,224), shuffle = False, testbb = True, true_rand = False):
        self.video_name = ['circleRegion', 'innerShop', 'movingView', 'park', 'playground', 'shopFrontGate', 'shopSecondFloor', 'shopSideGate', 'shopSideSquare', 'southGate']
        # self.video_name = ['park']
        self.views_name = ['Drone', 'View1', 'View2']
        self.true_rand = true_rand
        self.isShuffle = shuffle
        self.random_seed_num = 0
        self.views = views
        self.mode = mode
        self.testbb = testbb
        self.dataset = dataset
        self.down_sample = 1
        self.root_dir = os.path.join(C.ROOT_DIR)
        self.img_root = os.path.join(C.ROOT_DIR, 'images'+"_"+mode) #changed here!
        self.bb_size = bbox_size
        self.isCut = 0
        self.dataset_dir = os.path.join(self.img_root, dataset)
        self.cut_dict = {
                'circleRegion': [0, 1600],
                'innerShop': [0, 1100],
                'movingView': [0, 580],
                'park': [0, 600],
                'playground': [0, 900],
                'shopFrontGate': [0, 1250],
                'shopSecondFloor': [0, 825],
                'shopSideGate': [0, 750],
                'shopSideSquare': [0, 600],
                'southGate': [0, 800]
        }
        if self.mode == 'train':
            self.frames = frames
            # self.isCut = 1
        elif self.mode == 'test':
            self.frames = 1
            # self.isCut = 1

        self.view_ls = self.views_name
        self.set_epoch_img_dict(timestep_range = timestep_range)

        self.anno_dict, self.max_id, self.view_id_ls = self.gen_anno_dict()

    def set_epoch_img_dict(self, timestep_range =1):
        self.img_dict = self._gen_path_dict(timestep_range=timestep_range)
    

    def generate_shuffled_indices(self, n):
        indices = list(range(n))
        random.shuffle(indices)
        return indices

    def generate_rand_pairs(self, n, max_diff):
        shuffled_indices = self.generate_shuffled_indices(n)
        pairs = []
        used_indices = set()
        #Loop over the indices in a random order

        for index in shuffled_indices:
            if index in used_indices:
                continue
            #Generate a list of valid second indices within the max_diff range
            potential_pairs = [i for i in range(max(0, index - max_diff), min(n, index + max_diff + 1)) if i not in used_indices and i!=index]
            if potential_pairs:
                second_index = random.choice(potential_pairs)
                used_indices.add(index)
                used_indices.add(second_index)
                pairs.append((index, second_index))
        return pairs


    def _gen_path_dict(self, timestep_range = 1):
        path_dict = defaultdict(list)
        frames_all = glob(self.dataset_dir + "/*.jpg")
        
        for view in self.view_ls:
            path_ls = [img_path for img_path in frames_all if view in img_path]
            path_ls.sort()

            if self.isCut:
                start, end = self.cut_dict[self.dataset][0], self.cut_dict[self.dataset][1]
                path_ls = path_ls[start:end]

            # if drop_last:
            #     path_ls = path_ls[:-1]
            cut = len(path_ls) % self.frames
            if cut:
                path_ls = path_ls[:-cut]
            if self.isShuffle: 
                #Everytime the path dict is generated, the random seed num is consistent across the views
                random.seed(self.random_seed_num)
                random.shuffle(path_ls)
            path_dict[view] += path_ls

            
        if timestep_range == 1:
            #The original implementation, always uses timestep t-1 and t, does not randomize by definition
            path_dict = {view: [path_dict[view][i:i + self.frames] for i in range(0, len(path_dict[view]), self.frames)] for
                        view in path_dict}
        elif self.true_rand:
            #so timestep_range >1. 
            #Each batch consists of 2 indices with a random dt between 1 and timestep_range.
            total_steps = len(path_dict[self.view_ls[0]])
            index_pairs = self.generate_rand_pairs(total_steps, timestep_range)
            for view in path_dict:
                new_path_ls = []
                for pair in index_pairs:
                    new_path_ls.append([path_dict[view][pair[0]], path_dict[view][pair[1]]])
                path_dict[view] = new_path_ls
        else:
            #When we don't randomize, we always take timesteps [t-range, t], [t-range+1, t+1], [t-range+2, t+2], etc. when self.frames = 2. 
            #when self.frames =3 for example it becomes [t-range, t-range/2, t], ...
            
            # Using the same list [0, 1, 2, ..., 20], a timestep_range of 3, and self.frames of 3, the final output will handle the remaining indices as a group:
            # [0, 3, 6]
            # [1, 4, 7]
            # [2, 5, 8]
            # [9, 12, 15]
            # [10, 13, 16]
            # [11, 14, 17]
            # 18, 19 ,20 are skipped.
            for view in path_dict:
                new_path_ls = []
                total_steps = len(path_dict[view])
                used_indices = set()
                
                i = 0
                while i < total_steps:
                    frame_indices = [i + j * timestep_range for j in range(self.frames)]
                    if all(index < total_steps for index in frame_indices) and not any(index in used_indices for index in frame_indices):
                        new_path_ls.append([path_dict[view][index] for index in frame_indices])
                        used_indices.update(frame_indices)
                    i += 1
                
                path_dict[view] = new_path_ls
                                        
        self.random_seed_num +=1
        return path_dict

    def gen_anno_dict(self):
        anno_dict = {}
        max_id = -1
        view_maxid = -1
        view_id_ls = []

        for view in self.view_ls:
            anno_view_dict = defaultdict(list)
            if self.mode == 'train':
                anno_path = os.path.join(self.root_dir, 'train_gt', self.dataset, view + '.txt')
            elif self.mode == 'test':
                if self.testbb:
                    anno_path = os.path.join(C.TESTBB_DIR, '{}_{}.txt'.format(self.dataset, view))
                else:
                    anno_path = os.path.join(C.DETECTION_DIR, '{}_{}.txt'.format(self.dataset, view))

                	
            with open(anno_path, 'r') as anno_file:
                anno_lines = anno_file.readlines()
                for anno_line in anno_lines:
                    if self.mode == 'train': 
                        anno_line_ls = anno_line.split(' ')
                    else:
                        anno_line_ls = anno_line.split(',')
                    anno_key = str(int(anno_line_ls[0]))
                    anno_view_dict[anno_key].append(anno_line_ls)
                    if max_id < int(anno_line_ls[1]):
                        max_id = int(anno_line_ls[1])
                    if view_maxid < int(anno_line_ls[1]):
                        view_maxid = int(anno_line_ls[1])
            view_id_ls.append(view_maxid)
            view_maxid = -1
            anno_dict[view] = anno_view_dict
        return anno_dict, max_id, view_id_ls

    def read_anno(self, path: str):

        if type(path) == int:
            print('\n\n path is int')
            print(path)
            import pdb; pdb.set_trace()

        path_split = path.split('/')
        view = path_split[-1].split('.txt')[0].split("_")[1]

        frame = path_split[-1].split('.jpg')[0].split("_")[-1]
        
        annos = self.anno_dict[view][str(int(frame))]

        bbox_dict = {}
        for idx, anno in enumerate(annos):
            bbox = anno[2:6]

            xmin = int(float(bbox[0]))
            ymin = int(float(bbox[1]))
            xmax = int(float(bbox[2]))
            ymax = int(float(bbox[3]))
            bbox_trans = [xmin, ymin, xmax-xmin, ymax-ymin]
            if xmax - xmin <= 0 or ymax - ymin <= 0:
                continue
            bbox_dict[idx] = (bbox_trans, int(anno[1]))
        return bbox_dict

    def crop_img(self, frame_img, bbox_dict):
        img = cv2.imread(frame_img)
        c_img_ls = []
        bbox_ls = []
        label_ls = []
        for key in bbox_dict:
            bbox, lbl = bbox_dict[key]
            bbox = [0 if i < 0 else i for i in bbox]


            # img is in HWC layout. index y:y+h, x:x+w
            crop = img[bbox[1]:bbox[3] + bbox[1], bbox[0]:bbox[2] + bbox[0], :]

            # resize bbox to 224x224 and transpose to CHW layout, needed for pytorch
            crop = cv2.resize(crop, self.bb_size).transpose(2, 0, 1).astype(np.float32)
            c_img_ls.append(crop)
            bbox_ls.append(bbox)
            label_ls.append(lbl)
        #c_img_ls has dimensions [num_bboxes, 3, 224, 224], bbox_ls has dimensions [num_bboxes, 4], label_ls has dimensions [num_bboxes], frame_img is a string
        return np.stack(c_img_ls), bbox_ls, label_ls, frame_img

    def __len__(self):
        return min([len(self.img_dict[i]) for i in self.view_ls] + [10000])

    def __getitem__(self, item):
        ret = []

        img_ls = [self.img_dict[view][item] for view in self.view_ls]
    
        for img_view in img_ls:
            view_ls = []
            for img in img_view:
                anno = self.read_anno(img)
                if anno == {}:
                    #print('no anno', img)
                    if self.mode == 'train':
                        return self.__getitem__(item - 1)
                    else:
                        view_ls.append([])
                        continue
                view_ls.append(self.crop_img(img, anno))
            ret.append(view_ls)
        return ret


class SceneSampler(Sampler):
    def __init__(self, data_source, seed):
        self.data_source = data_source
        self.seed = seed
        self.scene_labels = ['circleRegion', 'innerShop', 'movingView', 'park', 'playground', 'shopFrontGate', 'shopSecondFloor', 'shopSideGate', 'shopSideSquare', 'southGate']

    def __iter__(self):
        # Map scenes to their batch indices
        scene_indices = {label: [] for label in set(self.scene_labels)}
        dataset_starts = [0]
        
        for dataset in self.data_source.datasets:
            dataset_starts.append(dataset_starts[-1] + len(dataset))
        
        for i, _ in enumerate(self.data_source.datasets):
            scene = self.scene_labels[i]
            start_idx = dataset_starts[i]
            end_idx = dataset_starts[i + 1]
            scene_indices[scene].extend(range(start_idx, end_idx))

        
        #Sort scene indices on keys
        scene_indices = dict(sorted(scene_indices.items()))

        ordered_indices = self._controlled_round_robin(scene_indices)
        
        return iter(ordered_indices)

    def __len__(self):
        return len(self.data_source)
    
    def _controlled_round_robin(self, scene_indices):
        num_scenes = len(scene_indices)
        total_batches = sum(len(indices) for indices in scene_indices.values())
        num_rounds = total_batches // num_scenes + (total_batches % num_scenes > 0)

        # Calculate the exact float number of batches each scene should contribute per round
        contributions_per_round = {scene: len(indices) / num_rounds for scene, indices in scene_indices.items()}
        
        # Shuffle the batches within each scene
        shuffled_scene_indices = {scene: random.sample(indices, len(indices)) for scene, indices in scene_indices.items()}
        #shuffled_scene_indices = scene_indices.copy()

        # This will store the final ordered list of batch indices
        epoch_indices = []
        current_contributions = defaultdict(float)

        for _ in range(num_rounds):
            round_indices = []

            for scene in scene_indices:
                # Floor contribution counts as the base number of batches per round
                num_to_add = int(np.floor(current_contributions[scene] + contributions_per_round[scene]))
                current_contributions[scene] += contributions_per_round[scene] - num_to_add
                
                # If cumulative contribution exceeds 1, add an extra batch
                if current_contributions[scene] >= 1:  
                    num_to_add += 1
                    current_contributions[scene] -= 1

                # Add the next set of batch indices for the scene
                if num_to_add > 0 and shuffled_scene_indices[scene]:
                    round_indices.extend(shuffled_scene_indices[scene][:num_to_add])
                    shuffled_scene_indices[scene] = shuffled_scene_indices[scene][num_to_add:]
            #random.seed(self.seed)
            random.shuffle(round_indices)  # Shuffle batches within the round to ensure mixing
            epoch_indices.extend(round_indices)

        #Handle the almost added indices
        for scene, idxls in shuffled_scene_indices.items():
            if len(idxls) == 1:
                idx = idxls[0]
                if current_contributions[scene] > 0.99:
                    epoch_indices.append(idx)
            if len(idxls) > 1:
                raise ValueError("There are still indices left in the shuffled scene indices")
        return epoch_indices


def create_dataset(mode = 'train', timestep_range = 1, frames=2, TDSS = False, bbox_size=(224,224), seed = 0):
    datasets = []
    for dataset in C.TRAIN_DATASET: #contains all the scenes in the training data.
        timestep_range_ = timestep_range
        datasets.append(Loader(views=C.VIEWS, timestep_range = timestep_range,  frames=frames, mode=mode, dataset=dataset, shuffle = False, bbox_size=bbox_size))
    concat_dataset = ConcatDataset(datasets)
    if TDSS:
        sampler = SceneSampler(concat_dataset, seed)
        dataset = DataLoader(concat_dataset, num_workers=8, sampler=sampler)
    else:
        dataset = DataLoader(concat_dataset, num_workers=8, shuffle=True)
    return dataset








