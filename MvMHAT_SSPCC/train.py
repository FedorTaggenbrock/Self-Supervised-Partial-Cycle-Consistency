import os
import random
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm

from torch.cuda.amp import autocast as autocast
import argparse
import torch.nn.functional as f

from itertools import cycle
from collections import defaultdict
import numpy as np
import sys
sys.path.append('.')

from MvMHAT_SSPCC.data_loader import create_dataset
from MvMHAT_SSPCC.loss import CompositeLoss
import MvMHAT_SSPCC.config as C

if torch.cuda.is_available():
    # Print the name of each GPU
    for i in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_name(i))
else:
    print("CUDA is not available.")


def train(epoch):
    model.train()
    epoch_loss = 0

    for step_i, data in tqdm(enumerate(dataset_train), total=len(dataset_train)): #total = 4501, pairs of the 9000 frames in total
        optimizer.zero_grad()
        feature_ls = []
        label_ls = []

        with autocast():
            for view_i in range(len(data)): #len(data) =3
                for frame_i in range(args.NUM_FRAMES):
                    # img is a tensor of all the cropped bounding boxes in the frame. 
                    # img has dimension (num_bounding_boxes, 3, 224, 224)
                    #c_img_ls has dimensions [num_bboxes, 3, 224, 224], bbox_ls has dimensions [num_bboxes, 4], lbl has dimensions [num_bboxes], frame_img is a string
                
                    img, box, lbl, scn = data[view_i][frame_i]
                    img = img.squeeze(0).cuda() #img has dimensions [num_bboxes, 3, 224, 224]

                    if args.PARTIAL_FOV < 1.0:
                        p_ind = partial_fov_indices(box, view_i, args.PARTIAL_FOV)
                        img = img[p_ind]
                        if len(img) == 0:
                            continue
                    
                    feature = model(img) # img.squeeze(0).cuda() has size [12, 3, 224, 224], feature has size [12, 1000]
                    feature_ls.append(feature)
                    label_ls.append(lbl)

            if len(feature_ls) < 3: # Coudl happen if the partial ratio is set too low. 
                continue

            # cbe = concatenated_boundingbox_embeddings, transformed with the model, has dimension (num_bounding_boxes, 1000) -> embedding dimension.
            # label has dimension [num_bboxes]
            #feature_ls = [cbe_view1_frame1, cbe_view1_frame2, cbe_view2_frame1, cbe_view2_frame2, cbe_view3_frame1, cbe_view3_frame2]
            #label_ls = [label_view1_frame1, label_view1_frame2, label_view2_frame1, label_view2_frame2, label_view3_frame1, label_view3_frame2]
            all_S = gen_S(feature_ls, args)

            step_loss = composite_loss(all_S)
            if step_loss != 0.0:
                epoch_loss += step_loss.item()
                if epoch >= 0:
                    step_loss.backward()
                    optimizer.step()
            if (step_i < 100 and step_i % 10 == 0) or (step_i % 500 == 0):
                print(make_write_str(composite_loss, epoch, step_loss.item(), epoch_loss))
    return epoch_loss 

def partial_fov_indices(box, view_i, ratio):
    #Returns a subset of the img tensor for which the boundingboxes that are on the right (1-ratio)% of the view are removed. Size is determined by view_i. 0 = 1920x1080, 1 = 3640x2048, 2 = 1920x1080
    #box has dimensions [num_bboxes, 4]
    #The bounding boxes are in the format [xmin, ymin, xmax, ymax]
    filtered_indices = []
    for i, bbox in enumerate(box):
        if view_i == 0:
            if bbox[0] < 1920*ratio:
                filtered_indices.append(i)
        elif view_i == 1:
            if bbox[0] < 3640*ratio:
                filtered_indices.append(i)
        elif view_i == 2:
            if bbox[0] < 1920*ratio:
                filtered_indices.append(i)
    return filtered_indices

def gen_S(feature_ls, args):
    # cbe = concatenated_boundingbox_embeddings, transformed with the model, has dimension (num_bounding_boxes, 1000) -> embedding dimension.
    #feature_ls = [cbe_view1_frame1, cbe_view1_frame2, cbe_view2_frame1, cbe_view2_frame2, cbe_view3_frame1, cbe_view3_frame2, cbe_view4_frame1, cbe_view4_frame2]
    norm_feature = [f.normalize(i, dim=-1) for i in feature_ls]  #normalize the feature vector
    all_blocks_S = []
    for feat1 in norm_feature:
        row_blocks_S = []
        for feat2 in norm_feature:
            S = torch.mm(feat1, feat2.transpose(0, 1))
            row_blocks_S.append(S)
        all_blocks_S.append(row_blocks_S)
        #all_blocks_S contains
        # [[u_t-1 x u_t-1, u_t-1 x u_t, u_t-1 x v_t-1, u_t-1 x v_t, u_t-1 x w_t-1, u_t-1 x w_t], 
        #  [u_t x u_t-1, .... ],
        #  [v_t-1 x u_t-1, .... ],
        #  [v_t c u_t-1, ....],
        #    ....  
    return all_blocks_S  

def make_write_str(composite_loss, epoch, step_loss, epoch_loss, final = False):
    if not final:
        newline = "\n"
        write_str = "\n experiment: {}".format(args.EX_ID)
    else:
        newline = ""
        write_str = "\n\n experiment: {}".format(args.EX_ID)

    write_str += newline+" epoch: {} losses used: {} \n current loss:{:.8f} epoch_loss:{:.8f}".format(epoch, args.LOSSES, step_loss, epoch_loss) + "   LR " +str(args.LR)
    return write_str



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Model settings.
    parser.add_argument('--EX_ID', type=str, help='Name for the saved model.')
    parser.add_argument('--LOSSES', type=str, nargs= "+", help='Losses to use in the model. Options are: ["pairwise_mvmhat", "triplewise_mvmhat", "cycle_variations_0123", "cycle_variations_0", ... ] ')
    parser.add_argument("--PARTIAL_MASKING", type=bool, default = False, help='Use partial masking')
    parser.add_argument("--LR",  type=float, default=1e-5, help='learning rate')
    parser.add_argument("--MAX_EPOCH",  type=int, default=10, help='Max number of epochs')
    parser.add_argument("--TDSS", type=bool, default = False, help='Use Time Divergent Scene Sampling')
    parser.add_argument("--SEED", type=int, default = 0, help='The seed for the random number generator')
    parser.add_argument("--PARTIAL_FOV", type= float, default = 1.0, help="The ratio of each frame that is disregarded during training to simulate more partial overlap")
    
    #Default parameters that don't need to be changed
    parser.add_argument('--PRETRAIN', type=bool, default=True, help='Start with pretrained model')
    parser.add_argument("--NUM_FRAMES", type=int, default = 2, help='The number of timesteps to use in a batch')
    parser.add_argument("--TIMESTEP_RANGE", type=int, default = 1, help='The distance between the timesteps in a batch')	
    # Parse the arguments
    args = parser.parse_args()

    # Set the seed for the random number generator
    random.seed(args.SEED)
    torch.manual_seed(args.SEED)

    if args.PRETRAIN:
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    else:
        model = resnet50(weights=None) 

    checkpoint_path = '/cross-view/DIVOTrack/Cross_view_Tracking/MvMHAT/models/pretrained.pth'
    ckp = torch.load(checkpoint_path)
        
    model.load_state_dict(ckp)
    model = nn.DataParallel(model).cuda()

    composite_loss = CompositeLoss(args.LOSSES, args)

    optimizer_params = [{'params': model.parameters()}]
    optimizer = torch.optim.Adam(optimizer_params, lr=args.LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    print('model: ' + args.EX_ID + ' '+
          'loss: ' + " ".join(args.LOSSES) + ' '+
          'lr: ' + str(args.LR))

    dataset_train = create_dataset(mode = 'train', timestep_range = args.TIMESTEP_RANGE, frames=args.NUM_FRAMES, TDSS = args.TDSS, seed = args.SEED)

    max_loss = 1e8
    for epoch_i in range(args.MAX_EPOCH):

        for l in composite_loss.losses:
            l.epoch = epoch_i

        print("Epoch {}".format(epoch_i+1))
        if args.TDSS:
            dataset_train = create_dataset(mode = 'train', timestep_range = epoch_i+1, frames=args.NUM_FRAMES, TDSS = args.TDSS)

        epoch_loss = train(epoch_i)
        avg_epoch_loss = epoch_loss / len(dataset_train) 
        if avg_epoch_loss < max_loss:
            max_loss = avg_epoch_loss
            print('save model')
            if isinstance(model, nn.DataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            save_dict =  {
                'epoch': epoch_i,
                'loss': avg_epoch_loss,
                'optimizer': optimizer.state_dict(),
                'model': state_dict,
            }

            torch.save(save_dict,
                "MvMHAT_SSPCC/models/" + args.EX_ID + '.pth'
            )
            write_str = make_write_str(composite_loss, epoch_i, avg_epoch_loss, epoch_loss, final = True)
            write_str +=  "all command line args: " +str(vars(args))


