import numpy as np
import argparse
from scipy.optimize import linear_sum_assignment
from sklearn.utils.extmath import softmax
import os
import re

#     {
# circleRegion:{
#     Drone:[[fid,pid,lx,ly,w,h,1,0,0,0,feature],...],   
#     View1:[...],   
#     View2:[...]
# }, 
#  innerShop:{
#     Drone:[[fid,pid,lx,ly,w,h,1,0,0,0,feature],...],   
#     View1:[...],   
#     View2:[...]
# }, 
#  ...
#  }

def match_features_across_views(dets_v1, dets_v2, cv_thresh):
    #Use a threshold to obtain partial matches with the Hungarian algorithm.
    #Code taken from Multi_view_Tracking/src/lib/deep_sort/update.py
    dets_v1 = np.array([det[10:] for det in dets_v1])
    dets_v2 = np.array([det[10:] for det in dets_v2])

    dets_v1 = dets_v1/np.linalg.norm(dets_v1, axis =1, keepdims=True)
    dets_v2 = dets_v2/np.linalg.norm(dets_v2, axis =1, keepdims=True)

    S12 = np.dot(dets_v1 , dets_v2.transpose(1, 0))

    scale12 = (
        np.log(0.5 / (1 - 0.5) * S12.shape[1])
        / 0.5
    )
    S12 = softmax(S12 * scale12)
    S12[S12 < cv_thresh] = 0

    assign_ls = linear_sum_assignment(-S12)
    assign_ls = np.asarray(assign_ls)
    assign_ls = np.transpose(assign_ls)
    X_12 = np.zeros((S12.shape[0], S12.shape[1]))
    for assign in assign_ls:
        if S12[assign[0], assign[1]] != 0:
            X_12[assign[0], assign[1]] = 1
    return X_12

def prec_recall_f1(TP, FP, FN):
    if TP+FP > 0:
        precision = TP/(TP+FP)
    else:
        precision = 0
    if TP+FN > 0:
        recall = TP/(TP+FN)
    else:
        recall = 0
    if precision+recall > 0:
        f1 = 2*precision*recall/(precision+recall)
    else:
        f1 = 0
    return precision, recall, f1

#python Cross_View_Association/Evaluate2.py --model my_L2_v5_p200_nosplit_a10_bTrue_easy1ep_start_10ep_s0_roundr__
 
if __name__ == '__main__':

    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model' , type=str,  help='Name of the test_features from the model')
    parser.add_argument('--thresholds', nargs='+', type=float, default=[0.1, 0.15, 0.2, 0.25, 0.3, 0.35], help='List of partial overlap/threshold values to do a gridsearch over')
    args = parser.parse_args()

    testbb_feature_file = './MVM_EVAL/test_features/'+args.model+'.npy'
    #create the result folders.
    save_dir = "./MVM_EVAL/Results/"+args.model
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    features = np.load(testbb_feature_file, allow_pickle=True).item()

    highest_f1 = 0

    for threshold in args.thresholds:
        
        res_dict = {}

        TP_all =0
        FP_all =0
        FN_all =0

        for scene in features:

            drone_feat = np.array(features[scene]["Drone"])
            view1_feat = np.array(features[scene]["View1"])
            view2_feat = np.array(features[scene]["View2"])

            all_feat = [drone_feat, view1_feat, view2_feat]

            min_frame = int(  max(min(drone_feat[:,0]),min(view1_feat[:,0]),min(view2_feat[:,0]))   )
            max_frame = int(  min(max(drone_feat[:,0]),max(view1_feat[:,0]),max(view2_feat[:,0]))   )
            
            TP = 0
            FP = 0
            FN = 0

            res_dict[scene] = {}

            #For every timframe, compute the scores for every viewpair.
            for frame_id in range(min_frame, max_frame+1):

                for pair0, pair1 in [[0,1], [0,2], [1,2]]:
                    feat1 = all_feat[pair0]
                    feat2 = all_feat[pair1]
                
                    frame_feat1 = feat1[feat1[:,0] == frame_id] 
                    frame_feat2 = feat2[feat2[:,0] == frame_id]

                    match_matrix_feat = match_features_across_views(frame_feat1, frame_feat2, threshold)

                    
                    for i in range(len(frame_feat1)):
                        for j in range(len(frame_feat2)):
                            if match_matrix_feat[i,j] == 1:
                                if frame_feat1[i][1] == frame_feat2[j][1]: #The true labels are stored as the pid in features. 
                                    TP += 1
                                else:  
                                    FP += 1
                            else:
                                if frame_feat1[i][1] == frame_feat2[j][1]:
                                    FN += 1

            precision, recall, f1 = prec_recall_f1(TP, FP, FN)

            res_dict[scene]["Precision"] = precision
            res_dict[scene]["Recall"] = recall
            res_dict[scene]["F1"] = f1

            TP_all += TP
            FP_all += FP
            FN_all += FN

        precision_all, recall_all, f1_all = prec_recall_f1(TP_all, FP_all, FN_all)

        save_file = save_dir+"/threshold_" + format(threshold, '.3f') + ".txt"
        
        with open(save_file, 'w') as f:
            pass
        with open(save_file, 'a') as file:
            for scene in res_dict:
                file.write("\n scene: "+scene+ " Precision: "+ format(res_dict[scene]["Precision"], '.4f')+ " Recall: "+format(res_dict[scene]["Recall"], '.4f')+ " F1: "+format(res_dict[scene]["F1"], '.4f')+"\n")
            file.write("\n Overall Precision: "+format(precision_all, '.4f')+ " Overall Recall: "+format(recall_all, '.4f')+ " Overall F1: "+format(f1_all, '.4f')+"\n")
        file.close()

        if f1_all > highest_f1:
            highest_f1 = f1_all

        print("Results saved to: ", save_file)

    os.rename(save_dir, save_dir+"_"+f"{highest_f1 * 100:.2f}")
    print("highest f1: ", highest_f1)
        




    

