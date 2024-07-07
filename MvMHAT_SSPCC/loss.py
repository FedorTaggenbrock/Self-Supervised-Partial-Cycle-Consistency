import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
from scipy.optimize import linear_sum_assignment
import numpy as np
import math
import re
from abc import ABC, abstractmethod
import os

class CompositeLoss(nn.Module):
    def __init__(self, loss_list, args):
        super(CompositeLoss, self).__init__()

        self.losses = []
        for loss_type in loss_list:
            if "pairwise_mvmhat" in loss_type or "triplewise_mvmhat" in loss_type:
                self.losses.append(MvMHATCycleLoss(loss_type, args))
            elif "cycle_variations" in loss_type:
                self.losses.append(CycleVariationsLoss(loss_type, args))

    def forward(self, all_S):
        loss = torch.tensor(0.0).cuda()
        for l in range(len(self.losses)):
            loss_class = self.losses[l]
            curr_loss = loss_class._compute_loss(all_S)
            loss += curr_loss
        return loss
   
class CycleLoss(ABC):
    def __init__(self, loss_type):
        self.epoch = 0

        self.m = 0.5
        self.delta = 0.5
        self.epsilon = 0.1
        
    @abstractmethod
    def _compute_loss(self, all_S):
        pass
    
    def _to_S_hat(self, S, custom_epsilon = None): 
        if custom_epsilon is not None:
            scale = np.log(self.delta / (1 - self.delta) * S.size(1)) / custom_epsilon
        else:
            scale = np.log(self.delta / (1 - self.delta) * S.size(1)) / self.epsilon
        S_hat = f.softmax(S * scale, dim=1)
        return S_hat
    
    def _compute_orig_cycle_loss(self, A_cycle):
        n = A_cycle.shape[0]
        I = torch.eye(n).cuda()
        pos = A_cycle * I
        neg = A_cycle * (1 - I)
        loss = torch.tensor(0.0).cuda()
        loss += torch.sum(f.relu(torch.max(neg, 1)[0] + self.m - torch.diag(pos)))
        loss += torch.sum(f.relu(torch.max(neg, 0)[0] + self.m - torch.diag(pos)))
        loss /= 2 * n
        return loss
    
    

class CycleVariationsLoss(CycleLoss):
    def __init__(self, loss_type, args):
        super(CycleVariationsLoss, self).__init__(loss_type)
        self.loss_type = loss_type
        self.cyc_vars = self._extract_cycle_vars(loss_type)

        if args.PARTIAL_MASKING:
            self.masking = True
            #For thresholding pseudolabels.
            self.M = 0.5 
            #For the masked loss
            self.m1 = 0.7
            self.m2 = 0.3
        else:
            self.masking = False

    def _extract_cycle_vars(self, loss_type):
        cyc_var_dict = {
            0: self._cycle_v0,
            1: self._cycle_v1,
            2: self._cycle_v2,
            3: self._cycle_v3,
        }
        #Extract the specific cycle variations used from the loss_type
        pattern = r'cycle_variations_([0-9]+)'
        matches = re.findall(pattern, loss_type)
        if matches:
            self.cyc_vars = list([cyc_var_dict[int(var)] for var in matches[0]])
        else:
            raise ValueError("Specify cycle variations in the loss type")
        return self.cyc_vars


    #Outer most function called to compute the loss
    def _compute_loss(self, all_S):
        pairwise_loss = self._pair_loss(all_S)

        triple_loss = self._triple_loss(all_S)
    
        return (pairwise_loss + triple_loss)/2
    

    def _pair_loss(self, all_S):
        loss = torch.tensor(0.0).cuda()
        symmetry_loss = torch.tensor(0.0).cuda()
        pairs = self._get_pairs(all_S)
        for i,j in pairs:  
            if self.masking:
                I_iji = self._get_pairwise_I_sudo(all_S, i, j)
            else:
                I_iji = None

            A_121 = self._pair_cycle(all_S, i, j)
            loss +=  self._cycle_loss(A_121, I_iji)

        return (loss +symmetry_loss)/len(pairs)
        
    def _triple_loss(self, all_S):
        triple_loss = torch.tensor(0.0).cuda()
        triples =  self._get_triples(all_S)
        for i,j,k in triples:    

            #One mask per unique triple
            if self.masking:
                I_ijki = self._get_triplewise_I_sudo(all_S, i,j,k)
            else:
                I_ijki = None

            for triple_cycle_func in self.cyc_vars:
                #Add the (masked) loss for each cycle
                triple_cycle = triple_cycle_func(all_S, i,j,k)
                triple_loss +=  self._cycle_loss(triple_cycle, I_ijki)
        return triple_loss/(len(triples))/len(self.cyc_vars)


    def _cycle_loss(self, A_cycle, I_mask=None):
        #Either use the masked loss or not.
        if self.masking:  
            return self._compute_psuedolabel_loss(A_cycle, I_mask)
        else:  
            return self._compute_orig_cycle_loss(A_cycle)
        

    def _get_pairwise_I_sudo(self, all_S, i, j):
        # Computes the diagonal elements of the partial overlap mask I_iji. 
        A_ij = (self._to_smallest_S_hat(all_S[i][j]) > self.M).float()
        I_iji = torch.mm(A_ij, A_ij.T)
        I_sudo_mask = torch.diag(I_iji) > 0
        return I_sudo_mask

    def _get_triplewise_I_sudo(self, all_S, i, j, k):
        # Computes the diagonal elements of the partial overlap mask I_ijki. 
        A_ij = self._to_smallest_S_hat(all_S[i][j]) 
        A_ij_pmatch = A_ij > self.M   #After the softmax, the elements higher than M are possible matches. 
        A_jk = self._to_smallest_S_hat(all_S[j][k])
        A_jk_pmatch = A_jk > self.M
        A_ki = self._to_smallest_S_hat(all_S[k][i])
        A_ki_pmatch = A_ki > self.M
        I_sudo_mask = torch.mm(torch.mm(A_ij_pmatch.float(), A_jk_pmatch.float()), A_ki_pmatch.float())
        I_sudo_mask = torch.diag(I_sudo_mask) > 0
        return I_sudo_mask

    def _to_smallest_S_hat(self, S_ij):
        if S_ij.shape[0] < S_ij.shape[1]:  
            A_ij = self._to_S_hat(S_ij, custom_epsilon=0.3)  
        else: 
            A_ji = self._to_S_hat(S_ij.T, custom_epsilon=0.3)
            A_ij = A_ji.T 
        return A_ij

    def _compute_psuedolabel_loss(self, A_cycle, I_sudo_mask):
        n = A_cycle.shape[0]
        I = torch.eye(n).cuda()
        pos = A_cycle * I
        diag_vec = torch.diag(pos)
        neg = A_cycle * (1 - I)
        loss = torch.tensor(0.0).cuda()

        #Compute the loss over both rows and columns.
        for max_dim in [0,1]:
            nondiag_max_vec = torch.max(neg, max_dim)[0]
            nondiag_min_vec = torch.min(neg, max_dim)[0]

            #L1_r, L2_r are row wise vectors, containing for each row the loss that is computed for that row.  
            L1_r = f.relu(nondiag_max_vec - diag_vec + self.m1)
            L2_r = f.relu(nondiag_max_vec - diag_vec + self.m2)
            loss += torch.sum(I_sudo_mask * L1_r + (~I_sudo_mask) * L2_r)
        loss /= 2*n #Divide by nr of elements for which the loss is computed.- 1x rows, 1x cols
        return loss

    def _pair_cycle(self, all_S, i,j):
        S12, S21 = all_S[i][j], all_S[j][i]
        S_12_hat = self._to_S_hat(S12)
        S_21_hat = self._to_S_hat(S21)
        A_121 = torch.mm(S_12_hat, S_21_hat)
        return A_121
    
    def _cycle_v1(self, all_S, i,j,k):
        S12 = torch.mm(all_S[i][j], all_S[j][k])  
        S21 = S12.T
        S12_hat = self._to_S_hat(S12)
        S21_hat = self._to_S_hat(S21)
        A_121 = torch.mm(S12_hat, S21_hat)
        return A_121

    def _cycle_v2(self, all_S, i,j,k):
        S12 = torch.mm(all_S[i][j], all_S[j][k])
        S21 = all_S[k][i]
        S12_hat = self._to_S_hat(S12)
        S21_hat = self._to_S_hat(S21)
        A_121 = torch.mm(S12_hat, S21_hat)
        return A_121

    def _cycle_v3(self, all_S, i,j,k):
        S_ij, S_jk, S_ki = all_S[i][j], all_S[j][k], all_S[k][i]
        S12 = torch.mm(S_ij, S_jk)
        S31 = torch.mm(S_jk, S_ki)
        S23 = torch.mm(S_ki, S_ij)
        S12_hat = self._to_S_hat(S12)
        S23_hat = self._to_S_hat(S23)
        S31_hat = self._to_S_hat(S31)
        A_121 = torch.mm(torch.mm(S12_hat, S23_hat), S31_hat)
        return A_121
    
    def _cycle_v0(self, all_S, i,j,k):
        S12, S23, S31 = all_S[i][j], all_S[j][k], all_S[k][i]
        S12_hat = self._to_S_hat(S12)
        S23_hat = self._to_S_hat(S23)
        S31_hat = self._to_S_hat(S31)
        A_cycle = torch.mm(torch.mm(S12_hat, S23_hat), S31_hat)
        return A_cycle

    def _get_pairs(self, all_S):
        num_views = len(all_S)
        pairs = []
        for i in range(num_views):
            for j in range(i+1, num_views):
                #Match only by starting and ending in the largest view. This helps to make the learning signal more informative. 
                if all_S[i][j].shape[0] < all_S[i][j].shape[1]:
                    pairs.append((j,i))
                else:
                    pairs.append((i,j))
        return pairs

    def _get_triples(self, all_S):
        num_views = len(all_S)
        triples = []
        for i in range(num_views):
            for j in range(i+1, num_views):
                for k in range(j+1, num_views):
                    #We only choose the triples where the largest view is the first and final one.
                    view_dims = [all_S[i][j].shape[0], all_S[j][k].shape[0], all_S[k][i].shape[0]]
                    largest_idx = np.argmax(view_dims)
                    if largest_idx  == 0:
                        triples.append((i,j,k))
                        triples.append((i,k,j))
                    elif largest_idx  == 1:
                        triples.append((j,k,i))
                        triples.append((j,i,k))
                    elif largest_idx  ==2:
                        triples.append((k,i,j))
                        triples.append((k,j,i))
        return triples


class MvMHATCycleLoss(CycleLoss):
    #Class for the pariwise and triplewise losses. Code from https://github.com/realgump/MvMHAT.
    #Class structure has been slightly modified to fit the current codebase, and tested to make sure exactly the same models are trained as with the original code.
    def __init__(self, loss_type, args):
        super(MvMHATCycleLoss, self).__init__(loss_type)
        self.loss_type = loss_type

    def _compute_loss(self, all_S):
        if "pairwise" in self.loss_type:
            return self.pairwise_loss(all_S)
        elif "triplewise" in self.loss_type :
            return self.triplewise_loss(all_S)
        else:
            raise ValueError("Loss type not recognized")

    def pairwise_loss(self, all_S):
        loss_num = 0
        loss_sum = 0
        for i in range(len(all_S)):
            for j in range(len(all_S)):
                if i < j:
                    # for len 3 does 0,1, 0,2, 1,2
                    loss_num += 1
                    S = all_S[i][j]
                    #It seems like S12 has more detections than S21, so that S1221 is larger than S2112, which could be a mistake
                    if S.shape[0] < S.shape[1]:
                        S21 = S
                        S12 = S21.transpose(1, 0)
                        dim0 = j
                    else:
                        S12 = S
                        S21 = S12.transpose(1, 0)
                        dim0 = i

                    scale12 = np.log(self.delta / (1 - self.delta) * S12.size(1)) / self.epsilon
                    scale21 = np.log(self.delta / (1 - self.delta) * S21.size(1)) / self.epsilon
                    S12_hat = f.softmax(S12 * scale12, dim=1)
                    S21_hat = f.softmax(S21 * scale21, dim=1)
                    S1221_hat = torch.mm(S12_hat, S21_hat)

                    loss = self._compute_orig_cycle_loss(S1221_hat)
                    loss_sum += loss
        return loss_sum / loss_num

    def triplewise_loss(self, all_S):
        loss_num = 0
        loss_sum = 0
        for i in range(len(all_S)):
            for j in range(len(all_S)):
                if i < j:
                    for k in range(len(all_S)):
                        if k != i and k != j :
                            #for len = 3, does 0,1,2, 0,2,1, 1,2,0.
                            loss_num += 1
                            S12_ = all_S[i][k]
                            S23_ = all_S[k][j]

                            S = torch.mm(S12_, S23_)
                            if S.shape[0] < S.shape[1]:
                                S21 = S
                                S12 = S21.transpose(1, 0)
                                dim0 = j
                            else:
                                S12 = S
                                S21 = S12.transpose(1, 0)
                                dim0 = i 

                            scale12 = np.log(self.delta / (1 - self.delta) * S12.size(1)) / self.epsilon
                            scale21 = np.log(self.delta / (1 - self.delta) * S21.size(1)) / self.epsilon
                            S12_hat = f.softmax(S12 * scale12, dim=1)
                            S21_hat = f.softmax(S21 * scale21, dim=1)
                            S1221_hat = torch.mm(S12_hat, S21_hat)

                            loss = self._compute_orig_cycle_loss(S1221_hat)
                            loss_sum += loss
        return loss_sum / loss_num









