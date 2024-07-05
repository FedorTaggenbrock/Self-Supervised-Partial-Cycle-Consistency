# Self-Supervised Partial Cycle-Consistency for Multi-View Matching

This repository contains the pytorch implementation of the Self-Supervised Partial Cycle-Consistency for Multi-View Matching paper. 

## Abstract
Matching objects across partially overlapping camera views is crucial in multi-camera systems and requires a view-invariant feature extraction network. Training such a network with cycle-consistency circumvents the need for labor-intensive labeling. In our paper, we extend the mathematical formulation of cycle-consistency to handle partial overlap. We then derive several cycle variants and introduce a pseudo-mask which directs the training loss to take partial cycle-consistency into account, consequently improving the self-supervised learning signal. We additionally present a time-divergent scene sampling scheme that improves the data input for self-supervised settings. Cross-camera matching experiments on the challenging DIVOTrack dataset show the merits of our approach. Compared to the self-supervised state-of-the-art, we achieve 4.3\% higher F1 score with our combined contributions. Our improvements are robust to reduced overlap in the training data, with substantial improvements in scenes with few matches between many people. Self-supervised feature networks trained with our method are effective at matching objects in a range of multi-camera settings, providing opportunities for complex tasks like large-scale multi-camera scene understanding.

## Citation
This code is built on top of [MvMHAT](https://github.com/realgump/MvMHAT) with the data from [DIVOTrack](https://github.com/shengyuhao/DIVOTrack). Thanks for their great work.

```
   @inproceedings{gan2021mvmhat,
   title={Self-supervised Multi-view Multi-Human Association and Tracking},
   author={Yiyang Gan, Ruize Han, Liqiang Yin, Wei Feng, Song Wang},
   booktitle={ACM MM},
   year={2021}
   }
```
Any use whatsoever of the DIVOTrack dataset and its associated software shall constitute your acceptance of the terms of this agreement. By using the dataset and its associated software, you agree to cite the papers of the authors, in any of your publications by you and your collaborators that make any use of the dataset, in the following format:
```
@article{hao2023divotrack,
  title={Divotrack: A novel dataset and baseline method for cross-view multi-object tracking in diverse open scenes},
  author={Hao, Shengyu and Liu, Peiyuan and Zhan, Yibing and Jin, Kaixun and Liu, Zuozhu and Song, Mingli and Hwang, Jenq-Neng and Wang, Gaoang},
  journal={International Journal of Computer Vision},
  pages={1--16},
  year={2023},
  publisher={Springer}
}
```
The license agreement for data usage implies the citation of the paper above. Please notice that citing the dataset URL instead of the publications would not be compliant with this license agreement. You can read the LICENSE from [LICENSE](https://github.com/shengyuhao/DIVOTrack/blob/main/LICENSE.md).


## Get Started
The code was tested on ubuntu with python 3.10, pytorch 2.0.1  An Nvidia GPU is needed for both training and testing. 
Install requirements.txt.


### Dataset Preparation
Send your filled-in LICENSE to the DIVOTrack authors (shengyuhao@zju.edu.cn, gaoangwang@intl.zju.edu.cn) to obtain the password with which to unzip their data.
Run the script below to automatically download and unzip their huggingface data with the password
~~~
sudo apt-get install unar
cd SSPCC/DATA
chmod +x download_and_extract.sh
./download_and_extract.sh the_password_here
~~~
Prepare the data for model training and evaluation:
~~~
python convert_divo.py
~~~

## Training
The first run of our model can be trained with the following python call:
~~~
cd SSPCC/
python MvMHAT_SSPCC/train.py --EX_ID Masked_CV_0123_TDSS_s0 --SEED 0 --TDSS True --LOSSES cycle_variations_0123 --PARTIAL_MASKING True
~~~
The masking can also be disabled, or different cyle variations can be used, for example with --LOSSES cycle_variations_12.


Models from MvMHAT can be trained as follows, either with or without TDSS:
~~~
python MvMHAT_SSPCC/train.py --EX_ID mvmhat_s0 --SEED 0  --LOSSES pairwise_mvmhat triplewise_mvmhat  --TDSS False
~~~

Models will be saved at MvMHAT_SSPCC/models/EX_ID.pth.

## Inference
Given a model like Masked_CV_0123_TDSS_s0.pth,
We extract features on the DIVOTrack testset with:
~~~
python MvMHAT_SSPCC/inference.py --model Masked_CV_0123_TDSS_s0
~~~

## Evaluation
The Matching quality of the extracted features is then evaluated with:
~~~
python MVM_EVAL/Evaluate.py --model Masked_CV_0123_TDSS_s0
~~~

## Generating Tables
We provide a script to generate the main results for the tables in our paper.
~~~
chmod +x main_results.sh
./main_results.sh
~~~
We also provide main_results.txt, which contains the multi-view matching F1 score of each method in our tables, reported per individual run. 
