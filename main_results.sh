cd SSPCC ;


#The main model variations for seed 0 are given below.
#Table 1 reports the average and std. dev. of these models over seeds 0-4.
#Table 2 reports the results per scene

#Original MvMHAT
python MvMHAT_SSPCC/train.py --EX_ID mvmhat_s0 --SEED 0 --LOSSES pairwise_mvmhat  triplewise_mvmhat; 
python MvMHAT_SSPCC/inference.py --model mvmhat_s0;
python MVM_EVAL/Evaluate.py --model mvmhat_s0;

#Original MvMHAT + TDSS
python MvMHAT_SSPCC/train.py --EX_ID mvmhat_TDSS_s0 --SEED 0 --LOSSES pairwise_mvmhat  triplewise_mvmhat; 
python MvMHAT_SSPCC/inference.py --model mvmhat_TDSS_s0;
python MVM_EVAL/Evaluate.py --model mvmhat_TDSS_s0;

#Cycle Variations
python MvMHAT_SSPCC/train.py --EX_ID CV_0123_s0 --SEED 0 --LOSSES cycle_variations_0123;
python MvMHAT_SSPCC/inference.py --model CV_0123_s0;
python MVM_EVAL/Evaluate.py --model CV_0123_s0;

#Cycle Variations + TDSS
python MvMHAT_SSPCC/train.py --EX_ID CV_0123_TDSS_s0 --SEED 0 --TDSS True --LOSSES cycle_variations_0123;
python MvMHAT_SSPCC/inference.py --model CV_0123_TDSS_s0;
python MVM_EVAL/Evaluate.py --model CV_0123_TDSS_s0;

#Masked Cycle Variations
python MvMHAT_SSPCC/train.py --EX_ID Masked_CV_0123_s0 --SEED 0  --LOSSES cycle_variations_0123 --PARTIAL_MASKING True;
python MvMHAT_SSPCC/inference.py --model Masked_CV_0123_s0;
python MVM_EVAL/Evaluate.py --model Masked_CV_0123_s0;

#Masked Cycle Variations + TDSS
python MvMHAT_SSPCC/train.py --EX_ID Masked_CV_0123_TDSS_s0 --SEED 0 --TDSS True --LOSSES cycle_variations_0123 --PARTIAL_MASKING True;
python MvMHAT_SSPCC/inference.py --model Masked_CV_0123_TDSS_s0;
python MVM_EVAL/Evaluate.py --model Masked_CV_0123_TDSS_s0;



# For Table 4, we instead train a specific model with the flag --PARTIAL_FOV 0.8 or --PARTIAL_FOV 0.6
#For example:

#Original MvMHAT 60% FOV
python MvMHAT_SSPCC/train.py --EX_ID mvmhat_s0_fov60 --SEED 0 --LOSSES pairwise_mvmhat  triplewise_mvmhat --PARTIAL_FOV 0.6;
python MvMHAT_SSPCC/inference.py --model mvmhat_s0_fov60;
python MVM_EVAL/Evaluate.py --model mvmhat_s0_fov60;

#Cycle Variations + TDSS  60% FOV
python MvMHAT_SSPCC/train.py --EX_ID CV_0123_TDSS_s0_fov60 --SEED 0 --TDSS True --LOSSES cycle_variations_0123  --PARTIAL_FOV 0.6;
python MvMHAT_SSPCC/inference.py --model CV_0123_TDSS_s0_fov60;
python MVM_EVAL/Evaluate.py --model CV_0123_TDSS_s0_fov60;
 
#Masked Cycle Variations + TDSS  60% FOV
python MvMHAT_SSPCC/train.py --EX_ID Masked_CV_0123_TDSS_s0 --SEED 0 --TDSS True --LOSSES cycle_variations_0123 --PARTIAL_MASKING True  --PARTIAL_FOV 0.6;
python MvMHAT_SSPCC/inference.py --model Masked_CV_0123_TDSS_s0;
python MVM_EVAL/Evaluate.py --model Masked_CV_0123_TDSS_s0;



# For Table 5, we perform cycle variation ablations report for runs 0-2. 
# For example for seed 0:

python MvMHAT_SSPCC/train.py --EX_ID CV_0_TDSS_s0 --SEED 0 --TDSS True --LOSSES cycle_variations_0;
python MvMHAT_SSPCC/inference.py --model CV_0_TDSS_s0;
python MVM_EVAL/Evaluate.py --model CV_0_TDSS_s0;

python MvMHAT_SSPCC/train.py --EX_ID CV_1_TDSS_s0 --SEED 0 --TDSS True --LOSSES cycle_variations_1;
python MvMHAT_SSPCC/inference.py --model CV_1_TDSS_s0;
python MVM_EVAL/Evaluate.py --model CV_1_TDSS_s0;

python MvMHAT_SSPCC/train.py --EX_ID CV_2_TDSS_s0 --SEED 0 --TDSS True --LOSSES cycle_variations_2;
python MvMHAT_SSPCC/inference.py --model CV_2_TDSS_s0;
python MVM_EVAL/Evaluate.py --model CV_2_TDSS_s0;

python MvMHAT_SSPCC/train.py --EX_ID CV_3_TDSS_s0 --SEED 0 --TDSS True --LOSSES cycle_variations_3;
python MvMHAT_SSPCC/inference.py --model CV_3_TDSS_s0;
python MVM_EVAL/Evaluate.py --model CV_3_TDSS_s0;

python MvMHAT_SSPCC/train.py --EX_ID CV_12_TDSS_s0 --SEED 0 --TDSS True --LOSSES cycle_variations_12;
python MvMHAT_SSPCC/inference.py --model CV_01_TDSS_s0;
python MVM_EVAL/Evaluate.py --model CV_01_TDSS_s0;

python MvMHAT_SSPCC/train.py --EX_ID CV_123_TDSS_s0 --SEED 0 --TDSS True --LOSSES cycle_variations_123;
python MvMHAT_SSPCC/inference.py --model CV_123_TDSS_s0;
python MVM_EVAL/Evaluate.py --model CV_123_TDSS_s0;

python MvMHAT_SSPCC/train.py --EX_ID CV_0123_TDSS_s0 --SEED 0 --TDSS True --LOSSES cycle_variations_0123;
python MvMHAT_SSPCC/inference.py --model CV_0123_TDSS_s0;
python MVM_EVAL/Evaluate.py --model CV_012_TDSS_s0;


