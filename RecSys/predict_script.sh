#!/bin/bash

# g=$1
# c=$2

data_dir="./PfastreXML/Data"
results_dir="./PfastreXML/Results"
model_dir="./PfastreXML/Results/model"

# trn_ft_file="${data_dir}/train_data_${3}.X"
# trn_lbl_file="${data_dir}/train_data_${3}.y"
tst_ft_file="${data_dir}/test_data.X"
# tst_lbl_file="${data_dir}/test_data_${3}.y"
# inv_prop_file="${data_dir}/inv_prop.txt"
score_file="${results_dir}/score_mat.txt"

#/usr/local/MATLAB/R2019b/bin/matlab -nodesktop -nodisplay -r "cd('$PWD'); addpath(genpath('../Tools'));addpath(genpath('/home/snehal/5thSem/CS771/assn2/Tree_Extreme_Classifiers/Tree_Extreme_Classifiers/Tools/matlab'));trn_X_Y = read_text_mat('$trn_lbl_file'); tst_X_Y = read_text_mat('$tst_lbl_file'); wts = inv_propensity(trn_X_Y,$1,$2); dlmwrite('$inv_prop_file',wts(:),'newline','unix'); exit;"


# training  
# Reads training features (in $trn_ft_file), training labels (in $trn_lbl_file), inverse propensity label weights (in $inv_prop_file), and writes Pfast(re)XML model (to $model_dir)
# ./PfastreXML_train $trn_ft_file $trn_lbl_file $inv_prop_file $model_dir -S 0 -T 4 -s 0 -t 20 -b 1.0 -c $c -m 10 -g $g -a 0.8 

# testing
# Reads test features (in $tst_ft_file) and model (in $model_dir), and writes test label scores (to $score_file)
./PfastreXML/PfastreXML_predict $tst_ft_file $score_file $model_dir -S 0 -T 4 -s 0 -t 20 -n 1000  -q 1 

# performance evaluation
#/usr/local/MATLAB/R2019b/bin/matlab -nodesktop -nodisplay -r "cd('$PWD'); addpath(genpath('../Tools'));addpath(genpath('/home/snehal/5thSem/CS771/assn2/Tree_Extreme_Classifiers/Tree_Extreme_Classifiers/Tools/matlab'));trn_X_Y = read_text_mat('$trn_lbl_file'); tst_X_Y = read_text_mat('$tst_lbl_file'); wts = inv_propensity(trn_X_Y,$1,$2); score_mat = read_text_mat('$score_file'); get_all_metrics(score_mat, tst_X_Y, wts); exit;"