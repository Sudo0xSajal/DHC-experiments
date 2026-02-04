#!/bin/bash


while getopts 'm:e:c:t:l:w:' OPT; do
    case $OPT in
        m) method=$OPTARG;;
        e) exp=$OPTARG;;
        c) cuda=$OPTARG;;
		    t) task=$OPTARG;;
		    l) lr=$OPTARG;;
		    w) cps_w=$OPTARG;;
    esac
done
echo $method
echo $cuda

epoch=500
echo $epoch

labeled_data="labeled_5p"
unlabeled_data="unlabeled_5p"
folder="Task_"${task}"_5p/"
cps="AB"

echo $folder

# FOLD 1 - Seed 0
echo "=== Starting Fold 1 (Seed 0) ==="
python3 code/train_${method}.py --task ${task} --exp ${folder}${method}${exp}/fold1 --seed 0 -g ${cuda} --base_lr ${lr} -w ${cps_w} -ep ${epoch} -sl ${labeled_data} -su ${unlabeled_data} --alpha_noise 0.7 --min_alpha 0.05 --grad_clip 1.0 --warmup_epochs 50 --val_freq 20 --noise_convergence_thresh 0.01 --noise_safe_mode -r
python3 code/test.py --task ${task} --exp ${folder}${method}${exp}/fold1 -g ${cuda} --cps ${cps}
python3 code/evaluate_Ntimes.py --task ${task} --exp ${folder}${method}${exp} --folds 1 --cps ${cps}

# FOLD 2 - Seed 1
# echo "=== Starting Fold 2 (Seed 1) ==="
# python3 code/train_${method}.py --task ${task} --exp ${folder}${method}${exp}/fold2 --seed 1 -g ${cuda} --base_lr ${lr} -w ${cps_w} -ep ${epoch} -sl ${labeled_data} -su ${unlabeled_data} --alpha_noise 0.7 --min_alpha 0.05 --grad_clip 1.0 --warmup_epochs 50 --val_freq 20 --noise_convergence_thresh 0.01 --noise_safe_mode -r
# python3 code/test.py --task ${task} --exp ${folder}${method}${exp}/fold2 -g ${cuda} --cps ${cps}
# python3 code/evaluate_Ntimes.py --task ${task} --exp ${folder}${method}${exp} --folds 2 --cps ${cps}

# FOLD 3 - Seed 666
# echo "=== Starting Fold 3 (Seed 666) ==="
# python3 code/train_${method}.py --task ${task} --exp ${folder}${method}${exp}/fold3 --seed 666 -g ${cuda} --base_lr ${lr} -w ${cps_w} -ep ${epoch} -sl ${labeled_data} -su ${unlabeled_data} --alpha_noise 0.7 --min_alpha 0.05 --grad_clip 1.0 --warmup_epochs 50 --val_freq 20 --noise_convergence_thresh 0.01 --noise_safe_mode -r
# python3 code/test.py --task ${task} --exp ${folder}${method}${exp}/fold3 -g ${cuda} --cps ${cps}

# FINAL EVALUATION ALL FOLDS
# echo "=== Final Evaluation All 3 Folds ==="
# python3 code/evaluate_Ntimes.py --task ${task} --exp ${folder}${method}${exp} --folds 3 --cps ${cps}

# echo "=== All 3 folds completed ==="