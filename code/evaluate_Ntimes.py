import os
import numpy as np
import argparse
from medpy import metric
from tqdm import tqdm
from utils import read_list, read_nifti
import torch
import torch.nn.functional as F
from utils.config import Config
from scipy import ndimage
from scipy.spatial.distance import cdist

# -----------------------------
# Argument parser
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='amos')
parser.add_argument('--exp', type=str, default="dhc")
parser.add_argument('--folds', type=int, default=3)
parser.add_argument('--cps', type=str, default="AB")
parser.add_argument('--spacing', type=float, nargs=3, default=(1.0, 1.0, 1.0),
                    help="Voxel spacing (z, y, x) in mm for Surface Dice")
parser.add_argument('--threshold', type=float, default=2.0, help="Boundary F1 distance threshold in mm")
args = parser.parse_args()

config = Config(args.task)

# -----------------------------
# Helper functions
# -----------------------------
def resize_array(arr, target_shape):
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float()
    resized_tensor = F.interpolate(tensor, size=target_shape, mode='nearest').squeeze(0).squeeze(0)
    return resized_tensor.numpy()

def compute_surface_dice(pred, label, tolerance=2.0, spacing=(1.0, 1.0, 1.0)):
    pred_border = pred ^ ndimage.binary_erosion(pred)
    label_border = label ^ ndimage.binary_erosion(label)
    
    pred_surface = np.argwhere(pred_border)
    label_surface = np.argwhere(label_border)
    
    if len(pred_surface) == 0 or len(label_surface) == 0:
        return 0.0

    pred_surface_scaled = pred_surface * np.array(spacing)
    label_surface_scaled = label_surface * np.array(spacing)

    distances_pred_to_label = cdist(pred_surface_scaled, label_surface_scaled).min(axis=1)
    distances_label_to_pred = cdist(label_surface_scaled, pred_surface_scaled).min(axis=1)

    pred_within_tolerance = np.sum(distances_pred_to_label <= tolerance)
    label_within_tolerance = np.sum(distances_label_to_pred <= tolerance)

    surface_dice = (pred_within_tolerance + label_within_tolerance) / (len(pred_surface) + len(label_surface))
    return surface_dice * 100

def compute_boundary_f1(pred, label, threshold=2.0, spacing=(1.0, 1.0, 1.0)):
    pred_border = pred ^ ndimage.binary_erosion(pred)
    label_border = label ^ ndimage.binary_erosion(label)
    
    pred_points = np.argwhere(pred_border)
    label_points = np.argwhere(label_border)
    
    if len(pred_points) == 0 or len(label_points) == 0:
        return 0.0, 0.0, 0.0
    
    pred_scaled = pred_points * np.array(spacing)
    label_scaled = label_points * np.array(spacing)
    
    distances_pred_to_label = cdist(pred_scaled, label_scaled).min(axis=1)
    distances_label_to_pred = cdist(label_scaled, pred_scaled).min(axis=1)
    
    precision = np.sum(distances_pred_to_label <= threshold) / len(pred_points)
    recall = np.sum(distances_label_to_pred <= threshold) / len(label_points)
    
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return f1 * 100, precision * 100, recall * 100

def print_metrics(title, mean_metrics, std_metrics):
    metric_names = ["Dice", "ASD", "HD95", "Surface Dice 1mm", "Surface Dice 2mm",
                    "Boundary F1", "Boundary Precision", "Boundary Recall"]
    print(f"\n{title}:")
    for i, name in enumerate(metric_names):
        print(f"{name}: {mean_metrics[i]:.2f} ± {std_metrics[i]:.2f}")

# -----------------------------
# Main evaluation
# -----------------------------
if __name__ == '__main__':
    ids_list = read_list('test', task=args.task)
    results_all_folds = []

    txt_path = f"./logs/{args.exp}/evaluation_res.txt"
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    print("\nEvaluating...")

    with open(txt_path, 'w') as fw:
        for fold in range(1, args.folds + 1):
            test_cls = [i for i in range(1, config.num_cls)]
            values = np.zeros((len(ids_list), len(test_cls), 8))

            # ✅ Track valid cases for this fold
            valid_cases = []

            for idx, data_id in enumerate(tqdm(ids_list)):
                file_path = os.path.join("./logs", args.exp, f"fold{fold}", f"predictions_{args.cps}", f'{data_id}.nii.gz')
                if not os.path.exists(file_path):
                    print(f"File does not exist: {file_path}")
                    continue

                pred = read_nifti(file_path)

                if args.task == "amos":
                    label_path = os.path.join(config.base_dir, 'labelsVa', f'{data_id}.nii.gz')
                else:
                    label_path = os.path.join(config.base_dir, 'labelsTr', f'label{data_id}.nii.gz')

                if not os.path.exists(label_path):
                    print(f"Label file does not exist: {label_path}")
                    continue

                label = read_nifti(label_path).astype(np.int8)
                label = resize_array(label, pred.shape).astype(np.int8)

                for i in test_cls:
                    pred_i = (pred == i).astype(bool)
                    label_i = (label == i).astype(bool)

                    if pred_i.sum() > 0 and label_i.sum() > 0:
                        dice = metric.binary.dc(pred_i, label_i) * 100
                        asd = metric.binary.asd(pred_i, label_i)
                        hd95 = metric.binary.hd95(pred_i, label_i)
                        sd_1mm = compute_surface_dice(pred_i, label_i, tolerance=1.0, spacing=args.spacing)
                        sd_2mm = compute_surface_dice(pred_i, label_i, tolerance=2.0, spacing=args.spacing)
                        bf1, bp, br = compute_boundary_f1(pred_i, label_i, threshold=args.threshold, spacing=args.spacing)
                    elif pred_i.sum() > 0 or label_i.sum() > 0:
                        dice, asd, hd95, sd_1mm, sd_2mm, bf1, bp, br = 0, 128, 128, 0, 0, 0, 0, 0
                    else:
                        dice, asd, hd95, sd_1mm, sd_2mm, bf1, bp, br = 1, 0, 0, 100, 100, 100, 100, 100

                    values[idx][i - 1] = np.array([dice, asd, hd95, sd_1mm, sd_2mm, bf1, bp, br])

                print(f"Intermediate metrics for {data_id}: {values[idx]}")

                # ✅ Add case to valid cases after processing
                valid_cases.append(idx)

            # ✅ Use only valid cases for averaging
            values_valid = values[valid_cases]
            values_mean_cases = np.mean(values_valid, axis=0)
            results_all_folds.append(values_valid)

            # Write fold metrics
            fw.write(f"\nFold {fold}\n")
            for metric_name, col in zip(
                ["Dice", "ASD", "HD95", "Surface Dice 1mm", "Surface Dice 2mm",
                 "Boundary F1", "Boundary Precision", "Boundary Recall"], range(8)
            ):
                fw.write(f"------ {metric_name} ------\n")
                fw.write(str(np.round(values_mean_cases[:, col], 2)) + '\n')
            fw.write("="*50 + "\n")

            print_metrics(f"Fold {fold} Average Metrics", np.mean(values_mean_cases, axis=0), np.std(values_mean_cases, axis=0))

        # ----------------------------- Combine all folds -----------------------------
        min_cases = min(len(r) for r in results_all_folds)
        results_all_folds = np.array([r[:min_cases] for r in results_all_folds])
        # results_all_folds = np.array(results_all_folds)
        final_per_class = np.mean(results_all_folds, axis=(0, 1))
        fold_means = np.mean(results_all_folds, axis=(1, 2))
        mean_metrics = np.mean(fold_means, axis=0)
        std_metrics = np.std(fold_means, axis=0,ddof=1)
        results_folds_mean = results_all_folds.mean(axis=0)

        # Write combined results
        fw.write("\nAll folds combined\n")
        # for i, data_id in enumerate(ids_list):
        for i, data_id in enumerate(ids_list[:min_cases]):
            fw.write("="*5 + f" Case-{data_id}\n")
            fw.write("Dice: " + str(np.round(results_folds_mean[i][:, 0], 2).tolist()) + '\n')
            fw.write("ASD: " + str(np.round(results_folds_mean[i][:, 1], 2).tolist()) + '\n')
            fw.write("HD95: " + str(np.round(results_folds_mean[i][:, 2], 2).tolist()) + '\n')
            fw.write("Surface Dice 1mm: " + str(np.round(results_folds_mean[i][:, 3], 2).tolist()) + '\n')
            fw.write("Surface Dice 2mm: " + str(np.round(results_folds_mean[i][:, 4], 2).tolist()) + '\n')
            fw.write("Boundary F1: " + str(np.round(results_folds_mean[i][:, 5], 2).tolist()) + '\n')
            fw.write("Boundary Precision: " + str(np.round(results_folds_mean[i][:, 6], 2).tolist()) + '\n')
            fw.write("Boundary Recall: " + str(np.round(results_folds_mean[i][:, 7], 2).tolist()) + '\n')
        
        # Final metrics
        fw.write("\n" + "="*50 + "\n")
        fw.write("FINAL RESULTS ACROSS ALL FOLDS\n")
        fw.write("="*50 + "\n")
        for idx, name in enumerate(["Dice", "ASD", "HD95", "Surface Dice 1mm", "Surface Dice 2mm", 
                                    "Boundary F1", "Boundary Precision", "Boundary Recall"]):
            fw.write(f"Final {name} of each class\n")
            fw.write(str(np.round(final_per_class[:, idx], 1).tolist()) + '\n')
        
        fw.write("\n" + "="*50 + "\n")
        fw.write(f"Final Avg Dice: {mean_metrics[0]:.2f}±{std_metrics[0]:.2f}\n")
        fw.write(f"Final Avg ASD: {mean_metrics[1]:.2f}±{std_metrics[1]:.2f}\n")
        fw.write(f"Final Avg HD95: {mean_metrics[2]:.2f}±{std_metrics[2]:.2f}\n")
        fw.write(f"Final Avg Surface Dice 1mm: {mean_metrics[3]:.2f}±{std_metrics[3]:.2f}\n")
        fw.write(f"Final Avg Surface Dice 2mm: {mean_metrics[4]:.2f}±{std_metrics[4]:.2f}\n")
        fw.write(f"Final Avg Boundary F1: {mean_metrics[5]:.2f}±{std_metrics[5]:.2f}\n")
        fw.write(f"Final Avg Boundary Precision: {mean_metrics[6]:.2f}±{std_metrics[6]:.2f}\n")
        fw.write(f"Final Avg Boundary Recall: {mean_metrics[7]:.2f}±{std_metrics[7]:.2f}\n")
        fw.write("="*50 + "\n")

        print_metrics("Final Average Metrics Across All Folds", mean_metrics, std_metrics)