import os
import json
import cv2
import numpy as np
from scipy.spatial.distance import directed_hausdorff

def calculate_metrics(gt, pred):
    gt_flat = gt.flatten()
    pred_flat = pred.flatten()

    TP = np.sum((gt_flat == 1) & (pred_flat == 1))
    TN = np.sum((gt_flat == 0) & (pred_flat == 0))
    FP = np.sum((gt_flat == 0) & (pred_flat == 1))
    FN = np.sum((gt_flat == 1) & (pred_flat == 0))

    epsilon = 1e-7

    Accuracy = (TP + TN) / (TP + TN + FP + FN + epsilon)
    Precision = TP / (TP + FP + epsilon)
    Recall = TP / (TP + FN + epsilon)
    Dice = (2 * TP + epsilon) / (2 * TP + FP + FN + epsilon)
    Jaccard = TP / (TP + FP + FN + epsilon)

    FDR = FP / (TP + FP + epsilon)
    FNR = FN / (TP + FN + epsilon)
    FOR = FN / (TN + FN + epsilon)
    FPR = FP / (FP + TN + epsilon)
    NPV = TN / (TN + FN + epsilon)
    TNR = TN / (TN + FP + epsilon)

    # Hausdorff Distance 95
    if np.any(gt) and np.any(pred):
        gt_pts = np.argwhere(gt == 1)
        pred_pts = np.argwhere(pred == 1)
        hd_95 = max(
            np.percentile([directed_hausdorff(gt_pts, pred_pts)[0]], 95),
            np.percentile([directed_hausdorff(pred_pts, gt_pts)[0]], 95)
        )
    else:
        hd_95 = 0.0

    return {
        "Accuracy": Accuracy,
        "Dice": Dice,
        "False Discovery Rate": FDR,
        "False Negative Rate": FNR,
        "False Omission Rate": FOR,
        "False Positive Rate": FPR,
        "Hausdorff Distance 95": hd_95,
        "Jaccard": Jaccard,
        "Negative Predictive Value": NPV,
        "Precision": Precision,
        "Recall": Recall,
        "Total Positives Reference": int(np.sum(gt)),
        "Total Positives Test": int(np.sum(pred)),
        "True Negative Rate": TNR
    }

def dice_summary_json(gt_folder, pred_folder):
    results = {"results": {"all": []}}
    all_metrics = []
    gt_files = sorted(os.listdir(gt_folder))
    pred_files = sorted(os.listdir(pred_folder))
    idx = 0

    for fname in gt_files:
        if not fname.endswith(".png"):
            continue

        gt_path = os.path.join(gt_folder, fname)
        pred_path = os.path.join(pred_folder, fname)
        if not os.path.exists(pred_path):
            print(f"Prediction for {fname} not found. Skipping.")
            continue

        # Load and binarize
        gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        pred_img = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        _, gt_bin = cv2.threshold(gt_img, 127, 1, cv2.THRESH_BINARY)
        _, pred_bin = cv2.threshold(pred_img, 127, 1, cv2.THRESH_BINARY)

        metrics = calculate_metrics(gt_bin, pred_bin)
        all_metrics.append(metrics)

        result_entry = {
            str(idx): metrics,
            "reference": gt_path,
            "test": pred_path
        }

        results["results"]["all"].append(result_entry)
        idx += 1

    # Compute mean of each metric
    mean_metrics = {}
    for key in all_metrics[0].keys():
        mean_metrics[key] = np.mean([m[key] for m in all_metrics])
    results["results"]["mean"] = mean_metrics

    # Save to JSON
    output_path = os.path.join(pred_folder, "summary1.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Saved summary to {output_path}")

current_script_directory = os.path.dirname(os.path.abspath(__file__))
# Example usage:
dice_summary_json(
    gt_folder="{current_script_directory}/input_mask",
    pred_folder="{current_script_directory}/pred_mask"
)
