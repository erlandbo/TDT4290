"""
Scanline classifier mean Average precision evaluator
Written as part of master thesis by Bendik Bogfjellmo
(github.com/bendikbo) (bendik.bogfjellmo@gmail.com)
"""
import torch
from statistics import mean
from matplotlib import pyplot as plt


def _calculate_AP(
    class_predictions: torch.Tensor,
    class_targets: torch.Tensor,
    recall_vals = 1000,
    conf_vals = 1000,
    plot_curves=False,
    plot_name="",
    plot_recall_start=0.6
    ):
    """
    calculates average precision for a single class
    Arguments:
    - class_predictions : torch.Tensor in shape of [num_preds]
    - class_targets     : torch.Tensor in shape of [num_targets]
    where num_preds == num_targets
    """
    #linear approximation of continuous confidence threshold
    confidence_thresholds = torch.linspace(0, 1, conf_vals)
    #Initalize array of predictions considered positive at each distinct confidence threshold
    pos_preds = torch.zeros(conf_vals, class_predictions.size()[0])
    for i in range(conf_vals):
        #confidence >= threshold => positive prediction
        pos_preds[i, class_predictions>=confidence_thresholds[i]] = 1
    #tensor of size [conf_vals] containing true positives for threshold
    num_true_positives = torch.sum((pos_preds*class_targets), dim=1)
    #tensor of size [conf_vals] containing false positives for each threshold
    num_false_positives = torch.sum(pos_preds, dim=1) - num_true_positives
    #The same for false negatives
    num_false_negatives = torch.sum(class_targets) - num_true_positives
    #initialize tensors for precision and recalls
    precisions = torch.zeros(conf_vals)
    recalls = torch.zeros_like(precisions)
    for i in range(conf_vals):
        num_tp = num_true_positives[i]
        num_fp = num_false_positives[i]
        num_fn = num_false_negatives[i]
        if (num_tp + num_fp) == 0:
            precisions[i] = 1
        else:
            precisions[i] = num_tp/(num_tp + num_fp)
        if (num_tp + num_fn) == 0:
            recalls[i] = 0
        else:
            recalls[i] = num_tp / (num_tp + num_fn)
    recall_levels = torch.linspace(0, 1, recall_vals)
    final_precisions = torch.zeros_like(recall_levels)
    for i in range(recall_vals):
        recall_level = recall_levels[i]
        recall_level_precisions = precisions[recalls >= recall_level]
        if not precisions.numel():
            final_precisions[i] = 0
        else:
            final_precisions[i] = torch.max(recall_level_precisions)
    if plot_curves:
        plot_pr_curves(final_precisions, recall_levels, plot_name, plot_recall_start)
    return torch.mean(final_precisions)



def plot_pr_curves(
    precisions,
    recalls,
    plot_name,
    plot_recall_start
    ):
    """
    Function to plot precision-recall curves
    Input:
    precisions - precisions at each recall point
    recalls - recalls at each recall point
    plot_recall_start - where at the recall axis to start the plot
    """
    y = precisions.tolist()
    x = recalls.tolist()
    plt.plot(x, y)
    plt.ylabel("Interpolated precision")
    plt.xlabel("Recall")
    plt.xlim(plot_recall_start, 1.00)
    plt.ylim(0.8, 1.0)
    plt.savefig(plot_name + ".pdf", format="pdf")
    plt.clf()


def calculate_mAP(
    predictions: torch.Tensor,
    targets : torch.Tensor,
    plot_curves=False,
    dereference_dict={}
    ):
    """
    calculates mean average precision based on predictions and targets.
    Arguments:
    - predictions   : torch.Tensor in shape of [num_preds, num_classes]
    - targets       : torch.Tensor in shape of [num_targets, num_classes]
    where num_targets == num_preds
    - plot_curves   : bool, decides wether precision/recall curves
    should be plotted, plots are stored as pdfs in folder where
    original scirpt has been called from
    - dereference_dict: dictionary to dereference from number to class name
    """
    ap_vals = {}
    for i in range(targets.size()[-1]):
        #print(f"class: {i}")
        class_predictions   = predictions[:, i]
        class_targets       = targets[:,i]
        if plot_curves:
            plot_name=dereference_dict[i]
        else:
            plot_name = ""
        class_AP = _calculate_AP(
            class_predictions,
            class_targets,
            plot_curves = plot_curves,
            plot_name = plot_name
            )
        #Tensors are a bit annoying to work without significant payoff.
        ap_vals[i] = float(class_AP)
    ap_vals["mAP"] = mean(ap_vals.values())
    return ap_vals
