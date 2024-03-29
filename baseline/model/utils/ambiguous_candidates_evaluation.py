"""Script evaluates ambiguous candidates for SIMMC 2.1 using golden labels.

Expected JSON format:

[
    "dialog_id": <dialog_id>,
    "predictions": [
        {
            "turn_id": <turn_id>,
            "disambiguation_candidates": <bool>,
        }
        ...
    ]
    ...
]

Author(s): Satwik Kottur
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import json

import numpy as np


def compute_precision_recall_f1(n_correct, n_true, n_pred):
    """Computes the precision, recall, and F1 scores.

    Args:
        n_correct: Number of correct (overlapping) predictions
        n_true: Number of ground truth items
        n_pred: Number of items predicted by a model

    Returns:
        rec: Recall
        prec: Precision
        f1: F1 score
    """
    rec = n_correct / n_true if n_true != 0 else 0.
    prec = n_correct / n_pred if n_pred != 0 else 0.
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) != 0 else 0.
    return rec, prec, f1


def evaluate_ambiguous_candidates(
    gt_labels, model_results, record_instance_results=None
):
    """Evaluates ambiguous candidates identification subtask.

    Uses golden labels and model predictions.

    Args:
        gt_labels: Ground truth labels.
        model_results: Generated labels.
        record_instance_results: Path to save instance-level metrics.
    """
    gt_label_pool = {ii["dialogue_idx"]: ii for ii in gt_labels["dialogue_data"]}

    predictions = []
    num_evaluations = 0
    num_target_candidates = 0
    num_pred_candidates = 0
    num_overlap_candidates = 0
    for model_datum in model_results:
        dialog_id = model_datum["dialog_id"]
        for round_datum in model_datum["predictions"]:
            round_id = round_datum["turn_id"]
            pred_set = set(round_datum["disambiguation_candidates"])
            gt_datum = gt_label_pool[dialog_id]["dialogue"][round_id]

            assert "disambiguation_label" in gt_datum["transcript_annotated"], (
                "Turn not to be evaluated!"
            )
            num_evaluations += 1
            target_set = set(
                gt_datum["transcript_annotated"]["disambiguation_candidates"]
            )
            num_target_candidates += len(target_set)
            num_pred_candidates += len(pred_set)
            num_overlap_candidates += len(pred_set.intersection(target_set))

            # Add the result to datum and save it back.
            if record_instance_results:
                round_datum["ambiguous_candidate_report"] = {
                    "num_pred": len(pred_set),
                    "num_target": len(target_set),
                    "num_overlap": len(pred_set.intersection(target_set)),
                }

    print(f"# Instances evaluated: {num_evaluations}")
    # Record and save per instance results.
    if record_instance_results:
        print("Saving per instance result: {}".format(record_instance_results))
        with open(record_instance_results, "w") as file_id:
            json.dump(model_results, file_id)
    recall, precision, f1 = compute_precision_recall_f1(
        num_overlap_candidates, num_target_candidates, num_pred_candidates
    )
    return {"recall": recall, "precision": precision, "f1": f1}


def main(args):
    print("Reading: {}".format(args["data_json_path"]))
    with open(args["data_json_path"], "r") as file_id:
        gt_labels = json.load(file_id)
    print("Reading: {}".format(args["model_result_path"]))
    with open(args["model_result_path"], "r") as file_id:
        model_results = json.load(file_id)

    if args["record_instance_results"]:
        instance_results_path = args["model_result_path"].replace(
            ".json", "_results.json"
        )
    else:
        instance_results_path = None

    report = evaluate_ambiguous_candidates(
        gt_labels, model_results, record_instance_results=instance_results_path
    )
    print(
        f"""Rec: {report["recall"]:.4f}  |  """
        f"""Prec: {report["precision"]:.4f}  |  """
        f"""F1: {report["f1"]:.4f}"""
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Disambiguation Evaluation")
    parser.add_argument(
        "--data_json_path",
        default="data/simmc2.1_dials_dstc11_devtest.json",
        help="Data with gold label for disambiguation",
    )
    parser.add_argument(
        "--model_result_path",
        default=None,
        help="Disambiguation labels generated by the model",
    )
    parser.add_argument(
        "--record_instance_results",
        dest="record_instance_results",
        action="store_true",
        default=False,
        help="Records per instance results and save it back",
    )
    try:
        parsed_args = vars(parser.parse_args())
    except (IOError) as msg:
        parser.error(str(msg))
    main(parsed_args)
