import argparse
import copy
import pickle

from gpt2_dst.utils.convert import parse_flattened_result, END_OF_BELIEF


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict", help="predict file path")
    parser.add_argument("--target", help="target file path")
    parser.add_argument("--output", help="output file path")

    args = parser.parse_args()
    predict_path = args.predict
    target_path = args.target
    output_path = args.output

    with open(predict_path, "rt") as f:
        predicts = f.read().splitlines()

    with open(target_path, "rt") as f:
        targets = f.read().splitlines()

    results = []

    for predict, target in zip(predicts, targets):
        act_attr = parse_flattened_result(target)[0]

        target_start_idx = target.index(END_OF_BELIEF) + len(END_OF_BELIEF) + 1
        target = target[target_start_idx:].strip()

        result = {
            "act_attr": act_attr,
            "input": predict,
            "target": target
        }
        results.append(copy.deepcopy(result))

    with open(output_path, "wb") as f:
        pickle.dump(results, f)
