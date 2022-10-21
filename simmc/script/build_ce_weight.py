import argparse
import pickle

import torch

import simmc.data.labels as L


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    with open(args.input, "rb") as f:
        data = pickle.load(f)

    counter = dict()

    act_set = set()
    for act in data["acts"]:
        act = L.ACTION[act]
        act_set.add(act)

        counter[act] = counter.get(act, 0) + 1

    weights = []
    for act in L.ACTION:
        weights.append(1 / counter[act])

    weights = torch.FloatTensor(weights)
    torch.save(weights, args.output)
