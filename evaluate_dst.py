import argparse
import pickle
from tqdm import tqdm
import numpy as np

import torch

from transformers import AlbertTokenizer

import simmc.data.labels as L
from simmc.model.osnet import OSNet, calc_object_similarity


START_BELIEF_STATE = "=> Belief State :"

END_OF_BELIEF = "<EOB>"
END_OF_SENTENCE = "<EOS>"

TEMPLATE_TARGET = (
    "{context} {START_BELIEF_STATE} {belief_state} "
    "{END_OF_BELIEF} {response} {END_OF_SENTENCE}"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Predictor:
    def __init__(self, net: OSNet):
        self.net = net

        self.tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")

    @torch.no_grad()
    def predict(self, context, objects, object_ids, metadata):
        context = self.tokenizer([context], padding=True, truncation=True, return_tensors="pt").to(device)
        objects = torch.FloatTensor(np.stack(objects)).unsqueeze(0).to(device)
        object_masks = torch.ones(1, objects.shape[1]).bool().to(device)

        output = self.net(context, objects, object_masks)

        # subtask 2
        disamb = output.disamb.item() > 0

        disamb_objs = []
        if disamb:
            disamb_obj = output.disamb_objs.squeeze()

            for i, obj_id in enumerate(object_ids):
                if disamb_obj[i] > 0:
                    disamb_objs.append(obj_id)

        # subtask 3
        act = L.ACTION[output.acts.argmax(dim=1).item()]
        is_req = output.is_request.item() > 0

        request_slots = []
        slots = []
        for i, slot in enumerate(output.slots.cpu().numpy()[0]):
            if slot > 0:
                if is_req:
                    request_slots.append(L.SLOT_KEY[i])
                else:
                    slots.append(L.SLOT_KEY[i])

        object_exists = output.object_exists.item() > 0

        slot_values = []
        objects = []
        if not is_req and object_exists:
            objects_proj = output.objects.squeeze()
            for i, obj_id in enumerate(object_ids):
                if objects_proj[i] > 0:
                    objects.append(obj_id)

            for obj_id in objects:
                for slot in slots:
                    if obj_id in metadata and slot in metadata[obj_id]:
                        slot_values.append([slot, metadata[obj_id][slot]])


        belief_state = "{act} [ {slot_values} ] ({request_slots}) < {objects} > | {disamb_candidates} |".format(
            act=act,
            slot_values=", ".join(
                [
                    f"{slot} = {str(value)}"
                    for slot, value in slot_values
                ]
            ),
            request_slots=", ".join(
                request_slots
            ),
            objects=", ".join(
                list(map(str, objects))
            ),
            disamb_candidates=", ".join(
                list(map(str, disamb_objs))
            )
        )

        return belief_state


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_path", help="path to index file", required=True)
    parser.add_argument("--input", help="input file", required=True)
    parser.add_argument("--output", help="output file", required=True)

    parser.add_argument("--ckpt", help="checkpoint file", required=True)
    
    args = parser.parse_args()

    indices = []
    with open(args.index_path, "rt") as f:
        for line in f.readlines():
            indices.append(tuple(map(int, line.split())))

    with open(args.input, "rb") as f:
        data = pickle.load(f)

    net = torch.load(args.ckpt).to(device)
    net.eval()
    predictor = Predictor(net)

    outputs = dict()

    for idx in tqdm(range(len(data["dialogue_idx"]))):
        dialogue_idx = data["dialogue_idx"][idx]
        turn_idx = data["turn_idx"][idx]

        context = data["context"][idx]
        objects = data["objects"][idx]
        object_ids = data["object_ids"][idx]
        metadata = data["raw_metadata"][idx]

        str_belief_state = predictor.predict(context, objects, object_ids, metadata)

        outputs[(dialogue_idx, turn_idx)] = TEMPLATE_TARGET.format(
            context=context,
            START_BELIEF_STATE=START_BELIEF_STATE,
            belief_state=str_belief_state,
            END_OF_BELIEF=END_OF_BELIEF,
            response="",
            END_OF_SENTENCE=END_OF_SENTENCE
        )

    sorted_outputs = []
    for idx in indices:
        if idx in outputs:
            sorted_outputs.append(outputs[idx])

    with open(args.output, "wt") as f:
        f.write("\n".join(sorted_outputs))
