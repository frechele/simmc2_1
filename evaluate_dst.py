import argparse
import pickle
from tqdm import tqdm
import numpy as np

import torch

import simmc.data.labels as L
from simmc.model.osnet import create_tokenizer, OSNet, calc_object_similarity


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

        self.tokenizer = create_tokenizer()

    @torch.no_grad()
    def predict(self, context, object_map, object_ids, metadata):
        context = self.tokenizer([context], padding=True, truncation=True, return_tensors="pt").to(device)
        object_map = torch.LongTensor(np.stack(object_map)).unsqueeze(0).to(device)
        object_masks = torch.zeros(1, object_map.shape[1]).bool().to(device)

        output = self.net(context, object_map, object_masks)

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

        request_slots = []
        if output.request_slot_exist.item() > 0:
            for i, slot in enumerate(output.request_slot.cpu().numpy()[0]):
                if slot > 0:
                    request_slots.append(L.SLOT_KEY[i])

        objects = []
        if output.object_exist.item() > 0:
            objects_sim = output.objects.squeeze()
            for i, obj_id in enumerate(object_ids):
                if objects_sim[i] > 0:
                    objects.append(obj_id)

        slot_values = []
        if act == "INFORM:GET":
            slots = []
            for i, slot in enumerate(output.slot_query.cpu().numpy()[0]):
                if slot > 0:
                    slots.append(L.SLOT_KEY[i])

            for obj_id in objects:
                for slot in slots:
                    if obj_id in metadata and slot in metadata[obj_id]:
                        slot_values.append((slot, metadata[obj_id][slot]))

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
        object_map = data["object_map"][idx]
        object_ids = data["object_ids"][idx]
        metadata = data["raw_metadata"][idx]

        str_belief_state = predictor.predict(context, object_map, object_ids, metadata)

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
