import argparse
import pickle
import numpy as np

import simmc.data.labels as L


START_BELIEF_STATE = "=> Belief State :"

END_OF_BELIEF = "<EOB>"
END_OF_SENTENCE = "<EOS>"

TEMPLATE_TARGET = (
    "{context} {START_BELIEF_STATE} {belief_state} "
    "{END_OF_BELIEF} {response} {END_OF_SENTENCE}"
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_path", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    indices = []
    with open(args.index_path, "rt") as f:
        for line in f.readlines():
            indices.append(tuple(map(int, line.split())))

    with open(args.input, "rb") as f:
        data = pickle.load(f)

    print(data.keys())

    outputs = dict()

    for idx in range(len(data["dialogue_idx"])):
        dialogue_idx = data["dialogue_idx"][idx]
        turn_idx = data["turn_idx"][idx]

        context = data["context"][idx]
        objects = data["objects"][idx]
        object_ids = data["object_ids"][idx]
        metadata = data["raw_metadata"][idx]

        # subtask 2
        disamb = data["disamb"][idx]
        disamb_objs = []
        if disamb:
            for obj_id in data["disamb_objects"][idx]:
                disamb_objs.append(data["object_ids"][obj_id])

        # subtask 3
        act = L.ACTION[data["acts"][idx]]
        is_req = data["is_request"][idx]

        request_slots = []
        slots = []
        for i, slot in enumerate(data["slots"][idx]):
            if slot > 0:
                if is_req:
                    request_slots.append(L.SLOT_KEY[i])
                else:
                    slots.append(L.SLOT_KEY[i])

        slot_values = []
        objects = []
        if not is_req:
            for obj_id in data["labels"][idx]:
                objects.append(data["object_ids"][idx][obj_id])

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

        outputs[(dialogue_idx, turn_idx)] = TEMPLATE_TARGET.format(
            context=context,
            START_BELIEF_STATE=START_BELIEF_STATE,
            belief_state=belief_state,
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
