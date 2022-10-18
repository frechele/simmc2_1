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
        inp_dialogue_idx = data["dialogue_idx"][idx]
        inp_turn_idx = data["turn_idx"][idx]

        inp_context = data["context"][idx]
        inp_objects = data["object_map"][idx]
        inp_object_ids = data["object_ids"][idx]
        inp_metadata = data["raw_metadata"][idx]

        # subtask 2
        disamb = data["disamb"][idx]
        disamb_objs = []
        if disamb:
            for obj_id in data["disamb_objects"][idx]:
                disamb_objs.append(data["object_ids"][idx][obj_id])

        # subtask 3
        act = L.ACTION[data["acts"][idx]]

        request_slots = []
        for i, slot in enumerate(data["request_slots"][idx]):
            if slot > 0:
                request_slots.append(L.SLOT_KEY[i])

        slot_values = []
        objects = []
        for obj_id in data["objects"][idx]:
            objects.append(inp_object_ids[obj_id])

        if act == "INFORM:GET":
            slots = []
            for i, slot in enumerate(data["slot_query"][idx]):
                if slot > 0:
                    slots.append(L.SLOT_KEY[i])

            for obj_id in objects:
                for slot in slots:
                    if obj_id in inp_metadata and slot in inp_metadata[obj_id]:
                        slot_values.append((slot, inp_metadata[obj_id][slot]))

        else:
            slot_values = data["slot_values"][idx]

        belief_state = "{act} [ {slot_values} ] ({request_slots}) < {object} > | {disamb_candidates} |".format(
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
            object=", ".join(
                list(map(str, objects))
            ),
            disamb_candidates=", ".join(
                list(map(str, disamb_objs))
            )
        )

        outputs[(inp_dialogue_idx, inp_turn_idx)] = TEMPLATE_TARGET.format(
            context=inp_context,
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
