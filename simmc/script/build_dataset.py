import argparse
import json
import pickle
from tqdm import tqdm
import numpy as np

from simmc.data.preprocess import metadata_to_vec
import simmc.data.labels as L
from simmc.data.labels import label_to_onehot, labels_to_vector


FIELDNAME_DIALOG = "dialogue"
FIELDNAME_USER_UTTR = "transcript"
FIELDNAME_ASST_UTTR = "system_transcript"
FIELDNAME_BELIEF_STATE = "transcript_annotated"
FIELDNAME_SYSTEM_STATE = "system_transcript_annotated"
FIELDNAME_DISAMB_LABEL = "disambiguation_label"
FIELDNAME_DISAMB_OBJS = "disambiguation_candidates"


def convert(dialogues, scenes, len_context):
    results = {
        "dialogue_idx": [],
        "turn_idx": [],

        "context": [],
        "objects": [],

        # subtask 2 outputs
        "disamb": [],
        "disamb_objects": [],

        # subtask 3 outputs
        "acts": [],
        "is_request": [],
        "slots": [],

        "labels": []
    }

    max_objects = 0
    for dialogue_data in tqdm(dialogues):
        prev_asst_uttr = None
        lst_context = []

        object_map = []
        id_to_idx = dict()

        last_idx = 0
        for scene_name in dialogue_data["scene_ids"].values():
            scene = scenes[scene_name]
            object_map += [metadata_to_vec(obj) for obj in scene["objects"]]
            mapping = { k: v for k, v in zip(scene["id_to_idx"].keys(), range(last_idx, last_idx + len(scene["id_to_idx"]))) }
            id_to_idx.update(mapping)

        max_objects = max(max_objects, len(object_map))

        for turn_id, turn in enumerate(dialogue_data[FIELDNAME_DIALOG]):
            user_uttr = turn.get(FIELDNAME_USER_UTTR, "").replace("\n", " ").strip()
            user_belief = turn.get(FIELDNAME_BELIEF_STATE, {})
            asst_uttr = turn.get(FIELDNAME_ASST_UTTR, "").replace("\n", " ").strip()

            context = ""
            if prev_asst_uttr:
                context += f"System : {prev_asst_uttr} "

            context += f"User : {user_uttr}"
            prev_asst_uttr = asst_uttr

            lst_context.append(context)
            context = " ".join(lst_context[-len_context:])

            results["dialogue_idx"].append(dialogue_data["dialogue_idx"])
            results["turn_idx"].append(turn_id)

            results["context"].append(context)

            # subtask 2 outputs
            results["disamb"].append(user_belief[FIELDNAME_DISAMB_LABEL])
            disamb_objs = [id_to_idx[obj_id] for obj_id in user_belief[FIELDNAME_DISAMB_OBJS]]
            results["disamb_objects"].append(disamb_objs)

            # subtask 3 outputs
            results["acts"].append(L.ACTION_MAPPING_TABLE[user_belief["act"]])

            is_request = len(user_belief["act_attributes"]["request_slots"]) > 0
            results["is_request"].append(is_request)

            if is_request:
                slots = user_belief["act_attributes"]["request_slots"]
            else:
                slots = list(user_belief["act_attributes"]["slot_values"].keys())
            
            if len(slots) == 0:
                slots = np.zeros(len(L.SLOT_KEY_MAPPING_TABLE))
            else:
                slots = labels_to_vector(slots, L.SLOT_KEY_MAPPING_TABLE)
            results["slots"].append(slots)

            objs = [id_to_idx[obj_id] for obj_id in user_belief["act_attributes"]["objects"]]
            results["objects"].append(object_map)

            results["labels"].append(objs)

    print("max objects:", max_objects)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dialog", help="dialog file", required=True)
    parser.add_argument("--scene", help="scene info file", required=True)

    parser.add_argument("--len_context", type=int, default=2)

    parser.add_argument("--output", help="output file path", required=True)

    args = parser.parse_args()

    with open(args.dialog, "rt") as f:
        dialogues = json.load(f)["dialogue_data"]

    with open(args.scene, "rb") as f:
        scenes = pickle.load(f)

    results = convert(dialogues, scenes, args.len_context)

    with open(args.output, "wb") as f:
        pickle.dump(results, f)
