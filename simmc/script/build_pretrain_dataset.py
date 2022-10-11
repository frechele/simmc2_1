import argparse
import json
import pickle
from tqdm import tqdm
import numpy as np

from simmc.data.preprocess import metadata_to_vec


FIELDNAME_DIALOG = "dialogue"
FIELDNAME_USER_UTTR = "transcript"
FIELDNAME_ASST_UTTR = "system_transcript"
FIELDNAME_BELIEF_STATE = "transcript_annotated"
FIELDNAME_SYSTEM_STATE = "system_transcript_annotated"


def convert(dialogues, scenes, len_context):
    results = {
        "context": [],
        "objects": [],
        "labels": []
    }

    max_objects = 0
    for dialogue_data in tqdm(dialogues):
        prev_asst_uttr = None
        prev_turn = None
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

        now_scene = None
        for turn_id, turn in enumerate(dialogue_data[FIELDNAME_DIALOG]):
            if str(turn_id) in dialogue_data["scene_ids"]:
                now_scene = scenes[dialogue_data["scene_ids"][str(turn_id)]]

            user_uttr = turn.get(FIELDNAME_USER_UTTR, "").replace("\n", " ").strip()
            user_belief = turn.get(FIELDNAME_BELIEF_STATE, {})
            asst_uttr = turn.get(FIELDNAME_ASST_UTTR, "").replace("\n", " ").strip()

            context = ""
            if prev_asst_uttr:
                context += f"System : {prev_asst_uttr} "

            context += f"User : {user_uttr}"
            prev_asst_uttr = asst_uttr
            prev_turn = turn

            lst_context.append(context)
            context = " ".join(lst_context[-len_context:])

            results["context"].append(context)
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
