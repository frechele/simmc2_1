import argparse
from collections import defaultdict
import json
import pickle
from tqdm import tqdm
import numpy as np

from simmc.data.metadata import MetadataDB
from simmc.data.preprocess import metadata_to_feat, labels_to_vector
import simmc.data.labels as L


FIELDNAME_DIALOG = "dialogue"
FIELDNAME_USER_UTTR = "transcript"
FIELDNAME_ASST_UTTR = "system_transcript"
FIELDNAME_BELIEF_STATE = "transcript_annotated"
FIELDNAME_SYSTEM_STATE = "system_transcript_annotated"
FIELDNAME_DISAMB_LABEL = "disambiguation_label"
FIELDNAME_DISAMB_OBJS = "disambiguation_candidates"


def convert(dialogues, scenes, len_context, db: MetadataDB):
    results = defaultdict(list)

    max_objects = 0
    for dialogue_data in tqdm(dialogues):
        prev_asst_uttr = None
        lst_context = []

        object_map = []
        raw_metadata = dict()
        id_to_idx = dict()

        last_idx = 0
        for scene_name in dialogue_data["scene_ids"].values():
            scene = scenes[scene_name]

            for object_id, old_idx in scene["id_to_idx"].items():
                if object_id in id_to_idx:
                    continue

                new_idx = len(object_map)
                object_map.append(metadata_to_feat(scene["objects"][old_idx], db))
                raw_metadata[object_id] = scene["objects"][old_idx]
                id_to_idx[object_id] = new_idx

        max_objects = max(max_objects, len(object_map))

        for turn_id, turn in enumerate(dialogue_data[FIELDNAME_DIALOG]):
            user_uttr = turn.get(FIELDNAME_USER_UTTR, "").replace("\n", " ").strip()
            user_belief = turn.get(FIELDNAME_BELIEF_STATE, {})
            asst_uttr = turn.get(FIELDNAME_ASST_UTTR, "").replace("\n", " ").strip()

            context = ""
            if prev_asst_uttr:
                context += f"{L.SYSTEM_UTTR_TOKEN} {prev_asst_uttr} "

            context += f"{L.USER_UTTR_TOKEN} {user_uttr}"
            prev_asst_uttr = asst_uttr

            lst_context.append(context)
            context = " ".join(lst_context[-len_context:])

            results["dialogue_idx"].append(dialogue_data["dialogue_idx"])
            results["turn_idx"].append(turn_id)

            results["context"].append(context)

            results["object_map"].append(object_map)
            results["object_ids"].append(list(id_to_idx.keys()))
            results["raw_metadata"].append(raw_metadata)

            # subtask 2 outputs
            results["disamb"].append(user_belief[FIELDNAME_DISAMB_LABEL])
            disamb_objs = [id_to_idx[obj_id] for obj_id in user_belief[FIELDNAME_DISAMB_OBJS]]
            results["disamb_objects"].append(disamb_objs)

            # subtask 3 outputs
            results["acts"].append(L.ACTION_MAPPING_TABLE[user_belief["act"]])

            objs = [id_to_idx[obj_id] for obj_id in user_belief["act_attributes"]["objects"]]
            results["objects"].append(objs)

            request_slots = user_belief["act_attributes"]["request_slots"]
            if len(request_slots) == 0:
                request_slots = np.zeros(len(L.SLOT_KEY_MAPPING_TABLE))
            else:
                request_slots = labels_to_vector(request_slots, L.SLOT_KEY_MAPPING_TABLE)

            results["request_slots"].append(request_slots)

            slot_values = []
            slot_query = np.zeros(len(L.SLOT_KEY_MAPPING_TABLE))
            if len(objs) == 0:
                # slot-value can be filled with only utterance
                for k, v in user_belief["act_attributes"]["slot_values"].items():
                    slot_values.append([k, str(v).strip()])
            else:
                # slot-value can be filled with utterance and metadata
                slots = list(user_belief["act_attributes"]["slot_values"].keys())

                if len(slots) > 0:
                    slot_query = labels_to_vector(slots, L.SLOT_KEY_MAPPING_TABLE)

            results["slot_values"].append(slot_values)
            results["slot_query"].append(slot_query)

    print("max objects:", max_objects)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dialog", help="dialog file", required=True)
    parser.add_argument("--scene", help="scene info file", required=True)
    parser.add_argument("--metadata-db", help="metadata db file", required=True)

    parser.add_argument("--len_context", type=int, default=2)

    parser.add_argument("--output", help="output file path", required=True)

    args = parser.parse_args()

    db = MetadataDB(args.metadata_db)

    with open(args.dialog, "rt") as f:
        dialogues = json.load(f)["dialogue_data"]

    with open(args.scene, "rb") as f:
        scenes = pickle.load(f)

    results = convert(dialogues, scenes, args.len_context, db)

    with open(args.output, "wb") as f:
        pickle.dump(results, f)
