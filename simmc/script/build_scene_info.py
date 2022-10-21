import argparse
from collections import defaultdict
import glob
import json
import pickle
import os
from tqdm import tqdm


def load_metadata(filename: str, domain: str):
    with open(filename, "rt") as f:
        data = json.load(f)

    for k in data:
        data[k]["domain"] = domain

    if domain == "fashion":
        for k in data:
            data[k]["color"] = data[k]["color"].split(", ")
    elif domain == "furniture":
        for k in data:
            data[k]["color"] = data[k]["color"].split(" and ")

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--fashion_metadata", help="fashion metadata file", required=True)
    parser.add_argument("--furniture_metadata", help="furniture metadata file", required=True)

    parser.add_argument("--scene_root", help="root path of scenes", required=True)
    parser.add_argument("--output", help="output file", required=True)
    parser.add_argument("--output-metadata-db", help="output metadata db file", required=True)

    args = parser.parse_args()
    fashion_metadata = load_metadata(args.fashion_metadata, "fashion")
    furniture_metadata = load_metadata(args.furniture_metadata, "furniture")

    metadata = dict()
    metadata.update(fashion_metadata)
    metadata.update(furniture_metadata)

    metadata_db = defaultdict(set)
    for dt in metadata.values():
        for k, v in dt.items():
            if type(v) == list:
                for vv in v:
                    metadata_db[k].add(str(vv))
            else:
                metadata_db[k].add(str(v))

    for k in metadata_db:
        metadata_db[k] = list(metadata_db[k])
        metadata_db[k] = sorted(metadata_db[k])

    with open(args.output_metadata_db, "wb") as f:
        pickle.dump(metadata_db, f)

    print("<Load metadata>")
    print("# of fashion:", len(fashion_metadata))
    print("# of furniture:", len(furniture_metadata))
    print("# of total:", len(metadata))
    print("=" * 30)
    print()

    scene_list = glob.glob(os.path.join(args.scene_root, "*scene.json"))
    print("<Load scenes>")
    print("# of scenes:", len(scene_list))
    print("=" * 30)
    print()

    scenes = dict()
    for scene_fname in tqdm(scene_list):
        with open(scene_fname, "rt") as f:
            scene_data = json.load(f)["scenes"][0]

        scene_name = os.path.basename(scene_fname)[:-11]

        scene = {
            "objects": [],
            "id_to_idx": dict(),
            "relationships": scene_data["relationships"],
        }

        for unique_id, obj in enumerate(scene_data["objects"]):
            obj_id = obj["index"]

            obj_info = metadata[obj["prefab_path"]]
            obj_info["object_id"] = obj_id
            obj_info["bbox"] = obj["bbox"]

            scene["objects"].append(obj_info)

            scene["id_to_idx"][obj_id] = unique_id

        scenes[scene_name] = scene

    with open(args.output, "wb") as f:
        pickle.dump(scenes, f)
