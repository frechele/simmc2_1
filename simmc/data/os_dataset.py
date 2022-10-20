from requests import request
import torch
import pickle
import numpy as np

from torch.utils.data import Dataset


class OSDataset(Dataset):
    def __init__(self, filename: str, object_padding: int = 140):
        super(OSDataset, self).__init__()

        with open(filename, "rb") as f:
            data = pickle.load(f)

        self.context = data["context"]
        self.object_map = data["object_map"]

        # subtask 2
        self.disamb = data["disamb"]
        self.disamb_objects = data["disamb_objects"]

        # subtask 3
        self.acts = data["acts"]
        self.request_slots = data["request_slots"]
        self.objects = data["objects"]
        self.slot_values = data["slot_values"]
        self.slot_query = data["slot_query"]

        self.object_padding = object_padding

    def __len__(self) -> int:
        return len(self.context)

    def __getitem__(self, index: int):
        context = self.context[index]

        object_map = self.object_map[index]
        object_padding_len = max(0, self.object_padding - len(object_map))
        object_padding_mask = np.array([0] * len(object_map) + [1] * object_padding_len)
        object_map += [np.zeros(object_map[0].shape[0])] * object_padding_len
        object_map = np.array(object_map)

        # subtask 2
        disamb = self.disamb[index]
        disamb_objects = self.disamb_objects[index]
        disamb_objects_ = np.zeros(object_map.shape[0])
        disamb_objects_[disamb_objects] = 1

        # subtask 3
        acts = self.acts[index]

        request_slot = self.request_slots[index]
        request_slot_exist = (request_slot.sum() > 0)

        objects = self.objects[index]
        objects_ = np.zeros(object_map.shape[0])
        objects_[objects] = 1

        object_exist = (len(objects) > 0)

        slot_query = self.slot_query[index]

        return {
            "context": context,
            "object_map": torch.LongTensor(object_map),
            "object_masks": torch.BoolTensor(object_padding_mask),

            "disamb": torch.tensor(disamb, dtype=torch.float).unsqueeze(-1),
            "disamb_objects": torch.LongTensor(disamb_objects_),

            "acts": torch.tensor(acts, dtype=torch.long),

            "request_slot": torch.FloatTensor(request_slot),
            "request_slot_exist": torch.tensor(request_slot_exist, dtype=torch.float).unsqueeze(-1),

            "objects": torch.FloatTensor(objects_),
            "object_exist": torch.tensor(object_exist, dtype=torch.float).unsqueeze(-1),

            "slot_query": torch.FloatTensor(slot_query),
        }


if __name__ == "__main__":
    from random import choice

    dataset = OSDataset("/data/simmc2/train_dials.pkl")
    print(len(dataset))
    print(choice(dataset))
