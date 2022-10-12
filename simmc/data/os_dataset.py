import torch
import pickle
import numpy as np

from torch.utils.data import Dataset

from simmc.data.preprocess import OBJECT_FEATURE_SIZE


class OSDataset(Dataset):
    def __init__(self, filename: str, object_padding: int = 140):
        super(OSDataset, self).__init__()

        with open(filename, "rb") as f:
            data = pickle.load(f)

        self.context = data["context"]
        self.objects = data["objects"]

        # subtask 2
        self.disamb = data["disamb"]
        self.disamb_objects = data["disamb_objects"]

        # subtask 3
        self.acts = data["acts"]
        self.is_request = data["is_request"]
        self.slots = data["slots"]

        self.labels = data["labels"]

        self.object_padding = object_padding

    def __len__(self) -> int:
        return len(self.context)

    def __getitem__(self, index: int):
        context = self.context[index]

        objects = self.objects[index]
        object_padding_len = max(0, self.object_padding - len(objects))
        object_padding_mask = np.array([0] * len(objects) + [1] * object_padding_len)
        objects += [np.zeros(OBJECT_FEATURE_SIZE)] * object_padding_len
        objects = np.array(objects)

        # subtask 2
        disamb = self.disamb[index]
        disamb_objects = self.disamb_objects[index]
        disamb_objects_ = np.zeros(objects.shape[0])
        disamb_objects_[disamb_objects] = 1

        # subtask 3
        acts = self.acts[index]
        is_request = int(self.is_request[index])
        slots = self.slots[index]

        labels = self.labels[index]
        labels_ = np.zeros(objects.shape[0])
        labels_[labels] = 1

        return {
            "context": context,
            "objects": torch.FloatTensor(objects),
            "object_masks": torch.BoolTensor(object_padding_mask),

            "disamb": torch.tensor(disamb).float().unsqueeze(-1),
            "disamb_objects": torch.LongTensor(disamb_objects_),

            "acts": torch.tensor(acts).long(),
            "is_request": torch.tensor(is_request).float().unsqueeze(-1),
            "slots": torch.FloatTensor(slots),

            "labels": torch.LongTensor(labels_),
        }


if __name__ == "__main__":
    from random import choice

    dataset = OSDataset("/data/simmc2/train_dials.pkl")
    print(len(dataset))
    print(choice(dataset))
