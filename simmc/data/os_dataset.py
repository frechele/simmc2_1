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

        labels = self.labels[index]
        labels_ = np.zeros(objects.shape[0])
        labels_[labels] = 1

        return {
            "context": context,
            "objects": torch.FloatTensor(objects),
            "labels": torch.LongTensor(labels_),
            "object_masks": torch.BoolTensor(object_padding_mask)
        }


if __name__ == "__main__":
    from random import choice

    dataset = OSDataset("/data/simmc2/train_dials.pkl")
    print(len(dataset))
    print(choice(dataset))
