import pickle


class MetadataDB:
    def __init__(self, path: str):
        with open(path, "rb") as f:
            self.db = pickle.load(f)

        self.pad_str = "<pad>"
        self.pad_idx = 0

        self.keys = []

        self.db_inv = dict()
        for k in self.db:
            self.keys.append(k)
            self.db[k].insert(0, self.pad_str)
            self.db_inv[k] = dict()
            for i, v in enumerate(self.db[k]):
                self.db_inv[k][v] = i

        self.keys = sorted(self.keys)
        self.keys.insert(self.pad_idx, self.pad_str)
        self.key_inv = { k: i for i, k in enumerate(self.keys) }

    def get_idx(self, slot: str, value: str) -> int:
        if slot == self.pad_str:
            return self.pad_idx

        return self.db_inv[slot][value]

    def get(self, slot: str, idx: int) -> str:
        if slot == self.pad_str:
            return self.pad_str

        return self.db[slot][idx]

    def get_key_idx(self, slot: str) -> int:
        return self.key_inv[slot]

    def get_key(self, idx: int) -> str:
        return self.keys[idx]


if __name__ == "__main__":
    db = MetadataDB("/data/simmc2/metadata_db.pkl")

    print(len(db.db["color"]))

