import argparse

from simmc.data.os_dataset import OSDataset
import simmc.data.labels as L


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)

    args = parser.parse_args()

    dataset = OSDataset(args.dataset)

    print("dataset size:", len(dataset))

    sum_disamb = 0
    sum_request_slot, cnt_request_slot, cnt_zero_request_slot = 0, 0, 0
    sum_objects, cnt_objects, cnt_zero_objects = 0, 0, 0

    for i in range(len(dataset)):
        data = dataset[i]

        sum_disamb += data["disamb"].item()

        sum_request_slot += data["request_slot"].sum().item()
        cnt_request_slot += data["request_slot"].size(0)
        cnt_zero_request_slot += (data["request_slot"].sum().item() < 1)

        sum_objects += data["objects"].sum().item()
        cnt_objects += data["object_masks"].sum().item()
        cnt_zero_objects += (data["objects"].sum().item() < 1)

    print("disamb mean:", sum_disamb / len(dataset))

    print("request slot mean:", sum_request_slot / cnt_request_slot)
    print("request slot zero:", cnt_zero_request_slot / len(dataset))

    print("objects mean:", sum_objects / cnt_objects)
    print("objects zero:", cnt_zero_objects / len(dataset))
