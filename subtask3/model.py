from collections import namedtuple

import torch
import torch.nn as nn

from data.labels import ACTION_STRINGS, SLOT_KEY_STRINGS, NUM_OF_OBJECTS


# act : action prediction
# request: 0 if action is request, 1 otherwise
# slots: slots prediction
# objects: objects prediction
DSTHeadOutput = namedtuple("DSTHeadOutput", ["act", "request", "slots", "objects"])


class DSTHead(nn.Module):
    def __init__(self, in_features: int):
        super(DSTHead, self).__init__()

        num_hidden = 256

        self.comm_fc = nn.Sequential(
            nn.Linear(in_features, num_hidden),
            nn.ReLU(inplace=True)
        )

        self.action_fc = nn.Linear(num_hidden, len(ACTION_STRINGS))  # multi-class

        self.request_fc = nn.Linear(num_hidden, 3)  # 3-aray 
        self.slots_fc = nn.Linear(num_hidden, len(SLOT_KEY_STRINGS))  # multi-label

        self.objects_fc = nn.Linear(num_hidden, NUM_OF_OBJECTS)  # multi-label

    def forward(self, x: torch.Tensor) -> DSTHeadOutput:
        x = self.comm_fc(x)

        act = self.action_fc(x)
        request = self.request_fc(x)
        slots = self.slots_fc(x)
        objects = self.objects_fc(x)

        return DSTHeadOutput(
            act=act,
            request=request,
            slots=slots,
            objects=objects
        )


if __name__ == "__main__":
    import torch.nn.functional as F

    num_features = 512

    model = DSTHead(num_features)

    dummy_input = torch.randn(1, num_features)

    # dummy targets
    action_t = torch.zeros(1, dtype=torch.long).random_(0, len(ACTION_STRINGS))

    request_t = torch.zeros(1, dtype=torch.long).random_(0, 3)
    slots_t = torch.zeros(1, len(SLOT_KEY_STRINGS)).random_(0, 2)

    objects_t = torch.zeros(1, NUM_OF_OBJECTS).random_(0, 2)

    outputs = model(dummy_input)

    loss_action = F.cross_entropy(outputs.act, action_t)
    
    loss_request = F.cross_entropy(outputs.request, request_t)
    loss_slots = F.binary_cross_entropy_with_logits(outputs.slots, slots_t)

    loss_objects = F.binary_cross_entropy_with_logits(outputs.objects, objects_t)

    loss = loss_action + loss_request + loss_slots + loss_objects
    print(loss.item())
