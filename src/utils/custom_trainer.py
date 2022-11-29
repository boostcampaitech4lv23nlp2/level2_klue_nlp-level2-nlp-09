import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser, Trainer

from src.utils.focal_loss import FocalLoss

# class CustomTrainer(Trainer):
#     def compute_loss(self, model, inputs, return_outputs=False):
#         labels = inputs.get("labels")
#         # forward pass
#         outputs = model(**inputs)
#         logits = outputs.get("logits")
#         # compute custom loss (suppose one has 3 labels with different weights)

#         weight = torch.tensor(
#             [
#                 0.7400,
#                 0.8608,
#                 0.9874,
#                 0.9903,
#                 0.9383,
#                 0.9559,
#                 0.8812,
#                 0.9619,
#                 0.9962,
#                 0.9986,
#                 0.9900,
#                 0.9952,
#                 0.9672,
#                 0.9940,
#                 0.9849,
#                 0.9594,
#                 0.9957,
#                 0.9734,
#                 0.9849,
#                 0.9969,
#                 0.9376,
#                 0.9826,
#                 0.9979,
#                 0.9974,
#                 0.9859,
#                 0.9615,
#                 0.9945,
#                 0.9988,
#                 0.9949,
#                 0.9969,
#             ],
#             device="cuda:0",
#         )

#         gamma = 0
#         loss_fct = FocalLoss(weight, gamma)
#         loss = loss_fct(logits.view(-1, 30), labels.view(-1))  # self.model.model_config.num_labels
#         return (loss, outputs) if return_outputs else loss


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = nn.CrossEntropyLoss(weight=None)
        loss = loss_fct(logits.view(-1, 30), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
