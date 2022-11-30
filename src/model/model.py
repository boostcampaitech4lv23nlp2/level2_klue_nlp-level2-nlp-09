import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
)


# BiGRU -> FC
class Model(torch.nn.Module):
    def __init__(self, MODEL_NAME):
        super().__init__()
        # self.model_config =  AutoConfig.from_pretrained(MODEL_NAME)
        self.model_config = AutoConfig.from_pretrained(MODEL_NAME)
        self.model_config.num_labels = 30
        self.model = AutoModel.from_pretrained(MODEL_NAME, config=self.model_config)
        self.hidden_dim = self.model_config.hidden_size
        self.lstm = nn.LSTM(
            input_size=self.hidden_dim, hidden_size=self.hidden_dim, num_layers=1, batch_first=True, bidirectional=True
        )
        self.fc = nn.Linear(self.hidden_dim * 2, self.model_config.num_labels)

    def forward(self, input_ids, attention_mask, labels, token_type_ids):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        # (batch, max_len, hidden_dim)

        hidden, (last_hidden, last_cell) = self.lstm(output)
        output = torch.cat((last_hidden[0], last_hidden[1]), dim=1)
        # hidden : (batch, max_len, hidden_dim * 2)
        # last_hidden : (2, batch, hidden_dim)
        # output : (batch, hidden_dim * 2)
        logits = self.fc(output)
        # logits : (batch, num_labels)

        return {"logits": logits}
