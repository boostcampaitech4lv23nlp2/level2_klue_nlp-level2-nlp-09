import torch
import torch.nn as nn
from transformers import AutoModel, PreTrainedModel, RobertaPreTrainedModel


class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.1, use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
        return self.linear(x)


class RBert(RobertaPreTrainedModel):
    def __init__(self, model_config, tokenizer, model_name_or_path):
        super(RBert, self).__init__(config=model_config)
        self.hidden_dim = model_config.hidden_size
        self.num_labels = model_config.num_labels
        self.tokenizer = tokenizer
        self.config = model_config
        self.model = AutoModel.from_pretrained(model_name_or_path, config=model_config)
        self.model.resize_token_embeddings(len(tokenizer))
        self.cls_fc = FCLayer(model_config.hidden_size, model_config.hidden_size, model_config.hidden_dropout_prob)
        self.entity_fc = FCLayer(model_config.hidden_size, model_config.hidden_size, model_config.hidden_dropout_prob)
        self.label_fc = FCLayer(
            model_config.hidden_size * 3,
            model_config.num_labels,
            model_config.hidden_dropout_prob,
            use_activation=False,
        )

    def forward(self, input_ids, attention_mask, subject_mask, object_mask, labels):
        # input_ids에서 subject 시작과 끝, object 시작과 끝 인덱스 구하기
        # input mask 구하기
        # average pooling 넘기기

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        pooled_output = outputs[1]

        subject_entity_hidden = self.entity_average(sequence_output, subject_mask)
        object_entity_hidden = self.entity_average(sequence_output, object_mask)

        cls_output = self.cls_fc(pooled_output)
        subject_entity_output = self.entity_fc(subject_entity_hidden)
        object_entity_output = self.entity_fc(object_entity_hidden)

        concat_hidden = torch.cat([cls_output, subject_entity_output, object_entity_output], dim=-1)
        logits = self.label_fc(concat_hidden)
        return {"logits": logits}

    def entity_average(self, hidden_output, entity_mask):
        # hidden_output: [batch_size, hidden_output_length, hidden_dim]
        entity_length = (entity_mask != 0).sum(dim=1).unsqueeze(1)
        # entity_mask: [batch_size, max_seq_len]
        # entity_length: [batch_size, 1]
        entity_mask_unsqueeze = entity_mask.unsqueeze(1)
        # entity_mask: [batch_size, 1, max_seq_len]

        sum_vector = torch.bmm(entity_mask_unsqueeze.float(), hidden_output).squeeze(1)
        avg_vector = sum_vector.float() / entity_length.float()
        return avg_vector
