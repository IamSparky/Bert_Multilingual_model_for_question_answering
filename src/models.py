from config import config
import transformers
import torch
import torch.nn as nn

class Bert_MultiLingual_Model(transformers.BertPreTrainedModel):
    def __init__(self, conf):
        super(Bert_MultiLingual_Model, self).__init__(conf)
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH, config = conf)
        self.drop_out = nn.Dropout(0.1)
        self.l0 = nn.Linear(768 * 2, 2)
        torch.nn.init.normal_(self.l0.weight, std=0.02)
    
    def forward(self, ids, mask, token_type_ids):
        _, _, out = self.bert(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )

        out = torch.cat((out[-1], out[-2]), dim=-1)
        out = self.drop_out(out)
        logits = self.l0(out)

        start_logits, end_logits = logits.split(1, dim=-1)

        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits

if __name__ == "__main__":

    model_config = transformers.BertConfig.from_pretrained(config.BERT_PATH)
    model_config.output_hidden_states = True
    model = Bert_MultiLingual_Model(conf=model_config)
    model = model.to(config.DEVICE)
    print(model)