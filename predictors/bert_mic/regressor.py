import os
from pathlib import Path

import torch.nn as nn
from transformers import BertModel


DEFAULT_BERT_MODEL_DIR = (
    Path(__file__).resolve().parent
    / "artifacts"
    / "backbone"
)

class REG(nn.Module):
    def __init__(self, bert_model_path=None):
        super(REG, self).__init__()
        model_path = bert_model_path or os.environ.get(
            "BERT_MODEL_PATH",
            os.environ.get("BERT_TOKENIZER_PATH", str(DEFAULT_BERT_MODEL_DIR)),
        )
        self.bert = BertModel.from_pretrained(
            model_path,
            output_attentions=True,
            local_files_only=True,
        )
        self.regressor= nn.Sequential(nn.LayerNorm(self.bert.config.hidden_size),
                                      #
                                      nn.Linear(self.bert.config.hidden_size, 512),
                                      nn.LeakyReLU(inplace=False),
                                      nn.Dropout(p=0.2),
                                      #
                                      nn.Linear(512, 128),
                                      nn.LeakyReLU(inplace=False),
                                      nn.Dropout(p=0.2),

                                      nn.Linear(128, 1))

    def forward(self, input_ids, attention_mask):
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return self.regressor(output.pooler_output), self.bert
