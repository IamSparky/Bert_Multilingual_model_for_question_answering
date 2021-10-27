import sys
sys.path.append("../../G2Wave_Classification/src/pip_installs_required/cpython-master/")

import dataset
import engine
import torch
import models
import pandas as pd
import torch.nn as nn
import numpy as np
import transformers

from config import config
from models import Bert_MultiLingual_Model
from sklearn import model_selection
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from Lib import copy
from create_folds import make_folds

def run():
    new_train = make_folds(config.TRAINING_FILE)
    
    model_config = transformers.BertConfig.from_pretrained(config.BERT_PATH)
    model_config.output_hidden_states = True
    model = Bert_MultiLingual_Model(conf=model_config)
    model = model.to(config.DEVICE)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    num_train_steps = int(223 / config.TRAIN_BATCH_SIZE * config.EPOCHS) # 223 is the size of each df_train with kfolds value from 0 to 4
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0, 
        num_training_steps=num_train_steps
    )

    a_string = "*" * 20
    for i in range(5):
        print(a_string, " FOLD NUMBER ", i, a_string)
        df_train = new_train[new_train.kfold != i].reset_index(drop=True)
        df_valid = new_train[new_train.kfold == i].reset_index(drop=True)

        train_dataset = dataset.ChaiiDataset(
            constant_func=config(),
            dataframe=df_train
        )

        train_data_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=4
        )

        valid_dataset = dataset.ChaiiDataset(
            constant_func=config(),
            dataframe=df_valid
        )

        valid_data_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=config.VALID_BATCH_SIZE, num_workers=1
        )

        all_jaccards = []
        for epoch in range(config.EPOCHS):
            print(f"Epoch --> {epoch+1} / {config.EPOCHS}")
            print(f"-------------------------------")
            engine.train_fn(train_data_loader, model, optimizer, config.DEVICE, scheduler)

            jaccard = engine.eval_fn(valid_data_loader, model, device)
            print(f"Jaccard Score = {jaccard:.2f}")
            all_jaccards.append(jaccard)
        print('\n')
        
        if i < 1:
            best_jaccard = max(all_jaccards)
            best_model = copy.deepcopy(model)
            all_jaccards = []
        else:
            if max(all_jaccards) > best_jaccard:
                best_jaccard = max(all_jaccards)
                best_model = copy.deepcopy(model)
                all_jaccards = []
                
    torch.save(best_model,'./best_question_answering_model.bin')
    print()
    print(f"The best jaccard metric score that we got across all the folds is : {best_jaccard}")

if __name__ == "__main__":
    run()