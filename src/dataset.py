from dataPreprocessing import prepare_train_features
from config import config
import pandas as pd
import torch

class ChaiiDataset:
    def __init__(self, constant_func, dataframe):
        self.constant_func = constant_func
        self.dataframe = dataframe
    
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        df= self.dataframe.iloc[item, :]

        data = prepare_train_features(args = self.constant_func,
                                      example = df)

        return {
            'context': data["context"],
            'question': data["question"],
            'answer': data["answer"],
            'padding_len': torch.tensor(data["padding_len"], dtype=torch.long),
            'ids': torch.tensor(data["ids"], dtype=torch.long),
            'mask': torch.tensor(data["mask"], dtype=torch.long),
            'token_type_ids': torch.tensor(data["token_type_ids"], dtype=torch.long),
            'targets_start': torch.tensor(data["targets_start"], dtype=torch.long),
            'targets_end': torch.tensor(data["targets_end"], dtype=torch.long),
            'offsets': torch.tensor(data["offsets"], dtype=torch.long)
        }


if __name__ == "__main__":
    df_train = pd.read_csv(config.TRAINING_FILE)

    train_dataset = ChaiiDataset(
            constant_func=config(),
            dataframe=df_train
        )

    idx = 111

    # print(train_dataset[idx]['context'],'\n')
    # print(train_dataset[idx]['question'],'\n')
    # print(train_dataset[idx]['answer'],'\n')
    # print(train_dataset[idx]['padding_len'],'\n')
    # print(train_dataset[idx]['ids'],'\n')
    # print(train_dataset[idx]['mask'],'\n')
    print(train_dataset[idx]['token_type_ids'],'\n')
    # print(train_dataset[idx]['targets_start'],'\n')
    # print(train_dataset[idx]['targets_end'],'\n')
    # print(train_dataset[idx]['offsets'])