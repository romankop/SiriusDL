import numpy as np
import pandas as pd
import pickle
import torch
import gc

from torch.utils.data import Dataset

def collate_fn(struct, N=50):
    batch, labels = {}, []
    lengths = [min(N, len(f_t[0]['Card_cat'])) for f_t in struct]
    max_len = max(lengths)
    for idx, row in enumerate(struct):
        labels.append(row[1])
        for feature in row[0]:
            batch[feature] = batch.get(feature, torch.Tensor())
            new_user_feature = torch.cat((row[0][feature][-N:],
                                        torch.zeros(max(max_len - len(row[0][feature]), 0), dtype=torch.float32)))
            batch[feature] = torch.cat((batch[feature], new_user_feature.unsqueeze(0)))
    return batch, torch.tensor(labels, dtype=torch.float32), lengths


class CustomDataset(Dataset):
    def __init__(self, dataset_path):
        with open(dataset_path, 'rb') as f:
            data, self.nrof_emb_categories, self.unique_categories = pickle.load(f)

        self.embedding_columns = ['Card_cat', 'Use Chip_cat', 'Merchant State_cat',
                                   'MCC_cat', 'Bad CVV_cat', 'Bad Card Number_cat', 'Bad Expiration_cat',
                                   'Bad PIN_cat', 'Bad Zipcode_cat', 'Insufficient Balance_cat',
                                   'Technical Glitch_cat', 'Weekday_cat', 'Weekend_cat', 'Daytime_cat']
        self.nrof_emb_categories = {key + '_cat': val for key, val in self.nrof_emb_categories.items()}
        
        self.numeric_columns = ['Year', 'Month', 'Day', 'Amount', 'Hour', 'Minute',
                               'Weekday_cos', 'Weekday_sin', 'Weekday_tan',
                               'Hour_cos', 'Hour_sin', 'Hour_tan', 'Month_cos', 'Month_sin',
                               'Month_tan']

        self.columns = self.embedding_columns + self.numeric_columns
        
        self.User = data.User.reset_index(drop=True)      
        self.users = data.User.unique()

        self.X = data[self.columns].reset_index(drop=True)
        self.y = np.asarray(data.groupby('User')['IsFraud_target'].max().values, dtype=np.int8)
        
        del data
        gc.collect()

        return

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):

        user = self.users[idx]
        
        X_slice = self.X[self.User.eq(user)]

        data = {col: torch.tensor(X_slice[col].values, dtype=torch.float32) for i, col in enumerate(self.columns)}
        
        del X_slice
        gc.collect()

        return data, np.float32(self.y[idx])