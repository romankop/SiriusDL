import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import tqdm
from tabnet import TabNet

class DenseFeatureLayer(nn.Module):

    def __init__(self, input_size, nrof_cat, emb_dim,
                 emb_columns, numeric_columns):
        super(DenseFeatureLayer, self).__init__()

        self.emb_columns = emb_columns
        self.numeric_columns = numeric_columns

        self.embeddings = nn.ModuleDict({})
        for i, col in enumerate(self.emb_columns):
            self.embeddings[col] = torch.nn.Embedding(nrof_cat[col], emb_dim)

        self.feature_bn = PackedSequenceBatchNorm1d(num_features=input_size)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform(m.weight)
            # m.bias.data.fill_(0.001)

    def forward(self, input_data, lengths):
        
        output = torch.Tensor()
        
        
        numeric_feats = torch.stack([input_data[col] for col in self.numeric_columns], dim=2)

        emb_output = None
        for k, col in enumerate(self.emb_columns):
            if emb_output is None:
                emb_output = self.embeddings[col](input_data[self.emb_columns[k]].long())
            else:
                emb_output = torch.cat(
                    [emb_output,
                     self.embeddings[col](input_data[self.emb_columns[k]].long())],
                    dim=2)
                
        output = torch.cat([torch.flatten(numeric_feats, start_dim=2), torch.flatten(emb_output, start_dim=2)], dim=2)
                
        output = nn.utils.rnn.pack_padded_sequence(output, lengths,
                                        enforce_sorted=False, batch_first=True)
        output = self.feature_bn(output)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        return output

class PackedSequenceBatchNorm1d(torch.nn.BatchNorm1d):
    def forward(self, x):

        return torch.nn.utils.rnn.PackedSequence(
            data=super().forward(x.data),
            batch_sizes=x.batch_sizes,
            sorted_indices=x.sorted_indices
        )    


class LSTM_Net(nn.Module):
    def __init__(self, nrof_out_classes, hidden_size,
                 nrof_cat, emb_dim, emb_columns, numeric_columns):
        super(LSTM_Net, self).__init__()
        
        self.input_size = len(numeric_columns) + len(emb_columns) * emb_dim
        
        self.dense_feature = DenseFeatureLayer(self.input_size, nrof_cat, emb_dim, emb_columns, numeric_columns)
        
        self.rnn = torch.nn.LSTM(self.input_size, hidden_size, batch_first=True)
        
        self.dense_layer = nn.Linear(hidden_size, nrof_out_classes)


    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform(m.weight)
            # m.bias.data.fill_(0.001)

    def forward(self, input_data, lengths):
                
        input_data = self.dense_feature(input_data, lengths)
                        
        hidden, _ = self.rnn(input_data)[1]
                
        hidden = hidden.squeeze(0)
                
        output = self.dense_layer(hidden)

        return output.flatten()
    
    
class LSTM_Tab_Net(nn.Module):
    def __init__(self, nrof_out_classes, hidden_size,
                 nrof_cat, emb_dim, emb_columns, numeric_columns,
                 n_steps, slice_ratio, output_size_ratio, nrof_glu):
        super(LSTM_Tab_Net, self).__init__()
        
        self.input_size = len(numeric_columns) + len(emb_columns) * emb_dim
        
        self.dense_feature = DenseFeatureLayer(self.input_size, nrof_cat, emb_dim, emb_columns, numeric_columns)
        
        self.rnn = torch.nn.LSTM(self.input_size, hidden_size, batch_first=True)
        
        self.tab_layer = TabNet(hidden_size, n_steps, slice_ratio, output_size_ratio, nrof_glu, nrof_out_classes)


    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform(m.weight)
            # m.bias.data.fill_(0.001)

    def forward(self, input_data, lengths):
                
        input_data = self.dense_feature(input_data, lengths)
                        
        hidden, _ = self.rnn(input_data)[1]
                
        hidden = hidden.squeeze(0)
                
        output = self.tab_layer(hidden)

        return output.flatten()
    
class Tab_LSTM_Net(nn.Module):
    def __init__(self, nrof_out_classes, hidden_size, tab_hidden_size,
                 nrof_cat, emb_dim, emb_columns, numeric_columns,
                 n_steps, slice_ratio, output_size_ratio, nrof_glu):
        super(Tab_LSTM_Net, self).__init__()
        
        self.input_size = len(numeric_columns) + len(emb_columns) * emb_dim
        
        self.tab_hidden_size = tab_hidden_size
        
        self.dense_feature = DenseFeatureLayer(self.input_size, nrof_cat, emb_dim, emb_columns, numeric_columns)
        
        self.tab_layer = TabNet(self.input_size, n_steps, slice_ratio, output_size_ratio, nrof_glu, tab_hidden_size)
        
        self.rnn = torch.nn.LSTM(tab_hidden_size, hidden_size, batch_first=True)
        
        self.dense_layer = nn.Linear(hidden_size, nrof_out_classes)
        

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform(m.weight)
            # m.bias.data.fill_(0.001)

    def forward(self, input_data, lengths):
                
        input_data = self.dense_feature(input_data, lengths)
        
        input_data_shape = input_data.shape
        
        input_data = input_data.view((input_data_shape[0] * input_data_shape[1], input_data_shape[2]))
        
        input_data = self.tab_layer(input_data).view((input_data_shape[0], input_data_shape[1], self.tab_hidden_size))
                        
        hidden, _ = self.rnn(input_data)[1]
                
        hidden = hidden.squeeze(0)
                
        output = self.dense_layer(hidden)

        return output.flatten()
    

class Conv_Net(nn.Module):
    def __init__(self, nrof_out_classes,
                 nrof_cat, emb_dim, emb_columns, numeric_columns):
        super(Conv_Net, self).__init__()
        
        self.input_size = len(numeric_columns) + len(emb_columns) * emb_dim
        
        self.dense_feature = DenseFeatureLayer(self.input_size, nrof_cat, emb_dim, emb_columns, numeric_columns)
        
        self.conv1 = nn.Conv2d(1, 3, 6)
        self.conv2 = nn.Conv2d(3, 6, 6)
        self.conv3 = nn.Conv2d(6, 9, 6)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.relu = nn.ReLU()
        
        self.dense_layer = nn.Linear(54, nrof_out_classes)


    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform(m.weight)
            # m.bias.data.fill_(0.001)

    def forward(self, input_data, lengths):
                
        output = self.dense_feature(input_data, lengths).unsqueeze(1)
                        
        output = self.pool(self.relu(self.conv1(output)))
        
        output = self.pool(self.relu(self.conv2(output)))
        
        output = torch.flatten(self.pool(self.relu(self.conv3(output))), 1)
                        
        output = self.dense_layer(output)

        return output.flatten()

    
class Tab_Conv_Net(nn.Module):
    def __init__(self, nrof_out_classes, n_steps, slice_ratio, output_size_ratio, nrof_glu, tab_hidden_size,
                 nrof_cat, emb_dim, emb_columns, numeric_columns):
        super(Tab_Conv_Net, self).__init__()
        
        self.input_size = len(numeric_columns) + len(emb_columns) * emb_dim
        self.tab_hidden_size = tab_hidden_size
        
        self.dense_feature = DenseFeatureLayer(self.input_size, nrof_cat, emb_dim, emb_columns, numeric_columns)
        
        self.tab_layer = TabNet(self.input_size, n_steps, slice_ratio, output_size_ratio, nrof_glu, tab_hidden_size)
        
        self.conv1 = nn.Conv2d(1, 3, 5)
        self.conv2 = nn.Conv2d(3, 6, 5)
        self.conv3 = nn.Conv2d(6, 9, 5)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.relu = nn.ReLU()
        
        self.dense_layer = nn.Linear(18, nrof_out_classes)


    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform(m.weight)
            # m.bias.data.fill_(0.001)

    def forward(self, input_data, lengths):
                
        input_data = self.dense_feature(input_data, lengths)
        
        input_data_shape = input_data.shape
        
        input_data = input_data.view((input_data_shape[0] * input_data_shape[1], input_data_shape[2]))
        
        output = self.tab_layer(input_data).view((input_data_shape[0], input_data_shape[1], self.tab_hidden_size))

        output = output.unsqueeze(1)
                        
        output = self.pool(self.relu(self.conv1(output)))
        
        output = self.pool(self.relu(self.conv2(output)))
        
        output = torch.flatten(self.pool(self.relu(self.conv3(output))), 1)
                        
        output = self.dense_layer(output)

        return output.flatten()
    
    
class Conv_Tab_Net(nn.Module):
    def __init__(self, nrof_out_classes, n_steps, slice_ratio, output_size_ratio, nrof_glu,
                 nrof_cat, emb_dim, emb_columns, numeric_columns):
        super(Conv_Tab_Net, self).__init__()
        
        self.input_size = len(numeric_columns) + len(emb_columns) * emb_dim
        
        self.dense_feature = DenseFeatureLayer(self.input_size, nrof_cat, emb_dim, emb_columns, numeric_columns)
                
        self.conv1 = nn.Conv2d(1, 3, 6)
        self.conv2 = nn.Conv2d(3, 6, 6)
        self.conv3 = nn.Conv2d(6, 9, 6)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.relu = nn.ReLU()
        
        self.tab_layer = TabNet(54, n_steps, slice_ratio, output_size_ratio, nrof_glu, nrof_out_classes)


    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform(m.weight)
            # m.bias.data.fill_(0.001)

    def forward(self, input_data, lengths):
                
        output = self.dense_feature(input_data, lengths).unsqueeze(1)
                        
        output = self.pool(self.relu(self.conv1(output)))
        
        output = self.pool(self.relu(self.conv2(output)))
        
        output = torch.flatten(self.pool(self.relu(self.conv3(output))), 1)
                        
        output = self.tab_layer(output)

        return output.flatten()

