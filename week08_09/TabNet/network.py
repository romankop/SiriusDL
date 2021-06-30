import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from sparsemax import Sparsemax

class DenseFeatureLayer(nn.Module):

    def __init__(self, input_size, nrof_cat, emb_dim,
                 emb_columns, numeric_columns):
        super(DenseFeatureLayer, self).__init__()

        self.emb_columns = emb_columns
        self.numeric_columns = numeric_columns

        self.embeddings = nn.ModuleDict({})
        for i, col in enumerate(self.emb_columns):
            self.embeddings[col] = torch.nn.Embedding(nrof_cat[col], emb_dim)

        self.feature_bn = torch.nn.BatchNorm1d(input_size)


    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform(m.weight)
            # m.bias.data.fill_(0.001)

    def forward(self, input_data):
        numeric_feats = torch.tensor(pd.DataFrame(input_data)[self.numeric_columns].values, dtype=torch.float32)

        emb_output = None
        for i, col in enumerate(self.emb_columns):
            if emb_output is None:
                emb_output = self.embeddings[col](torch.tensor(input_data[self.emb_columns[i]], dtype=torch.int64))
            else:
                emb_output = torch.cat(
                    [emb_output,
                     self.embeddings[col](torch.tensor(input_data[self.emb_columns[i]], dtype=torch.int64))],
                    dim=1)

        concat_input = torch.cat([numeric_feats, emb_output], dim=1)
        output = self.feature_bn(concat_input)

        return output

class GLULayer(nn.Module):

    def __init__(self, input_size, output_size):
        super(GLULayer, self).__init__()

        self.fc = torch.nn.Linear(input_size, output_size)
        self.fc_bn = torch.nn.BatchNorm1d(output_size)


    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform(m.weight)
            # m.bias.data.fill_(0.001)

    def forward(self, input_data):
        output = self.fc(input_data)
        output = self.fc_bn(output)
        output = torch.nn.functional.glu(output)

        return output

class FeatureTransformer(nn.Module):

    def __init__(self, nrof_glu, input_size, output_size):
        super(FeatureTransformer, self).__init__()

        self.scale = torch.sqrt(torch.FloatTensor([0.5]))

        self.nrof_glu = nrof_glu
        self.glu_layers = []

        for i in range(nrof_glu):
            self.glu_layers.append(GLULayer(input_size[i], output_size))

    def forward(self, input_data):
        layer_input_data = input_data
        for i in range(self.nrof_glu):
            layer_input_data = torch.add(layer_input_data, self.glu_layers[i](layer_input_data))
            layer_input_data = layer_input_data * self.scale
        
        return layer_input_data


class AttentiveTransformer(nn.Module):

    def __init__(self, input_size, output_size, relax_coef=2):
        super(AttentiveTransformer, self).__init__()
        
        self.fc = torch.nn.Linear(input_size, output_size)
        self.fc_bn = torch.nn.BatchNorm1d(output_size)
        
        self.prior = None
        self.relax_coef = relax_coef
        
        self.sparse = Sparsemax()


    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform(m.weight)
            # m.bias.data.fill_(0.001)

    def forward(self, input_data):
        
        input_data = self.fc_bn(self.fc(input_data))
        
        if self.prior is None:
            self.prior = torch.ones_like(input_data)
        
        mask = Sparsemax(input_data * self.prior)
        
        self.prior *= (self.relax_coef - mask)

        return mask

class TabNet(nn.Module):
    def __init__(self, n_steps, slice_ratio, output_size_ratio, nrof_glu, nrof_out_classes,
                 nrof_cat, emb_dim, emb_columns, numeric_columns):
        super(TabNet, self).__init__()
        
        self.n_steps = n_steps
        self.nrof_glu = nrof_glu
        self.input_size = (len(emb_columns) + len(numeric_columns)) * emb_dim
        self.output_size = int(self.input_size * output_size_ratio // 2 * 2)
        self.nrof_out_classes = nrof_out_classes
        
        self.dense_feature = DenseFeatureLayer(self.input_size, nrof_cat, emb_dim, emb_columns, numeric_columns)
        self.relu = nn.ReLU()
        self.attention_trans = AttentiveTransformer(self.split_size, self.input_size) 
        self.common_trans = FeatureTransformer(nrof_glu, [self.input_size] + [output_size // 2] * (nrof_glu - 1), output_size)
        self.step_trans = [FeatureTransformer(nrof_glu, [output_size // 2] * nrof_glu, output_size) for _ in range(n_steps + 1)]
        
        self.split_size = int(output_size // 2 * slice_ratio)
        self.dense_layer = nn.Linear(output_size // 2 - self.split_size, nrof_out_classes)


    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform(m.weight)
            # m.bias.data.fill_(0.001)

    def forward(self, input_data):
        
        input_data = self.dense_feature(input_data)
        
        input_data = self.common_trans(input_data)
        input_data = self.step_trans[0](input_data)[:, :, :self.slice_size]
        
        output = 0
        for i in range(1, self.n_steps + 1):
            mask = self.attention_trans(input_data)
            mask = self.common_trans(mask)
            mask = self.step_trans[i](mask)
            input_data = mask[:, :, :self.slice_size]
            output += self.relu(mask[:, :, self.slice_size:])
            
        output = self.dense_layer(output)
            
        output = torch.squeeze(output, 1)

        return output