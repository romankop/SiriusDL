import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from sparsemax import Sparsemax

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
        
        layer_input_data = self.glu_layers[0](input_data)
        for i in range(1, self.nrof_glu):
            layer_input_data = torch.add(layer_input_data, self.glu_layers[i](layer_input_data))
            layer_input_data = torch.mul(layer_input_data, self.scale)
        
        return layer_input_data


class AttentiveTransformer(nn.Module):

    def __init__(self, input_size, output_size, relax_coef=1):
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

    def forward(self, input_data, prior):
        
        input_data = self.fc_bn(self.fc(input_data))
                        
        mask = self.sparse(torch.mul(input_data, prior))
        prior = prior * (self.relax_coef - mask)
                                        
        return mask, prior

class TabNet(nn.Module):
    def __init__(self, input_size, n_steps, slice_ratio, output_size_ratio, nrof_glu, nrof_out_classes):
        super(TabNet, self).__init__()
        
        self.n_steps = n_steps
        self.nrof_glu = nrof_glu
        self.input_size = input_size
        self.output_size = int(self.input_size * output_size_ratio // 2 * 2)
        self.nrof_out_classes = nrof_out_classes
        self.split_size = int(self.output_size // 2 * slice_ratio)
        self.dense_size = self.output_size // 2 - self.split_size
        
        self.relu = nn.ReLU()
        self.attention_trans = AttentiveTransformer(self.split_size, self.input_size) 
        self.common_trans = FeatureTransformer(nrof_glu, [self.input_size] + [self.output_size // 2] * (nrof_glu - 1), self.output_size)
        
        self.step_trans = []
        for i in range(n_steps + 1):
            self.step_trans.append(FeatureTransformer(nrof_glu, [self.output_size // 2] * nrof_glu, self.output_size))
        
        self.dense_layer = nn.Linear(self.dense_size, nrof_out_classes)
        


    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform(m.weight)
            # m.bias.data.fill_(0.001)

    def forward(self, input_data):
                
        input_data = self.common_trans(input_data)
        input_data = self.step_trans[0](input_data)
        
        data, _ = torch.split(input_data, [self.split_size, self.dense_size], dim=1)
                        
        output = 0
        prior = torch.ones((data.shape[0], self.input_size))
        mask = torch.zeros((self.n_steps, data.shape[0], self.input_size))
        
        for i in range(1, self.n_steps + 1):
            new_mask, prior = self.attention_trans(data, prior)
            
            mask[i - 1] = new_mask
            
            new_mask = self.common_trans(new_mask)
            new_mask = self.step_trans[i](new_mask)
            data, output_data = torch.split(new_mask, [self.split_size, self.dense_size], dim=1)
            output = output + self.relu(output_data)
            
            
        output = self.dense_layer(output).flatten()
            
#         output = torch.squeeze(output, 1)

        return output