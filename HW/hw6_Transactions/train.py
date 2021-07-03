import torch

import torch.optim as optim
import torch.utils.data as data_utils

from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning.metrics import Accuracy

from data_loader import CustomDataset, collate_fn
import gc
import time

EPOCHS = 20
EMBEDDING_SIZE = 5
BATCH_SIZE = 512
NROF_OUT_CLASSES = 1
LEARNING_RATE = 3e-4
N_STEPS = 2
SLICE_RATIO = 0.5
OUTPUT_RATIO = 1
NROF_GLU = 2
TRAIN_PATH = 'train.pickle'
VALID_PATH = 'valid.pickle'
TEST_PATH = 'test.pickle'

def roc_auc_compute_fn(y_preds: torch.Tensor, y_targets: torch.Tensor) -> float:
    from sklearn.metrics import roc_auc_score

    y_true = y_targets.cpu().detach().numpy()
    y_pred = y_preds.cpu().detach().numpy()
    return roc_auc_score(y_true, y_pred)    
    

class Model_Pipe:
    def __init__(self, model, params, hyperparams, model_name):
        
        self.start = time.time()
        
        self.params = params
        self.hyperparams = hyperparams
        
        self.train_dataset = CustomDataset(TRAIN_PATH)
        self.train_loader = data_utils.DataLoader(dataset=self.train_dataset, collate_fn=collate_fn,
                                                  batch_size=self.hyperparams['batch_size'], shuffle=True)
        self.valid_dataset = CustomDataset(VALID_PATH)
        self.valid_loader = data_utils.DataLoader(dataset=self.valid_dataset, collate_fn=collate_fn,
                                                  batch_size=self.hyperparams['batch_size'], shuffle=True)
                
        self.network = model(nrof_cat=self.train_dataset.nrof_emb_categories, 
                             emb_columns=self.train_dataset.embedding_columns,
                             numeric_columns=self.train_dataset.numeric_columns,
                             **params)

        self.build_model()

        self.train_writer = SummaryWriter('./logs/' + model_name + '/train')
        self.valid_writer = SummaryWriter('./logs/' + model_name + '/valid')
                
        self.log_params(params)
        self.log_params(hyperparams)
        

        return

    def build_model(self):
        
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.hyperparams['lr'])

        return

    def log_params(self, params):
        
        for key in params:
            
            self.train_writer.add_text(key, str(params[key]))
            self.valid_writer.add_text(key, str(params[key]))
        
        return

    def load_model(self, restore_path=''):
        if restore_path == '':
            self.step = 0
        else:
            pass

        return

    def run_train(self):
        print('Run train ...')

        self.load_model()
        
        for epoch in range(self.hyperparams['epochs']):
            
            self.network.train()

            for features, label, lengths in self.train_loader:
                
                with torch.autograd.set_detect_anomaly(True):
                    
                    # Reset gradients
                    self.optimizer.zero_grad()
                    
                    output = self.network(features, lengths)
                    # Calculate error and backpropagate
                    loss = self.loss(output, label)

                    output = torch.sigmoid(output)

                    loss.backward(retain_graph=True)
                    label = label.long()
                    roc_auc = roc_auc_compute_fn(output, label)

                    # Update weights with gradients
                    self.optimizer.step()

                self.train_writer.add_scalar('CrossEntropyLoss', loss, self.step)
                self.train_writer.add_scalar('ROC-AUC', roc_auc, self.step)
                
                self.step += 1

                if self.step % 5 == 0:
                    print('EPOCH %d STEP %d : train_loss: %f train_roc_auc: %f' %
                          (epoch, self.step, loss.item(), roc_auc))

            # self.train_writer.add_histogram('hidden_layer', self.network.linear1.weight.data, self.step)
            
            # Run validation
            self.network.eval()
            
            y_true, y_pred = torch.Tensor(), torch.Tensor()

            for features, label, lengths in self.valid_loader:

                output = self.network(features, lengths)

                output = torch.sigmoid(output)
                
                label = label.long()
                
                y_true = torch.cat([y_true, label.flatten()])
                y_pred = torch.cat([y_pred, output.flatten()])
                                
            roc_auc = roc_auc_compute_fn(y_pred, y_true)
            self.valid_writer.add_scalar('ROC_AUC', roc_auc, epoch)

            print('EPOCH %d : valid_roc_auc: %f' % (epoch, roc_auc))
                        
        test_dataset = CustomDataset(TEST_PATH)
        test_loader = data_utils.DataLoader(dataset=self.valid_dataset, collate_fn=collate_fn,
                                                  batch_size=self.hyperparams['batch_size'], shuffle=True)
            
        self.network.eval()
            
        y_true, y_pred = torch.Tensor(), torch.Tensor()

        for features, label, lengths in test_loader:

            output = self.network(features, lengths)

            output = torch.sigmoid(output)

            label = label.long()

            y_true = torch.cat([y_true, label.flatten()])
            y_pred = torch.cat([y_pred, output.flatten()])

        roc_auc = roc_auc_compute_fn(y_pred, y_true)
        print('Test ROC-AUC: %f' % (roc_auc))
        
        del test_dataset, test_loader
        gc.collect()

        return
