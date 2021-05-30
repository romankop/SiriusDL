import torch

import torch.optim as optim
import torch.utils.data as data_utils

from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning.metrics import Accuracy

from network import DeepFMNet
from data_loader import CustomDataset

EPOCHS = 20
EMBEDDING_SIZE = 5
BATCH_SIZE = 512
NROF_LAYERS = 3
NROF_NEURONS = 50
DEEP_OUTPUT_SIZE = 50
NROF_OUT_CLASSES = 1
LEARNING_RATE = 3e-4
TRAIN_PATH = 'train_adult.pickle'
VALID_PATH = 'valid_adult.pickle'

class DeepFM:
    def __init__(self):
        self.train_dataset = CustomDataset(TRAIN_PATH)
        self.train_loader = data_utils.DataLoader(dataset=self.train_dataset,
                                                  batch_size=BATCH_SIZE, shuffle=True)
        self.valid_dataset = CustomDataset(VALID_PATH)
        self.valid_loader = data_utils.DataLoader(dataset=self.valid_dataset,
                                                  batch_size=BATCH_SIZE, shuffle=True)

        self.build_model()

        self.train_writer = SummaryWriter('./logs/train')
        self.valid_writer = SummaryWriter('./logs/valid')
        
        self.log_params()

        return

    def build_model(self):
        self.network = DeepFMNet(nrof_cat=self.train_dataset.nrof_emb_categories, emb_dim=EMBEDDING_SIZE,
                                 emb_columns=self.train_dataset.embedding_columns,
                                 numeric_columns=self.train_dataset.numeric_columns,
                                 nrof_layers=NROF_LAYERS, nrof_neurons=NROF_NEURONS,
                                 output_size=DEEP_OUTPUT_SIZE,
                                 nrof_out_classes=NROF_OUT_CLASSES)

        self.loss = torch.nn.BCEWithLogitsLoss()
        self.accuracy = Accuracy()
        self.optimizer = optim.Adam(self.network.parameters(), lr=LEARNING_RATE)

        return

    def log_params(self):
        
        self.train_writer.add_text('LEARNING_RATE', str(EPOCHS))
        self.train_writer.add_text('INPUT_SIZE', str(EMBEDDING_SIZE))
        self.train_writer.add_text('HIDDEN_SIZE', str(BATCH_SIZE))
        self.train_writer.add_text('OUTPUT_SIZE', str(NROF_LAYERS))
        self.train_writer.add_text('EMBEDDING_SIZE', str(DEEP_OUTPUT_SIZE))
        self.train_writer.add_text('BATCH_SIZE', str(NROF_OUT_CLASSES))
        self.train_writer.add_text('EPOCHS', str(LEARNING_RATE))
        
        self.valid_writer.add_text('LEARNING_RATE', str(EPOCHS))
        self.valid_writer.add_text('INPUT_SIZE', str(EMBEDDING_SIZE))
        self.valid_writer.add_text('HIDDEN_SIZE', str(BATCH_SIZE))
        self.valid_writer.add_text('OUTPUT_SIZE', str(NROF_LAYERS))
        self.valid_writer.add_text('EMBEDDING_SIZE', str(DEEP_OUTPUT_SIZE))
        self.valid_writer.add_text('BATCH_SIZE', str(NROF_OUT_CLASSES))
        self.valid_writer.add_text('EPOCHS', str(LEARNING_RATE))
        
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

        for epoch in range(EPOCHS):
            self.network.train()

            for features, label in self.train_loader:
                # Reset gradients
                self.optimizer.zero_grad()

                output = self.network(features)
                # Calculate error and backpropagate
                loss = self.loss(output, label)

                output = torch.sigmoid(output)
                
                loss.backward()
                label = label.long()
                acc = self.accuracy(output, label).item()

                # Update weights with gradients
                self.optimizer.step()

                self.train_writer.add_scalar('CrossEntropyLoss', loss, self.step)
                self.train_writer.add_scalar('Accuracy', acc, self.step)

                self.step += 1

                if self.step % 20 == 0:
                    print('EPOCH %d STEP %d : train_loss: %f train_acc: %f' %
                          (epoch, self.step, loss.item(), acc))

            # self.train_writer.add_histogram('hidden_layer', self.network.linear1.weight.data, self.step)

            # Run validation
            self.network.eval()
            
            correct, total = 0, 0

            for features, label in self.valid_loader:

                output = self.network(features)

                output = (torch.sigmoid(output) > 0.5).long()
                
                label = label.long()
                
                total += output.shape[0]
                correct += (output == label).sum().item()
                
            acс = correct / total
            self.valid_writer.add_scalar('Accuracy', acc, epoch)

            print('EPOCH %d : valid_acc: %f' % (epoch, acc))

        return
