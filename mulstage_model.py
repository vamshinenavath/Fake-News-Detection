import torch
import torchtext

from torch import nn
from torchtext.vocab import GloVe
import torch.optim as optim
import random

import torch
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder


class SVM_Classifier:
    def __init__(self, kernel = 'rbf', C= 1.0, **kwargs):
        self.kernel = kernel
        self.C = C
        self.model = SVC(kernel=self.kernel, C=self.C, **kwargs)
        self.label_encoder = LabelEncoder()

    def _to_numpy(self, X):
        if isinstance(X, torch.Tensor):
            return X.detach().cpu().numpy()
        return X
    

    def fit(self, X, y):
        X_np = self._to_numpy(X)
        if isinstance(y, torch.Tensor) and y.ndim > 1:
            y_np = y.argmax(dim=1).detach().cpu().numpy()
        else:
            y_np = self._to_numpy(y)
        self.model.fit(X_np, y_np)

    def predict(self, X):
        X_np = self._to_numpy(X)
        return self.model.predict(X_np)
    
    def score(self, X, y_true):
        y_true = self._to_numpy(y_true)
        y_pred = self.predict(X)
        return (y_pred == y_true).mean()
    

class CNN_BiLSTM(nn.Module):
    def __init__(self, vocab, vocab_size, embed_dim, hidden_dim, output_dim, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        # Load GloVe
        glove = GloVe(name='6B', dim=100)
        pretrained_embeddings = torch.zeros(vocab_size, embed_dim)
        for word, idx in vocab.get_stoi().items():
            if word in glove.stoi:
                pretrained_embeddings[idx] = glove[word]
        self.embedding.weight.data.copy_(pretrained_embeddings)
        self.embedding.weight.requires_grad = False  # freeze

        # CNN
        self.conv3 = nn.Conv1d(embed_dim, 100, kernel_size=3)
        self.conv5 = nn.Conv1d(embed_dim, 100, kernel_size=5)
        self.conv7 = nn.Conv1d(embed_dim, 100, kernel_size=7)

        # LSTM
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)

        # Combine and classify
        self.fc = nn.Linear(100 * 3 + hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x_embed = self.embedding(x)  # (B, T, E)
        x_cnn = x_embed.permute(0, 2, 1)  # (B, E, T)

        c3 = torch.relu(self.conv3(x_cnn)).max(dim=2)[0]
        c5 = torch.relu(self.conv5(x_cnn)).max(dim=2)[0]
        c7 = torch.relu(self.conv7(x_cnn)).max(dim=2)[0]

        cnn_out = torch.cat([c3, c5, c7], dim=1)

        lstm_out, _ = self.lstm(x_embed)
        lstm_out = lstm_out[:, -1, :]  # take last timestep

        combined = torch.cat([cnn_out, lstm_out], dim=1)
        out = self.fc(self.dropout(combined))
        return torch.sigmoid(out).squeeze(1)
    
    def extract_features(self, x):
        with torch.no_grad():
            x_embed = self.embedding(x)
            x_cnn = x_embed.permute(0, 2, 1)

            c3 = torch.relu(self.conv3(x_cnn)).max(dim=2)[0]
            c5 = torch.relu(self.conv5(x_cnn)).max(dim=2)[0]
            c7 = torch.relu(self.conv7(x_cnn)).max(dim=2)[0]

            cnn_out = torch.cat([c3, c5, c7], dim=1)

            lstm_out, _ = self.lstm(x_embed)
            lstm_out = lstm_out[:, -1, :]

            combined = torch.cat([cnn_out, lstm_out], dim=1)
            combined = self.dropout(combined)
            
        return combined
    

class SecondaryModel(nn.Module):
    def __init__(self, vocab, vocab_size, embed_dim, hidden_dim, output_dim, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        # Load GloVe
        glove = GloVe(name='6B', dim=100)
        pretrained_embeddings = torch.zeros(vocab_size, embed_dim)
        for word, idx in vocab.get_stoi().items():
            if word in glove.stoi:
                pretrained_embeddings[idx] = glove[word]
        self.embedding.weight.data.copy_(pretrained_embeddings)
        self.embedding.weight.requires_grad = False  # freeze

        # CNN
        self.conv3 = nn.Conv1d(embed_dim, 100, kernel_size=3)
        self.conv5 = nn.Conv1d(embed_dim, 100, kernel_size=5)
        self.conv7 = nn.Conv1d(embed_dim, 100, kernel_size=7)

        # LSTM
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)

        # Combine and classify
        self.fc1 = nn.Linear(100 * 3 + hidden_dim * 2, 512)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(256, 128)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.5)

        self.fc4 = nn.Linear(128, output_dim)

    def forward(self, x):
        x_embed = self.embedding(x)  # (B, T, E)
        x_cnn = x_embed.permute(0, 2, 1)  # (B, E, T)

        c3 = torch.relu(self.conv3(x_cnn)).max(dim=2)[0]
        c5 = torch.relu(self.conv5(x_cnn)).max(dim=2)[0]
        c7 = torch.relu(self.conv7(x_cnn)).max(dim=2)[0]

        cnn_out = torch.cat([c3, c5, c7], dim=1)

        lstm_out, _ = self.lstm(x_embed)
        lstm_out = lstm_out[:, -1, :]  # take last timestep

        combined = torch.cat([cnn_out, lstm_out], dim=1)
        x = self.fc1(combined)
        x = self.relu1(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.relu3(x)
        x = self.dropout3(x)

        x = self.fc4(x)
        return torch.sigmoid(x).squeeze(1)
    


# import torch
# import torch.nn as nn

# class FakeNewsClassifier(nn.Module):
#     def __init__(self, input_dim, num_classes):
#         super().__init__()
        
#         # First hidden layer
#         self.fc1 = nn.Linear(input_dim, 512)
#         self.relu1 = nn.ReLU()
#         self.dropout1 = nn.Dropout(0.5)  # Dropout to prevent overfitting
        
#         # Additional hidden layer
#         self.fc2 = nn.Linear(512, 256)  # The additional hidden layer
#         self.relu2 = nn.ReLU()
#         self.dropout2 = nn.Dropout(0.5)  # Dropout to prevent overfitting

#         self.fc3 = nn.Linear(256, 128)  # The additional hidden layer
#         ## tthis was previously nn.Linear(512, 128); changed because of an error
#         self.relu3 = nn.ReLU()
#         self.dropout3 = nn.Dropout(0.5)  # Dropout to prevent overfitting
        
#         # Output layer (final classification layer)
#         self.fc4 = nn.Linear(128, num_classes)
        
#     def forward(self, features):
#         x = self.fc1(features)
#         x = self.relu1(x)
#         x = self.dropout1(x)
        
#         # Pass through the additional hidden layer
#         x = self.fc2(x)
#         x = self.relu2(x)
#         x = self.dropout2(x)

#         # Final output layer
#         x = self.fc3(x)
#         x = self.relu3(x)
#         x = self.dropout3(x)

#         x = self.fc4(x)
#         return x
