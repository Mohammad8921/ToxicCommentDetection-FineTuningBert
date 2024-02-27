# pip install transformers==3.0.0
# pip install emoji

import gc
import torch
import torch.nn as nn
from transformers import BertModel


class BERT_CNN(nn.Module):

    def __init__(self):
        super(BERT_CNN, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.conv = nn.Conv2d(in_channels=13, out_channels=13,
                              kernel_size=(3, 768), padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(442, 3)
        self.flat = nn.Flatten()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
        _, _, all_layers = self.bert(
            sent_id, attention_mask=mask, output_hidden_states=True)
        # all_layers  = [13, 32, 64, 768]
        x = torch.transpose(
            torch.cat(tuple([t.unsqueeze(0) for t in all_layers]), 0), 0, 1)
        del all_layers
        gc.collect()
        torch.cuda.empty_cache()
        x = self.pool(self.dropout(self.relu(self.conv(self.dropout(x)))))
        x = self.fc(self.dropout(self.flat(self.dropout(x))))
        return self.softmax(x)


class BERT_CNN_1D(nn.Module):

    def __init__(self):
        super(BERT_CNN_1D, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.conv = nn.Conv1d(
            in_channels=768, out_channels=64, kernel_size=3, padding="same")
        self.pool = nn.MaxPool1d(kernel_size=3, stride=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(2176, 2)  # 64*34 = 2176
        self.flat = nn.Flatten()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
        last_hidden_state, _ = self.bert(sent_id, attention_mask=mask)
        x = last_hidden_state

        gc.collect()
        torch.cuda.empty_cache()

        x = self.dropout(x)
        x = torch.transpose(x, 1, 2)
        x = self.pool(self.dropout(self.relu(self.conv(x))))
        x = self.fc(self.dropout(self.flat(self.dropout(x))))
        return self.softmax(x)


class BERT_LSTM(nn.Module):

    def __init__(self):
        super(BERT_LSTM, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.lstm = nn.LSTM(768, 768, batch_first=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.3)  # For fully connected layer
        self.fc = nn.Linear(27648, 2)  # 36*768 = 27648
        self.flat = nn.Flatten()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
        last_hidden_state, _ = self.bert(sent_id, attention_mask=mask)
        x = last_hidden_state

        gc.collect()
        torch.cuda.empty_cache()

        x, _ = self.lstm(self.dropout(x))
        x = self.fc(self.dropout2(self.flat(self.dropout(x))))
        return self.softmax(x)


class BERT_LSTM_CNN(nn.Module):

    def __init__(self):
        super(BERT_LSTM_CNN, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.lstm = nn.LSTM(768, 768, batch_first=True)
        self.conv = nn.Conv1d(
            in_channels=768, out_channels=64, kernel_size=3, padding="same")
        self.pool = nn.MaxPool1d(kernel_size=3, stride=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(2176, 2)  # 64*34 = 2176
        self.flat = nn.Flatten()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
        last_hidden_state, _ = self.bert(sent_id, attention_mask=mask)
        x = last_hidden_state
        gc.collect()
        torch.cuda.empty_cache()

        x = self.lstm(self.dropout(x))
        x = x[0]  # output of lstm is a tuple.
        x = torch.transpose(x, 1, 2)
        x = self.pool(self.dropout(self.relu(self.conv(x))))
        x = self.fc(self.dropout(self.flat(self.dropout(x))))
        return self.softmax(x)


class BERT_CNN_LSTM(nn.Module):

    def __init__(self):
        super(BERT_CNN_LSTM, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.lstm = nn.LSTM(64, 64, batch_first=True)
        self.conv = nn.Conv1d(
            in_channels=768, out_channels=64, kernel_size=3, padding="same")
        self.pool = nn.MaxPool1d(kernel_size=3, stride=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(2176, 2)  # 64*34 = 2176
        self.flat = nn.Flatten()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
        last_hidden_state, _ = self.bert(sent_id, attention_mask=mask)
        x = last_hidden_state

        gc.collect()
        torch.cuda.empty_cache()

        x = torch.transpose(x, 1, 2)
        x = self.pool(self.dropout(self.relu(self.conv(self.dropout(x)))))
        x = torch.transpose(x, 1, 2)
        x, _ = self.lstm(self.dropout(x))
        x = self.fc(self.dropout(self.flat(self.dropout(x))))
        return self.softmax(x)


class BERT_2CNN_1D(nn.Module):

    def __init__(self):
        super(BERT_2CNN_1D, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.conv = nn.Conv1d(
            in_channels=768, out_channels=64, kernel_size=3, padding="same")
        self.conv2 = nn.Conv1d(
            in_channels=64, out_channels=32, kernel_size=3, padding="same")
        self.pool = nn.MaxPool1d(kernel_size=3, stride=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(1024, 2)  # 32*(36-2-2) = 1024
        self.flat = nn.Flatten()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
        last_hidden_state, _ = self.bert(sent_id, attention_mask=mask)
        x = last_hidden_state

        gc.collect()
        torch.cuda.empty_cache()

        x = torch.transpose(x, 1, 2)
        x = self.pool(self.dropout(self.relu(self.conv(self.dropout(x)))))
        x = self.pool(self.dropout(self.relu(self.conv2(self.dropout(x)))))
        x = self.fc(self.dropout(self.flat(self.dropout(x))))
        return self.softmax(x)
