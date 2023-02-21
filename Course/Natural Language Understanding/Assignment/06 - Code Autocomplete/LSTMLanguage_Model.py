import torch.nn as nn
import torch

class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, num_layers, dropout_rate):
                
        super().__init__()
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size,emb_dim)
        self.lstm = nn.LSTM(emb_dim, hid_dim, num_layers=num_layers, 
                                        dropout = dropout_rate, batch_first = True)
        self.dropout = nn.Dropout(dropout_rate)
        #when you do LM, you look forward, so it does not make sense to do bidirectional
        self.fc = nn.Linear(hid_dim,vocab_size)

    def init_hidden(self, batch_size, device):
        #this function gonna be run in the beginning of the epoch
        hidden = torch.zeros(self.num_layers, batch_size, self.hid_dim).to(device)
        cell   = torch.zeros(self.num_layers, batch_size, self.hid_dim).to(device)

        return hidden, cell #return as tuple

    def detach_hidden(self, hidden):
        hidden, cell = hidden
        hidden = hidden.detach() #removing this hidden from gradients graph
        cell =  cell.detach() #removing this hidden from gradients graph
        return hidden, cell

    def forward(self, src, hidden):
        #src: [batch_size, seq_len]

        #embed 
        embedded = self.embedding(src)
        #embed : [batch_size, seq_len, emb_dim]

        #send this to the lstm
        #we want to put hidden here... because we want to reset hidden .....
        output, hidden = self.lstm(embedded, hidden)
        #output : [batch_size, seq_len, hid_dim] ==> all hidden states
        #hidden : [batch_size, seq_len, hid_dim] ==> last hidden states from each layer

        output = self.dropout(output)
        prediction = self.fc(output)
        #prediction: [batch size, seq_len, vocab_size]
        return prediction, hidden