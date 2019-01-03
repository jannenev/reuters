
import torch.nn as nn
import torch.nn.functional as F
import torch as torch


# Conv1d - Maxpool or Avgpool
class CNNPool(nn.Module):
    def __init__(self, vocab_size, embedding_dim, nr_filters, filter_size, output_dim, pool='max'):
        super().__init__()
        
        # nn.Embedding(num_embeddings, : size of the dictionary of embeddings
        self.embedding = nn.Embedding(vocab_size, embedding_dim=embedding_dim)
                
        self.conv1 = nn.Conv1d(in_channels=50, out_channels=nr_filters, kernel_size=filter_size)
        self.fc = nn.Linear(in_features= nr_filters, out_features= output_dim)
        
        self.dropout = nn.Dropout(0.5) 
        
        self.nr_filters = nr_filters
        self.printmore= torch.zeros(1)
        self.pool=pool
        
    
    def forward(self,x):
        
        self.printmore=1
        
        if self.printmore:
            print('Begin -------')
            print(f'input x: {x.shape}')
        # x.shape [574, 64], [textlen for current batch, batch_size], textlen varies by batch
  

        # Permute to put batch first, to (N,W)
        x = x.permute(1, 0)
        if self.printmore:
            print(f'After permute: {x.shape}') # [64, 574], [batch, textlen]

        # EMBEDDING
        # nn.Embedding. input (N, W) mini-batch, Words per... out:(N, W, emb_dim)      
        x = self.embedding(x)
        if self.printmore:
            print(f'Embedded: {x.shape}') # [64, 574, 50], [batch, textlen, emb_dim]
        
        # CONVOLUTION
        #Weight we get is size [128, 1, 5], [nr_channels, in_chan, filter_size]
        
        #Conv input [batch, in_channels=, textlen, emb_dim]
        # chance dim 1 and 2 with each other, textlen and emb_dim
        x = x.transpose(1,2)
        if self.printmore:
            print(f'Textlen, emb switched with transpose: {x.shape}')     
        
        x = self.conv1(x)   
        x = F.relu(x)
        
        if self.printmore:
            print(f'After conv1d: {x.shape}')   
        #print(x.shape) # [64, 128, 1091] [batch, filters, nr_steps filter took]
        
        
        assert (this.pool=='max' or this.pool=='avg')
        
        if this.pool=='max':
            x = F.max_pool1d(x, x.size(2))          #print(x.shape) # [64, 128, 1]
        
        else : 
            x = F.avg_pool1d(x, x.size(2))          #print(x.shape) # [64, 128, 1]
        
        
        x = x.squeeze(2) # squeezes the last 1 dim away
        if self.printmore:
            print(f'After pool1d+squeeze(2): {x.shape}') #   [batch, nr_filters] [64, 128]  
       
        x = self.dropout(x)        
        x = self.fc(x)

        x = F.sigmoid(x)
        return(x)



# 1D conv MAX + AVG Pool concatenated
class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, nr_filters, filter_size, output_dim, dropout):
        super().__init__()
        
        # nn.Embedding(num_embeddings, : size of the dictionary of embeddings
        self.embedding = nn.Embedding(vocab_size, embedding_dim=embedding_dim)
        
        # 1 layer conv
        self.conv1 = nn.Conv1d(in_channels=50, out_channels=nr_filters, kernel_size=filter_size)      
        #self.conv1_bn = nn.BatchNorm2d(nr_filters)
        self.fc = nn.Linear(in_features= nr_filters*2, out_features= output_dim)
        
        # 2 layers conv
        #self.conv1 = nn.Conv1d(in_channels=50, out_channels=nr_filters, kernel_size=filter_size)
        #self.conv2 = nn.Conv1d(in_channels= nr_filters, out_channels=nr_filters*2, kernel_size=filter_size )
        #self.fc = nn.Linear(in_features= nr_filters*2, out_features= output_dim)
        
        self.dropout = nn.Dropout(dropout) 
        
        self.nr_filters = nr_filters
        self.printmore= torch.zeros(1)
        
    
    def forward(self,x):
        
        self.printmore=0
        
        if self.printmore:
            print('Begin -------')
            print(f'input x: {x.shape}')
            # x.shape [574, 64], [textlen for current batch, batch_size], textlen varies by batch      
  

        # Permute to put batch first, to (N,W)
        x = x.permute(1, 0)
        if self.printmore:
            print(f'After permute: {x.shape}') # [64, 574], [batch, textlen]

            
        # EMBEDDING
        # nn.Embedding. input (N, W) mini-batch, Words per... out:(N, W, emb_dim)      
        x = self.embedding(x)
        if self.printmore:
            print(f'Embedded: {x.shape}') # [64, 574, 50], [batch, textlen, emb_dim]
        
        # CONVOLUTION
        #Weight we get is size [128, 1, 5], [nr_channels, in_chan, filter_size]
        
        #Conv input [batch, in_channels=, textlen, emb_dim]
        # chance dim 1 and 2 with each other, textlen and emb_dim
        x = x.transpose(1,2)
        if self.printmore:
            print(f'Textlen, emb switched with transpose: {x.shape}')     
        
        x = self.conv1(x)           
        # = self.conv1_bn(self.conv1(x))
        if self.printmore:
            print(f'After conv1d: {x.shape}')   
        
            #print(x.shape) # [64, 128, 1091] [batch, filters, nr_steps filter took]
            
        #x = self.conv2(x)    

        #x = F.max_pool1d(x, x.size(2))          #print(x.shape) # [64, 128, 1]
        #x = x.squeeze(2) # squeezes the last 1 dim away

        # max + avg pool and concat
        # This will features 2* long, so next layer needs to input 2*len
        p1 = F.max_pool1d(x, x.size(2)).squeeze(2)          #print(x.shape) # [64, 128, 1]
        p2 = F.avg_pool1d(x, x.size(2)).squeeze(2)          #print(x.shape) # [64, 128, 1]
        x = torch.cat((p1,p2), 1)
        
        
        if self.printmore:
            print(f'After max_pool1d+squeeze(2): {x.shape}') #   [batch, nr_filters] [64, 128]  
       
        x = F.relu(x)        
       
        x = self.dropout(x)        
        
        x = self.fc(x)

        x = (x)

        return F.sigmoid(x)


# In[ ]:


# Calling
#VOCAB_SIZE=len(TEXT.vocab) # 25002
#EMBEDDING_DIM=50

#NR_FILTERS=128 
#FILTER_SIZE = 5
#OUTPUT_DIM = 126

#model = CNN(VOCAB_SIZE, EMBEDDING_DIM, NR_FILTERS, FILTER_SIZE, OUTPUT_DIM)
#model.embedding.weight.data.copy_(pretrained_embeddings)
#model = model.to(device)

#optimizer = optim.Adam(model.parameters())
#criterion = nn.BCEWithLogitsLoss()
#criterion = criterion.to(device)


# 2D conv model, with multi size kernels 
class CNN2(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout=0.5):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(fs,embedding_dim)) for fs in filter_sizes])
        self.fc = nn.Linear(len(filter_sizes)*n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        #x = [sent len, batch size]
        
        x = x.permute(1, 0)        
        #x = [batch size, sent len]
        
        embedded = self.embedding(x)
        #embedded = [batch size, sent len, emb dim]
        
        embedded = embedded.unsqueeze(1)        
        #embedded = [batch size, 1, sent len, emb dim]
        
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]            
        #conv_n = [batch size, n_filters, sent len - filter_sizes[n]]
        
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]        
        #pooled_n = [batch size, n_filters]
        
        cat = self.dropout(torch.cat(pooled, dim=1))
        #cat = [batch size, n_filters * len(filter_sizes)]
            
        return F.sigmoid(self.fc(cat))

    
#N_FILTERS = 100
#FILTER_SIZES = [3,4,5]
#model = CNN2(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT)


#N_FILTERS = 200
#FILTER_SIZES = [3,5,7]
#model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT)
#optimizer = optim.Adam(model.parameters())
# 14 epochs gave f1 0.84



# Simple model
import torch.nn as nn

class RNNSimple(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):

        #x = [sent len, batch size]
        
        embedded = self.embedding(x)
        
        #embedded = [sent len, batch size, emb dim]
        
        output, hidden = self.rnn(embedded)
        
        #output = [sent len, batch size, hid dim]
        #hidden = [1, batch size, hid dim]
        
        assert torch.equal(output[-1,:,:], hidden.squeeze(0))
        
        return self.fc(hidden.squeeze(0))

#HIDDEN_DIM = 256
#OUTPUT_DIM = 126
#model = RNNSimple(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)

# # LSTM model

# model
#HIDDEN_DIM = 250
#N_EPOCHS = 50
#N_LAYERS = 1

class LSTM(nn.Module):
    def __init__(self, input_dim, output_dim, embedding_dim, hidden_dim,
                n_layers, bidirectional, device):
        super(LSTM, self).__init__()
        self.n_layers = n_layers
        self.bidirectional=bidirectional
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional) # dropout=0.5)
        self.fc1 = nn.Linear(2*hidden_dim, 200) if bidirectional else nn.Linear(hidden_dim, 200)
        self.fc2 = nn.Linear(200, output_dim)
        self.hidden_dim = hidden_dim
        self.device=device
        
    def forward(self, x):
        x = self.embedding(x)
        x, (h, c) = self.lstm(x, self.hidden)
        x = F.relu(self.fc1(x[-1]))
        x = F.dropout(x)
        x = self.fc2(x)
        return F.sigmoid(x)
    
    def init_hidden(self, batch_size):
        if self.bidirectional:
            return (torch.zeros(2*self.n_layers, batch_size, self.hidden_dim).to(self.device),
                  torch.zeros(2*self.n_layers, batch_size, self.hidden_dim).to(self.device))
        else:
            return (torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device),
                  torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device))



#model = LSTM(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, N_LAYERS, BIDIRECTIONAL)
#model.embedding.weight.data.copy_(TEXT.vocab.vectors)
#optimizer = optim.SGD(model.parameters(), lr=0.02)
## optimizer = optim.Adam(model.parameters(), lr=0.01)
#model = model.to(device)
#criterion = criterion.to(device)
#model.hidden = model.init_hidden()



# LSTM Ver 2 - converges with 24k examples

class LSTMv2(nn.Module):
    def __init__(self, input_dim, output_dim, embedding_dim, hidden_dim,
                n_layers, bidirectional, device):
        super().__init__()
        self.n_layers = n_layers
        self.bidirectional=bidirectional
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional) # dropout=0.5)
        #self.fc1 = nn.Linear(2*hidden_dim, output_dim) if bidirectional else nn.Linear(hidden_dim, output_dim)
        self.fc1 = nn.Linear(2*hidden_dim, 200) if bidirectional else nn.Linear(hidden_dim, 200)
        self.fc2 = nn.Linear(200, output_dim)
        self.hidden_dim = hidden_dim
        self.device=device
        
    def forward(self, x):
        x = self.embedding(x)
        x, (h, c) = self.lstm(x, self.hidden)
        
        x = self.fc1(x[-1])
        x = F.relu(x)
        x = F.dropout(x)
        x = self.fc2(x)
        x = F.sigmoid(x)
        return x
    
    def init_hidden(self, batch_size):
        if self.bidirectional:
            return (torch.zeros(2*self.n_layers, batch_size, self.hidden_dim).to(self.device),
                  torch.zeros(2*self.n_layers, batch_size, self.hidden_dim).to(self.device))
        else:
            return (torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device),
                  torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device))
        

# LSTM ver3 batchnorm
class LSTMv3(nn.Module):
    def __init__(self, input_dim, output_dim, embedding_dim, hidden_dim,
                n_layers, bidirectional, device):
        super(LSTM, self).__init__()
        self.n_layers = n_layers
        self.bidirectional=bidirectional
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional) # dropout=0.5)
        self.fc1 = nn.Linear(2*hidden_dim, 200) if bidirectional else nn.Linear(hidden_dim, 200)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        self.fc2 = nn.Linear(200, output_dim)
        self.bn2 = nn.BatchNorm1d(200)
        self.hidden_dim = hidden_dim
        self.device=device
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.embedding(x)
        x, lstm_h = self.lstm(x, self.hidden)
        x = F.relu(self.fc1(x[-1]))
        x = self.dropout(x)
        output = self.fc2(self.bn2(x))
        return F.sigmoid(output)
    
    def init_hidden(self, batch_size):
        if self.bidirectional:
            return (torch.zeros(2*self.n_layers, batch_size, self.hidden_dim).to(self.device),
                  torch.zeros(2*self.n_layers, batch_size, self.hidden_dim).to(self.device))
        else:
            return (torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device),
                  torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device))


class LSTM_CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, hidden_dim, device):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=1, out_channels=n_filters, kernel_size=fs) for fs in filter_sizes])
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(len(filter_sizes)*n_filters, output_dim)
        self.hidden_dim = hidden_dim
        self.device=device
        
    def forward(self, x):
        # x = x.permute(1, 0)        
        x = self.embedding(x)
        x, (h, c) = self.lstm(x, self.hidden)
        x = self.fc1(x[-1]).unsqueeze(1)
        # embedded = embedded.unsqueeze(1)
        x = [F.relu(conv(x)) for conv in self.convs]  
        x = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in x]           
        x = torch.cat(x, dim=1)
        return torch.sigmoid(self.fc2(x))
      
    def init_hidden(self, batch_size):
            return (torch.zeros(1, batch_size, self.hidden_dim).to(self.device),
                  torch.zeros(1, batch_size, self.hidden_dim).to(self.device))



class CNN_LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, hidden_dim, device):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(fs,embedding_dim)) for fs in filter_sizes])
        self.fc = nn.Linear(len(filter_sizes)*n_filters, output_dim)
        self.lstm = nn.LSTM(n_filters*len(filter_sizes), hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.hidden_dim = hidden_dim
        self.device=device
        
    def forward(self, x):
        x = x.permute(1, 0)        
        embedded = self.embedding(x)
        embedded = embedded.unsqueeze(1)        
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]            
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]           
        cat = torch.cat(pooled, dim=1)
        x, (h, c) = self.lstm(cat.unsqueeze(0), self.hidden)
        return torch.sigmoid(self.fc2(x[-1]))
      
    def init_hidden(self, batch_size):
            return (torch.zeros(1, batch_size, self.hidden_dim).to(self.device),
                  torch.zeros(1, batch_size, self.hidden_dim).to(self.device))

