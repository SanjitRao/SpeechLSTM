import os
import zipfile
import pandas as pd
import torch
import numpy as np

from torch import nn, optim
import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import nltk
from nltk import word_tokenize

import time

directory_in_str = "TrumpLSTM_datasets"
df = pd.DataFrame()
directory = os.fsencode(directory_in_str)
#directory = directory.decode()
df = pd.DataFrame()
list_of_speeches = []
for file in os.listdir(directory):
    file = directory_in_str+ "/" + file.decode()
    filename = os.fsdecode(file)
    if filename.endswith("txt"):
        with open(file, "r") as f:
            #file.decode(encoding="utf-8")
            speech = f.readlines()
            f.close() # I thinkkk if u open using 'with' u dont gotta say f.close() but why not...
        list_of_speeches.append(speech)
    else:
        continue

### preprocessing data, create input_seq w/ added END and START tokens

vocab = set()
speech_data = []

for i in range(len(list_of_speeches)):
    # Add START and END tokens to each speech
    speech = "START " + list_of_speeches[i][0] + " END "
    # use speech to update vocab set, add 'START' and 'END' tokens to vocab
    vocab.update(speech.split())
    # Extend speech_data with current speech
    speech_data.extend(speech.split())
# Creating a dictionary that maps integers to the strings
int2string = dict(enumerate(vocab))
# convert vocab set into a list and convert to the np array
vocab = list(vocab)
vocab = np.array(vocab).reshape(-1,1)
# Creating another dictionary that maps strings to integers
string2int = {string: ind for ind, string in int2string.items()}


# define one-hot encoder and label encoder
#You should reshape your X to be a 2D array not 1D array. Fitting a model requires requires a 2D array. i.e (n_samples, n_features)
onehot_encoder = OneHotEncoder(sparse=False).fit(vocab)
#label_encoder  = {ch: i for i, ch in enumerate(vocab)}

device = "cuda:0" if torch.cuda.is_available() else "cpu"

class DataLSTM(): ## an iterable that allows you to sample bs number of words encoded as one-hot encoded vectors

  def __init__(self, speech_data, index=0, bs=1024, encoder=onehot_encoder):
      self.speech_data = speech_data
      self.index = index
      #self.bptt = bptt
      self.bs = bs
      self.encoder = encoder

  def __iter__(self):
    return self

  def __next__(self):
    input_seq = []
    output_seq = []

    if self.index + (self.bs) + 1 >= len(self.speech_data):
      raise StopIteration

    index = self.index

    #make a batch
    for _ in range(self.bs):
      
      input_line = []
      output_line = []

      #Deprecated
      '''#make a line
      for _ in range(self.bptt):
        input_line.append(self.train_data[index])
        #output line is 1 step ahead of input line
        output_line.append(self.train_data[index + 1])
        index += 1'''

      # Make a one-hot encoded vector of the given, leave output as string2int
      input_word = np.array([self.speech_data[index]])
      input_word = self.encoder.transform(input_word.reshape(-1,1)) 
      
      output_word = string2int[self.speech_data[index+1]]
      index +=1
      
      input_seq.append(input_word)
      output_seq.append(output_word)
    self.index = index

    #Deprecated
    '''for i in range(len(input_seq)):
      #need to map string to int
      input_seq[i] = [string2int[string] for string in input_seq[i]]
      output_seq[i] = [string2int[string] for string in output_seq[i]]'''
      
    #convert arrays into tensors and place on gpu
    input_seq = torch.tensor(input_seq, dtype=torch.float32).reshape(self.bs,1, -1).to(device) # (B=1024,N=20106, F=1)
    output_seq = torch.tensor(output_seq, dtype=torch.int64).to(device) #.reshape(self.bs,self.bptt,1)
    # Bacth will be of size (batch_size=10, num_in_seq = 64)
    # num_features = len(vocab)
    return input_seq, output_seq

# Data Flow Protocol:
# 1. network input shape: (batch_size, seq_length, num_features) == (1024, 1, 20106)
# 2. LSTM output shape: (batch_size, seq_length, hidden_size)
# 3. Linear input shape:  (batch_size * seq_length, hidden_size)
# 4. Linear output: (batch_size * seq_length, out_size)


speech_train, speech_valid = train_test_split(speech_data, train_size=int(0.8*len(speech_data)), test_size=int(0.2*len(speech_data)), shuffle=False)

speech_train_LSTM = DataLSTM(speech_train)
speech_valid_LSTM = DataLSTM(speech_valid)

### Define the LSTM class
class TrumpLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, output_size, encoder=onehot_encoder, vocab = vocab):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        self.encoder = encoder
        self.vocab = vocab



    def forward(self, x): ## x of shape (B, N, 1)
        '''If the initial states h_0 and c_0 are not provided, the RNN implementation assumes default values of zeros for both states.'''
        x, _ = self.lstm(x) # Output of shape (B, N, M), (h_n, c_n) <== dont need second output
        x = x.reshape(-1, hidden_size) ## gotta reshape b/c linear layer wants shape (B*N, M)
        x = self.linear(x) # -> Linear out: (batch_size * seq_length, out_size)
        return x

    def predict(self, char, top_k=None, hs=None):
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.to(device)
        self.eval()
        
        with torch.inference_mode():
            x = np.array([char])
            x = x.reshape(-1, 1)
            x = self.onehot_encode(x)
            x = x.reshape(1, 1, -1)
            x = torch.tensor(x, dtype=torch.float32)
            x = x.to(device)

            out = self(x)

            ps = F.softmax(out, dim=1).squeeze()
            
            if top_k is None:
                choices = np.arange(len(self.vocab))
            else:
                ps, choices = ps.topk(top_k)
                choices = choices.cpu().numpy()
            
            ps = ps.cpu().numpy()
            
            char = np.random.choice(choices, p=ps/ps.sum())
            char = int2string[char]

        return char, hs
    def sample(self, length, top_k=None, primer='Thank you'):
        hs = None
        primer = word_tokenize(primer)
        for px in primer:
            out, hs = self.predict(px, hs=hs)
        
        chars = [ch for ch in primer]
        for ix in range(length):
            char, hs = self.predict(chars[-1], top_k=top_k, hs=hs)
            chars.append(char)
        
        return ' '.join(chars)
    
    
    def label_encode(self, data):
        return np.array([string2int[ch] for ch in data])
    
    
    def label_decode(self, data):
        return np.array([int2string[i] for i in data])
    
    
    def onehot_encode(self, data):
        return self.encoder.transform(data)
    
    
    def onehot_decode(self, data):
        return self.encoder.inverse_transform(data)


## train the model


# Define architecture:
input_size  = len(vocab)  #  batch is of shape (64, 10)
hidden_size = 100 # number of hidden nodes in the LSTM layer, batch after passing through LSTM will have shape (64, 10, 100)
n_layers    = 2   # number of LSTM layers
output_size = len(vocab)  # output of len(vocab) scores for the next character

lr = 1e-5
model = TrumpLSTM(input_size, hidden_size, n_layers, output_size).to(device)

optimizer = AdamW(model.parameters(), lr=lr)
loss_fn = CrossEntropyLoss().to(device)
train_loss = []
valid_loss = []

def train_model(model, train_data, eval_data):
    start_time = time.time()
    #train_data = DataLM(data), I think data is already in DataLM form
    i = 0
    v_loss = 0
    t_loss = 0 # only looping through the data once, so initializing this and v_loss should be fine 
    for x, y in tqdm(train_data):
        elapsed_time = time.time() - start_time

        if elapsed_time >= 25200: # 7 hours
            MODEL_NAME = "TrumpLSTM_Model.pth"
            torch.save(model.state_dict(), MODEL_NAME)
            break

        print(x.shape, y.shape)
        model.train()
        y_hat = model(x)
        y_hat = y_hat.float().to(device)
        
        print("ur mom" + str(i))
        print(y_hat.shape) # Shape = (1024, 20106)
        
        #y_hat = torch.argmax(y_hat, dim=1).float().to(device)
        #y_indices = torch.argmax(y, dim=1)
        
        #Mid-training testing
        print(y.dtype, y_hat.dtype) # (torch.float32, torch.float32)
        print(y.shape, y_hat.shape)

        loss = loss_fn(y_hat, y) 
        #loss.requires_grad = True
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        model.eval()
        hs = None
        '''with torch.no_grad():
            for x, y in speech_valid_LSTM:
                #x = torch.tensor(x).float()
                #x = x.to(device)

                # invert one-hot of targets for use by cross-entropy loss function
                #y = y.reshape(-1, len(model.vocab))
                #y = model.onehot_decode(y)
                #y = model.label_encode(y.squeeze())
                #y = torch.from_numpy(y).long().to(device)

                out = model(x)
                out = torch.argmax(out, dim=1).float().to(device)

                loss = loss_fn(y, out)
                v_loss += loss.item()

                valid_loss.append(np.mean(v_loss))
        
        train_loss.append(np.mean(t_loss))
        
        if i % 2 == 0:
            print(f'------- Epoch {i} ---------')
            print(f'Training Loss: {train_loss[-1]}')
            if valid_loss:
                print(f'Valid Loss: {valid_loss[-1]}')
        
        print("reached here")'''
        i+=1
#train_model(model, speech_train_LSTM, speech_valid_LSTM)

#Plot training and validation loss 
'''plt.plot(train_loss, label="Training")
plt.plot(valid_loss, label="Validation")
plt.title("Loss vs Epochs")
plt.legend()
plt.show()'''

# save the model and export to desired PATH
#MODEL_NAME = "TrumpLSTM_Model.pth"
#torch.save(model.state_dict(), MODEL_NAME)