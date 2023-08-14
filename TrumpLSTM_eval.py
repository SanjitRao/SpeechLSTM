from TrumpLSTM import *

input_size  = 20106  #  batch is of shape (64, 10)
hidden_size = 100 # number of hidden nodes in the LSTM layer, batch after passing through LSTM will have shape (64, 10, 100)
n_layers    = 2   # number of LSTM layers
output_size = 20106  # output of len(vocab) scores for the next character

# load model
PATH = "TrumpLSTM_Model.pth"
model = TrumpLSTM(input_size, hidden_size, n_layers, output_size)
model.load_state_dict(torch.load(PATH))
#model.eval()
#print(model.state_dict())

print(vocab)
print(["Welcome"] in vocab)
print(model.sample(100))
