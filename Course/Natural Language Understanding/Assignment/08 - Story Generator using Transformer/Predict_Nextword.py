import torch
import pickle
from DecoderTransformer import Decoder
from torchtext.data.utils import get_tokenizer
import torch.nn as nn

#Load GPU
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

# Load data (deserialize)
with open('vocab_transform_hp.pickle', 'rb') as handle:
    vocab_transform = pickle.load(handle)
print(len(vocab_transform))

def generate(prompt, max_seq_len, temperature, model, tokenizer, vocab, device, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    model.eval()
    tokens = tokenizer(prompt)
    indices = [vocab[t] for t in tokens]
    batch_size = 1
    # hidden = model.init_hidden(batch_size, device)
    with torch.no_grad():
        for i in range(max_seq_len):
            src = torch.LongTensor([indices]).to(device)
            prediction, hidden = model(src)

            # print(prediction.shape)
            #prediction: [batch size, seq len, vocab size]
            #prediction[:, -1]: [batch size, vocab size] #probability of last vocab
            
            probs = torch.softmax(prediction[:, -1] / temperature, dim=-1)  
            prediction = torch.multinomial(probs, num_samples=1).item()    
            
            while prediction == vocab['<unk>']: #if it is unk, we sample again
                prediction = torch.multinomial(probs, num_samples=1).item()

            if prediction == vocab['<eos>']:    #if it is eos, we stop
                break

            indices.append(prediction) #autoregressive, thus output becomes input

    itos = vocab.get_itos()
    tokens = [itos[i] for i in indices]
    return tokens

def predict(prompt,temperature=1):
    max_seq_len = 30
    seed = 0
    generation = generate(prompt, max_seq_len, temperature, model, tokenizer, vocab_transform, device, seed)
    return ' '.join(generation)

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

output_dim  = len(vocab_transform)
hid_dim = 256
dec_layers = 12
dec_heads = 8
dec_pf_dim = 512
dec_dropout = 0.1

# Define special symbols and indices
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<sos>', '<eos>']

SRC_PAD_IDX = PAD_IDX
TRG_PAD_IDX = PAD_IDX

model = Decoder(output_dim, 
              hid_dim, 
              dec_layers, 
              dec_heads, 
              dec_pf_dim, 
              dec_dropout, 
              device,SRC_PAD_IDX,TRG_PAD_IDX).to(device)

model.apply(initialize_weights)
save_path = f'models/Decoder.pt'
model.load_state_dict(torch.load(save_path))