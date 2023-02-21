import torch
import pickle
from LSTMLanguage_Model import LSTMLanguageModel
from torchtext.data.utils import get_tokenizer
#Load GPU
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

# Load data (deserialize)
with open('vocab.pickle', 'rb') as handle:
    vocab = pickle.load(handle)
print(len(vocab))

def generate(prompt, max_seq_len, temperature, model, tokenizer, vocab, device, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    model.eval()
    tokens = tokenizer(prompt)
    indices = [vocab[t] for t in tokens]
    batch_size = 1
    hidden = model.init_hidden(batch_size, device)
    with torch.no_grad():
        for i in range(max_seq_len):
            src = torch.LongTensor([indices]).to(device)
            prediction, hidden = model(src, hidden)
            
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
                #superdiverse       more diverse
    # temperatures = [0.5, 0.7, 0.75, 0.8, 1.0] 
    #sample from this distribution higher probability will get more change
    # for temperature in temperatures:
    #     generation = generate(prompt, max_seq_len, temperature, model, tokenizer, 
    #                         vocab, device, seed)
    #     print(str(temperature)+'\n'+' '.join(generation)+'\n')
    generation = generate(prompt, max_seq_len, temperature, model, tokenizer, vocab, device, seed)
    return ' '.join(generation)

vocab_size = len(vocab)
emb_dim = 1024                # 400 in the paper
hid_dim = 1024                # 1150 in the paper
num_layers = 2                # 3 in the paper
dropout_rate = 0.65              
lr = 1e-3                     
model = LSTMLanguageModel(vocab_size, emb_dim, hid_dim, num_layers, dropout_rate).to(device)
save_path = f'models/best-val-auto.pt'
model.load_state_dict(torch.load(save_path))
