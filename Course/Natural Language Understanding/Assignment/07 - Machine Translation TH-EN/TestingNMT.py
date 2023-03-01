import torch
import pickle
from NMTAttention_model import *

SRC_LANGUAGE,TRG_LANGUAGE = 'th', 'en'


# Define special symbols and indices
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<sos>', '<eos>']

# Load data (deserialize)
with open('vocab_transform.pickle', 'rb') as handle:
    vocab_transform = pickle.load(handle)

with open('text_transform.pickle', 'rb') as handle:
    text_transform = pickle.load(handle)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

src_text = text_transform[SRC_LANGUAGE](' ').to(device)
trg_text = text_transform[TRG_LANGUAGE](' ').to(device)
src_text = src_text.reshape(-1, 1)  #because batch_size is 1
trg_text = trg_text.reshape(-1, 1)
text_length = torch.tensor([src_text.size(0)]).to(dtype=torch.int64)

save_path = './models/Seq2SeqPackedAttention_additive.pt'

input_dim   = len(vocab_transform[SRC_LANGUAGE])
output_dim  = len(vocab_transform[TRG_LANGUAGE])
emb_dim     = 256  
hid_dim     = 512  
dropout     = 0.5
SRC_PAD_IDX = PAD_IDX

def initialize_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)

attn = Attention(hid_dim, variants='additive')
enc  = Encoder(input_dim,  emb_dim,  hid_dim, dropout)
dec  = Decoder(output_dim, emb_dim,  hid_dim, dropout, attn)

model = Seq2SeqPackedAttention(enc, dec, SRC_PAD_IDX, device).to(device)
model.apply(initialize_weights)

model.load_state_dict(torch.load(save_path))
model.eval()

with torch.no_grad():
    output, attentions = model(src_text, text_length, trg_text, 0) #turn off teacher forcing
#trg_len, batch_size, trg_output_dim

output = output.squeeze(1)
#trg_len, trg_output_dim
output = output[1:]
output_max = output.argmax(1) #returns max indices
mapping = vocab_transform[TRG_LANGUAGE].get_itos()
for token in output_max:
    print(mapping[token.item()])


if __name__ == "__main__":
    pass
