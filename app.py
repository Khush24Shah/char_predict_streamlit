import streamlit as st
import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# random seed
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# Streamlit app on Text Generation
st.title('Text Generation')

# select embedding dimension (drop-down)
emb_dim = st.selectbox('Select the embedding dimension', [1, 2, 4, 8, 16])

# select activation function (drop-down)
activation = st.selectbox('Select the activation function', ['sigmoid', 'sin', 'tanh', 'relu'])

path = f"models/{activation}-model-{emb_dim}.pth"

# select "k" (number of characters to predict)
k = st.slider('Select the number of characters to predict', 1, 100, 1)

# Enter the text to predict upon
text = st.text_input('Enter the text to predict upon', 'Type here')

unique_chars = ['\n', ' ', '!', '$', '&', "'", ',', '-', '.', '3', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
vocab_dict = {i:ch for i, ch in enumerate(unique_chars)}
vocab_dict_inv = {ch:i for i, ch in enumerate(unique_chars)}
block_size = 8

# model
class NextChar(nn.Module):
  def __init__(self, block_size, vocab_size, emb_dim, hidden_dims=None):
    if hidden_dims is None:
      hidden_dims = [block_size * emb_dim, block_size * emb_dim]
    super().__init__()
    self.emb = nn.Embedding(vocab_size, emb_dim)
    self.lin1 = nn.Linear(block_size * emb_dim, hidden_dims[0])
    self.lin2 = nn.Linear(hidden_dims[0], hidden_dims[1])
    self.lin3 = nn.Linear(hidden_dims[1], vocab_size)

  def forward(self, x):
    x = self.emb(x)
    x = x.view(x.shape[0], -1)
    x = torch.sin(self.lin1(x))
    x = torch.sin(self.lin2(x))
    x = self.lin3(x)
    return x

# load model
model = NextChar(block_size, len(unique_chars), emb_dim)
model.load_state_dict(torch.load(path))
model.eval()

def generate_name(model, sentence, itos, stoi, block_size, max_len=10):
    original_sentence = sentence
    if len(sentence) < block_size:
        sentence = " " * (block_size - len(sentence)) + sentence
    using_for_predicction = sentence[-block_size:].lower()
    context = [stoi[word] for word in using_for_predicction]
    prediction = ""
    for i in range(max_len):
        x = torch.tensor(context).view(1, -1)
        y_pred = model(x)
        ix = torch.distributions.categorical.Categorical(logits=y_pred).sample().item()
        ch = itos[ix]
        prediction += ch
        context = context[1:] + [ix]

    return prediction

generation = generate_name(model, text, vocab_dict, vocab_dict_inv, block_size, k)

# write prediction
st.write(f'Predicted next {k} character{" is" if k==1 else "s are"}: ":blue[ {generation[:k]} ]"')

# feature visualization (t-SNE)

# with random state set to 42, the plot will be reproducible
emb = model.emb

# Assuming you have a tensor of higher-dimensional embeddings
embeddings = (emb.weight.data).detach().numpy()

# Apply t-SNE
tsne = TSNE(n_components=min(2, emb_dim), perplexity=30, n_iter=300)
embeddings_tsne = tsne.fit_transform(embeddings)

def plot_emb(embeddings_tsne, itos, dim, ax=None):
	if ax is None:
		_, ax = plt.subplots()
	
	if dim > 1:
		for i in range(len(itos)):
			x = embeddings_tsne[i, 0]
			y = embeddings_tsne[i, 1]
			ax.scatter(x, y, color='k')
			ax.text(x + 0.04, y + 0.04, itos[i])
	else:
		ax.scatter(embeddings_tsne, np.zeros_like(embeddings_tsne), color='k')
		for i in range(len(itos)):
			ax.text(embeddings_tsne[i] + 0.04, 0.04, itos[i])
	return ax

if emb_dim > 1:
	# convert the embeddings_tsne to a dataframe
	df = pd.DataFrame(embeddings_tsne, columns=['1', '2'])
else:
    # convert the embeddings_tsne to a dataframe
    df = pd.DataFrame(embeddings_tsne, columns=['1'])

fig, ax = plt.subplots()
ax = plot_emb(embeddings_tsne, vocab_dict, emb_dim, ax=ax)
st.pyplot(fig)