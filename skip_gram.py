import torch
import nltk
from nltk.tokenize import word_tokenize
import re
import collections

nltk.download('punkt') 
nltk.download('punkt_tab')

# Preprocess the raw text
def preprocess(raw_text, min_freq = 1, vocab_size = 5000):
    paragraphs = re.findall(r"([^\n]+)", raw_text)
    sentences = nltk.sent_tokenize(raw_text)
    tokens = word_tokenize(raw_text)
    counter = collections.Counter(tokens)
    vocab = {i: item for i, (item,_) in enumerate(counter.most_common(vocab_size))} 
    vocab_reverse = {item: i for i, (item,_) in enumerate(counter.most_common(vocab_size))} 
    return paragraphs, sentences, vocab, vocab_reverse

# Prepare training data, which consist of context-target pairs
def make_training_data(text, word2index, N=2):
    context = []
    target = []
    for sentence in text:
        tokens = word_tokenize(sentence)
        for i in range(len(tokens)):
            for j in range(max(0,i-N),min(i+N+1,len(tokens))):
                if i!=j and tokens[i] in word2index and tokens[j] in word2index:
                    context.append(word2index[tokens[i]])
                    target.append(word2index[tokens[j]])
    return context, target

# Define the training function
def train(net, dataloader, epochs, lr = 0.01):
    optim = torch.optim.Adam(net.parameters(),lr=lr)
    for i in range(epochs):
        total_loss = 0
        for (x,y) in dataloader:
            z = net(x)
            loss_fn = torch.nn.CrossEntropyLoss() 
            loss = loss_fn(z,y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()
        print(f"Epoch: {i+1}: loss={total_loss}")

# Define the neural network
class skip_gram_network(torch.nn.Module):
  def __init__(self,vocab_size, embedding_size=30):
    super().__init__()
    self.layer1 = torch.nn.Embedding(vocab_size,embedding_size)
    self.layer2 = torch.nn.Linear(embedding_size,vocab_size)

  def forward(self,x):
    x = self.layer1(x)
    x = self.layer2(x)
    return x

# Test the result on random words
def close_words(x, index2word, word2index, vecs, n=5):
  if x not in word2index:
    print(f"{x} is not contained in the dictionary.")
    return
  dist = []
  x_vec = vecs[word2index[x]]
  for i in range(len(vecs)):
    dist.append(torch.linalg.vector_norm(vecs[i]-x_vec))
  _, indices = torch.topk(torch.tensor(dist), k=n, largest=False)
  for i in indices:
    print(index2word[i.item()])
  return  

#raw_text = open("data/test-text.txt", "r", encoding="utf-8").read() 
raw_text = open("data/The_Adventures_of_Sherlock_Holmes.txt", "r", encoding="utf-8").read() 

vocab_size = 5000
context_size = 1
embedding_size = 50
epochs = 50 
learning_rate = 0.01
batch_size = 16

paragraphs, sentences, index2word, word2index = preprocess(raw_text, vocab_size=vocab_size)
vocab_size = min(len(index2word), vocab_size)
train_context, train_target = make_training_data(sentences, word2index)
dataset = torch.utils.data.TensorDataset(torch.tensor(train_context, dtype=torch.long),torch.tensor(train_target, dtype=torch.long))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

net = skip_gram_network(vocab_size=vocab_size, embedding_size=embedding_size)
train(net, dataloader, epochs)

vecs = net(torch.tensor(list(range(vocab_size))))

close_words('blood', index2word, word2index, vecs)


