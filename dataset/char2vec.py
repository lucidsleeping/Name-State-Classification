from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
warnings.filterwarnings(action = 'ignore') 
import gensim
from gensim.models import Word2Vec
 

sample = open("input.txt")
s = sample.read()
f = s.replace("\n", " ") 
data = []
for i in sent_tokenize(f):
    temp = []
    for j in word_tokenize(i):
        for k in j.lower():
            data.append(k)
    #data.append(temp)
print(data)
model = Word2Vec(data, min_count = 1, vector_size = 1)
print(model.corpus_count)
print(model.wv.vectors)
for i in data:
     print(i,model.wv[i])

