from nltk.tokenize import sent_tokenize, word_tokenize

from gensim.models import Word2Vec
 

sample = open("normalized_input.txt")
s = sample.read()
f = s.replace("\n", " ") 
data = []
for i in sent_tokenize(f):
    temp = []
    for j in word_tokenize(i):
        for k in j.lower():
            data.append(k)

model = Word2Vec(data, min_count = 1, vector_size = 1)
vec = ""
for i in data:
     vec += str(model.wv[i])

print(vec)
