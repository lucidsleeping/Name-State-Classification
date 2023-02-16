from nltk.tokenize import sent_tokenize, word_tokenize

#normalized file
out = open("normalized_input.txt","a")

#input file 
sample = open("input.txt")
s = sample.read()
f = s.replace("\n", " ") 
data = []
for i in sent_tokenize(f):
    temp = []
    for j in word_tokenize(i):
        data.append(j)

for i in data:
    print(len(i))
    if len(i)>=4:
        i.replace(".","#")
        i.replace(" ","#")
        zero = 30 - len(i)
        for j in range(zero):
            i = i + "0"
        out.write(i)
        out.write("\n")
    else:
        continue


