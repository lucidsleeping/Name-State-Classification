from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec
import csv

def normalization():
    out = open("newnormalized_input.txt","a")

    #input file 
    sample = open("newinput.txt")
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


def data_creation(key):
    with open("newnormalized_input.txt") as file:
        for item in file:
            data = item
            vec = []
            #appending name
            vec.append(data)
            data = list(data)
            model = Word2Vec(data, min_count = 1, vector_size = 200)
            for i in data:
                vec.append(list(model.wv[i]))
                # print(list(model.wv[i]))

            #vec.append("0")       
            inputvec = []
            inputstring = ""
            
            inputvec.append(vec[0])
        
            for i in range(1,len(vec)):
                vec[i] = str(vec[i])
                vec[i] = vec[i].replace("[","")
                vec[i] = vec[i].replace("]","")
                x = vec[i].split(",")
                for j in x:
                    print(j)
                    inputvec.append(j)
                # inputstring+=str(vec[i])
                # inputstring+=", "
                
        
            inputvec.append(key)

            with open("singledataset.csv", 'a') as csvfile:
                csvwriter = csv.writer(csvfile)
                # writing the data rows
                csvwriter.writerow(inputvec)




normalization()
#north = 1
#south = 2
#east = 3 
#west = 4
data_creation(1)
