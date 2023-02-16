from nltk.tokenize import sent_tokenize, word_tokenize

from gensim.models import Word2Vec
 
import csv

with open("normalized_input.txt") as file:
    for item in file:
        data = item
        vec = []
        #appending name
        vec.append(data)
        data = list(data)
        model = Word2Vec(data, min_count = 1, vector_size = 200)
        for i in data:
            vec.append(list(model.wv[i]))
           

        #name of the state (need to change manually)
        #jammu - 0001
        #nagaland - 0010
        #rajasthan - 0100
        #kerala - 1000
    
        vec.append("1000")


        inputvec = []
        inputvec.append(vec[0])
        for i in range(1,len(vec)):
            vec[i] = str(vec[i])
            vec[i] = vec[i].replace("[","")
            vec[i] = vec[i].replace("]","")
            print(vec[i])
            inputvec.append(vec[i])

        with open("dataset.csv", 'a') as csvfile:
            csvwriter = csv.writer(csvfile)
            # writing the data rows
            csvwriter.writerow(vec)

