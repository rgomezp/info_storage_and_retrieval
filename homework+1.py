#Homework 1 CSCE 470
#2-20-2017
#By: Rodrigo Gomez-Palacio

#Imports and Inputs
import nltk
#nltk.download()                          #downloads all nltk packages (may be needed if not installed on machine)
import string
import math

documents = [
		"Today the aggies won! Go aggies!", 
		"aggies have won today", 
		"the aggies lost last week", 
		"Find the latest Aggies news", 
		"An Aggie is a student at Texas A&M"
		]


# In[356]:

#Punctuation removal 
temp = []
for doc in documents:
    tokens = doc.translate(None,string.punctuation)
    temp.append(tokens)
    
documents = temp


# In[357]:

#Tokenize
temp = []
for doc in documents:
    tokens = nltk.word_tokenize(doc)
    temp.append(tokens)
    
documents = temp


# In[358]:

#Case Folding
for doc in documents:
    for term in range(0,len(doc)):
        doc[term] = doc[term].lower()
        
   


# In[359]:

#Stopword removal
from nltk.corpus import stopwords
stop = set(["the","go","have","an","is","a","at"])

for doc in range(0,len(documents)):
    new_doc = []
    for term in range(0,len(documents[doc])):
        if documents[doc][term] not in stop:
            new_doc.append(documents[doc][term])
    documents[doc] = new_doc


# In[360]:

#Lemmatization
from nltk.stem.wordnet import WordNetLemmatizer
lmtzr = WordNetLemmatizer()

for doc in documents:
    for term in range(0,len(doc)):
            doc[term] = lmtzr.lemmatize(doc[term])


# In[361]:

#Stemming with Porter Stemmer
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

for doc in documents:
    for term in range(0,len(doc)):
            doc[term] = stemmer.stem(doc[term])

print "---Preprocessing Output---"
for i in documents:
    print i


# In[362]:

#Build  term-document incidence matrix

#Build set with unique terms
terms = set([])
for doc in documents:
    for term in doc:
        if term not in terms:
            terms.add(term)
            
terms = list(terms)

incidence_matrix = []          #matrix elements represent terms appearing in documents

for term in terms:
    row = []
    for doc in documents:
        if term in doc:
            row.append(1)
        else:
            row.append(0)
    incidence_matrix.append(row)

print "---Incidence matrix---"
for i in incidence_matrix:
    print i


# In[363]:

#Build term-document count matrix (tf_td)

count_matrix = []

print "Terms:"
for i in terms:
    print " ", i
    
print "\n"
    

for term in terms:
    row = []
    for doc in documents:
        count = 0
        if term in doc:
            for index in range(0,len(doc)):
                if doc[index] == term:
                    count+=1
            row.append(count)
        else:
            row.append(0)
    count_matrix.append(row)
print "Count_matrix:"
for i in count_matrix:
    print i


# In[364]:

#Build document frequency matrix
document_freq_vec = []
for i in range(0,len(terms)):
    count = 0
    for doc in documents:
        if terms[i] in doc:
            count+=1
    document_freq_vec.append(count)
    
print "Doc_frequency_vector:\n", document_freq_vec, "\n\n"

#Build tf-weight vector
N = len(documents)
tf_weight_vec = []
for num in document_freq_vec:
    weight = math.log10(N/num)
    tf_weight_vec.append(weight)
    
print "Tf_weighted_vector:"
for i in tf_weight_vec:
    print i


# In[365]:

#Build Tf-Idf Matrix
tf_idf_matrix = []

for term in range(0,len(terms)):
    row = []
    for doc in range(0,len(documents)):
        weight = math.log10(1+count_matrix[term][doc])*tf_weight_vec[term]
        row.append(round(weight,3))
    tf_idf_matrix.append(row)
    
print "---Tf_idf_matrix---" 
for i in tf_idf_matrix:
    print i


# In[366]:

#Transpose tf_idf_matrix
tf_idf_matrix_T = zip(*tf_idf_matrix)
#print tf_idf_matrix_T

norm_tf_idf_matrix_T = []                #normalized tf_idf_matrix transpose
#Length Normalization
for doc_vec in tf_idf_matrix_T:
    #Calculate denominator
    sum_of_squares = 0
    for num in doc_vec:
        sum_of_squares += num**2
    denominator = sum_of_squares**(.5)
    new_vec = []
    for num in doc_vec:
        norm_val = num / denominator
        new_vec.append(norm_val)
    norm_tf_idf_matrix_T.append(new_vec)
        
    


# In[367]:

#Build 5 x 5 cosine similarity matrix
cosine_sim = []

for doc_vec1 in range(0,len(norm_tf_idf_matrix_T)):
    row = []
    for doc_vec2 in range(0,len(norm_tf_idf_matrix_T)):
        if doc_vec1 == doc_vec2:
            row.append(1)
        elif doc_vec1 < doc_vec2:                            #symmetric matrix, don't waste operations on half the mtrx
            score = 0
            for i in range(0,len(norm_tf_idf_matrix_T[doc_vec1])):
                score += (norm_tf_idf_matrix_T[doc_vec1][i]*norm_tf_idf_matrix_T[doc_vec2][i])
            row.append(round(score,4))
        else:
            row.append(0)
    cosine_sim.append(row)
    
#Because symmetric matrix, copy over
for doc_vec1 in range(0,len(norm_tf_idf_matrix_T)):
    for doc_vec2 in range(0,len(norm_tf_idf_matrix_T)):
        if doc_vec1 > doc_vec2:
            cosine_sim[doc_vec1][doc_vec2] = cosine_sim[doc_vec2][doc_vec1]

print "---Cosine Similarity Matrix---"
for row in cosine_sim:
    print row
            


# In[368]:

#---------Part 2------------
#--!--for visualization of part 2 (not required)--!--
#Linked List Class
class Node:
    def __init__(self,initdata):
        self.data = initdata
        self.next = None

    def getData(self):
        return self.data

    def getNext(self):
        return self.next

    def setData(self,newdata):
        self.data = newdata

    def setNext(self,newnext):
        self.next = newnext
        
#Term & Frequency Class
class Term:
    def __init__(self,initterm,initfreq):
        self.term = initterm
        self.freq = initfreq
        
#Build Inverted Index
head_container = []

for row in range(0,len(incidence_matrix)):
    count = 0
    for i in incidence_matrix[row]:
        count += i
    term_data = Term(terms[row],count)   #put term and count into struct
    node = Node(term_data)               #put struct into node
    head_container.append(node)          #put node into head_container
    for j in range(0,len(incidence_matrix[row])):
        if incidence_matrix[row][j]==1:
            temp_node = Node(j)
            node.next=temp_node
            node = node.next



# In[369]:

for i in head_container:
    print "[", i.data.term, i.data.freq,"] ->",
    node = i.next
    while node is not None:
        print node.data,"->",
        node = node.next
    print "//\n"


