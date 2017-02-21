
# coding: utf-8

# In[62]:

#Homework 1 CSCE 470
#2-20-2017
#By: Rodrigo Gomez-Palacio

	#*-------------Note---------------*#
	#Jupyter notebook (visual) outputs
	#can be found at
	#https://github.com/rgomezp/info_storage_and_retrieval
	#*--------------------------------*#

#Imports and Inputs
import nltk
#nltk.download()                          #downloads all nltk packages
import string
import math

documents = [
    "Today the aggies won! Go aggies!", 
    "aggies have won today", 
    "the aggies lost last week", 
    "Find the latest Aggies news", 
    "An Aggie is a student at Texas A&M"
]

documents2 = documents


# In[63]:

#Punctuation removal 
temp = []
for doc in documents:
    tokens = doc.translate(None,string.punctuation)
    temp.append(tokens)
    
documents = temp


# In[64]:

#Tokenize
temp = []
for doc in documents:
    tokens = nltk.word_tokenize(doc)
    temp.append(tokens)
    
documents = temp


# In[65]:

#Case Folding
for doc in documents:
    for term in range(0,len(doc)):
        doc[term] = doc[term].lower()
        


# In[66]:

#Stopword removal
#stopwords from nltk 'corpus' found here: http://www.nltk.org/book/ch02.html
stop = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'])

for doc in range(0,len(documents)):
    new_doc = []
    for term in range(0,len(documents[doc])):
        if documents[doc][term] not in stop:
        #if term is not in stop set, keep
            new_doc.append(documents[doc][term])
    documents[doc] = new_doc


# In[67]:

#Lemmatization using Wordnet Lemmatizer
from nltk.stem.wordnet import WordNetLemmatizer
lmtzr = WordNetLemmatizer()

for doc in documents:
    for term in range(0,len(doc)):
            doc[term] = lmtzr.lemmatize(doc[term])


# In[68]:

#Stemming with Porter Stemmer
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

for doc in documents:
    for term in range(0,len(doc)):
            doc[term] = stemmer.stem(doc[term])

print "---Preprocessing Output---"
for i in documents:
    print i


# In[69]:

#Build  Term-Document Incidence Matrix

#Build set with unique terms
terms = []
for doc in documents:
    for term in doc:
        if term not in terms:
            terms.append(term)
terms.sort()
            
incidence_matrix = []          #cols: documents,  rows: terms

for term in terms:
    row = []
    for doc in documents:
    #if term appears in document: true...else: false
        if term in doc:      
            row.append(1)
        else:
            row.append(0)
    incidence_matrix.append(row)

print "---Incidence matrix---"
for i in incidence_matrix:
    print i


# In[70]:

#Build term-document count matrix (tf_td)

count_matrix = []    

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
print "---Count_matrix:---"
for i in count_matrix:
    print i
    
print count_matrix[0][1]


# In[71]:

#Build document frequency vector
document_freq_vec = []
for i in range(0,len(terms)):
    count = 0
    for doc in documents:
        if terms[i] in doc:
            count+=1
    document_freq_vec.append(count)
    
print "Doc_frequency_vector:\n", document_freq_vec, "\n\n"

#Build idf vector
N = float(len(documents))
idf_vec = []
for num in document_freq_vec:
    weight = math.log10(N/num)
    idf_vec.append(weight)
    
print "Idf_vector:"
for i in idf_vec:
    print i


# In[72]:

#Build Tf-Idf Matrix
tf_idf_matrix = []

for term in range(0,len(terms)):
    row = []
    for doc in range(0,len(documents)):
        weight = math.log10(1.0+float(count_matrix[term][doc]))*idf_vec[term]
        row.append(round(weight,5))
    tf_idf_matrix.append(row)
    
print "---Tf_idf_matrix---" 
for i in tf_idf_matrix:
    print i


# In[73]:

#Log-Frequency Matrix
log_freq_matrix = []
for term_vec in count_matrix:
    row = []
    for i in range(0,len(term_vec)):
        temp = math.log10(1+term_vec[i])
        row.append(temp)
    log_freq_matrix.append(row)      


# In[74]:

#Transpose log_freq_matrix
log_freq_matrix = zip(*log_freq_matrix)

normalized_lfm = []                #normalized log_freq_matrix
#Length Normalization
for doc_vec in log_freq_matrix:
    #Calculate denominator
    sum_of_squares = 0
    for num in doc_vec:
        sum_of_squares += num**2
    denominator = math.sqrt(sum_of_squares)
    new_vec = []
    for num in doc_vec:
        norm_val = num / denominator
        new_vec.append(norm_val)
    normalized_lfm.append(new_vec)



# In[75]:

#Build 5 x 5 cosine similarity matrix
cosine_sim = []

for doc_vec1 in range(0,len(normalized_lfm)):
    row = []
    for doc_vec2 in range(0,len(normalized_lfm)):
        if doc_vec1 == doc_vec2:
            row.append(1)
        elif doc_vec1 < doc_vec2:                            #symmetric matrix, don't waste operations on half the mtrx
            score = 0
            for i in range(0,len(normalized_lfm[doc_vec1])):
                score += (normalized_lfm[doc_vec1][i]*normalized_lfm[doc_vec2][i])
            row.append(round(score,3))
        else:
            row.append(0)
    cosine_sim.append(row)
    
#Because symmetric matrix, copy over
for doc_vec1 in range(0,len(normalized_lfm)):
    for doc_vec2 in range(0,len(normalized_lfm)):
        if doc_vec1 > doc_vec2:
            cosine_sim[doc_vec1][doc_vec2] = cosine_sim[doc_vec2][doc_vec1]

print "---Cosine Similarity Matrix---"
for row in range(0,len(cosine_sim)):
    print row+1,cosine_sim[row]
print "\nDocuments"
for doc in range(0,len(documents2)):
    print doc+1,":", documents2[doc]


# In[76]:

#Part 2------------(OPTIONAL)
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



# In[77]:

for i in head_container:
    print "[", i.data.term, i.data.freq,"] ->",
    node = i.next
    while node is not None:
        print node.data,"->",
        node = node.next
    print "//\n"


# In[ ]:



