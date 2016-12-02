import nltk
import re
import csv
import sklearn.svm as svm
from pymongo import MongoClient
from svmutil import *
import pickle

stopWords = ['a','able','about','across','after','all','almost','also','am','among','an','and','any','are','as','at','be','because','been','but','by','can','cannot','could','dear','did','do','does','either','else','ever','every','for','from','get','got','had','has','have','he','her','hers','him','his','how','however','i','if','in','into','is','it','its','just','least','let','like','likely','may','me','might','most','must','my','neither','no','nor','not','of','off','often','on','only','or','other','our','own','rather','said','say','says','she','should','since','so','some','than','that','the','their','them','then','there','these','they','this','tis','to','too','twas','us','wants','was','we','were','what','when','where','which','while','who','whom','why','will','with','would','yet','you','your']
client = MongoClient()
db = client.test
featureList = []
count_pos = 0
count_neg = 0
mem_pos = 0.0
mem_neg = 0.0
correct_pos_mem = 0.0
correct_neg_mem = 0.0

cursor_dev = db.final_data3.find({}, { "_id": 0 }).limit(5000)

#Feature Vector Generation
def generateFeatureVector(review): 
    reviewfeatureVector = []
    review_words = review.split()
    for word in review_words:
        if (word in stopWords):
            continue
        else:
            final_word = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", word)
            if(final_word is None):
                continue
            else:
                    reviewfeatureVector.append(word)
    return reviewfeatureVector

def fetch_features(review):
    review_words = set(review)
    allfeatures = {}
    for word in featureList:
        if word in review_words : 
            allfeatures[word] = True
        else:
            allfeatures[word] = False
    return allfeatures

f = open('megamClassifier.pickle', 'rb')
megamClassifier = pickle.load(f)
f.close()

f = open('featureList.pickle', 'rb')
featureList = pickle.load(f)
f.close()

#Prediction the sentiment using MegaM
for row in cursor_dev:
    review = row['restaurant']['review']
    if row['restaurant']['verdict'] == "Positive":
        count_pos += 1.0
    else:
        count_neg += 1.0
    review = re.sub('[\s]+', ' ', review)
    review = re.sub('[^A-Za-z0-9\s]+', '', review)
    featureVector = generateFeatureVector(review)
    c = megamClassifier.classify(fetch_features(featureVector))
    if c == "Positive":
        mem_pos += 1.0
    else:
        mem_neg += 1.0
    if row['restaurant']['verdict'] == c and c == "Positive":
        correct_pos_mem += 1.0
    elif row['restaurant']['verdict'] == c and c == "Negative":
        correct_neg_mem += 1.0

precision_pos = float(correct_pos_mem/mem_pos)
precision_neg = float(correct_neg_mem/mem_neg)
recall_pos = float(correct_pos_mem/count_pos)
recall_neg = float(correct_neg_mem/count_neg)
print "PRECISION POS", precision_pos
print "PRECISION NEG", precision_neg
print "RECALL POS", recall_pos
print "RECALL NEG", recall_neg