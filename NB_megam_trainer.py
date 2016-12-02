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
reviews = []
featureList = []

cursor_train = db.final_data3.find({}, { "_id": 0 }).limit(10000)

def removeDuplicatesFromList(myList)
    newList = list(set(myList))
    return newList

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

for row in cursor_train:
    sentiment = row['restaurant']['verdict']
    review = row['restaurant']['review'].lower()
    review = re.sub('[\s]+', ' ', review)
    review = re.sub('[^A-Za-z0-9\s]+', '', review)
    reviewfeatureVector = generateFeatureVector(review)
    featureList.extend(reviewfeatureVector)
    reviews.append((reviewfeatureVector, sentiment));

featureList = removeDuplicatesFromList(featureList)
trainFeatureSet = nltk.classify.util.apply_features(fetch_features, reviews)

#Code for Maximum Entropy Classifier using MEGAM algorithm
megamClassifier = nltk.classify.maxent.MaxentClassifier.train(trainFeatureSet, 'MEGAM', trace=3,encoding=None, labels=None, gaussian_prior_sigma=0, max_iter = 50)
f = open('megamClassifier.pickle', 'wb')
pickle.dump(megamClassifier, f)
f.close()

#Code for Naive Bayes Classifier
naiveBayesClassifier = nltk.NaiveBayesClassifier.train(trainFeatureSet)
f = open('naiveBayesClassifier.pickle', 'wb')
pickle.dump(naiveBayesClassifier, f)
f.close()

#Dumping featureList
f = open('featureList.pickle', 'wb')
pickle.dump(featureList, f)
f.close()