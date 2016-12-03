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

#Below function generates base SVM vector with values of all features set as zero
def generateEmptyfeatureDict()
    featureDict = {}
    for word in featureList
        featureDict[word] = 0
    return featureDict
    
def generateSVMFeatures(reviews)
    SVMfeatureDict = generateEmptyfeatureDict() 
    feature_vector =[]
    sentiment_class = []
    for eachReview in reviews : 
        review_statement = eachReview[0]
        sentiment_label = eachReview[1]
        if(sentiment_label == 'Positive'):
            label = 0
        elif(sentiment_label == 'Negative'):
            label = 1
        sentiment_class.append(sentiment_label)
        for eachWord in review_statement :
            if eachWord in SVMfeatureDict : 
                SVMfeatureDict[eachWord] = SVMfeatureDict.get(eachWord) + 1
        feature_vector.append(SVMfeatureDict.values())
    return {'feature_vector' : feature_vector, 'sentiment_class': sentiment_class}

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
            final_word = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", word) #remove words not starting with an alphanumeric character
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

parameters = svm_parameter()
parameters.kernel_type = LINEAR
parameters.C = 10

trainingFeatureVector = generateSVMFeatures(reviews)
problemDefinition = svm_problem(trainingFeatureVector['sentiment_class'], trainingFeatureVector['feature_vector'])

SVMclassifier = svm_train(problemDefinition, parameters)
svm_save_model('svmClassifier', SVMclassifier)