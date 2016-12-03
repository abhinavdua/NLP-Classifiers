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

cursor_dev = db.final_data3.find({}, { "_id": 0 }).limit(5000)

#Below function generates base SVM vector with values of all features set as zero
def generateEmptyfeatureDict()
    featureDict = {}
    for word in featureList
        featureDict[word] = 0
    return featureDict

#Generate final SVM vector for all reviews with word count	
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

f = open('featureList.pickle', 'rb')
featureList = pickle.load(f)
f.close()

reviews =[]
lab = []
for row in cursor_dev:
    review = row['restaurant']['review']
    sent = row['restaurant']['verdict']
    if sent == "Positive":
        lab.append(0.0)
        count_pos += 1
    else:
        lab.append(1.0)
        count_neg += 1
    review = re.sub('[\s]+', ' ', review)
    review = re.sub('[^A-Za-z0-9\s]+', '', review)
    reviews.append(review)
dev_feature_vector = generateSVMFeatures(test_tweets)
class_labels, class_accs, class_vals = svm_predict([0] * len(dev_feature_vector['feature_vector']),dev_feature_vector['feature_vector'], svmClassifier)
correct_pos = 0.0
correct_neg = 0.0
for item, item1 in zip(class_labels, lab):
    if item == item1 and item == 0.0:
        correct_pos += 1.0
    elif item == item1 and item == 1.0:
        correct_neg += 1.0
 
svm_pos = float(class_labels.count(0.0))
svm_neg = float(class_labels.count(1.0))

#Print evaluation metrics
precision_pos = float(correct_pos/svm_pos)
precision_neg = float(correct_neg/svm_neg)
recall_pos = float(correct_pos/count_pos)
recall_neg = float(correct_neg/count_neg)
print "PRECISION POS", precision_pos
print "PRECISION NEG", precision_neg
print "RECALL POS", recall_pos
print "RECALL NEG", recall_neg