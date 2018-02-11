from nltk.corpus import movie_reviews
import nltk
import random
#prepare data set with labels
documents = [(list(movie_reviews.words(fileid)),category) for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)
all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words()) #words,FreqDist() 方法获取到每个单词的出现次数
word_features = list(all_words)[:2000]
#checks whether each of these words is present in a given document.
def document_features(document):
    #The reason that we compute the set of all words in a document in [3], rather than just checking if word in document, is that checking whether a word occurs in a set is much faster than checking whether it occurs in a list (4.7).
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features
#print(document_features(movie_reviews.words('pos/cv957_8737.txt')))
featuresets = [(document_features(d),c) for (d,c) in documents]
train_set =featuresets[100:]
test_set = featuresets[:100]
classfier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classfier,test_set))