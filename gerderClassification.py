from nltk.corpus import names
from  nltk.classify import apply_features
import nltk
labeled_names = ([(name,'male')for name in names.words('male.txt')]+[(name,'female')for name in names.words('female.txt')])
import random
random.shuffle(labeled_names)
def gender_features(word):
    return {'last_letter':word[-1]}
#featuresets = [(gender_features(n),gender) for (n,gender) in labeled_names]
#train_set,test_set = featuresets[500:],featuresets[:500]

train_set = apply_features(gender_features,labeled_names[500:])
test_set = apply_features(gender_features,labeled_names[:500])
classfier = nltk.NaiveBayesClassifier.train(train_set)
print(classfier.classify(gender_features('Neo')))
print(classfier.classify(gender_features('Trinity')))
print(nltk.classify.accuracy(classfier,test_set))
classfier.show_most_informative_features(5)

