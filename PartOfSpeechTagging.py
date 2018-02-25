'''
加入整个句子
根据上下文（前一个单词）判断目标单词的词性
accuracy约0.789
'''
from nltk.corpus import brown
import nltk
#特征提取，i是句子中第几个单词，从0开始
def pos_features(sentence,i):

    features={'suffix(1)':sentence[i][-1:],
              'suffix(2)':sentence[i][-2:],
              'suffix(3)':sentence[i][-3:]}
    if i == 0:
        features['prev_word']='start'
    else:
        features['prev_word']=sentence[i-1]
    return features
#语料库中自带的，句子中每一个单词都带有词性
tagged_sents = brown.tagged_sents(categories='news')
#构建特征集合
featuresets = []
for tagged_sent in tagged_sents:
    #去掉句子中单词的词性
    untagged_sent = nltk.tag.untag(tagged_sent)
    for i,(word,tag) in enumerate(tagged_sent):
        featuresets.append((pos_features(untagged_sent,i),tag))
#划分训练集和测试集
size = int(len(featuresets)*0.1)
train_set,test_set = featuresets[size:],featuresets[:size]
#训练决策树
classfier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classfier,test_set))
