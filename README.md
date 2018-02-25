# nltk-Python
simple programs about nltk using python
## genderClassfication:
使用Python nltk中names语料库，给定英文姓名，分类为male或femal  
使用简单贝叶斯的分类器，未添加dev_set测试
## DocumentClaasification:
还是用python nltk中的语料库，movie_reviews，根据文档中出现的高频词汇对文档进行分类，特征是是否包括一些高频词汇
##PartOfSpeechTagging:
使用python nltk中语料库brown，根据上下文（句子中目标单词的前一个单词）来判断目标单词的词性；  
特征提取包括后缀和句子中的前一个单词  
语料库中的tagged_sents提供了每一个句子中每一个单词的词性，可以用于训练和检验  
使用 nltk.NaiveBayesClassifier训练
