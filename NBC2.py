# coding: utf-8
from __future__ import print_function, unicode_literals
import os
import time
import random
import jieba
import jieba.analyse
import numpy as np
from collections import defaultdict
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB


# 主要用来获取停用词
def MakeWordsSet(words_file):
    words_set = set()
    with open(words_file, 'r') as fp:
        for line in fp.readlines():
            word = line.strip()
            if len(word) > 0 and word not in words_set:  # 去重
                words_set.add(word)
    return words_set


def TextProcessing(folder_path, test_size=0.2):
    folder_list = os.listdir(folder_path)
    data_list = []
    class_list = []

    # 类间循环
    for folder in folder_list:
        new_folder_path = os.path.join(folder_path, folder)
        # 输出相关路径和时间
        print("路径 = ", new_folder_path, time.asctime((time.localtime(time.time()))))
        files = os.listdir(new_folder_path)
        # 类内循环
        for file in files:
            with open(os.path.join(new_folder_path, file), 'r') as fp:
                raw = fp.read()
            word_cut = jieba.cut(raw, cut_all=False)  # 精确模式，返回的结构是一个可迭代的genertor
            word_list = list(word_cut)  # genertor转化为list，每个词unicode格式
            data_list.append(word_list)
            class_list.append(folder)

            # 划分训练集和测试集
    data_class_list = list(zip(data_list, class_list))
    # 返回随机排列后的序列，没有返回值，会直接修改data_class_list
    random.shuffle(data_class_list)
    index = int(len(data_class_list) * test_size) + 1  # 获取部分序列位置（index） (train:test)4 : 1
    train_list = data_class_list[index:]
    test_list = data_class_list[:index]
    train_data_list, train_class_list = zip(*train_list)
    test_data_list, test_class_list = zip(*test_list)

    # 统计词频放入all_words
    # all_words_dict = {}
    all_words = ''
    for word_list in train_data_list:
        for word in word_list:
            all_words += word
            # key函数利用词频进行降序排序
    # 内建函数sorted参数需为list
    # all_words_tuple_list = sorted(all_words_dict.items(), key=lambda f:f[1], reverse=True)
    # all_words_list = list(zip(*all_words_tuple_list))[0]
    return all_words, train_data_list, test_data_list, train_class_list, test_class_list


def words_dict(all_words_list, deleteN, stopwords_set=set()):
    # 选取特征词
    feature_words = []
    n = 1
    for t in range(deleteN, len(all_words_list), 1):
        if n > 2000:  # feature_words的维度1500
            break
        if not all_words_list[t].isdigit() and all_words_list[t] not in stopwords_set and 1 < len(
                all_words_list[t]) < 5:
            feature_words.append(all_words_list[t])
            n += 1
    return feature_words


def TextFeatures(train_data_list, test_data_list, feature_words, flag='nltk'):
    def text_features(text, feature_words):
        text_words = set(text)
        features = [1 if word in text_words else 0 for word in feature_words]
        return features

    train_feature_list = [text_features(text, feature_words) for text in train_data_list]
    test_feature_list = [text_features(text, feature_words) for text in test_data_list]
    return train_feature_list, test_feature_list


def TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list):
    ## sklearn分类器
    ## 构建朴素贝叶斯模型
    model = GaussianNB()
    model.fit(train_feature_list, train_class_list)
    ## 使用测试集进行测试
    expected = train_class_list
    predicted = model.predict(test_class_list)
    # 输出测试效果
    print(metrics.classification_report(expected, predicted))
    print(metrics.confusion_matrix(expected, predicted))
    exit()
    test_accuracy = classifier.score(test_feature_list, test_class_list)
    return test_accuracy


if __name__ == '__main__':
    print('STARTING TIME : ', time.asctime((time.localtime(time.time()))))

    # 文本预处理
    folder_path = './Database/SogouC/Sample'
    all_words, train_data_list, test_data_list, train_class_list, test_class_list = TextProcessing(folder_path,
                                                                                                   test_size=0.2)

    # 生成stopwords_set
    # stopwords_file = './stopwords.txt'
    # stopwords_set = MakeWordsSet(stopwords_file)

    ## 文本特征提取和分类
    # feature_words = words_dict(all_words_list, 20, stopwords_set)
    feature_words = jieba.analyse.extract_tags(all_words, 1500)
    train_feature_list, test_feature_list = TextFeatures(train_data_list, test_data_list, feature_words)
    test_accuracy = TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list)
    print('准确率 : ', test_accuracy * 100, '%')
    print('ENDING TIME : ', time.asctime((time.localtime(time.time()))))
    print("finished")
