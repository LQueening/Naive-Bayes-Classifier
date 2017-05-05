# coding: utf-8
from __future__ import print_function, unicode_literals
import os
import time
import random
import jieba
import jieba.analyse
import numpy as np
from collections import defaultdict
import sklearn
from sklearn.naive_bayes import MultinomialNB


# 文本预处理，遍历新闻文件进行分词，同时划分测试集和训练集
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
    index = int(len(data_class_list) * test_size) + 1
    train_list = data_class_list[index:]
    test_list = data_class_list[:index]
    train_data_list, train_class_list = zip(*train_list)
    test_data_list, test_class_list = zip(*test_list)

    # 统计词频放入all_words
    all_words = ''
    for word_list in train_data_list:
        for word in word_list:
            all_words += word
    return all_words, train_data_list, test_data_list, train_class_list, test_class_list


# 获取新闻的特征
def getNewsFeatures(news, feature_words):
    print('开始获取新闻特征')
    def text_features(text, feature_words):
        text_words = set(text)
        features = [1 if word in text_words else 0 for word in feature_words]
        return features

    feature_result = [text_features(text, feature_words) for text in news]
    return feature_result


# 新闻分类器
def newsClassifier(train_data, train_type):
    print('构建新闻分类器')
    model = MultinomialNB()
    model.fit(train_data, train_type)
    return model


# 使用分类器进行训练，同时使用训练之后的分类器对测试集进行分类，验证正确率
def TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list):
    print('开始训练')
    classifier = newsClassifier(train_feature_list, train_class_list)
    print('开始分类')
    res = classifier.fit(train_feature_list, train_class_list)
    print('获取结果')
    test_accuracy = res.score(test_feature_list, test_class_list)
    return test_accuracy


if __name__ == '__main__':
    print('STARTING TIME : ', time.asctime((time.localtime(time.time()))))
    # 文本预处理
    folder_path = './Database/SogouC/Sample'
    all_words, train_data_list, test_data_list, train_class_list, test_class_list = TextProcessing(folder_path,
                                                                                                   test_size=0.2)
    ## 文本特征提取和分类
    feature_words = jieba.analyse.extract_tags(all_words, 1500)
    train_feature_list = getNewsFeatures(train_data_list, feature_words)
    test_feature_list = getNewsFeatures(test_data_list, feature_words)
    test_accuracy = TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list)
    print('准确率 : ', test_accuracy * 100, '%')
    print('ENDING TIME : ', time.asctime((time.localtime(time.time()))))
    print("finished")
