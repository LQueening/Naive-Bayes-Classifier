# coding: utf-8
import os
import time
import random
import jieba
import sklearn
from sklearn.naive_bayes import MultinomialNB
import numpy as np
# import pylab as pl
import matplotlib.pyplot as plt


# 获取停用词
def MakeWordsSet(words_file):
    words_set = set()
    with open(words_file, 'r') as fp:
        for line in fp.readlines():
            word = line.strip().decode("utf-8")
            if len(word) > 0 and word not in words_set:  # 去重
                words_set.add(word)
    return words_set


# 使用结巴分词对文本进行分词
def cutWordByJieba(new_text):
    word_cut = jieba.cut(new_text, cut_all=False)
    word_list = list(word_cut)
    return word_list


# 遍历目录获取所有新闻，处理各分类文件夹中的各篇新闻
def TextProcessing(folder_path, test_size=0.2):
    folder_list = os.listdir(folder_path)
    data_list = []
    class_list = []

    # 类间循环
    for folder in folder_list:
        new_folder_path = os.path.join(folder_path, folder)
        files = os.listdir(new_folder_path)
        # 类内循环
        j = 1
        for file in files:
            if j > 300:
                break
            with open(os.path.join(new_folder_path, file), 'r') as fp:
                raw = fp.read()
            ## ---------------------------------------------------------------------------------------------------------
            word_list = cutWordByJieba(raw)
            # print word_list
            ## ---------------------------------------------------------------------------------------------------------
            data_list.append(word_list)
            class_list.append(folder.decode('utf-8'))
            j += 1
    ## 划分训练集和测试集
    data_class_list = zip(data_list, class_list)
    random.shuffle(data_class_list)
    index = int(len(data_class_list) * test_size) + 1
    train_list = data_class_list[index:]
    train_data_list, train_class_list = zip(*train_list)

    # 统计词频放入all_words_dict
    all_words_dict = {}
    for word_list in train_data_list:
        for word in word_list:
            if all_words_dict.has_key(word):
                all_words_dict[word] += 1
            else:
                all_words_dict[word] = 1

    # key函数利用词频进行降序排序
    all_words_tuple_list = sorted(all_words_dict.items(), key=lambda f: f[1], reverse=True)  # 内建函数sorted参数需为list
    all_words_list = list(zip(*all_words_tuple_list)[0])

    return data_class_list, all_words_list, train_data_list, train_class_list


# 随机获取测试集数据
def getTestDataByRandom(data_class_list, test_size=0.2):
    random.shuffle(data_class_list)
    index = int(len(data_class_list) * test_size) + 1
    test_list = data_class_list[:index]

    # 输出测试集数据
    print('-------------------------------------------------')
    for i in test_list:
        print("".join(i[0]))
        print ('-------------------------------------------------')

    test_data_list, test_class_list = zip(*test_list)

    return test_data_list, test_class_list


# 从训练集分词结果中去除停用词，选取特征词
def words_dict(all_words_list, stopwords_set=set()):
    feature_words = []
    n = 1
    for t in range(0, len(all_words_list), 1):
        if n > 2000:  # feature_words的维度
            break
        if not all_words_list[t].isdigit() and all_words_list[t] not in stopwords_set and 1 < len(
                all_words_list[t]) < 5:
            feature_words.append(all_words_list[t])
            n += 1
    return feature_words


# 获取文本特征集
def getNewsFeatures(news_data, feature_words):
    def text_features(text, feature_words):
        text_words = set(text)
        ## sklearn特征 list
        features = [1 if word in text_words else 0 for word in feature_words]
        return features

    new_features = [text_features(text, feature_words) for text in news_data]
    return new_features


# 创建贝叶斯分类器
def createClassifier(train_feature_list, train_class_list):
    classifier = MultinomialNB().fit(train_feature_list, train_class_list)
    return classifier


# 对提取出特征集的文本进行分类
def TextClassifier(new_classifier, test_feature_list, test_class_list):
    # ------------------------------------------------------------------------------------------------------------------
    # 输出分类结果
    predictRes = new_classifier.predict(test_feature_list)
    print('测试集：')
    print predictRes
    exportTypeByRes(predictRes)
    print ('测试集新闻条数为：')
    print (len(predictRes))
    print('准确率为：')
    print (new_classifier.score(test_feature_list, test_class_list))
    print('***********************************************************************************************************')
    # print('输入任意字符继续程序')
    # continue_str = raw_input()
    # ------------------------------------------------------------------------------------------------------------------
    test_accuracy = new_classifier.score(test_feature_list, test_class_list)
    return test_accuracy


# 对文本特征集进行类别的预测
def predictNewsType(new_classifier, new_feature):
    predictRes = new_classifier.predict(new_feature)
    print predictRes
    exportTypeByRes(predictRes)
    return predictRes


# 由于文件夹对应类别为一串字符，需要按照最后两位输出对应的中文类别
def exportTypeByRes(preidct_res):
    for res in preidct_res:
        tempStr = str(res)
        if tempStr.find('08') > -1:
            print ('财经')
        elif tempStr.find('10') > -1:
            print ('IT')
        elif tempStr.find('13') > -1:
            print ('健康')
        elif tempStr.find('14') > -1:
            print ('体育')
        elif tempStr.find('16') > -1:
            print ('旅游')
        elif tempStr.find('20') > -1:
            print ('教育')
        elif tempStr.find('22') > -1:
            print ('招聘')
        elif tempStr.find('23') > -1:
            print ('文化')
        elif tempStr.find('24') > -1:
            print ('军事')


# 对外部输入的新闻文本预测类别
def inputNewsClassifier():
    print '请输入新闻文本：'
    news_text = raw_input()
    news_cut_word = cutWordByJieba(news_text)
    testTuple = (news_cut_word,)
    news_text_feature = getNewsFeatures(testTuple, feature_words)
    print('----------------  result  -----------------------')
    res = predictNewsType(newsClassifier, news_text_feature)
    print('输入任意字符继续新闻分类，长度超过3结束')
    continue_input = raw_input()
    if (len(continue_input) < 3):
        inputNewsClassifier()
    else:
        return


# 入口
if __name__ == '__main__':
    # 对自带的文本进行测试集和训练集的分类，同时根据训练集对测试集进行分类，验证结果
    print "start"
    ## 文本预处理
    folder_path = './Database/SogouC/Sample'
    data_class_list, all_words_list, train_data_list, train_class_list = TextProcessing(folder_path, test_size=0.02)
    # ------------------------------------------------------------------------------------------------------------------
    # 测试集的分词结果
    # for cut_res in test_data_list[0]:
    #     print(cut_res)
    # exit()
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # 获取停用词
    stopwords_file = './stopwords_cn.txt'
    stopwords_set = MakeWordsSet(stopwords_file)
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    ## 文本特征提取和分类
    # test_accuracy_list = []
    feature_words = words_dict(all_words_list, stopwords_set)

    # 去除停用词之后按词频降序排序的词集
    # for feature_word in feature_words:
    #     print feature_word
    # exit()

    train_feature_list = getNewsFeatures(train_data_list, feature_words)
    newsClassifier = createClassifier(train_feature_list, train_class_list)

    # 对测试集进行分类
    test_accuracy_list = []
    accuracy_sum = 0
    test_counts = range(0, 20, 1)
    for test_count in test_counts:
        test_data_list, test_class_list = getTestDataByRandom(data_class_list, test_size=0.02)
        test_feature_list = getNewsFeatures(test_data_list, feature_words)
        test_accuracy = TextClassifier(newsClassifier, test_feature_list, test_class_list)
        test_accuracy = "%.2f" % test_accuracy
        accuracy_sum += float(test_accuracy)
        test_accuracy_list.append(test_accuracy)

print test_accuracy_list
print('平均准确率为：')
print ("%.2f" % (accuracy_sum / 20))
# ------------------------------------------------------------------------------------------------------------------
# 对外部输入的文本进行分类
# inputNewsClassifier()
# exit()
# --------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------------------
# 结果评价
plt.figure()
plt.plot(test_counts, test_accuracy_list)
plt.title('test accuracy')
plt.xlabel('test_time')
plt.ylabel('accuracy')
plt.savefig('sort_result.png')
plt.show()
# ------------------------------------------------------------------------------------------------------------------

print "finished"
