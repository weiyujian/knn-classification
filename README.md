# knn-classification
knn text classification

#通过tfidf计算文本相似度，从而预测问句所属类别

#实现过程

#1.根据训练语料（标签\t问句），进行分词，获得（标签\t标签分词\t问句\t问句分词）

#2.根据输入的训练语料分词结果，产生ngram和skipgram的特征，基于此生成tfidf模型

#3.对于测试集，进行分词，获取测试问句的tfidf表征，计算训练语料中与其最相似的topn问句，根据topn问句的标签来预测测试问句的标签

#准备训练语料:

#python get_knn_input.py

#训练过程:

#python knn_classification.py --model_version=knn_model_v3 --is_train=true --input_file=./data/cnews.train.knn_seg

#测试过程:

#python knn_classification.py --model_version=knn_model_v3 --is_train=false --input_file=./data/cnews.test.knn_seg
