# svm-tracin
use svm to analise tracin_score vs distances
使用数据集：iris

使用模型：linear svm（手写）

使用的距离计算：欧氏距离，余弦距离，曼哈顿城市距离，切比雪夫距离，相关因素距离，马氏距离

主要修改的文件：压缩包中linear_classifier.py

epoch:目前跑了50个，每个epoch生成了一个csv文件，文件中包含每对样本间tracin_score和各种距离

svm epoch文档：主要是每轮epoch中svm loss随样本不断输入而不断变换的情况，第一轮epoch表现正常，后续出现过拟合
