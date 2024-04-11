# 基于SVD和SGNS的词向量学习实验

```
.
│  2021213382.pdf
│  2021213382.txt
│  list.txt
│  README.md
│  
├─code
│      data_pre.py
│      model.py
│      sgns.py
│      svd.py
│      svd_sparse.py
│      
├─data
│      pku_sim_test.txt
│      stopwords.txt
│      training.txt
│      
└─result
```
## 1.实验报告和导出相似度文件
    分别为2021213382.pdf和2021213382.txt
## 2.SVD代码
    svd.py为过滤了低频词后的svd代码，用于计算奇异值数量等信息
    svd_saprse.py为使用了稀疏矩阵后的代码
## 3.SGNS代码
    data_pre.py：通过对数据集training.txt预处理，得到正负样本对
    model.py：SGNS的网络模型类
    sgns.py：对SGNS网络进行训练和测试


