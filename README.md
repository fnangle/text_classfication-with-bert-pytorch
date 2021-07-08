# text_classfication
nlp text classification task program on IMDB dataset

## 数据集
共50000条电影评论，正负各一半，训练和测试各一半。
下载：http://ai.stanford.edu/~amaas/data/sentiment/

## 模型说明：
大概是自己nlp学习的一点记录，去年用比较经典的模型写的文本分类，比较粗糙，没有特意调参：  

### TextCNN-pytorch实现、keras实现

文件说明：

- TextCNN_keras.py 用tf2中的keras写的CNN文本分类，acc不太高. 训练集划出来0.1的比例作为验证集
- TextCNN.py 是pytorch版本，用了glove词向量，测试时把训练那部分注释掉
- TextCNN_parallel.py 用分布式并行的相关代码改造上个文件，提升了训练的效率（约5倍），详见博客：https://www.cnblogs.com/yh-blog/p/12877922.html

### Bert-pytorch实现

文件说明：

- process_imdb.py 是数据处理函数，把评论文本和标签转为list格式
- Bert.py 是主函数
- test_bert.py 用来测试模型accuracy

借助了huggingface的开源仓库 https://github.com/huggingface/transformers. 

官方fine-tune参数推荐 https://github.com/google-research/bert

- batch sizes: 8, 16, 32, 64, 128
- learning rates(Adam): 3e-4, 1e-4, 5e-5, 3e-5
- Number of epochs: 3, 4

## Results

| result  | acc |       
| :----: | :----: |
| TextCNN（keras）  | 87.27% |
| Bert  | 95.18%| 



## Bert-classification-new说明：

预训练模型：bert_base_multilingual_cased

重构了用mbert分类的代码，可以用自己的数据训练得到的分类模型，根据分类预测结果，卡阈值选取句子，注意这里是二分类。已测试候选句量级为40w时能正常运行

- process：数据处理函数，把评论文本和标签转为list格式，注意训练和测试accuracy的文件都在代码中指定，自行修改,其中outdomain代表训练的负样本，indomain代表训练的正样本。

- classify：主函数，用于训练，用默认参数时，训练脚本:

  `CUDA_VISIBLE_DEVICES=1 python classify.py`

  也可以自己定义参数

- test.py：用来测试模型accuracy或者用模型进行句子的筛选

  测acc：
  `CUDA_VISIBLE_DEVICES=1 python test.py`

  筛选句子：
  `CUDA_VISIBLE_DEVICES=1 python test.py --is-select True --file candidate`
