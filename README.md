# text_classfication
nlp text classification task program on IMDB dataset

## 数据集
共50000条电影评论，正负各一半，训练和测试各一半。
下载：http://ai.stanford.edu/~amaas/data/sentiment/

## 模型说明：
大概是自己nlp学习的一点记录，去年用比较经典的模型来做文本分类，写的比较粗糙，没有特意调参：  

### TextCNN-pytorch实现、keras实现

文件说明：

- TextCNN.py 是整体文件，测试时把训练那部分注释掉
- TextCNN_parallel.py 用分布式并行的相关代码改造上个文件，提升了训练的效率（约5倍），详见博客：https://www.cnblogs.com/yh-blog/p/12877922.html

### Bert-pytorch实现

文件说明：

- process_imdb.py 是数据处理函数，把评论文本和标签转为list格式
- Bert.py 是主函数
- test_bert.py 用来测试模型accuracy

借助了huggingface的开源仓库 https://github.com/huggingface/transformers. 

官方fine-tune参数推荐

- Batch size: 16, 32
- Learning rate (Adam): 5e-5, 3e-5, 2e-5
- Number of epochs: 3, 4



| result  | acc |       
| :----: | :----: |
| TextCNN  | 87.27% |
| Bert  | 95.18%| 
