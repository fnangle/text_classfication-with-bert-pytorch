# text_classfication
nlp text classification task program on IMDB dataset


## 自己nlp学习的记录：
用比较经典的模型来做文本分类，写的比较粗糙，没有特意调参：  

### TextCNN-keras实现     

### Bert-pytorch实现

借助了huggingface的开源仓库 https://github.com/huggingface/transformers. 

官方fine-tune参数推荐

- Batch size: 16, 32
- Learning rate (Adam): 5e-5, 3e-5, 2e-5
- Number of epochs: 3, 4



| result  | acc |       
| :----: | :----: |
| TextCNN  | 87.27% |
| Bert  | 95.18%| 
