import os
import random
vocab=[]  #词表
"""
将原始的文件处理成 评论列表和标签列表
"""
# 从pos以及neg样例中共抽取25000个样本
imdb_dir = './aclImdb'
train_dir=os.path.join(imdb_dir,'train')
test_dir=os.path.join(imdb_dir,'test')

def file_list(f_dir):
    labels=[];texts=[]
    for label_type in ['neg','pos']:
        dir_name=os.path.join(f_dir,label_type)
        for fname in os.listdir(dir_name):
            if fname[-4:] =='.txt':
                fo=open(os.path.join(dir_name,fname))
                texts.append(fo.read())
                fo.close()
                if label_type=='pos':
                    labels.append(1)
                else:
                    labels.append(0)
    return texts,labels

train_texts,train_labels=file_list(train_dir)
test_texts,test_labels=file_list(test_dir)
print(train_labels[:3],test_labels[-3:])
# 由于之前我们处理数据的时候得到的数据集前12500个是neg样本后12500个是pos样本，因此我们需要将其随机打乱：
random.seed(1)
idx=[i for i in range(len(train_texts))]
# print(idx[-3:])
random.shuffle(idx)
# print(len(idx),len(texts),len(labels))

x=[]    #打乱后的文本列表
y=[]    #打乱后对应的标签列表
#x,y对应评论和标签的列表，已打乱
for id in idx:
    x.append(train_texts[id])
    y.append(train_labels[id])
# x=texts
# y=labels
print(x[-1:],y[-1:])

TRAINSET_SIZE = 25000
TESTSET_SIZE = 25000

train_samples = x[:TRAINSET_SIZE]
train_labels = y[:TRAINSET_SIZE]

test_samples = test_texts[:TESTSET_SIZE]  #测试集不用打乱
test_labels = test_labels[:TESTSET_SIZE]
# print(eval_labels)









