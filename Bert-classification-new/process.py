import os
import random

def file_list(dir_name,label):
    texts=[];labels=[]  
    with open(dir_name) as fr:
        for line in fr.readlines():
            if len(line)<=1: #konghang
                continue
            else:
                texts.append(str(line).strip())
                labels.append(label)
    return texts,labels


traindir='.'
devdir='.'

neg_texts,neg_labels=file_list(os.path.join(traindir,'outdomain'),0)
pos_texts,pos_labels=file_list(os.path.join(traindir,'indomain'),1)
train_texts,train_labels=[],[]
train_texts.extend(pos_texts); train_texts.extend(neg_texts)
train_labels.extend(pos_labels); train_labels.extend(neg_labels)

# dev_texts0,dev_labels0=file_list(os.path.join(devdir,'test_neg.ca'),0)
dev_texts1,dev_labels1=file_list(os.path.join(devdir,'test_pos.ca'),1)
dev_texts,dev_labels=[],[]
# dev_texts.extend(dev_texts0); 
dev_texts.extend(dev_texts1)
# dev_labels.extend(dev_labels0); 
dev_labels.extend(dev_labels1)

random.seed(1)
idx=[i for i in range(len(train_texts))]
random.shuffle(idx)

x=[]    
y=[]   

for id in idx:
    x.append(train_texts[id])
    y.append(train_labels[id])
train_samples = x
train_labels = y
# print(train_samples[-3:],train_labels[-3:])

########## test acc or apply  #########
"""test"""
test_samples = dev_texts 
test_labels = dev_labels

# print(len(train_samples),len(train_labels))