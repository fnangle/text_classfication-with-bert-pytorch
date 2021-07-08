import process
from torch.utils.data import TensorDataset,DataLoader
import torch
import torch.nn as nn
from torch import optim
import logging
import numpy as np
import pickle
import os
import torch.nn.functional as F
# import matplotlib.pyplot as plt
from transformers import BertForSequenceClassification, AdamW,BertTokenizer,BertModel
from transformers import get_linear_schedule_with_warmup
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--maxlen',type=int,default=512,help='sentence length in train files')
parser.add_argument('--batch-size',type=int,default=16)
parser.add_argument('--lr',type=float,default=3e-5)
parser.add_argument('--epoch',type=int,default=4)
parser.add_argument('--train-dir',default='train_ids.pkl',help="save train file in pkl format,convenient for loading train files in next time")
parser.add_argument('--path',default='bert_base_multilingual_cased',help="pretrained model path")
parser.add_argument('--params-dir',default='mbert_base_bs16_beta.pkl',help="save trained model params to this file")
parser.add_argument('--num-labels',type=int,default=2,help="nums of labels for classification")

parser.add_argument('--is-select',default=False,help='test or apply select')
parser.add_argument('--treshold',type=float,default=0.7,help="treshold for select sentence of binary classification")
parser.add_argument('--file',default=None,help='set candidate file for selecting')
args = parser.parse_args()

MAXLEN=args.maxlen - 2
batchsize = args.batch_size 

path=args.path
tokenizer = BertTokenizer.from_pretrained(path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def convert_text_to_ids(tokenizer,sentence,limit_size=MAXLEN):
    
    t=tokenizer.tokenize(sentence)[:limit_size]
    # print(len(sentence),len(t))
    encoded_ids= tokenizer.encode(t) 
    if len(encoded_ids)<limit_size+2:
        tmp=[0]*(limit_size+2-len(encoded_ids))
        encoded_ids.extend(tmp)
    return encoded_ids

# def convert_ids_to_text(tokenizer,ids,limit_size=MAXLEN):
#     t=ids[:limit_size]
#     tokens= tokenizer.convert_ids_to_tokens(t) 
#     if len(tokens)<limit_size+2:
#         tmp=[0]*(limit_size+2-len(tokens))
#         tokens.extend(tmp)
#     return tokens
    

def attention_masks(input_ids):
    atten_masks = []
    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        atten_masks.append(seq_mask)
    return atten_masks

def predict(logits):
    res=torch.argmax(logits,dim=1)  #
    probs=F.softmax(logits,dim=1)   #binary classification prob
    return res,probs

def train_model(net, epoch=args.epoch,lr=args.lr,train_pkl=args.train_dir):
    # ------------------------------
    if os.path.exists(train_pkl):
        with open(train_pkl,'rb') as fr:
            input_ids=pickle.load(fr)
    else:
        input_ids= [convert_text_to_ids(tokenizer,sen) for sen in process.train_samples]
        with open(train_pkl,'wb') as fw:
            pickle.dump(input_ids,fw)
    # input_ids= [convert_text_to_ids(tokenizer,sen) for sen in process.train_samples]
    input_labels = process.train_labels
    atten_token_train=attention_masks(input_ids)
    train_set = TensorDataset(torch.LongTensor(input_ids),torch.LongTensor(atten_token_train),torch.LongTensor(input_labels))
    train_loader = DataLoader(dataset=train_set,
                          batch_size=batchsize,
                          shuffle=True,
                          num_workers=4
                          )

    for i, (train,mask, label) in enumerate(train_loader):
        print(train.shape,mask.shape, label.shape)  ##torch.Size([8, 512]) torch.Size([8,512]) torch.Size([8, 1])
        break
    # --------------------------------
    avg_loss = []
    net.train()  # 
    net.to(device)
    optimizer = AdamW(net.parameters(), lr)

    accumulation_steps = 8
    for e in range(args.epoch):
        for batch_idx, (data, mask, target) in enumerate(train_loader):
            # optimizer.zero_grad()
            data, mask, target = data.to(device), mask.to(device), target.to(device)
            output = net(input_ids=data, token_type_ids=None, attention_mask=mask, labels=target)
            #logits
            loss,logits=output[0],output[1]
            loss = loss / accumulation_steps  # 
            avg_loss.append(loss.item())
            loss.backward()

            if ((batch_idx + 1) % accumulation_steps) == 0:
                optimizer.step()
                optimizer.zero_grad()

            if batch_idx % 10 == 0:
                logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss:{:.6f}'.format(
                    e + 1, batch_idx, len(train_loader), 100. *
                    batch_idx / len(train_loader), np.array(avg_loss).mean()
                ))

    print('Finished Training')
    return net


def test_model(net,file=None,is_select=False,output=None):
    batchsize=64
    #--------------------------------
    if is_select:   #to select data
        assert file is not None
        test_samples,test_labels=process.file_list(file,-1)
    else:   # to test acc
        test_samples,test_labels=process.test_samples,process.test_labels
    print(len(test_samples),len(test_labels))
    input_ids2= [convert_text_to_ids(tokenizer,sen) for sen in test_samples]
    input_labels2 = torch.unsqueeze(torch.tensor(test_labels),dim=1)

    atten_tokens_eval=attention_masks(input_ids2)
    test_set = TensorDataset(torch.LongTensor(input_ids2),torch.LongTensor(atten_tokens_eval), torch.LongTensor(input_labels2))
    test_loader = DataLoader(dataset=test_set,
                            batch_size=batchsize, 
                            num_workers=4)
    for i, (train,mask, label) in enumerate(test_loader):
        print(train.shape,mask.shape, label.shape)               #
        break
    #--------------------------------
    net.eval()
    net=net.to(device)
    correct=0
    total=0
    with torch.no_grad():
        if is_select:
            if not output:
                output=file+".res"
            fw=open(output,'w')
        count = 0
        for batch_idx, (data,mask,label) in enumerate(test_loader):
            # logging.info("test batch_id=" + str(batch_idx))
            data, mask ,label =data.to(device), mask.to(device), label.to(device)
            output = net(input_ids=data, token_type_ids=None, attention_mask=mask) #
            # print(output[0].size(),label.shape)
            total += label.size(0)  #
            res , probs=predict(output[0])[0],predict(output[0])[1]
            probs_pos=probs[:,1]
            if is_select:
                for i,value in enumerate(probs_pos):
                    if value>args.treshold:
                        fw.write(str(count+i)+"\t"+str(value.item())+"\t"+test_samples[count+i])
                        fw.write('\n')
            count+=len(probs_pos) 
            if not is_select:
                correct += ( res == label.flatten()).sum().item()
        if is_select:
            fw.close()
            print('Finished Writing')
        else:
            print(f'correct: {correct} all-sum: {total} accuracy: {100.*correct/total:.3f}%')


if __name__=="__main__":
    logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.INFO)

    pre_net=BertForSequenceClassification.from_pretrained(args.path)    
    model=train_model(pre_net,args.num_labels)
    torch.save(model.state_dict(), args.params_dir)  
