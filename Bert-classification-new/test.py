import torch
import classify 
import os
from transformers import BertForSequenceClassification, AdamW,BertTokenizer,BertModel
import argparse
 # 测试
if __name__=="__main__":
    args = classify.args
    model=BertForSequenceClassification.from_pretrained(args.path)
    model.load_state_dict(torch.load(args.params_dir))
    if args.is_select:
        classify.test_model(model,file=args.file,is_select=args.is_select)
    else:
        classify.test_model(model)