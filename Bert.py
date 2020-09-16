from transformers import BertTokenizer, BertModel
import torch
path='bert_base_uncased/'
config_dir=path
tokenizer = BertTokenizer.from_pretrained(path)

model = BertModel.from_pretrained(path)

texts = ["Replace me by any text you'd like."]
tokenized_text = [tokenizer.tokenize(i) for i in texts]
print(tokenized_text)
#[['replace', 'me', 'by', 'any', 'text', 'you', "'", 'd', 'like', '.']]

input_ids = [tokenizer.convert_tokens_to_ids(i) for i in tokenized_text]
print(input_ids)
#[[5672, 2033, 2011, 2151, 3793, 2017, 1005, 1040, 2066, 1012]]

encoded_input = tokenizer(texts, return_tensors='pt')
#{'input_ids': tensor([[ 101, 5672, 2033, 2011, 2151, 3793, 2017, 1005, 1040, 2066, 1012,  102]]), 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])}

output = model(**encoded_input)
# print(output)

def encode_fulltext(text_list):
    all_input_ids=[]
    for text in text_list:
        input_ids=tokenizer.encode(text,add_special_tokens=True,max_length=120,pad_to_max_length=True,return_tensors='pt')
        all_input_ids.append(input_ids)
    # all_input_ids=torch.cat(all_input_ids, dim=0) #按维数0拼接（竖着拼）  按维数1拼接（横着拼）
    return all_input_ids

texts = ["Replace me by any text you'd like.",'I am chinese.']
all_input_ids=encode_fulltext(texts)
print(all_input_ids)
