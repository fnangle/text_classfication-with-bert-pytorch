import sys

bi_candi=sys.argv[1]
num=sys.argv[2]
bi_res=bi_candi+'.res'
with open(bi_candi,encoding='utf-8')as fr1,open(num,encoding='utf-8') as fr2,open(bi_res,'w',encoding='utf-8')as fw:
    nums=fr2.readlines()
    nums=[ str(num).strip() for num in nums]
    lines=fr1.readlines()
    for i,line in enumerate(lines):
        if str(i) in nums:
            fw.write(line)