lines  = open('./test/6k_benign.txt','r')
a=[]
for line in lines:
    a.append(int(line))
lines2  = open('./test/6k_sybil.txt','r')
b=[]
for line in lines2:
    b.append(int(line))
with open('./test/twitter_edges.txt','w') as f: # 如果filename不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据！
    for i in range(0,len(a)):
        f.write(str(a[i])+' '+'0'+'\n')
    for i in range(0,len(b)):
        f.write(str(b[i])+' '+'1'+'\n')
