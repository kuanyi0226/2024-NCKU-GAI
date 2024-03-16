import numpy as np

#practice1: softmax function

# 設定總共類別
c = 10                    
# 模擬輸出 logits
x = np.random.rand(c)
print('logits:')  
print(x) #print logits

#todo
exp_x = np.exp(x)
softmax_x =  exp_x/exp_x.sum() #softmax formula

print('softmax result:')
print(softmax_x)

