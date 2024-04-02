import numpy as np

#practice2: Linear Layer + ReLU Activation
# 設定輸入維度
d_in = 10                                
# 設定輸出維度
d_out = 30                               

# 模擬神經網路輸入
x = np.ones((d_in, 1))                   
# 模擬神經網路權重
W = np.random.rand(d_out, d_in) * 10 - 5 
print('模擬神經網路輸入:')
print(x)
# 模擬神經網路偏差值
b = np.random.rand(d_out, 1) * 10 - 5  
# todo
Wx_b = (W @ x) + b
result = np.maximum(0, Wx_b)
print('result:')
print(result)