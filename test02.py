import numpy as np

a = np.arange(56).reshape(7,2,2,2)
b = np.reshape(a,(-1,8))
print(a.shape)
print(b.shape)