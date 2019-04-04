import numpy as np
from keras import backend as kbe
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


data = kbe.variable(np.random.random((4,2)))   # create 4 x 2 tensor with random number
zero_data = kbe.zeros_like(data)               # create 4 x 2 tensor of zeros
print(kbe.eval(zero_data))