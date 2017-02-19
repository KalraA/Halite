import time
start = time.time()
from networking import *
import os
import sys
import numpy as np
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Convolution2D#, Conv2D
from keras.models import Sequential
from operator import itemgetter
import json
import random

from keras.models import load_model
model2 = load_model('re-erd11True.h5')


weights = []
w = model2.get_weights()#[np.array(x) for x in json.load(open("erd_weights.w"))]
for i, m in enumerate(w):
  if i % 2 == 0:
    if i < 100:
      # print( m.shape)
      weights.append(m)#.transpose((2, 3, 1, 0)))
    else:
      weights.append(m)
  else:
    weights.append(m)
# model.summary()
# print ([x.shape for x in weights])

f = open('erdy.json', 'w')
json.dump([x.tolist() for x in weights], f)
f.close()
