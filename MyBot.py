import tensorflow as tf
import time
start = time.time()
# from networking import *
from hlt import *
import os
import sys
import numpy as np
from operator import itemgetter
import json
import random
import gc
# set_floatx('float64')

VISIBLE_DISTANCE = 0
myID, gameMap = get_init()

check1 = time.time() - start
pad = 9
x = gameMap.height
y = gameMap.width
name = "erdy.json"

g2 = tf.Graph()
with g2.as_default():
  f = open(name, 'r')
  weights = [tf.constant(x, dtype=tf.float32) for x in json.load(f)]
  # print ([x.get_shape() for x in weights])
  input_shape = (1, x+pad*2, y+pad*2, 4)
  inp = tf.placeholder(tf.float32, input_shape)
  curr = inp
  for i in range(len(weights)//2):
    j = i*2
    conv = tf.nn.conv2d(curr, weights[j], [1, 1, 1, 1], padding='VALID') + weights[j+1]
    if i != len(weights)//2-1:
      curr = tf.nn.relu(conv)
    else:
      curr = tf.nn.softmax(conv)
  out = curr
  f.close()

config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)


check2 = time.time() - start

second_net = False

def wrap(arr, pad):
    # print (arr.shape)
    orig_dim1 = arr.shape[0]
    orig_dim2 = arr.shape[1]
    n1 = np.concatenate((arr, arr, arr), axis=0)
    n2 = np.concatenate((n1, n1, n1), axis=1)
    a = orig_dim1 - pad
    b = pad + 2*orig_dim1
    c = orig_dim2 - pad
    d = pad + 2*orig_dim2
    return n2[a:b, c:d, :]

def stack_to_input(stack, position, pos=1):
    return np.take(np.take(stack,
                np.arange(-VISIBLE_DISTANCE,VISIBLE_DISTANCE + 1)+position[0],axis=1,mode='wrap'),
                np.arange(-VISIBLE_DISTANCE,VISIBLE_DISTANCE + 1)+position[1],axis=2,mode='wrap').flatten()

def frame_to_stack(frame):
    game_map = np.array([[(x.owner, x.production, x.strength) for x in row] for row in frame.contents])
    return np.array([(game_map[:, :, 0] == myID),  # 0 : owner is me
                      ((game_map[:, :, 0] != 0) & (game_map[:, :, 0] != myID)),  # 1 : owner is enemy
                      game_map[:, :, 1]/8-1,  # 2 : production
                      game_map[:, :, 2]/128 - 1,  # 3 : strength
                      ]).astype(np.float32)
#print ("bob is a thing")
switch = False
send_init('Conv9erdfullprep')
turn = 0
seed = random.random()
check3 = time.time() - start
cap = 500
# loggy = open('loggy', 'w')
if check3 > 130:
  sendFrame([])
frame = gameMap
next_strength = np.zeros((gameMap.width, gameMap.height))
prev_frame = None

stack = None
import copy
with tf.Session(graph=g2, config=config) as session:
  session.run(tf.global_variables_initializer())
  while True:
      # move_start = time.time()
      turn += 1
      num_used = 0
      # turn += 1
      prev_stack = stack
      frame.get_frame()
      move_start = time.time()
      # if turn == 1:
      #   send_frame([])
      #   continue
      # time1 = time.time()
      stack = frame_to_stack(frame)
      if turn == 1:
          send_frame([])
          continue
      # time2 = time.time()
      a = wrap(stack.transpose(1, 2, 0), pad) 
      # time3 = time.time()
      out1 = session.run(out, {inp: np.array([a])})[0]
      # time4 = time.time()
      moves = [Move(square, out1[square.y][square.x].argmax()) for square in frame if square.strength > 0 and square.owner == myID]
      # time5 = time.time()
      if time.time() - move_start > 1.4:
        send_frame(moves)
        continue
      warnings = []
      for square in gameMap:
        if square.owner == myID:
          next_strength[square.x][square.y] = square.production + square.strength
        else:
          next_strength[square.x][square.y] = -square.strength
      if time.time() - move_start > 1.4:
        send_frame(moves)
        continue
      for m in moves:
        if m.direction == 0:
          warnings.append((m, m.square, m.square))
          continue
        nsite = frame.get_target(m.square, m.direction)
        if nsite.owner == 0 or nsite.owner == myID:
          csite = m.square
          next_strength[nsite.x][nsite.y] += csite.strength
          next_strength[csite.x][csite.y] -= csite.strength + csite.production
          nns = next_strength[nsite.x][nsite.y]
          if nns <= 0 and nsite.owner == 0:
            warnings.append((m, nsite, csite))

          elif nns > 300 and nsite.owner == myID:
            warnings.append((m, nsite, csite))
      if time.time() - move_start > 1.4:
        send_frame(moves)
        continue
      for w in warnings:
        nns = next_strength[w[1].x][w[1].y]
        if nns <= 0 and w[1].owner == 0:
          moves.append(Move(w[0][0], 0))
      # time6 = time.time()
      send_frame(moves)
