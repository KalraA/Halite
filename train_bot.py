import datetime
import os
import sys
from keras.layers import Dropout
import numpy as np
import json
import tensorflow as tf
from keras.regularizers import l2
from keras.optimizers import Nadam, RMSprop
from keras.models import Sequential,load_model
from keras.layers import Dense, Reshape, Flatten, Activation, BatchNormalization
from keras.layers.convolutional import Convolution2D, Conv2D
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping,ModelCheckpoint,TensorBoard
import keras.backend as K


K.dim_ordering = 'tf' # This is like a 10x speedup
opt = Adam(0.0001)
a = int(sys.argv[2])
b = int(sys.argv[3])
REPLAY_FOLDER = sys.argv[1]
training_input = []
training_target = []
tf.python.control_flow_ops = tf
print (K.image_dim_ordering())
VISIBLE_DISTANCE = 9
input_dim=(2*VISIBLE_DISTANCE+1, 2*VISIBLE_DISTANCE+1, 4)
#input_dim = (None, None, 4)
np.random.seed(0) # for reproducibility

model = Sequential([Convolution2D(150, 1, 3, activation='relu', border_mode = 'valid', input_shape=input_dim),
                     Conv2D(150, 3, 1, activation='relu', border_mode='valid'), 
Conv2D(75, 1, 1), Activation('relu'),
Conv2D(150, 1, 3, activation='relu', border_mode='valid')
,Convolution2D(150, 3, 1, activation='relu', border_mode= 'valid'),
Conv2D(75, 1, 1), Activation('relu'),
Convolution2D(150, 1, 3, activation='relu', border_mode= 'valid'), 
Convolution2D(150, 3, 1, activation='relu', border_mode='valid'),
Conv2D(75, 1, 1), Activation('relu'),
		    	Convolution2D(150, 3, 3, activation='relu', border_mode= 'valid', W_regularizer=l2(0.0001)),
		    Conv2D(5, 11, 11, activation='linear', border_mode='valid'), Flatten(), Activation('softmax')])

model.compile(opt,'categorical_crossentropy', metrics=['accuracy'])

def stack_to_input(stack, position):
    return np.take(np.take(stack,
                np.arange(-VISIBLE_DISTANCE,VISIBLE_DISTANCE + 1)+position[0],axis=1,mode='wrap'),
                np.arange(-VISIBLE_DISTANCE,VISIBLE_DISTANCE + 1)+position[1],axis=2,mode='wrap')#.flatten()


dude = "erdman v19" # or your own bot for RL
wz = ['width":' + str(x) for x in list(range(a, b))]
m = 25.0
n = 10000
i = 0;
count = 0
wins = 0
rew = 1
size = len(os.listdir(REPLAY_FOLDER))    
for index, replay_name in enumerate(os.listdir(REPLAY_FOLDER)):
    rew = 1
    if i > n: break;
    i += 1
    if replay_name[-4:]!='.hlt' and replay_name[-4:] != '.txt':continue
    f = open('{}/{}'.format(REPLAY_FOLDER,replay_name)).read() 
    b = False
    for w in wz:
       if w in f:
            b = True
      # print ("not 2 players")
    if not b:
         print("wrong_size")
         continue
    try:
        replay = json.load(open('{}/{}'.format(REPLAY_FOLDER,replay_name)))
    except ValueError:
        continue
    playing = replay['player_names']
    if len(replay['moves']) < 10: continue
    if dude not in playing:
         print ("No Mag")
         continue
    target_idd =  playing.index(dude)
    num_players = replay['num_players']
    if num_players != 4 and False:
        continue
    frames=np.array(replay['frames'])
    player=frames[:,:,:,0]
    daframe = frames[-1]
    win = True
    if target_id != target_idd+1: 
        win = False
    
    target_id = target_idd + 1
    print("laoding: " + replay_name)
    prod = np.repeat(np.array(replay['productions'])[np.newaxis],replay['num_frames'],axis=0)
    strength = frames[:,:,:,1]
    dis = 0.995
    moves = (np.arange(5) == np.array(replay['moves'])[:,:,:,None]).astype(float)[:200]
    if win:
        rew = 1.0
    else:
        rew = -0.1
    # This is the RL code.  Uncomment this
    # for i in range(len(moves)):
    #   moves[-1-i] *= rew*np.array([0.06, 1., 1., 1., 1.])
    #   rew *= dis
    stacks = np.array([player==target_id ,(player!=target_id) & (player!=0), prod/8-1, strength/128-1]).astype(float)
    print (stacks.shape)
    stacks = stacks.transpose(1,0,2,3)[:len(moves)].astype(np.float32)

    s2 = stacks#stacks_a.transpose(1, 0, 2,: 3)[:len(moves)]
    if len(moves) < 5:
        continue
    position_indices = s2[:,0].nonzero()
    sampling_rate = np.sqrt(1/s2[:,0].mean(axis=(1,2)))[position_indices[0]]
    sampling_rate *= moves[position_indices].dot(np.array([1,m,m,m,m])) # weight moves 10 times higher than still
    sampling_rate /= sampling_rate.sum()
    sample_indices = np.transpose(position_indices)[np.random.choice(np.arange(len(sampling_rate)),
                                                                    min(len(sampling_rate),int(1.5*2048)),p=sampling_rate,replace=False)]
    replay_input = np.array([stack_to_input(stacks[i],[j,k]) for i,j,k in sample_indices])
    replay_target = moves[tuple(sample_indices.T)]
    replay_input = replay_input.transpose(0, 2, 3, 1)
    training_input.append(replay_input.astype(np.float32))
    training_target.append(replay_target.astype(np.float32))


ew = [0, 1, 4, 3, 2]
ns = [0, 3, 2, 1, 4]

now = datetime.datetime.now()
training_input = np.concatenate(training_input,axis=0)
training_target = np.concatenate(training_target,axis=0)
# 4 flips
training_input = np.concatenate([training_input, training_input[:, ::-1, :, :], training_input[:, :, ::-1, :], training_input[:, ::-1, ::-1, :]],axis=0)
training_target = np.concatenate([training_target, training_target[:, ns], training_target[:, ew], training_target[:, ns][:, ew]])
#indices = np.arange(len(training_input))
#np.random.shuffle(indices) #shuffle training samples
#training_input = training_input[indices]
#training_target = training_target[indices]
#model = load_model('model.h5')

print(np.sum(training_target, axis=0)) #print number of each category

model.fit(training_input,training_target,validation_split=0.1,
          callbacks=[EarlyStopping(patience=4),
                     ModelCheckpoint('re-erd'+str(a) + str(b) +'.h5',verbose=1,save_best_only=True)]#,
                     #tensorboard],
          , batch_size=1024, nb_epoch=1000)