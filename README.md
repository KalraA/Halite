# Halite Bot!

### About
halite.io is an online AI competition, and this bot was a submission. Check it out at halite.io. This bot uses a combination of supervised learning, transfer learning, and reinforcement learning to build the #1 Machine Learning Bot.

### Results
This bot recieved the following achievements, even though there were major timeout issues during the finals: #1 Machine Learning Bot, #1 Undergraduate Submission, #12 Overall, #9 Overall w/o extra final timeouts, #1 best language name. (Ok I gave myself that last one). This was also the first and only ML bot in diamond Since November and until I made a post revealing all the secrets of that bot: http://forums.halite.io/t/building-a-good-ml-bot/776 . The following a summary of the final bot, but doesn't include everything. The link above describes the details omitted here. Some things were changed from then. 

### Replays
Here are some of my favourite game replays:

Favourite Win:

https://halite.io/game.php?replay=ar1487294314-2974319356.hlt

ML 1v1s

https://halite.io/game.php?replay=ar1487263413-2138886622.hlt

https://halite.io/game.php?replay=ar1487290872-3827664987.hlt

https://halite.io/game.php?replay=ar1487293806-2467046108.hlt

Power of Non Aggression Pact:

https://halite.io/game.php?replay=ar1487271549-1684906881.hlt

### Algorithm/Architecture

The algorithm consisted of one feed forward fully convolutional network. In the forward pass, the padding was done by taking the opposite edge and concatenating it on because the map reflects itself:

i.e:

0 1 4

3 2 3

4 1 2

Gives you

2 4 1 2 4

4 0 1 4 0

3 3 2 3 3

2 4 1 2 4

4 0 1 4 0

The network was of the following format:

Input ?x?x?x4 -> [150x3x1 -> 150x1x3 -> 75x1x1]*3 -> 150x3x3 -> 5x11x11 -> Output ?x?x?x5

The reason for this archetecture is that simple 3x3 convolution kernels have more computation. They were timing out. Also In CNNs, depth is extremely important, because 4 layer CNN only applied 5 non linearities total, and that's not a lot of representational power. Factoring out the small convolutions into 1x1s and 3x1s similar to Inception v4 allows for more efficient representation with less compute. This archetecture improved performance significantly even when trained on the same data. The receptive field is a 19x19 block for each output.

Then these outputs were checked for 0 piece moves, and moves onto untakable neutral squares. 

### Training

The training pipeline was seperated into 2 parts, supervised learning and reinforcement learning.

#### Supervised Learning:

This is the supervised learning process:

##### Data:

The data is all games of erdman v19. Each game was processed to find the moves that erdman made, and a 19x19 block extracted where the input was of the format batchx19x19x4 where the 4 was id==myId, id==enemyId, production/8-1, strength/255-1 (centering around 0). Then to remove bias, each example was flipped in 3 ways creating 4 data points. Rotations were not included because if you look at the maps, the symmetries are created via flips, not rotations, and so adding rotations would force the net to learn something it doesnt have too, and when you only have a small network due to forward pass time constraints, you need to make sure it learns the most it can. The data was also balanced so that there was a 20% split in terms of STILL, NORTH, WEST, SOUTH, EAST moves. A lot of people didn't do this, but I found this leads to better play. Approx 5million datapoints were used

Improvements to make:

Duplicates should be removed.

##### Waterfalling:

To improve convergence speed and accuracy, instead of training a new model each time, I would initialize the current model by starting from a previous one. This works because the old model is already is in a nearby weight space, and so the distance the weights need to travel is much less, so more iterations can be focussed on fine tuning. This gave about a 1.5% boost in accuracy.

##### Other:

Interestingly, dropout and l2 regularization tended to lead to worse perforance and less confident bots overall. not 100% sure why, but those were my observations. I believe it is because the model was already too small to overfit, so this was making it even worse. The optimizers tried were Adam, Nadam, and RMSProp. I found RMSProp to be the most successful for waterfalling, and Adam is the best for training from scratch. Step size of 3e-3 for training from scratch, and 1e-4 for fine tuning. beta1 = 0.993 for Adam. Batch Normalization worked, and improved performance, however the forward pass implementation seemed to be slower, and I didn't have time to rebuild it in raw tensorflow. 

other details were already presented in this post: 

#### Reinforcement Learning

So in the last few days, I realized alhough my bot was doing really well, it was timing out, and so wasn't clearly better than other bots, and because they were all also training from erdman v19, so I needed to do something extra. So I submitted a bot 2 days before the deadline. Let it play games for a day. Then applied RL an iteration of poilcy gradient learning on this games, then let it play another day, and let it run another iteration, and then submit. This bot finished a solid 12th even though there were more timeouts in finals than before, so I believe it worked.

##### Setup:

So after downloading all the games of myself, I had it load up my previous model, and train it with a decaying reward starting at the last turn. 

##### Reward Picking:

So the way it worked, is 1.0 was the positive reward for winning, and -0.1 was the negative reward for losing. Since I was only doing a few iterations, and since anything that was first place was considered a loss, the negative reward was small to stop huge jumps from happening. Then the reward was discounted with a factor of 0.97 from move -1. This was then multiplied into the label, which is equivalent to multiplying the gradient since the derivative of crossentropy loss propogates this multiplication. This was the Implementation of policy gradient learning. RMSProp with 0.0001 step size was used. 

### Code

Based on ML starter bot

python3:
MyBot.py -> actual bot code
train_bot.py -> training code
convert.py -> takes keras h5 model and extracts weights to json for MyBot
hlt.py -> communications module provided by competition

python2:
get_games.py -> game downloading script

Other:
LANGUAGE -> ai lmao (it's a play on words from ayyy lmao, not AL lmao ...).
erdy.json -> final weights

