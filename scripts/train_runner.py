import os
import sys
min_seed = 5
max_seed = 14
#filename = "train_SafeMADDPG_soft.py" #ziyad 
#filename = "train_SafeMADDPG_soft_reward.py"
filename = "train_SafeMADDPG_hard.py" #athina
#filename = "train_MADDPG.py" #kostas

for i in range(min_seed,max_seed):
    print("running script " + filename  + " with seed " + str(i))
    os.system('python3 ' + filename + " " + str(i))


