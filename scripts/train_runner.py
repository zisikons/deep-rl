import os
import sys
min_seed = 5
max_seed = 14
<<<<<<< HEAD
#filename = "train_SafeMADDPG_soft.py" #ziyad 
#filename = "train_SafeMADDPG_soft_reward.py"
filename = "train_SafeMADDPG_hard.py" #athina
#filename = "train_MADDPG.py" #kostas
=======
filename = "train_SafeMADDPG_soft.py" #ziyad 
#filename = "train_SafeMADDPG_soft_reward.py
#filename = "train_SafeMADDPG_hard.py" #athina
filename = "train_MADDPG.py"
#kostas
>>>>>>> 3b235a3b4a4783f6fae90803f09de8200b6f7254

for i in range(min_seed,max_seed):
    print("running script " + filename  + " with seed " + str(i))
    os.system('python3 ' + filename + " " + str(i))


