import os
import sys

######
# 0: Ziyad
# 1: Dimitris
# 2: Kostas
# 3: Athina
user = 3

if user == 0:
    min_seed = 5
    max_seed = 10

elif user == 1:
    min_seed = 10
    max_seed = 15
else:
    min_seed = 5
    max_seed = 15

simulations = {0: "train_SafeMADDPG_soft.py",
               1: "train_SafeMADDPG_soft.py",
               2: "train_SafeMADDPG_hard.py",
               3: "train_MADDPG.py"}

filename = simulations[user]
#filename = "train_SafeMADDPG_soft.py" #ziyad 
#filename = "train_SafeMADDPG_soft_reward.py"
#filename = "train_SafeMADDPG_hard.py" # kostas
# filename = "train_MADDPG.py" # athina

for i in range(min_seed,max_seed):
    print("running script " + filename  + " with seed " + str(i))
    os.system('python3 ' + filename + " " + str(i))


