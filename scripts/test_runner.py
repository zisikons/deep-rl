import os
import sys
min_seed = 5
max_seed = 15
filename = "test_agents.py"
for i in range(min_seed,max_seed):
    print("running script " + filename  + " with seed " + str(i))
    os.system('python3 ' + filename + " " + str(i))


