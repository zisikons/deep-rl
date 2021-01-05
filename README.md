# deep-rl
Deep Learning Project (Fall Semester 2020) :rocket:

This repository contains the source code and the implementation details for the [Deep Learning](http://www.da.inf.ethz.ch/teaching/2020/DeepLearning/) course offered at ETH Zurich.

#### Useful Links:
- Course [webpage](http://www.da.inf.ethz.ch/teaching/2020/DeepLearning/)

## Project Description

## Installation
To install and execute the project's source code follow the steps in the following snippet:
**WARNING:** Installs packages

```
git clone git@github.com:zisikons/deep-rl.git
cd ./deep-rl
sh install_requirements.sh  # installs all the requirements
```

Alternatively if you don't want to install the new packages on your computer, then make sure to at least:
* Download the code from the forked [multiagent-particle-envs](https://github.com/zisikons/multiagent-particle-envs/tree/a8ba7c4c49edbb7c164426fb90e141af465380b1)
```
git submodule update --init --recursive
```
* Download the following python packages:
```
gym==0.10.5
pyglet==1.3.2
qpsolvers
```

## Execution
Once the code is downloaded and everything is set, in order to train an agent you need to do the following:
```
python3 scripts/collect_data.py        # Uses the simulator to generate the datasets for the
                                       # constraint sensitivity Neural Networks
                                    
python3 core/constraint_network.py     # Trains the constraint sensitivity Neural Networks
                                       # (not required for the vanilla DDPG agent) 

python3 scripts/train_<agent_type>.py  # Trains one of the 3 RL agents that were developed
                                       # during this project
                                       
python3 scripts/plot_results.py        # Generates plot to compare the 3 agents
                                       # (requires training of all agents)
```
*Note: The above sequence takes a considerable amount of time.*


## References
<Add papers here>

## Contributors
- Athina Nisioti (@anisioti)
- Dimitris Gkouletsos (@dgkoul)
- Konstantinos Zisis (@zisikons)
- Ziyad Sheebaelhamd (@ziyadsheeba)
