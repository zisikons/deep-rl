# Safe Deep Reinforcement Learning for Multi-Agent Systems with Continuous Action Spaces

This repository contains the source code and the implementation details for the paper (add paper link)
accepted at [RL4RealLife @ICML2021](https://sites.google.com/view/RL4RealLife)


#### Useful Links:
- Course [webpage](http://www.da.inf.ethz.ch/teaching/2020/DeepLearning/)

## Project Description
The objective of this project is to develop a safe variation of the Deep Deterministic Policy Gradient (DDPG). More specifically, the goal is to modify the potentially unsafe DDPG-based agent's action via projecting it on a safe subspace space using a QP Solver. More details can be found in [1]. This project relies heavily on the [OpenAI’s Multi-Agent Particle Environments](https://github.com/openai/multiagent-particle-envs) [2], which is the simulator used to train and evaluate the agents.

## Installation
To install and execute the project's source code follow the steps described in the following snippet:
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
<a id="1">[1]</a>
Gal Dalal, Krishnamurthy Dvijotham, Matej Vecerik, Todd Hester, Cosmin Paduraru, Yuval Tassa (2018). 
Safe Exploration in Continuous Action Spaces

<a id="1">[2]</a>
Ryan Lowe, Yi Wu, Aviv Tamar, Jean Harb, Pieter Abbeel, Igor Mordatch (2017).
Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments


## Contributors
- Athina Nisioti (@anisioti)
- Dimitris Gkouletsos (@dgkoul)
- Konstantinos Zisis (@zisikons)
- Ziyad Sheebaelhamd (@ziyadsheeba)
