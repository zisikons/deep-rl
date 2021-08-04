# Safe Deep Reinforcement Learning for Multi-Agent Systems with Continuous Action Spaces

This repository contains the source code and the implementation details for the paper (add paper link)
accepted at [RL4RealLife @ICML2021](https://sites.google.com/view/RL4RealLife)

## Project Description
The objective of this project is to develop a safe variation of the Multiagent Deep Deterministic Policy Gradient (MADDPG). More specifically, the goal is to modify the potentially unsafe MADDPG-based agents' action via projecting it on a safe subspace space using a QP Solver. More details can be found in [1]. This project relies heavily on the [OpenAI’s Multi-Agent Particle Environments](https://github.com/openai/multiagent-particle-envs) [2], which is the simulator used to train and evaluate the agents.

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
python3 scripts/collect_data.py                  # Uses the simulator to generate the datasets for the
                                                 # constraint sensitivity Neural Networks
                                    
python3 scripts/train_constraint_networks.py     # Trains the constraint sensitivity Neural Networks
                                                 # (not required for the vanilla MADDPG agent) 

python3 scripts/train_<agent_type>.py            # Trains one of the 3 RL agents that were developed
                                       
python3 scripts/test_<agent_type>.py             # Tests one of the 3 RL agents that were developed            
```
*Note: The above sequence takes a considerable amount of time.*

## Approach Summary
![alt text](https://github.com/zisikons/deep-rl/blob/main/poster.jpeg)


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
