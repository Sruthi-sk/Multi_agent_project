# Accelerating Human-AI Co-adaptation (Multi-agent Group 12)

This repository is for Accelerating Human-AI Co-adaptation. Adapted our code from 'https://github.com/VT-Collab/RILI_co-adaptation/tree/main' which is the main repository for the paper "Learning Latent Representations to Co-Adapt to Humans" [http://arxiv.org/abs/2212.09586].

In this repository we include codes for the two strategies:
- RILI vs SAC
- Multiple Agents

## Instructions 
You can install the packages by running the following command

`
pip install -r requirements.txt
`

Then, install the gym environment from any folder:

```
cd multiple_agents
cd 2_Agents
cd gym-rili
pip install -e .
cd ..
```

## Codes for RILI vs SAC 
- `main.py` : Contains code for training the Hider and Seeker in Circle environment
- `replaymemory.py` : Contains code for storing RILI agent's memory 
- `replaymemory_SAC.py` : Contains code for storing memory of SAC agent
- `models/`: directory for saved agents
- `runs/`: directory to visualize losses and rewards using `tensorboard --logdir runs`
- `gym-rili/gym_rili/envs/circle.py`: main environment for our code
- `algos/sac_agent.py` and `algos/sac_model_networks.py`: Codes for the SAC Agent
- `algos/rili.py`, `algos/model_rili.py` and `algos/model_sac.py`: Codes for the RILI Agent

## Code files for Multiple Agents
### common subfolders
#### algos
- code for RILI Agent
### 2_Agents
- `main.py` : Contains code for pretraining the model with 2 agents
- `maintest.py` : Contains code for testing the pre-trained model on circle environment
- `replaymemory.py` : Contains code for storing memory 
- `env/circle.py` : Contains code for circle-N environment used to pretrain 3 agents
### 3_Agents
- `main.py` : Contains code for pretraining the model with 3 agents
- `maintest.py` : Contains code for testing the pre-trained model on circle environment
- `replaymemory.py` : Contains code for storing memory 
- `env/circle.py` : Contains code for circle-N environment used to pretrain 3 agents


