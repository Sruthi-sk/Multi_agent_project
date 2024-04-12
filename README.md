# Accelerating Human-AI Co-adaptation (Multi-agent Group 12)

This repository is for Accelerating Human-AI Co-adaptation. For initial setup refer to 'https://github.com/VT-Collab/RILI_co-adaptation/tree/main' which is the main repository for the paper "Learning Latent Representations to Co-Adapt to Humans" [http://arxiv.org/abs/2212.09586].

In this repository we include codes for the two strategies:
- RILI vs SAC
- Multiple Agents

Codes for RILI vs SAC 


Code files for Multiple Agents
- `maintest.py` : Contains code for testing the pre-trained model on circle environment
- `main3agents.py` : Contains code for pretraining the model with 3 agents
- `main2agents.py` : Contains code for pretraining the model with 2 agents
- `replaymemory.py` : Contains code for storing memory//
Within env folder
- `circle.py` : Contains code for circle environment
- `circle-N-3agents.py` : Contains code for circle-N environment used to pretrain with 3 agents
- `circle-N-2agents.py` :   //
Within algos folder
- contains code for RILI taken from the above paper

