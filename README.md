# CAV-MARL-HighwayMerge: Multi-Agent Ramp Merging with MAPPO and MADQN

**CAV-MARL-HighwayMerge** implements *Multi-Agent Reinforcement Learning* (MARL) algorithms — **MAPPO** and **MADQN** — for **cooperative ramp merging** of *Connected and Autonomous Vehicles (CAVs)* using the [HighwayEnv](https://github.com/Farama-Foundation/HighwayEnv) simulation environment.
The project aims to reproduce and extend the results from the referenced research paper - *Cooperative merging for connected automated vehicles in mixed traffic: A multi-agent reinforcement learning approach* - to achieve efficient, collision-free merging behavior under mixed traffic scenarios.

---

## Overview

This repository contains:

- **Custom environment wrapper** that adapts `highway_env` to a multi-agent ramp merging scenario.  
- **MAPPO and MADQN agents** implemented from scratch using PyTorch.  
- **Training and evaluation pipelines** with configurable hyperparameters.  
- **Support for ramp merging**, highway driving, and future mixed autonomy experiments.

---

## Repository Structure

MARL_CAVs/<br>
├── agents/<br>
│ ├── mappo.py # MAPPO implementation (actor-critic)<br>
│ ├── madqn.py # MADQN implementation (deep Q-learning)<br>
│ ├── networks.py # Shared neural network architectures<br>
|<br>
├── highway_env/ # HighwayEnv environment<br>
|<br>
├── env_wrapper.py # Custom wrapper for MARL-style observation & reward<br>
├── train_mappo.py # Train MAPPO agent<br>
├── train_madqn.py # Train MADQN agent<br>
├── evaluate_mappo.py # Evaluate trained MAPPO policy<br>
├── evaluate_madqn.py # Evaluate trained MADQN policy<br>
├── test_wrapper.py # Sanity test for environment wrapper<br>
│<br>
├── checkpoints/ # Saved model weights (.pth files)<br>

---

## Installation

### Clone the repository
```
git clone https://github.com/<your-username>/CAV-MARL-HighwayMerge.git
cd CAV-MARL-HighwayMerge
```

### Create a Conda environment
```
conda create -n marl_env python=3.10 -y
conda activate marl_env
```

### Install dependencies
```
pip install torch gymnasium highway-env pygame numpy matplotlib
```

### Test your environment
```
python test_wrapper.py
```
Expected output:
```
✅ Reset successful.
Wrapped observation shape: (25,)
&nbsp;&nbsp;&nbsp;&nbsp;Taking random action: 2
&nbsp;&nbsp;&nbsp;&nbsp;Reward from compute_paper_reward(): 3.72
```

## Training

### Train MAPPO and MADQN
```
python train_mappo.py
python train_madqn.py
```

Both scripts will:
- Initialize the ramp-merging scenario (merge-v0)
- Train the corresponding agent (MAPPO or MADQN)
- Save model checkpoints (.pth files in the checkpoints/ directory)

## Evaluation

### Evaluate MAPPO and MADQN
```
python evaluate_MAPPO.py
python evaluate_MADQN.py
```

---

## Notes
- If you encounter a size mismatch error while loading a checkpoint, it means the action space size changed between training and evaluation.
- To force the environment to match the original 4-action setup, add this line before wrapping the environment:
  ```
  raw_env.unwrapped.configure({
    "action": {
      "type": "DiscreteMetaAction",
      "actions": ["LANE_LEFT", "LANE_RIGHT", "FASTER", "SLOWER"]
    }
  })
  ```
