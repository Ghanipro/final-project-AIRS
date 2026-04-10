# AIRS (Autonomous Intrusion Response System)

## Overview
This repository contains the initial project scaffold for an Autonomous Intrusion Response System implemented in Python.

## Structure
- `airs/`: Main package containing the Gymnasium environment, training scripts, evaluation pipeline, and Streamlit dashboard.
- `config/`: Configuration files including dependencies.
- `tests/`: Basic test scripts for the project.

## Requirements
To install the required packages, run:
```
pip install -r requirements.txt
```
### Train (full mode)
python scripts/train_all.py

### Performance Boost (Curriculum + Warm Start)
```bash
python scripts/performance_boost.py --seeds 0,1,2,3,4
```

### Red-vs-Blue Self-Play
```bash
python scripts/train_self_play.py --rounds 3 --blue-timesteps 100000 --red-timesteps 100000
```

### MARL Evaluation
```bash
python scripts/evaluate_marl.py --blue-algo PPO --episodes 200
```

### Evaluate (full mode)
python scripts/evaluate_all.py

### Dashboard
streamlit run src/dashboard/app.py
