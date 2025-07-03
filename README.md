# Reinforcement Learning Playbook

A flexible and modular Python library for reinforcement learning experimentation and research.

## Features

- 🎮 Easy integration with Gymnasium environments
- 🤖 Modular agent architecture supporting multiple RL algorithms
- 📊 Built-in evaluation and logging capabilities
- 🔧 Customizable environment wrappers
- 📈 TensorBoard logging support

## Installation

```bash
git clone https://github.com/keivalya/rl_playbook.git
```

## Usage

```python
from rl_playbook import TrainLoop, EnvWrapper, Evaluator
loop = TrainLoop(
    env=EnvWrapper("CartPole-v1", normalize=True, reward_scale=0.1),
    agent="PPO",
    total_timesteps=100_000,
    evaluator=Evaluator(eval_env="CartPole-v1", every=5_000),
    log_to="tensorboard"
)
loop.run()
```
