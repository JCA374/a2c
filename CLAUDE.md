# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A2C (Advantage Actor-Critic) reinforcement learning agent for the CarRacing-v3 Gymnasium environment. The agent uses a compact observation space with lidar-style ray casting instead of raw pixel input.

## Development Commands

### Training
```bash
# Basic training with default parameters (500k steps, auto envs based on CPU cores)
python train_a2c.py

# Custom training configuration
python train_a2c.py --num-envs auto --cpu-threads auto --total-steps 1000000 --rollout-length 256

# Specify log directory
python train_a2c.py --log-dir logs/experiment_name
```

Training features:
- **Progress bar**: Real-time training progress with live metrics (loss, return, episodes completed, steps)
- **Periodic logging**: Detailed metrics printed at log intervals (configurable with `--log-interval`)
- Logs saved to `logs/a2c_<timestamp>/` by default
- `config.json`: Training configuration
- `metrics.csv`: Per-update metrics
- `best_model.pt`: Best model checkpoint based on mean return

### Playing Trained Models
```bash
# Play a saved model
python play_model.py --model-path logs/a2c_20241226_143000/best_model.pt

# Play with custom rendering
python play_model.py --model-path logs/a2c_20241226_143000/best_model.pt --episodes 10 --render-mode human
```

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_a2c.py

# Run with verbose output
pytest -v

# Run specific test function
pytest tests/test_a2c.py::test_a2c_trainer_runs
```

## Architecture

### Core Components

**A2C Agent** ([a2c_agent.py](a2c_agent.py)):
- `ActorCritic`: Neural network with separate policy and value heads (256 hidden units, tanh activation)
- `RolloutBuffer`: Fixed-size storage for rollouts with GAE computation
- `A2CTrainer`: Main training loop with vectorized environments
- Uses tanh-squashed Gaussian policy for continuous actions
- Includes action space biasing for CarRacing (encourages throttle, discourages brake)

**Environment Stack** ([env/make_env.py](env/make_env.py)):
Wrappers applied in this order:
1. Base `CarRacing-v3` environment
2. `ActionClipWrapper`: Clips actions to valid bounds
3. `CarRacingRewardWrapper`: Competition-style reward shaping
4. `TimeLimit`: Step-based episode limit (1000 steps default)
5. `EpisodeTimeLimit`: Wall-clock time limit (60 seconds default)
6. `CompactObservation`: Converts pixel obs to compact state + lidar rays

**Compact Observation** ([env/wrappers.py](env/wrappers.py)):
Vector size: 8 base features + 32 lidar rays (default)

Base features (all float32, normalized):
- Speed, local velocity (x,y), heading (cos, sin)
- Angular velocity, mean steer angle, traction fraction

Lidar rays:
- 32 rays spanning -0.9π to 0.9π relative to car heading
- Each ray returns normalized distance to road edge [0,1]
- Uses ray-casting against road polygon geometry

**Reward Shaping** ([env/wrappers.py](env/wrappers.py):52):
- Progress: +2.0 per tile of track progress
- Off-track penalty: -40.0 (when traction < 0.95)
- Collision penalty: -20.0
- Control cost: -0.01 × ||action||
- Time penalty: -0.05 per step
- Finish bonus: +400.0 (if track completed)
- All configurable via `RewardConfig` dataclass

**Track Seeds** ([env/tracks.py](env/tracks.py)):
- Training: seeds 0-19 (20 tracks)
- Validation: seeds 101-105 (5 tracks)
- Test: seeds 2024-2033 (10 tracks)

### Key Implementation Details

**Vectorized Training**:
- `SimpleVectorEnv` ([env/vector_env.py](env/vector_env.py)) provides parallel environment execution
- Automatically resets environments on episode termination
- Training uses multiple parallel envs (default: 4) for faster data collection

**GAE (Generalized Advantage Estimation)**:
- Implemented in `RolloutBuffer.compute_returns_and_advantages`
- Default λ=0.95, γ=0.99
- Advantages normalized before policy update

**Action Space Handling**:
- Raw actions sampled from Gaussian distribution
- Tanh squashing maps to [-1, 1]
- Linear transformation to environment action bounds [low, high]
- Log-prob corrected for tanh transformation (important for policy gradient)

**Logging** ([log_utils.py](log_utils.py)):
- CSV-based metrics logging (not TensorBoard)
- JSON config snapshot
- Tracks all hyperparameters and training metrics per update

## Configuration

All configurations use dataclasses:
- `A2CConfig`: Training hyperparameters
- `EnvConfig`: Environment setup
- `RewardConfig`: Reward shaping parameters
- `EpisodeConfig`: Episode limits
- `CompactStateConfig`: Observation space configuration

Modify configs by:
1. Creating dataclass instances with custom values in training scripts
2. Passing command-line arguments (for training script)
3. Directly editing dataclass defaults for global changes

## Common Development Tasks

### Adding New Environment Wrappers
1. Create wrapper class in [env/wrappers.py](env/wrappers.py)
2. Add to wrapper stack in `_build_env()` in [env/make_env.py](env/make_env.py)
3. Order matters: reward shaping before observation transformation
4. Add tests in [tests/test_wrappers.py](tests/test_wrappers.py)

### Modifying Network Architecture
- Edit `ActorCritic.__init__()` in [a2c_agent.py](a2c_agent.py):86
- Current: 2-layer MLP with 256 hidden units and tanh activation
- Separate policy and value networks share no parameters
- Action biases for CarRacing environment in `_init_action_biases()`

### Changing Training Algorithm
- Core update logic in `A2CTrainer.train()` at [a2c_agent.py](a2c_agent.py):214
- Loss computation at [a2c_agent.py](a2c_agent.py):264-268
- Policy loss: -advantages * log_probs (A2C objective)
- Value loss: MSE between predicted values and returns
- Entropy bonus: encourages exploration

### Debugging Environment Issues
- Check observation space: `CompactObservation._compact_state()` in [env/wrappers.py](env/wrappers.py):186
- Verify reward shaping: `CarRacingRewardWrapper.step()` in [env/wrappers.py](env/wrappers.py):75
- Lidar ray visualization: rays cast in `_cast_lidar()` at [env/wrappers.py](env/wrappers.py):237
- Road polygon detection: `_point_on_road()` at [env/wrappers.py](env/wrappers.py):256
