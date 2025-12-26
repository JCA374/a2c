# Training Duration Guide

## Understanding Episode Completion and Training Metrics

### The "Zero Mean Return" Issue

If you see `mean_return_last10=0.0` and `episodes_completed=0` in your training metrics, this is **NOT a bug** - it means episodes haven't completed yet during your training run.

### Why Episodes Don't Complete Early in Training

**Key insight**: An untrained agent performs randomly and receives negative rewards in CarRacing, causing episodes to run for the full `max_steps` (default: 1000 steps) before truncation.

#### Example Calculation

With your configuration:
```python
rollout_length = 64
num_envs = 4
total_steps = 20000
max_episode_steps = 1000  # From EnvConfig
```

- **Steps per update**: `rollout_length × num_envs = 64 × 4 = 256 steps`
- **Total updates**: `total_steps ÷ 256 = 78 updates`
- **Steps for one episode** (worst case): 1000 steps
- **Updates until first episode completes**: `1000 ÷ 256 ≈ 4 updates`

If you stop training after just 8 updates (2048 steps), you won't see any episodes complete because 2048 < 1000!

### Recommended Training Durations

| Goal | Minimum `total_steps` | Reasoning |
|------|----------------------|-----------|
| **Quick test** | 20,000 | See a few episodes complete (with 4 envs) |
| **Basic training** | 500,000 | Default, allows meaningful learning |
| **Good results** | 1,000,000+ | Recommended for competitive performance |

#### Formula for Minimum Steps

To see at least N episodes complete (worst case):
```
min_total_steps = max_episode_steps × num_envs × N
```

For example, to see at least 5 episodes with 4 environments:
```
min_total_steps = 1000 × 4 × 5 = 20,000 steps
```

### How to Verify Episode Completion

1. **Check metrics.csv**: Look for `episodes_completed` > 0
2. **Monitor mean_return_last10**: Should become non-zero once episodes complete
3. **Early training returns**: Expect negative returns (-60 to -80) as agent runs to max_steps

### Reducing Time to First Episode (for testing)

If you want to quickly verify training works:

```python
# Option 1: Reduce max_steps
env_config = EnvConfig(render_mode="rgb_array")
env_config.episode.max_steps = 100  # Much shorter episodes

# Option 2: Increase rollout_length
cfg = A2CConfig(
    total_steps=20000,
    rollout_length=256,  # Collect more steps per update
    num_envs=4
)
```

### Expected Training Progression

1. **Updates 1-4**: `episodes_completed=0`, learning from partial episodes
2. **Update ~4-8**: First episodes complete (all ~1000 steps, negative returns)
3. **Updates 10+**: More episodes complete, returns may improve slightly
4. **Updates 50+**: Agent starts learning, episode lengths decrease, returns increase
5. **Updates 200+**: Meaningful progress, some episodes finish track successfully

### Interpreting Metrics When Episodes Haven't Completed

Even with `episodes_completed=0`, training is still working:
- `rollout_reward_mean`: Shows average reward per step in rollouts
- `loss_policy`, `loss_value`, `entropy`: Should be finite and changing
- `action_mean_*`: Actions should vary, not stuck at extremes

The agent learns from partial episode experiences via the rollout buffer, so progress happens even before first episode completion.

## Quick Checklist

Before reporting zero episode completions as a bug:

- [ ] Ran for at least `max_episode_steps × num_envs × 2` total steps?
- [ ] Checked `episodes_completed` in the last row of metrics.csv?
- [ ] Verified other metrics (loss, entropy) are changing normally?
- [ ] Confirmed rollout_reward_mean is non-zero (even if small/negative)?

If all above are "yes" and still no episodes after sufficient training, then investigate further.
