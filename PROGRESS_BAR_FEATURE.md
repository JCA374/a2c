# Progress Bar Feature

## Overview

Training now includes a real-time progress bar powered by `tqdm` that shows live training metrics.

## What You'll See

When running training, you'll see:

```
Training:  60%|██████    | 150/250 [02:15<01:30, 1.1update/s, loss=0.297, return=-45.2, eps=12, steps=19200]
```

### Progress Bar Components

1. **Percentage Complete**: `60%` - How much of training is done
2. **Visual Bar**: `██████    ` - Visual progress indicator
3. **Update Counter**: `150/250` - Current update / Total updates
4. **Time Elapsed**: `[02:15<01:30]` - Time elapsed / Time remaining estimate
5. **Speed**: `1.1update/s` - Updates per second
6. **Live Metrics**:
   - `loss`: Current total loss
   - `return`: Mean return over last 10 completed episodes (0.0 until episodes complete)
   - `eps`: Total episodes completed so far
   - `steps`: Total environment steps taken

## Detailed Logging

At regular intervals (configured with `--log-interval`), detailed metrics are printed:

```
Update 10/250 | step=1280 | loss=0.297 | return=-45.23 | eps=12 | best=-43.50
Update 20/250 | step=2560 | loss=0.245 | return=-42.18 | eps=28 | best=-40.12
```

This provides more detailed information without cluttering the progress bar.

## Understanding the Metrics

### Early Training (First ~10 Updates)

```
Training:   8%|▊         | 20/250 [00:30<05:45, loss=3.142, return=0.0, eps=0, steps=2560]
```

- `return=0.0` and `eps=0`: Normal! Episodes haven't completed yet
- See [TRAINING_DURATION_GUIDE.md](TRAINING_DURATION_GUIDE.md) for why this happens
- Training is still working - watch the `loss` and `steps` changing

### Mid Training (After Episodes Start Completing)

```
Training:  40%|████      | 100/250 [03:20<05:00, loss=0.542, return=-58.3, eps=45, steps=12800]
```

- `return=-58.3`: Average return over last 10 episodes (negative is expected early on)
- `eps=45`: 45 episodes have completed
- `loss` should be decreasing over time

### Late Training (Agent Improving)

```
Training:  90%|█████████ | 225/250 [07:30<00:50, loss=0.183, return=25.7, eps=203, steps=28800]
```

- `return=25.7`: Positive returns indicate agent is learning to complete the track
- More episodes completing as agent gets better
- Lower loss indicates more stable learning

## Configuration

Control logging frequency:

```bash
# Log detailed metrics every 5 updates (default: 10)
python train_a2c.py --log-interval 5

# Longer interval for less console output
python train_a2c.py --log-interval 50
```

## Benefits

1. **Real-time feedback**: Know exactly where training is at any moment
2. **Time estimates**: See how long training will take
3. **Quick diagnostics**: Spot issues immediately (stuck loss, no episodes completing)
4. **Less clutter**: Detailed logs only at intervals, progress bar shows essentials
5. **Professional appearance**: Clean, informative output

## Technical Details

- Implemented in [a2c_agent.py](a2c_agent.py):214-340
- Uses `tqdm` library (added to [requirements.txt](requirements.txt))
- Progress bar updates every update step
- Detailed logs use `tqdm.write()` to avoid interfering with progress bar
- Automatically closes at end of training

## Example Output Flow

```
Training:   0%|          | 0/250 [00:00<?, ?update/s]
Training:   4%|▍         | 10/250 [00:15<06:00, loss=2.453, return=0.0, eps=0, steps=1280]
Update 10/250 | step=1280 | loss=2.453 | return=0.00 | eps=0 | best=0.00

Training:   8%|▊         | 20/250 [00:30<05:45, loss=1.876, return=0.0, eps=0, steps=2560]
Update 20/250 | step=2560 | loss=1.876 | return=0.00 | eps=0 | best=0.00

Training:  12%|█▏        | 30/250 [00:45<05:30, loss=0.942, return=-62.5, eps=4, steps=3840]
Update 30/250 | step=3840 | loss=0.942 | return=-62.45 | eps=4 | best=-62.45

...

Training: 100%|██████████| 250/250 [08:20<00:00, 2.0s/update, loss=0.234, return=45.8, eps=215, steps=32000]
Update 250/250 | step=32000 | loss=0.234 | return=45.82 | eps=215 | best=58.93
```

Clean and informative!
