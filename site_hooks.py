"""Hooks for publish.py (ml-project-site skill): replays for the log, models and games for the Play tab."""
import glob
import json
import os
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))


def encode_mp4(frames, out, fps=25, hold=25):
    """frames: list of PIL images (same size, even width/height). Plays once; the last frame is held `hold` frames."""
    w, h = frames[0].size
    p = subprocess.Popen(['nice', '-n', '15', 'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                          '-s', f'{w}x{h}', '-r', str(fps), '-i', '-', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '28',
                          '-pix_fmt', 'yuv420p', '-movflags', '+faststart', out], stdin=subprocess.PIPE)
    for f in frames + [frames[-1]] * hold:
        p.stdin.write(f.convert('RGB').tobytes())
    p.stdin.close()
    p.wait()


def replay_gif(policy_path, seed, out_gif):
    """Every 4th frame at 12.5 fps (2x speed), 300x200; plays once and stops on the outcome."""
    os.nice(10)
    from a2c_agent import Agent
    from viz.render import episode_frames
    frames, _ = episode_frames(Agent.load(policy_path), seed, every=4, size=(300, 200), label='2x speed')
    frames = [f.convert('P', palette=0, colors=96) for f in frames]
    frames[0].save(out_gif, save_all=True, append_images=frames[1:] + [frames[-1]] * 12, duration=80, optimize=True)


def _champions(run_dir):
    return sorted(glob.glob(os.path.join(run_dir, 'champions', 'step_*.pt')))


def models():
    """The best model of the newest run, two earlier stages of it (to see the learning) and the best of older runs."""
    runs = sorted((d for d in glob.glob(os.path.join(ROOT, 'runs', '*')) if os.path.isdir(d)), key=os.path.getmtime)
    out = []
    for i, run in enumerate(reversed(runs)):
        champs = _champions(run)
        if not champs:
            continue
        name = os.path.basename(run)
        mid = next((c for c in champs if int(os.path.basename(c)[5:-3]) >= 1_000_000), None)
        picks = [champs[-1]] if i else [c for c in (champs[-1], mid, champs[0]) if c]   # stable: only "best" moves
        for k, c in enumerate(dict.fromkeys(picks)):
            step = int(os.path.basename(c)[5:-3])
            label = f"{name} @ {step / 1e6:.1f} M" + (" (best)" if k == 0 else "")
            note = ("Best validation score of this run" if k == 0 else "An earlier best of the same run, to compare") + \
                   f": a checkpoint after {step:,} steps of driving."
            out.append({'id': f"{name}_s{step}", 'label': label, 'path': os.path.relpath(c, ROOT), 'note': note})
        if len(out) >= 4:
            break
    return out[:4]


def games():
    """Ten fixed test tracks (seeds 2024-2033), never used in training or selection."""
    return json.load(open(os.path.join(ROOT, 'experiments', 'play_tracks.json')))


def render_play(policy_path, game, out_mp4):
    """One lap attempt at 2x speed (every 2nd frame at 25 fps), 480x320, plays once; last frame shows the outcome."""
    os.nice(10)
    from a2c_agent import Agent
    from viz.render import episode_frames
    frames, r = episode_frames(Agent.load(os.path.join(ROOT, policy_path)), game['seed'], every=2, size=(480, 320), label='2x speed')
    encode_mp4(frames, out_mp4, fps=25, hold=25)
    result = 'lap' if r['lap'] else f"{100 * r['tiles']:.0f}% of track"
    return {'ok': bool(r['lap']), 'result': result, 'score': r['score'],
            'detail': f"score {r['score']:.1f} in {r['steps']} frames"}
