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


def models():
    return []


def games():
    return []
