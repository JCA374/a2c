"""Build the narrated films for the site from real episodes, real logs and real results.

    python video/build_film.py howto        # needs voice/film-howto exported (per-scene MP3s)
    python video/build_film.py explainer    # needs voice/film-explainer exported
Output: docs/video/<film>.mp4 (1280x720, 25 fps) + a poster, listed in docs/video/films.json for The model tab.
Scenes match the ## sections of video/<film>.md one to one. Footage is never repeated: each scene drives its own
track. When the narration runs longer than the footage, the last frame is held.
"""
import csv
import glob
import io
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from a2c_agent import Agent  # noqa: E402
from viz.render import episode_frames  # noqa: E402

FPS, SW, SH, GAP = 25, 1280, 720, 0.7
BG, PANEL, INK, MUTED = (14, 17, 22), (26, 30, 38), (232, 232, 236), (150, 152, 160)
ACC, GOOD, BAD, WARN = (120, 160, 255), (105, 219, 124), (255, 120, 120), (255, 169, 77)
RUN = "runs/ppo_v2_r1"
TMP = "video/build"
OUT = "docs/video"


def font(size, bold=False, mono=False):
    name = "DejaVuSansMono.ttf" if mono else ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
    return ImageFont.truetype(name, size)


def canvas():
    return Image.new("RGB", (SW, SH), BG)


def wrap(d, text, f, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if d.textlength(t, font=f) > width and cur:
            lines.append(cur)
            cur = w
        else:
            cur = t
    return lines + ([cur] if cur else [])


def header(d, title, sub=None):
    d.text((60, 40), title, font=font(40, True), fill=INK)
    if sub:
        d.text((60, 96), sub, font=font(22), fill=MUTED)


def champions():
    return sorted(glob.glob(f"{RUN}/champions/step_*.pt"))


def step_of(path):
    return int(os.path.basename(path)[5:-3])


def best_model():
    goals = sorted(glob.glob("results/goal_*.json"), key=lambda p: json.load(open(p))["mean"])
    if goals:
        return json.load(open(goals[-1]))["policy"]
    return f"{RUN}/best.pt"


def jpeg(img):
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=92)
    return buf.getvalue()


class Frames:
    """An episode's frames kept JPEG-compressed in memory (1000 raw 900x600 frames would be 1.6 GB)."""

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return Image.open(io.BytesIO(self.data[i])).convert("RGB")


def footage(model, seed, every=1, rays=True, label=None):
    frames, res = episode_frames(Agent.load(model), seed, every=every, size=(600, 400), rays=rays, label=label, store=jpeg)
    return Frames(frames), res


def play(frames, n, i):
    return frames[min(i, len(frames) - 1)].resize((900, 600))


def terminal(lines, n_visible, title="terminal", size=(1160, 520), fsize=21):
    img = Image.new("RGB", size, (10, 12, 16))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, size[0], 34], fill=PANEL)
    for k, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        d.ellipse([14 + 22 * k, 11, 26 + 22 * k, 23], fill=c)
    d.text((size[0] / 2, 17), title, font=font(16), fill=MUTED, anchor="mm")
    y = 50
    f = font(fsize, mono=True)
    for line in lines[:n_visible]:
        col = GOOD if line.startswith("$") else INK if not line.startswith("#") else MUTED
        for part in wrap(d, line, f, size[0] - 40) or [""]:
            d.text((20, y), part, font=f, fill=col)
            y += fsize + 9
    return img


def chart(xs, ys, size, xlab, ylab, upto=None, goal=None, ylim=None, color=ACC):
    w, h = size
    img = Image.new("RGB", size, PANEL)
    d = ImageDraw.Draw(img)
    L, R, T, B = 80, 20, 20, 60
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    y0, y1 = ylim or (min(ys.min(), 0), max(ys.max(), goal or 0) * 1.05)
    x0, x1 = 0, xs.max()
    X = lambda x: L + (x - x0) / (x1 - x0) * (w - L - R)
    Y = lambda y: h - B - (y - y0) / (y1 - y0) * (h - T - B)
    f = font(16)
    for v in np.linspace(y0, y1, 6):
        d.line([L, Y(v), w - R, Y(v)], fill=(45, 50, 60))
        d.text((L - 8, Y(v)), f"{v:.0f}", font=f, fill=MUTED, anchor="rm")
    for v in np.linspace(x0, x1, 6):
        d.text((X(v), h - B + 10), f"{v:.1f}", font=f, fill=MUTED, anchor="ma")
    d.text(((L + w - R) / 2, h - 22), xlab, font=f, fill=MUTED, anchor="mm")
    d.text((10, T), ylab, font=f, fill=MUTED)
    if goal is not None:
        d.line([L, Y(goal), w - R, Y(goal)], fill=BAD, width=2)
        d.text((w - R - 6, Y(goal) - 6), f"goal {goal:.0f}", font=f, fill=BAD, anchor="rb")
    k = len(xs) if upto is None else max(2, int(upto * len(xs)))
    pts = [(X(x), Y(y)) for x, y in zip(xs[:k], ys[:k])]
    if len(pts) > 1:
        d.line(pts, fill=color, width=3)
    for p in pts:
        d.ellipse([p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4], fill=color)
    return img


def val_series():
    rows = list(csv.DictReader(open(f"{RUN}/stats.csv")))
    pts = [(int(r["step"]) / 1e6, float(r["val_score"])) for r in rows if r["val_score"]]
    return [p[0] for p in pts], [p[1] for p in pts]


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def bullets(d, items, x, y, width, shown, size=26):
    f = font(size)
    for k, it in enumerate(items[:shown]):
        d.ellipse([x, y + 12, x + 10, y + 22], fill=ACC)
        for line in wrap(d, it, f, width - 30):
            d.text((x + 26, y), line, font=f, fill=INK)
            y += size + 10
        y += 14
    return y


# ---------------- the instruction film ----------------

def h_intro(n):
    frames, res = footage(best_model(), 2030, every=1, label="the best model, test track 2030")
    for i in range(n):
        img = canvas()
        d = ImageDraw.Draw(img)
        d.text((60, 70), "Drive A2C", font=font(56, True), fill=INK)
        d.text((60, 140), "how to use it", font=font(30), fill=ACC)
        bullets(d, ["set it up", "train a driver", "watch it race", "measure its score", "run the tests"], 60, 220, 280,
                1 + int(5 * i / max(1, n * 0.6)))
        d.text((60, 620), "49 numbers in, 2 out:", font=font(22), fill=MUTED)
        d.text((60, 650), "steer and pedal", font=font(22), fill=MUTED)
        img.paste(play(frames, n, i).resize((810, 540)), (420, 90))
        yield img


def h_install(n):
    lines = ["$ git clone https://github.com/JCA374/a2c.git && cd a2c",
             "$ python -m venv .venv",
             "$ .venv/bin/python -m pip install -r requirements.txt",
             "", "# gymnasium[box2d]  torch  numba  numpy  pillow  pytest",
             "# no GPU needed: trained on an 8-thread laptop"]
    for i in range(n):
        img = canvas()
        d = ImageDraw.Draw(img)
        header(d, "1. Set it up", "three commands")
        img.paste(terminal(lines, 1 + int(len(lines) * i / max(1, n * 0.7))), (60, 150))
        yield img


def h_train(n):
    log = [l.rstrip() for l in open(f"{RUN}.log") if l.startswith("upd") and " val " in l and "val  " not in l][:10]
    lines = ["$ .venv/bin/python train_a2c.py --run my_run", *log]
    xs, ys = val_series()
    for i in range(n):
        img = canvas()
        d = ImageDraw.Draw(img)
        t = i / n
        if t < 0.55:
            header(d, "2. Train", "8 cars on random tracks; validation on 16 fixed tracks every 100,000 frames")
            img.paste(terminal(lines, 1 + int(len(lines) * t / 0.45), title="training log (real)", fsize=17), (60, 150))
        else:
            header(d, "2. Train", "runs/my_run/: stats.csv, events.jsonl, curriculum.json, best.pt, champions/")
            img.paste(chart(xs, ys, (1160, 500), "million frames of driving", "validation score (16 tracks)",
                            upto=ease((t - 0.55) / 0.35), goal=900, ylim=(-100, 1000)), (60, 160))
        yield img


def h_watch(n):
    ch = champions()
    frames, res = footage(ch[len(ch) // 2], 2027, every=1, rays=False, label="test track 2027: four hairpins")
    for i in range(n):
        img = canvas()
        d = ImageDraw.Draw(img)
        header(d, "3. Watch it", None)
        d.text((60, 110), "$ .venv/bin/python play.py runs/my_run/best.pt", font=font(22, mono=True), fill=GOOD)
        img.paste(play(frames, n, i).resize((780, 520)), (60, 170))
        d.text((880, 200), "On the site:", font=font(26, True), fill=INK)
        bullets(d, ["the Play tab", "pick a model", "pick one of 10 fixed test tracks", "the clip plays once and ends on its result"],
                880, 250, 360, 4, size=22)
        yield img


def h_evaluate(n):
    goals = sorted(glob.glob("results/goal_*.json"))
    g = json.load(open(goals[-1])) if goals else None
    lines = ["$ .venv/bin/python evaluate.py runs/my_run/best.pt", "# 100 unseen tracks, deterministic, the game's own rules,",
             "# in the reference env (pixels drawn, pure-Python lidar)"]
    if g:
        lines += [f"{g['label']}: mean {g['mean']:.1f} (95% {g['ci'][0]:.1f}-{g['ci'][1]:.1f}), median {g['median']:.1f},",
                  f"laps {100 * g['laps']:.0f}%, >=900 {100 * g['share_900']:.0f}%, worst {g['min']:.1f}",
                  "# -> results/goal_*.json -> the goal table on the Log tab"]
    scores = sorted(r["score"] for r in g["rows"]) if g else []
    for i in range(n):
        img = canvas()
        d = ImageDraw.Draw(img)
        header(d, "4. Measure it", "score = +1000/N per new tile, -0.1 per frame, 1000 frames")
        img.paste(terminal(lines, 1 + int(len(lines) * i / max(1, n * 0.5)), size=(1160, 280), fsize=19), (60, 150))
        if scores and i > n * 0.4:
            k = max(2, int(len(scores) * ease((i - 0.4 * n) / (0.4 * n))))
            img.paste(chart(np.arange(1, k + 1) / 100, scores[:k], (1160, 250),
                            "the 100 goal tracks, sorted by score (share of tracks)", "score", goal=900,
                            ylim=(min(min(scores), 0), 1000), color=WARN), (60, 450))
        yield img


def h_tests(n):
    out = subprocess.run([".venv/bin/python", "-m", "pytest", "-q", "-p", "no:warnings", "tests"],
                         capture_output=True, text=True).stdout.strip().splitlines()
    lines = ["$ .venv/bin/python -m pytest", *[l for l in out if l.strip()][-3:],
             "# test_lidar: numba ray cast == pure Python, bit for bit",
             "# test_equivalence: training env == reference env, bit for bit",
             "# test_agent: log-probabilities, GAE, PPO and A2C updates, save/load"]
    for i in range(n):
        img = canvas()
        d = ImageDraw.Draw(img)
        header(d, "5. Test it", "install, train, watch, evaluate, test")
        img.paste(terminal(lines, 1 + int(len(lines) * i / max(1, n * 0.6)), size=(1160, 330), fsize=19), (60, 150))
        d.text((60, 520), "The site follows the current run, rebuilt every 15 minutes:", font=font(24), fill=INK)
        d.text((60, 560), "Log · Curriculum · How it sees · The model · Play · Speed · Lessons · Podcast", font=font(24, True), fill=ACC)
        d.text((60, 610), "jca374.github.io/a2c", font=font(24), fill=MUTED)
        yield img


# ---------------- the explainer film ----------------

def e_task(n):
    frames, res = footage(best_model(), 2031, every=1, rays=False, label="test track 2031")
    for i in range(n):
        img = canvas()
        d = ImageDraw.Draw(img)
        header(d, "The task", "Gymnasium's CarRacing: a new random track every time")
        img.paste(play(frames, n, i).resize((780, 520)), (60, 160))
        bullets(d, ["+1000/N for each of the N tiles, the first time it touches one",
                    "-0.1 for every frame", "1000 frames, then time is up", "900+ on average: the classic 'solved'"],
                880, 190, 370, 1 + int(4 * i / max(1, n * 0.6)), size=22)
        yield img


def e_blind(n):
    frames, _ = footage(champions()[0], 2025, every=1, rays=False, label="illustration: an early v2 car, greyed out")
    for i in range(n):
        img = canvas()
        d = ImageDraw.Draw(img)
        header(d, "Version one was blind", "December 2025: an A2C agent with 32 lidar rays")
        fr = play(frames, n, i).resize((780, 520)).convert("L").convert("RGB")
        img.paste(fr, (60, 160))
        lines = ["obs[8:40] = [0. 0. 0. 0. 0. 0. 0. 0.", "            0. 0. 0. 0. 0. 0. 0. 0.",
                 "            0. 0. 0. 0. 0. 0. 0. 0.", "            0. 0. 0. 0. 0. 0. 0. 0.]"]
        img.paste(terminal(lines, 4, title="every ray, every frame", size=(380, 200), fsize=15), (870, 170))
        bullets(d, ["it tested Box2D bodies as polygons: never 'on the road'", "traction always 1, finish bonus never paid",
                    "score on 100 unseen tracks: -42.9"], 880, 400, 370, 1 + int(3 * i / max(1, n * 0.7)), size=21)
        yield img


def e_eyes(n):
    frames, _ = footage(best_model(), 2024, every=1, rays=True, label="blue: lidar rays · yellow: centre line ahead")
    for i in range(n):
        img = canvas()
        d = ImageDraw.Draw(img)
        header(d, "New eyes: 49 numbers", None)
        img.paste(play(frames, n, i).resize((780, 520)), (60, 120))
        bullets(d, ["speed, slide and spin", "which wheels are on the grass", "where it sits on the road, and its angle to it",
                    "8 centre-line points up to 40 tiles ahead", "19 rays to the road edge"], 880, 140, 370,
                1 + int(5 * i / max(1, n * 0.7)), size=22)
        yield img


def e_learn(n):
    for i in range(n):
        img = canvas()
        d = ImageDraw.Draw(img)
        header(d, "How it learns", "actor-critic: one network drives, one judges")
        t = i / n
        boxes = [("drive", "8 cars, 256 frames each", 0.0), ("judge", "advantage: better or worse than the critic expected?", 0.25),
                 ("improve", "more of the good actions, less of the bad", 0.5), ("repeat", "PPO: 10 careful passes per batch", 0.75)]
        for k, (a, b, t0) in enumerate(boxes):
            if t < t0 * 0.8:
                continue
            x = 60 + k * 300
            d.rounded_rectangle([x, 220, x + 270, 420], 16, fill=PANEL, outline=ACC, width=2)
            d.text((x + 135, 260), a, font=font(32, True), fill=INK, anchor="mm")
            for j, line in enumerate(wrap(d, b, font(20), 230)):
                d.text((x + 135, 310 + 28 * j), line, font=font(20), fill=MUTED, anchor="mm")
            if k:
                d.line([x - 26, 320, x - 4, 320], fill=MUTED, width=3)
        d.text((60, 500), "Paid for one thing only: the game's own score.", font=font(28), fill=WARN)
        yield img


def e_progress(n):
    ch = champions()
    picks = [ch[0], ch[len(ch) // 2], best_model()]
    clips = [footage(p, 2026, every=2, rays=False)[0] for p in picks]
    labels = [f"{step_of(picks[0]) / 1e6:.1f} M frames", f"{step_of(picks[1]) / 1e6:.1f} M frames", "best"]
    xs, ys = val_series()
    for i in range(n):
        img = canvas()
        d = ImageDraw.Draw(img)
        header(d, "Learning to race", "the same test track at three moments of one run (2x speed)")
        for k, (c, lab) in enumerate(zip(clips, labels)):
            img.paste(play(c, n, i).resize((390, 260)), (50 + k * 400, 140))
            d.text((50 + k * 400, 408), lab, font=font(22, True), fill=INK)
        img.paste(chart(xs, ys, (1180, 250), "million frames", "validation", upto=ease(i / (0.8 * n)), goal=900,
                        ylim=(-100, 1000)), (50, 450))
        yield img


def e_result(n):
    goals = sorted((json.load(open(p)) for p in glob.glob("results/goal_*.json")), key=lambda g: g["mean"])
    g = goals[-1] if goals else None
    for i in range(n):
        img = canvas()
        d = ImageDraw.Draw(img)
        header(d, "The result", "100 tracks it never saw, driven once, deterministically")
        rows = [("v1 A2C, blind lidar", -42.9, MUTED)] + ([(f"v2 PPO, {step_of(g['policy']) / 1e6:.1f} M steps", g["mean"], GOOD)] if g else [])
        X0 = 420
        Xv = lambda v: X0 + (v + 100) / 1100 * 760
        for k, (lab, v, col) in enumerate(rows):
            y = 220 + 120 * k
            d.text((60, y + 10), lab[:34], font=font(22), fill=INK)
            grow = ease(i / (0.5 * n))
            xa, xb = Xv(0), Xv(0) + (Xv(v) - Xv(0)) * grow
            d.rectangle([min(xa, xb), y, max(xa, xb), y + 50], fill=col)
            d.text((Xv(v) + (10 if v >= 0 else -10), y + 25), f"{v:.1f}", font=font(26, True), fill=INK,
                   anchor="lm" if v >= 0 else "rm")
        d.line([Xv(900), 190, Xv(900), 470], fill=BAD, width=2)
        d.text((Xv(900), 180), "goal 900", font=font(18), fill=BAD, anchor="mb")
        if g:
            d.text((60, 520), f"95% interval {g['ci'][0]:.0f} to {g['ci'][1]:.0f} · laps finished {100 * g['laps']:.0f}% · "
                              f"tracks at 900 or more {100 * g['share_900']:.0f}% · worst {g['min']:.0f}", font=font(22), fill=MUTED)
        yield img


FILMS = {
    "howto": ("How to use this project", "Set up, train, watch, evaluate and test, in about three minutes.",
              [h_intro, h_install, h_train, h_watch, h_evaluate, h_tests]),
    "explainer": ("How the car learned to race", "Why version one was blind, what the car sees now, how it learns, and how well it drives.",
                  [e_task, e_blind, e_eyes, e_learn, e_progress, e_result]),
}


def duration(path):
    out = subprocess.run(["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                         capture_output=True, text=True).stdout
    return float(out)


def render_scene(film, k, fn):
    a = f"voice/film-{film}/audio/voiceover/ch{k:02d}.mp3"
    a = a if os.path.exists(a) else None
    dur = (duration(a) if a else 12.0) + GAP
    n = int(dur * FPS)
    out = os.path.join(TMP, f"{film}_scene{k:02d}.mp4")
    cmd = ["nice", "-n", "10", "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{SW}x{SH}",
           "-r", str(FPS), "-i", "-"]
    if a:
        cmd += ["-i", a, "-af", f"apad=pad_dur={GAP + 1}", "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "1"]
    cmd += ["-t", f"{n / FPS:.3f}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p",
            "-r", str(FPS), out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    count, last = 0, None
    for img in fn(n):
        if count >= n:
            break
        if count == n // 2:
            img.save(os.path.join(TMP, f"{film}_preview{k:02d}.png"))
        p.stdin.write(img.tobytes())
        last, count = img, count + 1
    while count < n:
        p.stdin.write(last.tobytes())
        count += 1
    p.stdin.close()
    p.wait()
    print(f"scene {k}: {fn.__name__} {dur:.1f} s", flush=True)
    return out


def main():
    film = sys.argv[1]
    os.nice(10)
    title, desc, scenes = FILMS[film]
    os.makedirs(TMP, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    resume = "--resume" in sys.argv  # keep scene files that already exist
    files = [os.path.join(TMP, f"{film}_scene{k:02d}.mp4") if resume and os.path.exists(os.path.join(TMP, f"{film}_scene{k:02d}.mp4"))
             else render_scene(film, k, fn) for k, fn in enumerate(scenes, 1)]
    lst = os.path.join(TMP, f"{film}_list.txt")
    open(lst, "w").write("".join(f"file '{os.path.abspath(f)}'\n" for f in files))
    mp4 = os.path.join(OUT, f"{film}.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy",
                    "-movflags", "+faststart", mp4], check=True)
    Image.open(os.path.join(TMP, f"{film}_preview01.png")).resize((640, 360)).save(os.path.join(OUT, f"{film}.jpg"), quality=85)
    reg = os.path.join(OUT, "films.json")
    films = [f for f in (json.load(open(reg)) if os.path.exists(reg) else []) if f["file"] != f"{film}.mp4"]
    films.append({"file": f"{film}.mp4", "poster": f"{film}.jpg", "title": title,
                  "desc": f"{desc} {duration(mp4) / 60:.1f} minutes, narrated."})
    films.sort(key=lambda f: list(FILMS).index(f["file"][:-4]))
    json.dump(films, open(reg, "w"), indent=1)
    print("film:", mp4, f"{duration(mp4):.0f} s")


if __name__ == "__main__":
    main()
