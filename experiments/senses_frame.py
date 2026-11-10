"""An example frame for the 'How it sees' tab: the model driving, its rays and look-ahead points drawn, and the 49
numbers it received at that moment -> docs/img/sees.png, docs/data/senses.json."""
import json, sys
import numpy as np
from PIL import Image
from a2c_agent import Agent
from env.config import EnvConfig
from env.make_env import make_env
from viz.render import draw_rays

model, seed, frame = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
agent = Agent.load(model)
env = make_env(EnvConfig(training=False, render_mode="rgb_array"))
obs, _ = env.reset(seed=seed)
for _ in range(frame):
    obs, *_ = env.step(agent.drive(obs))
img = Image.fromarray(env.render())
draw_rays(img, env, obs)
img.save("docs/img/sees.png")
names = ["speed", "forward velocity", "sideways velocity", "yaw rate", "steer angle",
         "front-left wheel on road", "front-right wheel on road", "rear-left wheel on road", "rear-right wheel on road",
         "front wheel rim speed", "rear wheel rim speed", "offset from centre line", "heading error (sin)", "heading error (cos)"]
for k in env.cfg.lookahead:
    names += [f"point {k} tiles ahead: forward", f"point {k} tiles ahead: left"]
names += [f"ray {i + 1} ({a:+.0f} deg)" for i, a in enumerate(np.degrees(env.ray_angles))]
act = agent.drive(obs)
json.dump({"model": model, "seed": seed, "frame": frame, "names": names, "values": [round(float(v), 3) for v in obs],
           "action": [round(float(a), 3) for a in act]}, open("docs/data/senses.json", "w"), indent=0)
print("ok", act)
