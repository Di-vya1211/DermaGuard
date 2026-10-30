import random
import cv2
import numpy as np
from quality_agent import ImageQualityAgent, EVAL_DIR

agent = ImageQualityAgent()
files = random.Random(7).sample(sorted(EVAL_DIR.glob("*.jpg")), 40)

def blur(i):   return cv2.GaussianBlur(i, (0, 0), 6)
def dark(i):   return (i * 0.3).astype(np.uint8)
def bright(i): return np.clip(i.astype(int) + 100, 0, 255).astype(np.uint8)
def glare(i):
    o = i.copy()
    for cx, cy in [(150, 120), (400, 200), (300, 350)]:
        cv2.circle(o, (cx, cy), 60, (255, 255, 255), -1)
    return o
def occlude(i):
    o = i.copy(); h, w = o.shape[:2]
    o[:, : int(w * 0.4)] = 0
    return o

tests = {"original": (lambda i: i, None), "blur": (blur, "blur"), "dark": (dark, "brightness"),
         "bright": (bright, "brightness"), "glare": (glare, "glare"), "occlusion": (occlude, "occlusion")}

print(f"{'variant':10s} {'not GOOD':>9s} {'right check found':>18s}")
for name, (fn, expect) in tests.items():
    notgood = hit = 0
    for f in files:
        r = agent.assess(fn(cv2.imread(str(f))))
        notgood += r["status"] != "GOOD"
        hit += (expect in r["failed_checks"]) if expect else 0
    hit_txt = f"{hit/len(files)*100:.0f}%" if expect else "-"
    print(f"{name:10s} {notgood/len(files)*100:8.0f}% {hit_txt:>18s}")