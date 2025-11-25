# visualize/pygame_view.py  (slow + step mode)
import os, math, argparse
import numpy as np
import pygame
from stable_baselines3 import PPO, DQN, A2C
from environment.learning_path_env import LearningPathEnv

DEFAULT_TOPIC_NAMES = ["Loops", "Functions", "Data", "OOP", "Algebra", "Stats"]
MODEL_PATHS = {"PPO": "results/ppo_model.zip", "DQN": "results/dqn_model.zip", "A2C": "results/a2c_model.zip"}
ACTION_NAMES = ["move-0", "move-1", "move-2", "review", "rest"]

def load_model(alg):
    if alg == "PPO": return PPO.load(MODEL_PATHS[alg])
    if alg == "DQN": return DQN.load(MODEL_PATHS[alg])
    if alg == "A2C": return A2C.load(MODEL_PATHS[alg])
    raise ValueError(f"Unknown alg {alg}")

def polar_layout(center, radius, n):
    cx, cy = center
    pts = []
    for i in range(n):
        theta = 2*math.pi*i/n - math.pi/2
        pts.append((int(cx + radius*math.cos(theta)), int(cy + radius*math.sin(theta))))
    return pts

def draw_bar(screen, x, y, w, h, frac, fg=(50,200,120), bg=(60,60,60), border=(255,255,255)):
    pygame.draw.rect(screen, bg, (x, y, w, h), border_radius=4)
    pygame.draw.rect(screen, fg, (x, y, max(0, int(w*frac)), h), border_radius=4)
    pygame.draw.rect(screen, border, (x, y, w, h), width=1, border_radius=4)

def run(alg="PPO", time_budget=120, steps=0, fps=24, slowdown_ms=400, step_mode=False, save_frames=False, out_dir="results/videos"):
    assert alg in MODEL_PATHS, f"Unknown alg {alg}"
    if not os.path.exists(MODEL_PATHS[alg]):
        raise SystemExit(f"Model not found: {MODEL_PATHS[alg]}")

    pygame.init()
    W, H = 900, 700
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption(f"{alg} — Learning Path Viewer (slow/step)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("arial", 22)
    small = pygame.font.SysFont("arial", 16)
    big = pygame.font.SysFont("arial", 28)

    center = (W//2, H//2 + 20)
    ring_radius = 230
    topic_radius = 36
    student_r = 18

    env = LearningPathEnv(time_budget=time_budget)
    model = load_model(alg)
    obs, _ = env.reset()
    n = env.n
    topic_names = DEFAULT_TOPIC_NAMES[:n] if len(DEFAULT_TOPIC_NAMES) >= n else [f"Topic {i}" for i in range(n)]
    points = polar_layout(center, ring_radius, n)
    trail = []

    if save_frames:
        os.makedirs(out_dir, exist_ok=True)

    t = 0
    last_reward = 0.0
    last_action_name = ""
    running = True

    def draw_frame():
        screen.fill((24, 26, 32))
        # edges
        edge_color = (90, 90, 110)
        for i, nbrs in env.adj.items():
            x1, y1 = points[i]
            for j in nbrs:
                x2, y2 = points[j]
                pygame.draw.line(screen, edge_color, (x1, y1), (x2, y2), 2)
        # trail
        if len(trail) > 1:
            for i in range(1, len(trail)):
                a = trail[i-1]; b = trail[i]
                pygame.draw.line(screen, (255, 120, 90), a, b, 3)

        # topics
        for i, (x, y) in enumerate(points):
            mastery = float(env.mastery[i])
            # base node
            pygame.draw.circle(screen, (50, 55, 70), (x, y), topic_radius)
            # mastery rim
            rim = (int(120+135*mastery), int(150+80*mastery), 220)
            pygame.draw.circle(screen, rim, (x, y), topic_radius, width=3)
            # label + mastery bar
            s = small.render(topic_names[i], True, (210,210,210))
            screen.blit(s, (x - s.get_width()//2, y + topic_radius + 8))
            draw_bar(screen, x-40, y + topic_radius + 28, 80, 8, mastery)

        # student (red ring) at current topic
        sx, sy = points[env.pos]
        pygame.draw.circle(screen, (255, 80, 80), (sx, sy), topic_radius+8, width=3)
        pygame.draw.circle(screen, (230, 230, 255), (sx, sy), student_r)

        # HUD
        pygame.draw.rect(screen, (34, 36, 44), (0, 0, W, 80))
        screen.blit(big.render(f"{alg} — Mission-Based Learning", True, (240,240,240)), (20, 16))
        hud = f"t={t}   time_left={env.time_left}   fatigue={env.fatigue:.2f}   reward={last_reward:.2f}   action={last_action_name}"
        screen.blit(font.render(hud, True, (230,230,230)), (20, 48))

        pygame.display.flip()

    while running:
        # Step control: either wait for SPACE (step_mode) or advance after a delay
        waiting_for_step = True
        while waiting_for_step:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False; waiting_for_step = False
                elif step_mode and event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    waiting_for_step = False
            if not step_mode:
                waiting_for_step = False
            draw_frame()
            # tick low fps while waiting so UI stays responsive
            clock.tick(max(10, fps))

        if not running:
            break

        # agent acts
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, _, _ = env.step(int(action))
        last_reward = r
        last_action_name = ACTION_NAMES[int(action)] if int(action) < len(ACTION_NAMES) else str(int(action))
        t += 1

        trail.append(points[env.pos])
        if len(trail) > 1200:
            trail.pop(0)

        draw_frame()

        # slow it down:
        clock.tick(fps)            # frame rate cap
        if slowdown_ms > 0:        # extra per-step delay
            pygame.time.delay(slowdown_ms)

        # save frame if requested
        if save_frames:
            pygame.image.save(screen, os.path.join(out_dir, f"{alg.lower()}_pg_{t:05d}.png"))

        if done or (steps > 0 and t >= steps):
            break

    pygame.quit()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alg", choices=["PPO","DQN","A2C"], default="PPO")
    ap.add_argument("--time-budget", type=int, default=120)
    ap.add_argument("--steps", type=int, default=0, help="0 = run until episode ends")
    ap.add_argument("--fps", type=int, default=24, help="refresh rate; lower = slower")
    ap.add_argument("--slowdown-ms", type=int, default=400, help="extra delay after each action")
    ap.add_argument("--step-mode", action="store_true", help="press SPACE to advance one action")
    ap.add_argument("--save-frames", action="store_true")
    ap.add_argument("--out-dir", default="results/videos")
    args = ap.parse_args()

    run(
        alg=args.alg,
        time_budget=args.time_budget,
        steps=args.steps,
        fps=args.fps,
        slowdown_ms=args.slowdown_ms,
        step_mode=args.step_mode,
        save_frames=args.save_frames,
        out_dir=args.out_dir,
    )

if __name__ == "__main__":
    main()
