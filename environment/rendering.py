import os
import math
import numpy as np
import pygame
 
from learning_path_env import LearningPathEnv
 
DEFAULT_TOPIC_NAMES = ["Loops", "Functions", "Data", "OOP", "Algebra", "Stats"]
ACTION_NAMES = ["move-0", "move-1", "move-2", "review", "rest"]
 
 
def polar_layout(center, radius, n):
    cx, cy = center
    pts = []
    for i in range(n):
        theta = 2 * math.pi * i / n - math.pi / 2
        pts.append(
            (
                int(cx + radius * math.cos(theta)),
                int(cy + radius * math.sin(theta)),
            )
        )
    return pts
 
 
def draw_bar(screen, x, y, w, h, frac,
             fg=(50, 200, 120), bg=(60, 60, 60), border=(255, 255, 255)):
    pygame.draw.rect(screen, bg, (x, y, w, h), border_radius=4)
    pygame.draw.rect(screen, fg, (x, y, max(0, int(w * frac)), h), border_radius=4)
    pygame.draw.rect(screen, border, (x, y, w, h), width=1, border_radius=4)
 
 
def run_random_env(time_budget=120, fps=24, slowdown_ms=400,
                   steps=0, save_frames=False, out_dir="results/videos"):
    pygame.init()
    W, H = 900, 700
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("LearningPathEnv — Random Policy Viewer")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("arial", 22)
    small = pygame.font.SysFont("arial", 16)
    big = pygame.font.SysFont("arial", 28)
 
    center = (W // 2, H // 2 + 20)
    ring_radius = 230
    topic_radius = 36
    student_r = 18
 
    env = LearningPathEnv(time_budget=time_budget)
    obs, _ = env.reset()
 
    n = env.n
    topic_names = (
        DEFAULT_TOPIC_NAMES[:n]
        if len(DEFAULT_TOPIC_NAMES) >= n
        else [f"Topic {i}" for i in range(n)]
    )
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
                a = trail[i - 1]
                b = trail[i]
                pygame.draw.line(screen, (255, 120, 90), a, b, 3)
 
        # topics
        for i, (x, y) in enumerate(points):
            mastery = float(env.mastery[i])
            # base node
            pygame.draw.circle(screen, (50, 55, 70), (x, y), topic_radius)
            # mastery rim
            rim = (
                int(120 + 135 * mastery),
                int(150 + 80 * mastery),
                220,
            )
            pygame.draw.circle(screen, rim, (x, y), topic_radius, width=3)
            # label + mastery bar
            s_txt = small.render(topic_names[i], True, (210, 210, 210))
            screen.blit(s_txt, (x - s_txt.get_width() // 2, y + topic_radius + 8))
            draw_bar(screen, x - 40, y + topic_radius + 28, 80, 8, mastery)
 
        # student (red ring) at current topic
        sx, sy = points[env.pos]
        pygame.draw.circle(
            screen, (255, 80, 80), (sx, sy), topic_radius + 8, width=3
        )
        pygame.draw.circle(screen, (230, 230, 255), (sx, sy), student_r)
 
        # HUD
        pygame.draw.rect(screen, (34, 36, 44), (0, 0, W, 80))
        screen.blit(
            big.render(
                "LearningPathEnv — Random Policy", True, (240, 240, 240)
            ),
            (20, 16),
        )
        hud = (
            f"t={t}   time_left={env.time_left}   "
            f"fatigue={env.fatigue:.2f}   reward={last_reward:.3f}   "
            f"action={last_action_name}   avg_mastery={float(np.mean(env.mastery)):.3f}"
        )
        screen.blit(font.render(hud, True, (230, 230, 230)), (20, 48))
 
        pygame.display.flip()
 
    while running:
        # Handle quit events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
 
        if not running:
            break
 
        # random action
        action = env.action_space.sample()
        obs, r, done, _, _ = env.step(int(action))
        last_reward = r
        last_action_name = (
            ACTION_NAMES[int(action)]
            if int(action) < len(ACTION_NAMES)
            else str(int(action))
        )
        t += 1
 
        trail.append(points[env.pos])
        if len(trail) > 1200:
            trail.pop(0)
 
        draw_frame()
 
        clock.tick(fps)
        if slowdown_ms > 0:
            pygame.time.delay(slowdown_ms)
 
        if save_frames:
            pygame.image.save(
                screen,
                os.path.join(out_dir, f"random_env_frame_{t:05d}.png"),
            )
 
        if done or (steps > 0 and t >= steps):
            break
 
    print(
        f"Episode done. Steps={t}, final avg mastery={float(np.mean(env.mastery)):.3f}"
    )
 
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or event.type == pygame.KEYDOWN:
                waiting = False
 
    pygame.quit()
 
 
if __name__ == "__main__":
    run_random_env(
        time_budget=40,   # shorter episode just to visualise
        fps=16,
        slowdown_ms=400,
        steps=0,
        save_frames=False,
    )