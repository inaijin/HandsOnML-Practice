# For Normal Play Do python main.py --mode play --render
# For Evaluation Do python main.py --mode eval --checkpoint checkpoints/dqn_ep15550.pth --render
# For Training Do python main.py --mode train --episodes 20000 --double_dqn --render_every 200 --render

import torch
import pygame
import argparse
from train import Trainer
from env.pacman_env import PacmanEnv
from agents.dqn_agent import DQNAgent
from game.game import UP, DOWN, LEFT, RIGHT, STAY

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval", "play"], default="train")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--render_every", type=int, default=50)
    parser.add_argument("--no_render", action="store_true")
    parser.add_argument("--double_dqn", action="store_true")
    return parser.parse_args()

# MPS For Macbook Users :)
def select_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def human_play(env: PacmanEnv):
    print("=== Human Play Mode ===")
    print("Arrow keys or WASD to move Pac-Man.")
    print("Press R to restart, ESC to quit.")

    env.reset()
    done = False
    clock = pygame.time.Clock()

    key_to_action = {
        pygame.K_UP: UP,
        pygame.K_w: UP,
        pygame.K_DOWN: DOWN,
        pygame.K_s: DOWN,
        pygame.K_LEFT: LEFT,
        pygame.K_a: LEFT,
        pygame.K_RIGHT: RIGHT,
        pygame.K_d: RIGHT
    }

    current_action = STAY

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit
            elif event.type == pygame.KEYDOWN:
                if event.key in key_to_action:
                    current_action = key_to_action[event.key]
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    raise SystemExit
                elif event.key == pygame.K_r:
                    obs = env.reset()
                    done = False
                    current_action = STAY

        if not done:
            _, _, done, info = env.step(current_action)
            env.render(delay=100)
        else:
            # Game over screen
            env.render(delay=0)
            print("Game over!", info)
            print("Press R to restart or ESC to quit.")
            clock.tick(5)

def main():
    args = parse_args()
    device = select_device()
    env = PacmanEnv(grid_size=10)
    obs_shape = env.observation_space_shape()
    action_size = env.action_size()
    agent = DQNAgent(obs_shape, action_size, device=device, double_dqn=args.double_dqn)

    if args.mode == "train":
        trainer = Trainer(env, agent, device=device)
        trainer.train(
            episodes=args.episodes,
            render=(not args.no_render and args.render),
            render_every=args.render_every
        )
    elif args.mode == "play":
        human_play(env)
    else:
        if args.checkpoint:
            agent.load(args.checkpoint)
        else:
            print("No checkpoint provided for evaluation.")
            return
        # run evaluation/visualization
        obs = env.reset()
        done = False
        total = 0
        while True:
            action = agent.act(obs, epsilon=0.0)  # greedy
            obs, reward, done, info = env.step(action)
            total += reward
            env.render()
            if done:
                print("Episode finished, score:", total)
                total = 0
                _ = input("Press Enter to play again, Ctrl+C to exit...")
                obs = env.reset()
                done = False

if __name__ == "__main__":
    main()
