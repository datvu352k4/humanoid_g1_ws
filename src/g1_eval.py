import argparse
import os
import pickle
from importlib import metadata

import torch

try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib") != "2.2.4":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError(
        "Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.2.4'."
    ) from e
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from g1_env import g1Env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="g1-push")
    parser.add_argument("--ckpt", type=int, default=999)
    args = parser.parse_args()

    gs.init(backend=gs.gpu, precision="32", logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(
        open(f"logs/{args.exp_name}/cfgs.pkl", "rb")
    )

    if "push_params" not in env_cfg:
        env_cfg["push_params"] = {}

    env_cfg["push_params"].update(
        {
            "enable": True,
            "min_force": 1500.0,  # Apply max force
            "max_force": 1500.0,
            "start_interval": 100,  # Push every 1 seconds
            "end_interval": 100,
            "curriculum_steps": 1,  # Skip curriculum
        }
    )
    # Disable reward calculation
    reward_cfg["reward_scales"] = {}

    env = g1Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")

    print(f"Loading checkpoint: {resume_path}")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=gs.device)

    obs, _ = env.reset()

    with torch.no_grad():
        while True:
            actions = policy(obs)
            obs, rews, dones, infos = env.step(actions)


if __name__ == "__main__":
    main()
