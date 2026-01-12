import math
import torch
import genesis as gs
from genesis.utils.geom import (
    quat_to_xyz,
    transform_by_quat,
    inv_quat,
    transform_quat_by_quat,
)
from genesis.utils.misc import tensor_to_array
import random


def gs_rand(lower, upper, batch_shape):
    assert lower.shape == upper.shape
    return (upper - lower) * torch.rand(
        size=(*batch_shape, *lower.shape), dtype=gs.tc_float, device=gs.device
    ) + lower


"""
    RL Environment for Unitree G1 Humanoid Locomotion task using Genesis Simulator.
    
    Task: The robot must learn to stand up, track velocity commands, and recover from 
          external push forces (Push Recovery Robustness).
    
    Observation Space (96 dims): 
        - Proprioception (Joint pos/vel, Base ang vel, Projected gravity)
        - Commands (Target velocities)
        - Previous actions
        
    Action Space (29 dims): 
        - Joint position targets (PD Control) relative to the default standing pose.
"""


class g1Env:
    def __init__(
        self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False
    ):
        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]
        self.device = gs.device

        self.simulate_action_latency = True  # there is a 1 step latency on real robot
        self.dt = 0.02  # control frequency on real robot is 50hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        # ---CONFIG ---
        push_cfg = self.env_cfg.get(
            "push_params",
            {
                "enable": False,
                "max_force": 1000.0,
                "min_force": 50.0,
                "start_interval": 100,  # Number of simulation steps between pushes
                "end_interval": 20,
                "curriculum_steps": 10000,
            },
        )

        self.push_enable = push_cfg["enable"]
        self.max_push_force = push_cfg["max_force"]
        self.min_push_force = push_cfg["min_force"]
        self.start_push_interval = push_cfg["start_interval"]
        self.end_push_interval = push_cfg["end_interval"]
        self.curriculum_steps = push_cfg["curriculum_steps"]
        self.global_step_counter = 0
        self.curriculum_force_log = 0.0
        self.curriculum_progress_log = 0.0
        # -----------------------------

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.dt,
                substeps=8,
            ),
            rigid_options=gs.options.RigidOptions(
                enable_self_collision=False,
                tolerance=1e-5,
                max_collision_pairs=200,
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
                max_FPS=int(1.0 / self.dt),
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=[0]),
            show_viewer=show_viewer,
        )

        # add plane
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        # add robot
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="g1_description/g1_29dof_mode_11_g.urdf",
                pos=self.env_cfg["base_init_pos"],
                quat=self.env_cfg["base_init_quat"],
            ),
        )
        # build
        self.scene.build(n_envs=num_envs)

        # -------------------------------
        self.rigid_solver = self.scene.sim.rigid_solver

        # List of link to apply force
        target_link_names = [
            "pelvis",
            "torso_link",
            "left_shoulder_pitch_link",
            "right_shoulder_pitch_link",
            "head_link",
            "left_hip_pitch_link",
            "right_hip_pitch_link",
        ]

        self.push_link_indices = []
        self.idx_to_name_map = {}
        for name in target_link_names:
            try:
                idx = self.robot.get_link(name).idx
                self.push_link_indices.append(idx)
                self.idx_to_name_map[idx] = name
            except:
                pass
        if not self.push_link_indices:
            self.push_link_indices = [1]
        # -------------------------------

        # names to indices
        self.motors_dof_idx = torch.tensor(
            [
                self.robot.get_joint(name).dof_start
                for name in self.env_cfg["joint_names"]
            ],
            dtype=gs.tc_int,
            device=gs.device,
        )
        self.actions_dof_idx = torch.argsort(self.motors_dof_idx)

        # PD control parameters
        self.robot.set_dofs_kp(
            [self.env_cfg["kp"]] * self.num_actions, self.motors_dof_idx
        )
        self.robot.set_dofs_kv(
            [self.env_cfg["kd"]] * self.num_actions, self.motors_dof_idx
        )

        # Define global gravity direction vector
        self.global_gravity = torch.tensor(
            [0.0, 0.0, -1.0], dtype=gs.tc_float, device=gs.device
        )

        # Initial state
        self.init_base_pos = torch.tensor(
            self.env_cfg["base_init_pos"], dtype=gs.tc_float, device=gs.device
        )
        self.init_base_quat = torch.tensor(
            self.env_cfg["base_init_quat"], dtype=gs.tc_float, device=gs.device
        )
        self.inv_base_init_quat = inv_quat(self.init_base_quat)
        self.init_dof_pos = torch.tensor(
            [
                self.env_cfg["default_joint_angles"][joint.name]
                for joint in self.robot.joints[1:]
            ],
            dtype=gs.tc_float,
            device=gs.device,
        )
        self.init_qpos = torch.concatenate(
            (self.init_base_pos, self.init_base_quat, self.init_dof_pos)
        )
        self.init_projected_gravity = transform_by_quat(
            self.global_gravity, self.inv_base_init_quat
        )

        # initialize buffers
        self.base_lin_vel = torch.empty(
            (self.num_envs, 3), dtype=gs.tc_float, device=gs.device
        )
        self.base_ang_vel = torch.empty(
            (self.num_envs, 3), dtype=gs.tc_float, device=gs.device
        )
        self.projected_gravity = torch.empty(
            (self.num_envs, 3), dtype=gs.tc_float, device=gs.device
        )
        self.obs_buf = torch.empty(
            (self.num_envs, self.num_obs), dtype=gs.tc_float, device=gs.device
        )
        self.rew_buf = torch.empty(
            (self.num_envs,), dtype=gs.tc_float, device=gs.device
        )
        self.reset_buf = torch.ones(
            (self.num_envs,), dtype=gs.tc_bool, device=gs.device
        )
        self.episode_length_buf = torch.empty(
            (self.num_envs,), dtype=gs.tc_int, device=gs.device
        )
        self.commands = torch.empty(
            (self.num_envs, self.num_commands), dtype=gs.tc_float, device=gs.device
        )
        self.commands_scale = torch.tensor(
            [
                self.obs_scales["lin_vel"],
                self.obs_scales["lin_vel"],
                self.obs_scales["ang_vel"],
            ],
            device=gs.device,
            dtype=gs.tc_float,
        )
        self.commands_limits = [
            torch.tensor(values, dtype=gs.tc_float, device=gs.device)
            for values in zip(
                self.command_cfg["lin_vel_x_range"],
                self.command_cfg["lin_vel_y_range"],
                self.command_cfg["ang_vel_range"],
            )
        ]
        self.actions = torch.zeros(
            (self.num_envs, self.num_actions), dtype=gs.tc_float, device=gs.device
        )
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.empty_like(self.actions)
        self.dof_vel = torch.empty_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.empty(
            (self.num_envs, 3), dtype=gs.tc_float, device=gs.device
        )
        self.base_quat = torch.empty(
            (self.num_envs, 4), dtype=gs.tc_float, device=gs.device
        )
        self.default_dof_pos = torch.tensor(
            [
                self.env_cfg["default_joint_angles"][name]
                for name in self.env_cfg["joint_names"]
            ],
            dtype=gs.tc_float,
            device=gs.device,
        )
        self.extras = dict()
        self.extras["observations"] = dict()

        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros(
                (self.num_envs,), dtype=gs.tc_float, device=gs.device
            )

    def _resample_commands(self, envs_idx):
        commands = gs_rand(*self.commands_limits, (self.num_envs,))
        if envs_idx is None:
            self.commands.copy_(commands)
        else:
            torch.where(envs_idx[:, None], commands, self.commands, out=self.commands)

    def step(self, actions):
        self.actions = torch.clip(
            actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"]
        )
        # Simulate hardware latency (1-step delay)
        exec_actions = (
            self.last_actions if self.simulate_action_latency else self.actions
        )
        target_dof_pos = (
            exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        )
        self.robot.control_dofs_position(
            target_dof_pos[:, self.actions_dof_idx], self.motors_dof_idx
        )

        # ====================================================================
        # Push logic
        if self.push_enable:
            self.global_step_counter += 1
            # Calculate Curriculum Progress
            progress = min(1.0, self.global_step_counter / self.curriculum_steps)
            # Incraese the maximum allowed force (0% -> 100%)
            current_force = self.min_push_force + progress * (
                self.max_push_force - self.min_push_force
            )
            # Decrease the interval between pushes
            current_interval = int(
                self.start_push_interval
                - progress * (self.start_push_interval - self.end_push_interval)
            )
            self.curriculum_progress_log = progress
            self.curriculum_force_log = current_force
            # Apply Disturbance
            if self.global_step_counter % current_interval == 0:
                # Apply force from [80% Max, 100% Max]
                scale = 0.8 + 0.2 * torch.rand(1, device=self.device).item()
                force_magnitude = scale * current_force
                # Generate random 3D vector
                random_dir = torch.randn((self.num_envs, 1, 3), device=self.device)
                random_dir[..., 2] = 0.0  # Constraint: Horizontal push only
                norm = torch.norm(random_dir, dim=-1, keepdim=True)
                push_forces = (random_dir / (norm + 1e-6)) * force_magnitude
                # Target Link Selection
                chosen_link_idx = random.choice(self.push_link_indices)
                link_name = self.idx_to_name_map.get(chosen_link_idx, "Unknown")

                print(
                    f"[Push] Target: {link_name:20s} | Force: {force_magnitude:.1f} N (Max allowed: {current_force:.1f})"
                )
                # Apply External Force to Physics Engine
                self.rigid_solver.apply_links_external_force(
                    force=push_forces, links_idx=[chosen_link_idx]
                )

        # ====================================================================

        self.scene.step()

        self.episode_length_buf += 1
        self.base_pos = self.robot.get_pos()
        self.base_quat = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(self.inv_base_init_quat, self.base_quat),
            rpy=True,
            degrees=True,
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos = self.robot.get_dofs_position(self.motors_dof_idx)
        self.dof_vel = self.robot.get_dofs_velocity(self.motors_dof_idx)

        # compute reward
        self.rew_buf.zero_()
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # resample commands
        self._resample_commands(
            self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt)
            == 0
        )

        # check termination and reset
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= (
            torch.abs(self.base_euler[:, 1])
            > self.env_cfg["termination_if_pitch_greater_than"]
        )
        self.reset_buf |= (
            torch.abs(self.base_euler[:, 0])
            > self.env_cfg["termination_if_roll_greater_than"]
        )

        # Compute timeout
        self.extras["time_outs"] = (
            self.episode_length_buf > self.max_episode_length
        ).to(dtype=gs.tc_float)

        # Reset environment if necessary
        self._reset_idx(self.reset_buf)

        # update observations
        self._update_observation()

        self.last_actions.copy_(self.actions)
        self.last_dof_vel.copy_(self.dof_vel)
        self.extras["observations"]["critic"] = self.obs_buf

        if self.push_enable:
            self.extras["episode"]["curr_force_max"] = torch.tensor(
                self.curriculum_force_log, device=self.device
            )
            self.extras["episode"]["curr_progress"] = torch.tensor(
                self.curriculum_progress_log, device=self.device
            )

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        return None

    def _reset_idx(self, envs_idx=None):
        self.robot.set_qpos(
            self.init_qpos, envs_idx=envs_idx, zero_velocity=True, skip_forward=True
        )
        if envs_idx is None:
            self.base_pos.copy_(self.init_base_pos)
            self.base_quat.copy_(self.init_base_quat)
            self.projected_gravity.copy_(self.init_projected_gravity)
            self.dof_pos.copy_(self.init_dof_pos)
            self.base_lin_vel.zero_()
            self.base_ang_vel.zero_()
            self.dof_vel.zero_()
            self.actions.zero_()
            self.last_actions.zero_()
            self.last_dof_vel.zero_()
            self.episode_length_buf.zero_()
            self.reset_buf.fill_(True)
        else:
            torch.where(
                envs_idx[:, None], self.init_base_pos, self.base_pos, out=self.base_pos
            )
            torch.where(
                envs_idx[:, None],
                self.init_base_quat,
                self.base_quat,
                out=self.base_quat,
            )
            torch.where(
                envs_idx[:, None],
                self.init_projected_gravity,
                self.projected_gravity,
                out=self.projected_gravity,
            )
            torch.where(
                envs_idx[:, None], self.init_dof_pos, self.dof_pos, out=self.dof_pos
            )
            self.base_lin_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.base_ang_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.dof_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.actions.masked_fill_(envs_idx[:, None], 0.0)
            self.last_actions.masked_fill_(envs_idx[:, None], 0.0)
            self.last_dof_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.episode_length_buf.masked_fill_(envs_idx, 0)
            self.reset_buf.masked_fill_(envs_idx, True)

        n_envs = envs_idx.sum() if envs_idx is not None else self.num_envs
        self.extras["episode"] = {}
        for key, value in self.episode_sums.items():
            if envs_idx is None:
                mean = value.mean()
            else:
                mean = torch.where(n_envs > 0, value[envs_idx].sum() / n_envs, 0.0)
            self.extras["episode"]["rew_" + key] = (
                mean / self.env_cfg["episode_length_s"]
            )
            value.masked_fill_(envs_idx, 0.0)
        self._resample_commands(envs_idx)

    def _update_observation(self):
        self.obs_buf = torch.concatenate(
            (
                self.base_ang_vel * self.obs_scales["ang_vel"],
                self.projected_gravity,
                self.commands * self.commands_scale,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],
                self.dof_vel * self.obs_scales["dof_vel"],
                self.actions,
            ),
            dim=-1,
        )

    def reset(self):
        self._reset_idx()
        self._update_observation()
        return self.obs_buf, None

    # ------------ reward functions----------------
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(
            torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1
        )
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_base_height(self):
        # Penalize base height away from target
        return torch.square(self.base_pos[:, 2] - self.reward_cfg["base_height_target"])

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_similar_to_default(self):
        # Penalize joint poses far away from default pose
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_orientation(self):
        # Penalize body tilt (roll and pitch orientation error).
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
