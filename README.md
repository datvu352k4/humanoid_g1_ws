
# Unitree G1 Robust Push Recovery



## 1. Project Overview
his project implements a **Reinforcement Learning (RL)** policy for the **Unitree G1 Humanoid** robot using **Genesis Sim**. 

The goal is to achieve **Robust Push Recovery**: training the robot to maintain balance and recover from significant external forces (up to 2000N in 1 step) while tracking zero-velocity commands (standing still).

## 2. CLI Documentation (How to Run)
### Installation
```bash
git clone https://github.com/datvu352k4/humanoid_g1_ws.git 
cd humanoid_g1_ws
pip install -r requirements.txt
```
### Training new policy 
To start training the policy with Curriculum Learning:
```bash 
cd humanoid_g1_ws
python src/g1_train.py --exp_name g1-push --num_envs 8192 --max_iterations 1000
```
Output: Checkpoints are saved in logs/g1-push.

### Evaluation 
To visualize the trained policy and test robustness against Maximum Force:
```bash 
cd humanoid_g1_ws
python src/g1_eval.py --exp_name g1-push --ckpt 900
```

## 3.Technical Summary
### 3.1 Simulation Setup
Simulator: Genesis Sim.

Physics Config:
- dt: 0.02s (50Hz Control Frequency).

- substeps: 12 (600Hz Physics Frequency): high substeps are crucial to prevent simulation instability during high-impact collisions.


### 3.2 RL Configuration

#### The agent is trained using PPO
- Network Architecture: Actor-Critic MLP [512, 256, 128] with ELU activation.
- Control Scheme: PD Position Control (Kp=100, Kd=5).
#### Observation Space (96 dimensions)
The robot perceives the environment through the following normalized signals:
- Joint positions (relative to nominal pose) (29). 
- Joint velocities (29).
- Base State: Base angular velocity (3) and Projected gravity vector (3).
- History: Last actions (29).
- Command: Target velocities (Vx, Vy, Wz) (3).
#### Action Space (29 dimensions) 
- Output: Target joint positions for the 29-DoF robot.
- Scaling: Actions are scaled by a factor of 0.25 before being added to the nominal standing pose.
```
action_total = self.q_homing + residual_action_nn
```
### 3.3 Robustness Strategy (Curriculum Learning)
To ensure the robot can withstand 2000N pushes without falling early in training, a sliding-window curriculum is implemented:

#### Force Magnitude: 
- Linearly increases from 50N (Start) to 2000N (End).

#### Push Frequency:

- Early Stage: Pushes occur every 2.0s (Interval=100). Helps the robot adapt to constant small noise.

- Late Stage: Pushes occur every 1.0s (Interval=50). Allows the robot sufficient time to perform recovery steps and stabilize after massive impacts.
### 3.4 Reward Shaping Strategy
The reward function is designed to help robot standing while enforcing stability constraints:
| Component           | Weight   | Description                |
| :--------           | :------- | :------------------------- |
| `tracking_lin_vel`  | `1.5`    | Encourages the robot to follow target linear velocities (Vxy = 0). |
| `tracking_ang_vel`  | `0.5`    | Encourages the robot to follow target angular velocities (Wz = 0). |             
| `lin_vel_z`         | `-2.0`   | Penalizes vertical velocity of the robot base. Discourages hopping, jumping, or jittery vertical motions. Ensures the robot maintains stable ground contact |
| `base_height`       | `-50.0`  | Heavy penalty based on the deviation of the base height from the target standing height. Teaches the robot that maintaining an upright standing posture is the most critical objective |
| `action_rate`       | `-0.005` | Penalizes the difference between the current action and the previous action. Encourages smooth control policies |
| `similar_to_default`| `-0.1`   | Penalizes joint positions that deviate significantly from the nominal standing pose.                            |
| `orientation`       | `-5.0`   | Penalizes body tilt (pitch and roll) relative to the gravity vector. Enforces an upright torso. Keeping the body vertically aligned is crucial for push recovery |

#### a. **Linear Velocity Tracking Reward**

The robot is encouraged to track $v_x, v_y$ references commanded by the user.

```math
R_{lin\_vel} = \exp[-\|v^{ref}_{xy} - v_{xy}\|^2]
```

Where:
- $v^{ref}_{xy} = [v_x^{ref}, v_y^{ref}]$ is the commanded velocity.
- $v_{xy} = [v_x, v_y]$ is the actual velocity.

#### b. **Angular Velocity Tracking Reward**

The robot is encouraged to track $w_z$ reference commanded by the user.

```math
R_{ang\_vel} = \exp[-(w^{ref}_{z} - w_{z})^2]
```

Where:
- $w_{cmd,z}$ is the commanded yaw velocity.
- $w_{base,z}$ is the actual yaw velocity.

#### 3. **Height Penalty**

The robot is encouraged to maintain a desired height as specified by the commanded altitude. A penalty is applied for deviations from this target height:

$$
R_{z} = (z - z_{ref})^2
$$

Where:
- $z$ is the current base height.
- $z_{ref}$ is the target height specified in the commands.

#### 4. **Pose Similarity Reward**

To keep the robot's joint poses close to a default configuration, a penalty is applied for large deviations from the default joint positions:

```math
R_{pose\_similarity} = \|q - q_{default}\|^2
```

Where:
- $q$ is the current joint position.
- $q_{default}$ is the default joint position.

#### 5. **Action Rate Penalty**

To ensure smooth control and discourage abrupt changes in actions, a penalty is applied based on the difference between consecutive actions:

```math
R_{action\_rate} = \|a_{t} - a_{t-1}\|^2
```

Where:
- $a_t$ and $a_{t-1}$ are the actions at the current and previous time steps, respectively.

#### 6. **Vertical Velocity Penalty**

To discourage unnecessary movement along the vertical ($z$) axis, a penalty is applied to the squared $z$-axis velocity of the base when the robot is not actively jumping. The reward is:

```math
R_{lin\_vel\_z} = v_{z}^2
```

Where:
- $v_{z}$ is the vertical velocity of the base.

#### 7. **Roll and Pitch Stabilization Penalty**

To ensure the robot maintains stability, a penalty is applied to discourage large roll and pitch deviations of the base. This reward is:

```math
R_{roll\_pitch} = roll^2 + pitch^2
```

Where:
- $roll$ is the roll angle of the base.
- $pitch$ is the pitch angle of the base.

---

Although the primary task is to stand still, the Tracking reward is essential. By setting the target velocity to zero, these terms effectively function as a stationarity objective, training the policy to actively cancel out any momentum induced by external pushes to return to a halt.

## 4. Project Structure

```
.
├── src/
│   ├── g1_env.py                              # Environment Logic: Physics, Rewards, Curriculum
│   ├── g1_train.py                            # PPO Configuration & Training Loop
│   └── g1_eval.py                             # Evaluation Script & Robustness Test
├── g1_description/g1_29dof_mode_11_g.urdf     # Robot Assets
├── requirements.txt
└── README.md
```

## References

 - [Genesis Sim](https://github.com/Genesis-Embodied-AI/Genesis)


