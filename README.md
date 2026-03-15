# Double Inverted Pendulum RL Agent

### Environment Design
This project uses `pymunk` to simulate rigid body dynamics. The setup consists of a static track segment and a cart constrained to it via a `GrooveJoint`. Two 100-length poles of mass 0.5 are connected in series using `PivotJoints`. The physics loop runs at a fixed timestep of 1/60s.

### Reward Function Design
**Baseline Reward:** The fundamental goal is to balance the poles. The baseline formula strictly measures uprightness:
$Reward = \cos(\theta_1) + \cos(\theta_2)$
This peaks at 2.0 when perfectly vertical.

**Shaped Reward:** To speed up learning, the shaped reward incorporates domain knowledge penalties:
$Reward = (\cos(\theta_1) + \cos(\theta_2)) - 0.1|x_{cart}| - 0.01(|\omega_1| + |\omega_2|) - 0.001|action|^2$
This actively penalizes drifting off-screen, wild erratic swinging, and excessive energy usage.

### How to Run
1. Build the image: `docker-compose build`
2. Train Baseline: `docker-compose run train python train.py --reward_type baseline --save_path models/baseline`
3. Train Shaped: `docker-compose run train python train.py --reward_type shaped --save_path models/shaped`
4. Evaluate & Render: `docker-compose run evaluate python evaluate.py --model_path models/shaped.zip --gif_path media/agent_final.gif`
