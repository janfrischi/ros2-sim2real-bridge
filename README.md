# franka_rl_bridge

`franka_rl_bridge` is a ROS 2 package designed to bridge Franka Emika robot state and control interfaces with reinforcement learning (RL) workflows. It provides utilities for listening to joint states, running policy inference, and executing RL policies on the robot or in simulation. The package is intended for research and development in robot learning, enabling seamless integration between ROS 2 and RL pipelines.

## Project Structure

- `franka_rl_bridge/`
  - [`__init__.py`](franka_rl_bridge/__init__.py)
  - [`imitation_learning/`](franka_rl_bridge/imitation_learning)
    - [`__init__.py`](franka_rl_bridge/imitation_learning/__init__.py)
    - [`bc_policy_runner.py`](franka_rl_bridge/imitation_learning/bc_policy_runner.py) – Behavior Cloning (BC) policy execution.
    - `utils/`
      - [`__init__.py`](franka_rl_bridge/imitation_learning/utils/__init__.py)
      - `...` BC utilities (data loading, preprocessing, model helpers).
  - [`reinforcement_learning/`](franka_rl_bridge/reinforcement_learning)
    - [`__init__.py`](franka_rl_bridge/reinforcement_learning/__init__.py)
    - `joint_state_listener.py` – Subscribes to robot joint states and publishes standardized RL observations.
    - `policy_inference.py` – Loads a trained RL policy and runs inference.
    - `policy_runner.py` – Executes a policy in closed-loop control.
    - `policy_plotter.py` – Live plotting / monitoring (entry point).
    - `run_bc_tests.py`, `interactive_bc_tester.py`, `test_bc_policy.py` – BC / RL diagnostic and evaluation tools.
    - `verify_topics.py` – Validates required ROS 2 topics.
    - `monitor_observations.py` – Streams and inspects observation vectors.
    - `pose_sender.py` – Sends target poses for testing / teleoperation.
    - `...` additional helpers and experiment scripts.
  - [`plotting/`](franka_rl_bridge/plotting)
    - [`plot_data_json.py`](franka_rl_bridge/plotting/plot_data_json.py)
    - [`plot_inference_real.py`](franka_rl_bridge/plotting/plot_inference_real.py)
    - [`plot_inference_sim.py`](franka_rl_bridge/plotting/plot_inference_sim.py)
    - [`plot_success_rate_inference.py`](franka_rl_bridge/plotting/plot_success_rate_inference.py)
    - [`policy_plotter.py`](franka_rl_bridge/plotting/policy_plotter.py) – Offline plotting / post‑run analysis.
    - Logs (CSV): `observations_test.csv`, `policy_inference_17_06_real_robot.csv`, `policy_inference_17_06.csv`, `policy_inference_24_07.csv`
    - Figures (SVG): `results_detailed_analysis.svg`, `results_stacked_bar_chart.svg`
  - `models/`
    - `imitation_learning/` – Stored BC checkpoints.
    - `reinforcement_learning/` – RL checkpoints (`*.pt` such as `model_1999.pt`, `franka_lift.pt`, `franka_lift2.pt`, etc.).
  - Project metadata & packaging:
    - [`package.xml`](franka_rl_bridge/package.xml)
    - [`setup.py`](franka_rl_bridge/setup.py)
    - [`setup.cfg`](franka_rl_bridge/setup.cfg)
    - [`requirements.txt`](franka_rl_bridge/requirements.txt)

## Installation

### Prerequisites

Before installing this package, ensure you have the following dependencies installed:

1. **ROS 2 Humble**
   - Follow the official installation guide: [https://docs.ros.org/en/humble/index.html](https://docs.ros.org/en/humble/index.html)

2. **libfranka v0.13.2**
   - Install from source: [https://github.com/frankaemika/libfranka](https://github.com/frankaemika/libfranka)
   - Make sure to check out version v0.13.2:
     ```bash
     git clone https://github.com/frankaemika/libfranka.git
     cd libfranka
     git checkout 0.13.2
     # Follow build instructions from the repository
     ```

3. **franka_ros2 v0.1.15**
   - Install from source: [https://github.com/frankaemika/franka_ros2](https://github.com/frankaemika/franka_ros2)
   - Make sure to check out version v0.1.15:
     ```bash
     git clone https://github.com/frankaemika/franka_ros2.git
     cd franka_ros2
     git checkout 0.1.15
     # Build and install according to the repository instructions
     ```

### Installation Steps

1. **Clone the repository**
   ```bash
   cd ~/franka_ros2_ws/src
   git clone <this-repo-url>
   ```

2. **Install dependencies**:
   Make sure you have ROS 2 Humble installed and sourced. Install Python dependencies if needed:
   ```bash
   pip install -r requirements.txt  # if such a file exists
   ```

3. **Build the workspace**:
   ```bash
   cd ~/franka_ros2_ws
   colcon build --packages-select franka_rl_bridge
   ```

4. **Source the workspace**:
   ```bash
   source ~/franka_ros2_ws/install/setup.bash
   ```

## Sim2Real Bridging

This package successfully bridges the simulation-to-reality (sim2real) gap by implementing:

- Standardized state representation that works across both simulated and real robots
- Consistent action space mapping for seamless policy transfer
- Automated domain adaptation to account for dynamics differences
- Real-time performance optimization for controller execution
- Robust error handling and safety measures for real robot deployment

With these features, policies trained in simulation can be directly deployed on the physical Franka robot with minimal adjustments, reducing development time and hardware wear.

## Running the Package

```bash
# Launch the cartesian impedance controller
ros2 launch cartesian_impedance_control cartesian_impedance_controller.launch.py

# Start the joint_state_listener node
ros2 run franka_rl_bridge joint_state_listener

# Start the PolicyRunner node to deploy a trained policy (replace path and policy accordingly)
ros2 run franka_rl_bridge policy_runner --policy /home/pdzuser/franka_ros2_ws/src/franka_rl_bridge/models/model_2999.pt --device cpu

# Launch visualization tools
ros2 run franka_rl_bridge policy_plotter
```

