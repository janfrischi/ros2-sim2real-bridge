# franka_rl_bridge

`franka_rl_bridge` is a ROS 2 package designed to bridge Franka Emika robot state and control interfaces with reinforcement learning (RL) workflows. It provides utilities for listening to joint states, running policy inference, and executing RL policies on the robot or in simulation. The package is intended for research and development in robot learning, enabling seamless integration between ROS 2 and RL pipelines.

## Project Structure

- `franka_rl_bridge/`
  - `__init__.py`: Package marker file.
  - `joint_state_listener.py`: Node/script to subscribe to joint states and publish or log them for RL use.
  - `plot_policy_logs.py`: Utility for plotting logs generated during policy execution or training.
  - `policy_inference.py`: Script for running inference using a trained RL policy.
  - `policy_runner.py`: Node/script to execute a policy in a ROS 2 environment.
- `resource/franka_rl_bridge`: Resource file for ROS 2 package indexing.
- `test/`
  - `test_copyright.py`: Copyright compliance test.
  - `test_flake8.py`: Code style test using flake8.
  - `test_pep257.py`: Docstring style test using pep257.
- `setup.py`: Python package setup script.
- `setup.cfg`: Configuration for packaging.
- `package.xml`: ROS 2 package manifest.

## Installation

1. **Clone the repository** (if not already in your workspace):
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

## Usage

You can run the provided scripts as ROS 2 nodes. For example:
```bash
ros2 run franka_rl_bridge joint_state_listener
ros2 run franka_rl_bridge policy_inference
ros2 run franka_rl_bridge policy_runner
```

## License

See `package.xml` for license information.

## Maintainer

- pdzuser (<frijan@ethz.ch>)

