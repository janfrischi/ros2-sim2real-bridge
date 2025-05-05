# franka_rl_bridge

`franka_rl_bridge` is a ROS 2 package designed to bridge Franka Emika robot state and control interfaces with reinforcement learning (RL) workflows. It provides utilities for listening to joint states, running policy inference, and executing RL policies on the robot or in simulation. The package is intended for research and development in robot learning, enabling seamless integration between ROS 2 and RL pipelines.

## Project Structure

- `franka_rl_bridge/`
  - `__init__.py`: Package marker file.
  - `joint_state_listener.py`: Node/script to subscribe to joint states and publish or log them for RL use.
  - `plot_policy_logs.py`: Utility for plotting logs generated during policy execution or training.
  - `policy_inference.py`: Script for running inference using a trained RL policy.
  - `policy_runner.py`: Node/script to execute a policy in a ROS 2 environment.

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

