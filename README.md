# franka_rl_bridge

`franka_rl_bridge` is a modular ROS 2 (Humble) package that unifies Franka Emika Panda robot state & control interfaces with modern Imitation and Reinforcement learning workflows. It standardizes observations (EEF, objects, gripper, kinematics) and action mapping across simulation and real hardware, enabling fast policy deployment with minimal code changes. A mixin-based utility layer (imitation_learning/utils) encapsulates gripper control, object (cube) management, kinematics/Jacobian access, keyboard interaction, policy lifecycle, and structured observation building. The package emphasizes: 
- Sim2Real consistency (identical observation & action schemas) 
- Safe, rate-controlled policy execution 
- Deterministic or stochastic inference modes 
- Rapid iteration via modular mixins 
- Lightweight logging & plotting for evaluation 

## Project Structure

- `franka_rl_bridge/`
  - [`__init__.py`](franka_rl_bridge/__init__.py)
  - `imitation_learning/`
    - [`__init__.py`](franka_rl_bridge/imitation_learning/__init__.py)
    - [`bc_policy_runner.py`](franka_rl_bridge/imitation_learning/bc_policy_runner.py) – Behavior Cloning (BC) policy execution node.
    - `utils/` (core mixins & helpers used by bc_policy_runner)
      - [`__init__.py`](franka_rl_bridge/imitation_learning/utils/__init__.py)
      - `cube_manager.py` – Cube pose management, attachment/detachment logic.
      - `gripper_control.py` – Gripper action clients, open/close sequencing, cooldowns.
      - `observations.py` – Builds structured observation dict (EEF, cubes, gripper, indices).
      - `robot_state.py` – Robot kinematics, Jacobian subscription, manipulability computation.
      - `policy_control.py` – Policy loading (robomimic), episode lifecycle, action post‑processing.
      - `instructions.py` – User instruction & help text (normal vs testing modes).
      - `keyboard.py` – Low‑level non‑blocking keyboard input utility.
      - `keyboard_handler.py` – High‑level keyboard command mapping & state flags.
  - `reinforcement_learning/` 
    - `joint_state_listener.py`
    - `policy_inference.py`
    - `policy_runner.py`
    - (other experimental / diagnostic scripts omitted for brevity)
  - `plotting/`
    - `plot_data_json.py`, `plot_inference_real.py`, `plot_inference_sim.py`
    - `plot_success_rate_inference.py`, `policy_plotter.py`
    - Example logs (CSV) & figures (SVG)
  - `models/`
    - `imitation_learning/` – Stored BC checkpoints.
    - `reinforcement_learning/` – RL checkpoints (`*.pt`).
  - Project metadata & packaging:
    - `package.xml`, `setup.py`, `setup.cfg`, `requirements.txt`

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
     ```

3. **franka_ros2 v0.1.15**
   - Install from source: [https://github.com/frankaemika/franka_ros2](https://github.com/frankaemika/franka_ros2)
   - Make sure to check out version v0.1.15:
     ```bash
     git clone https://github.com/frankaemika/franka_ros2.git
     cd franka_ros2
     git checkout 0.1.15
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

## Running the Imitation Learning Pipeline

```bash
# Launch the cartesian impedance controller
ros2 launch cartesian_impedance_control cartesian_impedance_controller.launch.py

# Switch on the Imitation Learning mode
ros2 param set /cartesian_impedance_controller imitation_learning_mode true

# Start the PolicyRunner node to deploy a trained policy, add --deterministic flag if needed
python3 franka_rl_bridge/imitation_learning/bc_policy_runner.py --policy /path/to/policy_checkpoint.pth 
```

