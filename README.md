# ROS 2 Sim2Real Bridge for Franka Robots

This package provides a bridge to run Reinforcement Learning (RL) policies trained in simulation on a real Franka Emika robot using ROS 2.

## Key Components

*   **`joint_state_listener.py`**: Subscribes to the robot's joint states from ROS 2 topics.
*   **`policy_inference.py`**: Performs inference using a trained RL policy model to generate robot commands based on observations.
*   **`policy_runner.py`**: Orchestrates the overall process of reading robot state, running policy inference, and sending commands to the robot.
*   **`plot_policy_logs.py`**: A utility script to visualize logged data from policy execution.

## Nodes

The package installs the following executable nodes:

*   `joint_state_listener`
*   `policy_inference`
*   `policy_runner`
