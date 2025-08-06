from setuptools import find_packages, setup

package_name = 'franka_rl_bridge'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='pdzuser',
    maintainer_email='frijan@ethz.ch',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'joint_state_listener = franka_rl_bridge.reinforcement_learning.joint_state_listener:main',
            'policy_inference = franka_rl_bridge.reinforcement_learning.policy_inference:main',
            'policy_runner = franka_rl_bridge.reinforcement_learning.policy_runner:main',
            'policy_plotter = franka_rl_bridge.reinforcement_learning.policy_plotter:main',
            'bc_policy_runner = franka_rl_bridge.imitation_learning.bc_policy_runner:main',
            'bc_policy_runner_legacy = franka_rl_bridge.imitation_learning.bc_policy_runner_legacy:main',
            'run_bc_tests = franka_rl_bridge.reinforcement_learning.run_bc_tests:main',
            'interactive_bc_tester = franka_rl_bridge.reinforcement_learning.interactive_bc_tester:main',
            'test_bc_policy = franka_rl_bridge.reinforcement_learning.test_bc_policy:main',
            'verify_topics = franka_rl_bridge.reinforcement_learning.verify_topics:main',
            'monitor_observations = franka_rl_bridge.reinforcement_learning.monitor_observations:main',
            'pose_sender = franka_rl_bridge.reinforcement_learning.pose_sender:main'
        ],
    },
)
