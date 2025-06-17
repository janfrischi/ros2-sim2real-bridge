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
            'joint_state_listener = franka_rl_bridge.joint_state_listener:main',
            'policy_inference = franka_rl_bridge.policy_inference:main',
            'policy_runner = franka_rl_bridge.policy_runner:main',
            'policy_plotter = franka_rl_bridge.policy_plotter:main',
            'bc_policy_runner = franka_rl_bridge.bc_policy_runner:main',
            'test_bc_policy = franka_rl_bridge.test_bc_policy:main',
            'verify_topics = franka_rl_bridge.verify_topics:main',
            'monitor_observations = franka_rl_bridge.monitor_observations:main',
            'pose_sender = franka_rl_bridge.pose_sender:main',
        ],
    },
)
