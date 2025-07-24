#!/usr/bin/env python3
"""
BC Policy Testing Module
Provides comprehensive testing capabilities for BC policies without cluttering the main runner.
"""

import numpy as np
import time
import os
from typing import Dict, List, Optional, Any


class BCPolicyTester:
    """Comprehensive testing framework for BC policies"""
    
    def __init__(self, policy_runner):
        """Initialize tester with reference to the main policy runner"""
        self.runner = policy_runner
        self.test_results = {}
        self.analytics = self._initialize_analytics()
        
    def _initialize_analytics(self):
        """Initialize performance tracking analytics"""
        return {
            'action_history': [],
            'observation_history': [],
            'success_rate': [],
            'completion_times': [],
            'cube_trajectories': {'cube_1': [], 'cube_2': [], 'cube_3': []},
            'gripper_commands': [],
            'policy_confidences': [],
            'failure_modes': [],
            'workspace_coverage': []
        }

    def run_automated_test_suite(self, num_trials: int = 10) -> Dict:
        """Run automated test suite across multiple scenarios"""
        test_scenarios = [
            "default", "wide_spread", "tight_cluster", "corner_formation",
            "stacking_ready", "reach_challenge", "precision_test"
        ]
        
        results = {}
        
        print(f"\n🧪 AUTOMATED TEST SUITE - {num_trials} trials per scenario")
        print("="*80)
        
        for scenario in test_scenarios:
            print(f"\n🧪 TESTING SCENARIO: {scenario.upper()}")
            print("="*60)
            
            scenario_results = {
                'successes': 0,
                'failures': 0,
                'timeout': 0,
                'avg_completion_time': 0,
                'cube_distances': [],
                'final_positions': []
            }
            
            for trial in range(num_trials):
                print(f"  Trial {trial + 1}/{num_trials}...", end=" ")
                
                # Reset environment
                self.runner.spawn_cubes_preset(scenario)
                self.runner.reset_to_home()
                time.sleep(1.0)  # Settle time
                
                # Run trial with timeout
                result = self._run_single_trial(max_steps=400, timeout=20.0)
                
                # Evaluate success
                if self._evaluate_stacking_success():
                    scenario_results['successes'] += 1
                    print("✅ SUCCESS")
                elif result['timeout']:
                    scenario_results['timeout'] += 1
                    print("⏰ TIMEOUT")
                else:
                    scenario_results['failures'] += 1
                    print("❌ FAILED")
                    
                scenario_results['cube_distances'].append(self._get_final_cube_distances())
                scenario_results['final_positions'].append(self.runner.cube_positions.copy())
            
            results[scenario] = scenario_results
            self._print_scenario_results(scenario, scenario_results)
        
        self._generate_test_report(results)
        return results

    def run_stress_tests(self) -> Dict:
        """Run stress tests with challenging scenarios"""
        stress_scenarios = {
            "extreme_corners": {
                'cube_1': np.array([0.35, -0.30, 0.0203]),  # Far corner
                'cube_2': np.array([0.65, 0.30, 0.0203]),   # Opposite corner
                'cube_3': np.array([0.35, 0.30, 0.0203])    # Another corner
            },
            "minimum_spacing": {
                'cube_1': np.array([0.50, 0.0, 0.0203]),
                'cube_2': np.array([0.50, 0.045, 0.0203]),  # Just above min distance
                'cube_3': np.array([0.50, -0.045, 0.0203])  # Just below min distance
            },
            "occlusion_test": {
                'cube_1': np.array([0.45, 0.0, 0.0203]),    # Target
                'cube_2': np.array([0.50, 0.0, 0.0203]),    # Blocking cube
                'cube_3': np.array([0.55, 0.0, 0.0203])     # Far cube
            },
            "reach_limits": {
                'cube_1': np.array([0.65, 0.0, 0.0203]),    # Maximum reach
                'cube_2': np.array([0.35, 0.28, 0.0203]),   # Corner reach
                'cube_3': np.array([0.35, -0.28, 0.0203])   # Corner reach
            }
        }
        
        print(f"\n🔥 STRESS TEST SUITE")
        print("="*80)
        
        stress_results = {}
        for scenario_name, positions in stress_scenarios.items():
            print(f"\n🔥 STRESS TEST: {scenario_name.upper()}")
            
            # Apply custom positions
            self.runner.cube_positions.update(positions)
            
            # Run multiple trials
            results = []
            for trial in range(5):
                result = self._run_single_trial(max_steps=500, timeout=30.0)
                results.append(result)
            
            stress_results[scenario_name] = results
            success_rate = sum(r['success'] for r in results) / len(results)
            avg_time = np.mean([r['time'] for r in results])
            
            print(f"   Success Rate: {success_rate:.1%}")
            print(f"   Average Time: {avg_time:.1f}s")
        
        return stress_results

    def run_comparative_analysis(self) -> Dict:
        """Compare policy performance with baseline behaviors"""
        baselines = {
            'random_policy': self._run_random_baseline,
            'hardcoded_sequence': self._run_hardcoded_baseline,
        }
        
        print(f"\n🏁 COMPARATIVE ANALYSIS")
        print("="*80)
        
        comparison_results = {}
        
        for baseline_name, baseline_func in baselines.items():
            print(f"\n🏁 RUNNING BASELINE: {baseline_name}")
            baseline_results = baseline_func(num_trials=10)
            comparison_results[baseline_name] = baseline_results
        
        # Add current policy results
        current_results = self.run_automated_test_suite(num_trials=10)
        comparison_results['current_policy'] = current_results
        
        # Generate comparison report
        self._generate_comparison_report(comparison_results)
        return comparison_results

    def start_live_monitoring(self):
        """Start live performance monitoring with matplotlib"""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.animation import FuncAnimation
            
            print("📊 Starting live monitoring dashboard...")
            
            # Create monitoring plots
            self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(12, 8))
            
            # Plot 1: EEF trajectory
            self.ax1.set_title("End-Effector Trajectory")
            self.ax1.set_xlabel("X Position (m)")
            self.ax1.set_ylabel("Y Position (m)")
            
            # Plot 2: Action smoothness
            self.ax2.set_title("Action Smoothness")
            self.ax2.set_xlabel("Time Step")
            self.ax2.set_ylabel("Position Change (m)")
            
            # Plot 3: Cube distances
            self.ax3.set_title("Inter-Cube Distances")
            self.ax3.set_xlabel("Time Step")
            self.ax3.set_ylabel("Distance (m)")
            
            # Plot 4: Policy confidence
            self.ax4.set_title("Policy Confidence")
            self.ax4.set_xlabel("Time Step")
            self.ax4.set_ylabel("Max Mode Probability")
            
            # Animation update function
            def update_plots(frame):
                if len(self.analytics['action_history']) > 10:
                    self._update_trajectory_plot()
                    self._update_smoothness_plot()
                    self._update_distance_plot()
                    self._update_confidence_plot()
            
            self.animation = FuncAnimation(self.fig, update_plots, interval=100)
            plt.tight_layout()
            plt.show(block=False)
            
        except ImportError:
            print("❌ Matplotlib not available. Install with: pip install matplotlib")
        except Exception as e:
            print(f"❌ Live monitoring failed: {e}")

    def track_performance(self, obs_dict: Dict, action_np: np.ndarray, policy_output=None):
        """Track detailed performance metrics during execution"""
        timestamp = time.time()
        
        # Store trajectories
        self.analytics['action_history'].append({
            'timestamp': timestamp,
            'action': action_np.copy(),
            'eef_pos': obs_dict['eef_pos'].copy(),
            'gripper_cmd': action_np[7]
        })
        
        # Track cube positions
        for i, cube_name in enumerate(['cube_1', 'cube_2', 'cube_3']):
            cube_pos = self.runner.cube_positions[cube_name].copy()
            self.analytics['cube_trajectories'][cube_name].append({
                'timestamp': timestamp,
                'position': cube_pos,
                'attached': cube_name == self.runner.cube_attached
            })
        
        # Workspace coverage
        eef_pos_2d = obs_dict['eef_pos'][:2]
        self.analytics['workspace_coverage'].append(eef_pos_2d.copy())

    def generate_performance_report(self):
        """Generate comprehensive performance report"""
        print("\n📊 PERFORMANCE ANALYTICS REPORT")
        print("="*80)
        
        # Success rate analysis
        if self.analytics['success_rate']:
            success_rate = np.mean(self.analytics['success_rate']) * 100
            print(f"Overall Success Rate: {success_rate:.1f}%")
        
        # Action smoothness
        if len(self.analytics['action_history']) > 1:
            actions = np.array([a['action'] for a in self.analytics['action_history']])
            action_changes = np.diff(actions, axis=0)
            pos_smoothness = np.mean(np.linalg.norm(action_changes[:, :3], axis=1))
            print(f"Average Position Change: {pos_smoothness:.6f}m/step")
        
        # Gripper usage
        if self.analytics['action_history']:
            gripper_commands = [a['gripper_cmd'] for a in self.analytics['action_history']]
            open_ratio = sum(1 for g in gripper_commands if g > 0) / len(gripper_commands)
            print(f"Gripper Open Ratio: {open_ratio:.1%}")
        
        # Workspace coverage
        if self.analytics['workspace_coverage']:
            positions = np.array(self.analytics['workspace_coverage'])
            coverage_area = self._calculate_convex_hull_area(positions)
            print(f"Workspace Coverage: {coverage_area:.4f}m²")

    # Private helper methods
    def _run_single_trial(self, max_steps: int = 400, timeout: float = 20.0) -> Dict:
        """Run a single trial with timeout and step limit"""
        start_time = time.time()
        step_count = 0
        
        self.runner.start_policy()
        
        while (time.time() - start_time < timeout and 
               step_count < max_steps and 
               not self._evaluate_stacking_success()):
            time.sleep(1.0 / self.runner.control_frequency)
            step_count += 1
        
        self.runner.stop_policy()
        
        return {
            'success': self._evaluate_stacking_success(),
            'timeout': time.time() - start_time >= timeout,
            'steps': step_count,
            'time': time.time() - start_time
        }

    def _evaluate_stacking_success(self) -> bool:
        """Evaluate if the stacking task was completed successfully"""
        cube1_pos = self.runner.cube_positions['cube_1']
        cube2_pos = self.runner.cube_positions['cube_2']
        cube3_pos = self.runner.cube_positions['cube_3']
        
        # Cube 2 should be above cube 1
        height_diff_21 = cube2_pos[2] - cube1_pos[2]
        lateral_dist_21 = np.linalg.norm(cube2_pos[:2] - cube1_pos[:2])
        
        # Cube 3 should be above cube 2
        height_diff_32 = cube3_pos[2] - cube2_pos[2]
        lateral_dist_32 = np.linalg.norm(cube3_pos[:2] - cube2_pos[:2])
        
        # Success criteria
        cube_size = 0.0406  # Cube height
        height_tolerance = 0.005
        lateral_tolerance = 0.02
        
        stack_21 = (height_diff_21 > cube_size - height_tolerance and 
                    height_diff_21 < cube_size + height_tolerance and
                    lateral_dist_21 < lateral_tolerance)
        
        stack_32 = (height_diff_32 > cube_size - height_tolerance and 
                    height_diff_32 < cube_size + height_tolerance and
                    lateral_dist_32 < lateral_tolerance)
        
        return stack_21 and stack_32

    def _get_final_cube_distances(self) -> List[float]:
        """Get final distances between all cubes"""
        cube_positions = [self.runner.cube_positions[f'cube_{i}'] for i in range(1, 4)]
        distances = []
        for i in range(len(cube_positions)):
            for j in range(i+1, len(cube_positions)):
                dist = np.linalg.norm(cube_positions[i] - cube_positions[j])
                distances.append(dist)
        return distances

    def _print_scenario_results(self, scenario: str, results: Dict):
        """Print results for a specific scenario"""
        total_trials = results['successes'] + results['failures'] + results['timeout']
        success_rate = (results['successes'] / total_trials) * 100
        
        print(f"   📊 Results: {success_rate:.1f}% success ({results['successes']}/{total_trials})")
        print(f"   ❌ Failures: {results['failures']}, ⏰ Timeouts: {results['timeout']}")

    def _generate_test_report(self, results: Dict):
        """Generate comprehensive test report"""
        print(f"\n📋 COMPREHENSIVE TEST REPORT")
        print("="*80)
        
        overall_successes = sum(r['successes'] for r in results.values())
        overall_trials = sum(r['successes'] + r['failures'] + r['timeout'] for r in results.values())
        overall_success_rate = (overall_successes / overall_trials) * 100
        
        print(f"Overall Success Rate: {overall_success_rate:.1f}% ({overall_successes}/{overall_trials})")
        print("\nScenario Breakdown:")
        
        for scenario, result in results.items():
            total = result['successes'] + result['failures'] + result['timeout']
            rate = (result['successes'] / total) * 100
            print(f"  {scenario:<20}: {rate:5.1f}% ({result['successes']}/{total})")

    def _generate_comparison_report(self, comparison_results: Dict):
        """Generate comparison report between different approaches"""
        print(f"\n🏆 COMPARATIVE ANALYSIS REPORT")
        print("="*80)
        
        for approach, results in comparison_results.items():
            if isinstance(results, dict) and 'default' in results:
                # Handle test suite results
                total_successes = sum(r['successes'] for r in results.values())
                total_trials = sum(r['successes'] + r['failures'] + r['timeout'] for r in results.values())
                success_rate = (total_successes / total_trials) * 100 if total_trials > 0 else 0
            else:
                # Handle baseline results
                total_successes = sum(1 for r in results if r['success'])
                total_trials = len(results)
                success_rate = (total_successes / total_trials) * 100 if total_trials > 0 else 0
            
            print(f"{approach:<20}: {success_rate:5.1f}% ({total_successes}/{total_trials})")

    def _run_random_baseline(self, num_trials: int) -> List[Dict]:
        """Run random action baseline for comparison"""
        print("   Running random action baseline...")
        results = []
        
        for trial in range(num_trials):
            self.runner.reset_to_home()
            
            for step in range(200):
                # Generate random action
                random_action = np.random.uniform(-0.1, 0.1, size=8)
                random_action[7] = np.random.choice([-1, 1])  # Random gripper
                
                self.runner.execute_action(random_action)
                time.sleep(1.0 / self.runner.control_frequency)
            
            success = self._evaluate_stacking_success()
            results.append({'success': success, 'steps': 200})
        
        return results

    def _run_hardcoded_baseline(self, num_trials: int) -> List[Dict]:
        """Run hardcoded sequence baseline"""
        print("   Running hardcoded sequence baseline...")
        results = []
        
        for trial in range(num_trials):
            self.runner.reset_to_home()
            success = False
            
            # Simple hardcoded sequence (placeholder)
            try:
                cube2_pos = self.runner.cube_positions['cube_2']
                approach_pos = cube2_pos + np.array([0, 0, 0.05])
                
                for step in range(100):
                    if step < 20:
                        # Approach cube_2
                        action = np.zeros(8)
                        action[:3] = approach_pos
                        action[3:7] = [1, 0, 0, 0]  # Identity quaternion
                        action[7] = 1  # Open gripper
                    elif step < 40:
                        # Grasp cube_2
                        action = np.zeros(8)
                        action[:3] = cube2_pos
                        action[3:7] = [1, 0, 0, 0]
                        action[7] = -1  # Close gripper
                    else:
                        # Move to cube_1
                        cube1_pos = self.runner.cube_positions['cube_1']
                        stack_pos = cube1_pos + np.array([0, 0, 0.04])
                        action = np.zeros(8)
                        action[:3] = stack_pos
                        action[3:7] = [1, 0, 0, 0]
                        action[7] = 1  # Open gripper
                    
                    self.runner.execute_action(action)
                    time.sleep(1.0 / self.runner.control_frequency)
                
                success = self._evaluate_stacking_success()
            except:
                success = False
            
            results.append({'success': success, 'steps': 100})
        
        return results

    def _calculate_convex_hull_area(self, positions: np.ndarray) -> float:
        """Calculate convex hull area of workspace coverage"""
        try:
            from scipy.spatial import ConvexHull
            if len(positions) < 3:
                return 0.0
            hull = ConvexHull(positions)
            return hull.volume  # In 2D, volume is area
        except:
            return 0.0

    # Placeholder methods for live monitoring plots
    def _update_trajectory_plot(self):
        """Update trajectory plot"""
        pass

    def _update_smoothness_plot(self):
        """Update smoothness plot"""
        pass

    def _update_distance_plot(self):
        """Update distance plot"""
        pass

    def _update_confidence_plot(self):
        """Update confidence plot"""
        pass