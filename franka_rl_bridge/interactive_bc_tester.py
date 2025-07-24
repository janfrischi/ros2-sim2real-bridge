#!/usr/bin/env python3
"""
Interactive BC Policy Testing
Combines the main runner with testing capabilities in an interactive mode.
"""

import argparse
import rclpy
from bc_policy_runner import BCPolicyRunner
from bc_policy_tester import BCPolicyTester


def main():
    """Main interactive testing function"""
    parser = argparse.ArgumentParser(description="Interactive BC Policy Testing")
    parser.add_argument("--policy", type=str, required=True,
                       help="Path to the trained BC policy file (.pth)")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to run inference on (cpu or cuda)")
    parser.add_argument("--frequency", type=float, default=20.0,
                       help="Control frequency in Hz")
    
    args = parser.parse_args()
    
    # Initialize ROS2
    rclpy.init()
    
    try:
        # Create policy runner
        runner = BCPolicyRunner(
            policy_path=args.policy,
            device=args.device,
            deterministic=True,
            control_frequency=args.frequency,
            replay_file=None
        )
        
        # Create and attach tester
        tester = BCPolicyTester(runner)
        runner.attach_tester(tester)
        
        print("🎮 Interactive BC Policy Testing Mode")
        print("   - Use normal policy runner controls (Space, s, r, o, etc.)")
        print("   - Use testing controls (t, m, x, n)")
        print("   - Press 'q' to quit")
        
        # Run interactive mode
        executor = rclpy.executors.MultiThreadedExecutor()
        executor.add_node(runner)
        
        try:
            executor.spin()
        except KeyboardInterrupt:
            print("\n\nShutting down interactive tester...")
        finally:
            executor.shutdown()
        
    except Exception as e:
        print(f"❌ Interactive testing failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()