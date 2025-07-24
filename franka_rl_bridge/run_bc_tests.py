#!/usr/bin/env python3
"""
Standalone BC Policy Testing Script
Run comprehensive tests without the main policy runner interface.
"""

import argparse
import rclpy
import signal
import sys
from bc_policy_runner import BCPolicyRunner
from bc_policy_tester import BCPolicyTester


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\n🛑 Received interrupt signal. Cleaning up...")
    # Force terminal restoration
    try:
        import termios
        import tty
        # Try to restore normal terminal settings
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, 
                         termios.tcgetattr(sys.stdin))
    except:
        pass
    
    # Exit cleanly
    if rclpy.ok():
        rclpy.shutdown()
    sys.exit(0)


def main():
    """Main testing function"""
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    parser = argparse.ArgumentParser(description="BC Policy Testing Suite")
    parser.add_argument("--policy", type=str, required=True,
                       help="Path to the trained BC policy file (.pth)")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to run inference on (cpu or cuda)")
    parser.add_argument("--test-mode", choices=['automated', 'stress', 'comparative', 'monitor'], 
                       required=True, help="Type of test to run")
    parser.add_argument("--trials", type=int, default=10,
                       help="Number of trials for testing")
    parser.add_argument("--frequency", type=float, default=20.0,
                       help="Control frequency in Hz")
    
    args = parser.parse_args()
    
    # Initialize ROS2
    rclpy.init()
    
    runner = None
    try:
        # Create policy runner (without interactive mode)
        runner = BCPolicyRunner(
            policy_path=args.policy,
            device=args.device,
            deterministic=True,
            control_frequency=args.frequency,
            replay_file=None
        )
        
        # Create tester
        tester = BCPolicyTester(runner)
        
        # Run specified test
        if args.test_mode == 'automated':
            print(f"🧪 Running Automated Test Suite with {args.trials} trials per scenario")
            results = tester.run_automated_test_suite(args.trials)
            
        elif args.test_mode == 'stress':
            print("🔥 Running Stress Tests")
            results = tester.run_stress_tests()
            
        elif args.test_mode == 'comparative':
            print("🏁 Running Comparative Analysis")
            results = tester.run_comparative_analysis()
            
        elif args.test_mode == 'monitor':
            print("📊 Starting Live Monitoring Mode")
            tester.start_live_monitoring()
            
            # Keep running for monitoring
            import time
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nMonitoring stopped.")
        
        print("\n✅ Testing completed successfully!")
        
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Ensure proper cleanup
        if runner is not None:
            try:
                runner.cleanup()
                runner.destroy_node()
            except:
                pass
        
        if rclpy.ok():
            rclpy.shutdown()
        
        # Final terminal restoration attempt
        try:
            import termios
            import os
            # Reset terminal to sane defaults
            os.system('stty sane')
        except:
            pass


if __name__ == "__main__":
    main()