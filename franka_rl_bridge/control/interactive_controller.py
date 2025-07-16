#!/usr/bin/env python3
"""
Interactive Controller - Handles keyboard input and user interface
"""
import sys
import termios
import select
from typing import Callable, Optional

class InteractiveController:
    """Handles keyboard input and interactive commands"""
    
    def __init__(self, command_handler: Optional[Callable] = None):
        self.command_handler = command_handler
        self.old_settings = None
        self.setup_terminal()
        
        # Command mappings
        self.normal_commands = {
            ' ': 'start_stop_policy',
            's': 'stop_policy',
            'r': 'reset_to_home',
            'o': 'toggle_gripper',
            'c': 'random_spawn',
            '1': 'spawn_line',
            '2': 'spawn_triangle',
            '3': 'spawn_stack_ready',
            'q': 'quit'
        }
        
        self.replay_commands = {
            ' ': 'replay_step',
            'a': 'toggle_auto_step',
            'e': 'toggle_action_execution',
            't': 'toggle_auto_trial',
            'r': 'replay_reset',
            'h': 'reset_to_home',
            'n': 'next_trial',
            'p': 'prev_trial',
            'i': 'show_trial_info',
            'o': 'toggle_gripper',
            'q': 'quit'
        }
    
    def setup_terminal(self):
        """Setup terminal for non-blocking input"""
        try:
            self.old_settings = termios.tcgetattr(sys.stdin)
            new_settings = termios.tcgetattr(sys.stdin)
            
            new_settings[3] = new_settings[3] & ~(termios.ECHO | termios.ICANON)
            new_settings[6][termios.VMIN] = 0
            new_settings[6][termios.VTIME] = 0
            
            termios.tcsetattr(sys.stdin, termios.TCSANOW, new_settings)
        except:
            pass
    
    def restore_terminal(self):
        """Restore terminal settings"""
        try:
            if self.old_settings:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
                sys.stdout.flush()
                sys.stderr.flush()
        except:
            pass
    
    def get_key(self):
        """Get a single keypress if available"""
        try:
            if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                key = sys.stdin.read(1)
                sys.stdout.flush()
                return key
            return None
        except:
            return None
    
    def check_input(self, replay_mode: bool = False) -> Optional[str]:
        """Check for keyboard input and return command"""
        key = self.get_key()
        if key is None:
            return None
        
        key = key.lower()
        commands = self.replay_commands if replay_mode else self.normal_commands
        
        return commands.get(key, None)
    
    def print_instructions(self, replay_mode: bool = False, replay_data: Optional[dict] = None):
        """Print control instructions"""
        print("\n" + "=" * 80)
        print("BC POLICY RUNNER - CONTROL INSTRUCTIONS".center(80))
        print("=" * 80)
        
        if replay_mode and replay_data:
            print("Replay Mode Controls:")
            print("  Space: Next step")
            print("  a: Toggle auto-step")
            print("  p: Previous trial")
            print("  n: Next trial")
            print("  i: Show trial info")
            print("  e: Toggle action execution")
            print("  t: Toggle auto-next trial")
            print("  r: Reset to beginning of trial")
            print("-" * 80)
        
        print("Robot Controls:")
        print("  Space-Bar: Start policy execution")
        print("  s: Stop policy execution")
        print("  r: Reset to home position")
        print("  o: Toggle gripper (open/close)")
        print("-" * 80)
        print("Environment Controls:")
        print("  c: Randomly spawn cubes")
        print("  1: Spawn cubes in line pattern")
        print("  2: Spawn cubes in triangle pattern")
        print("  3: Spawn cubes in stack-ready pattern")
        print("-" * 80)
        print("General:")
        print("  q: Quit")
        print("=" * 80)