#!/usr/bin/env python3
"""
Instruction Mixin for BC Policy Runner
Handles all instruction printing and configuration listing functionality.
"""

class InstructionMixin:
    """Mixin class for handling instruction display and configuration listings"""
    
    def print_instructions(self):
        """Print control instructions for normal mode"""
        print("\n" + "=" * 90)
        print("BC POLICY RUNNER - CONTROL INSTRUCTIONS".center(90))
        print("=" * 90)
        
        print("Robot Controls:")
        print("  Space-Bar: Start policy execution")
        print("  s: Stop policy execution")
        print("  r: Reset to home position AND increment trial ID")
        print("  o: Toggle gripper (open/close)")
        print("  v: Update cubes from camera")
        print("-" * 90)
        print("Environment Controls - Basic Patterns (config changes only):")
        print("  c: Random spawn      │ 1: Line pattern      │ 2: Triangle pattern  │ 3: Stack-ready")
        print("-" * 90)
        print("Environment Controls - Number Key Presets:")
        print("  4: Default           │ 5: Custom positions  │ 6: Wide spread       │ 7: Tight cluster")
        print("  8: Corner formation  │ 9: Stacking ready    │ 0: Manipulation test")
        print("-" * 90)
        print("Environment Controls - Letter Key Presets:")
        print("  a: Reach challenge   │ b: Pick-place demo   │ d: Sorting task       │ e: Assembly line")
        print("  f: Circular arrange  │ h: Precision test    │ i: Learning prog 1    │ j: Learning prog 2")
        print("  k: Workspace corners")
        print("-" * 90)
        print("Information & Control:")
        print("  l: List all presets  │ g: Emergency gripper reset │ q: Quit")
        print("=" * 90)
        print("💡 TIP: Use 'l' to see detailed descriptions of all presets")
        print()

    def print_instructions_testing(self):
        """Print control instructions for testing mode"""
        print("\n" + "=" * 90)
        print("BC POLICY TESTING MODE - CONTROL INSTRUCTIONS".center(90))
        print("=" * 90)
        
        print("Robot Controls:")
        print("  Space-Bar: Start policy execution")
        print("  s: Stop policy execution")
        print("  r: Reset to home position AND increment trial ID")
        print("  o: Toggle gripper (open/close)")
        print("-" * 90)
        print("Testing Configuration Controls:")
        print("  c: Random spawn │  v: Update cubes from camera")
        print("-" * 90)
        print("Basic Test Cases (0-9):")
        print("  0: Config 0       │ 1: Config 1       │ 2: Config 2")
        print("  3: Config 3       │ 4: Config 4       │ 5: Config 5")
        print("  6: Config 6       │ 7: Config 7       │ 8: Config 8")
        print("  9: Config 9")
        print("-" * 90)
        print("Extended Test Cases (Letters):")
        print("  y: Config 10       │ u: Config 11       │ i: Config 12")
        print("  p: Config 13       │ a: Config 14       │ d: Config 15")
        print("-" * 90)
        print("Information & Control:")
        print("  l: List all testing configs │ g: Emergency gripper reset │ q: Quit")
        print("=" * 90)
        print("🧪 TESTING MODE: Systematic evaluation of behavior cloning policy")
        print("💡 TIP: Use 'l' to see detailed descriptions of all testing configurations")
        print()

    def list_testing_configs(self):
        """List all available testing configurations with descriptions"""
        configs_info = {
            "config_0": "Line, centered",
            "config_1": "Spread triangle",
            "config_2": "Vertical line",
            "config_3": "Random triangle",
            "config_4": "Long diagonals",
            "config_5": "Line with offset",
            "config_6": "Close cluster",
            "config_7": "Far y-range",
            "config_8": "Edge-to-edge test",
            "config_9": "Increasing height",
            "config_10": "Centered close",
            "config_11": "Side-to-side",
            "config_12": "Opposite corners",
            "config_13": "Full spread triangle",
            "config_14": "Short diagonal pattern in bottom-right",
            "config_15": "Inverted triangle in top-center",
            "random_config_1": "Random test case 1 - mixed positioning",
            "random_config_2": "Random test case 2 - edge case testing",
            "random_config_3": "Random test case 3 - corner emphasis",
            "validation_1": "Validation set 1 - standard evaluation",
            "validation_2": "Validation set 2 - advanced evaluation"
        }
        
        print("\n🧪 AVAILABLE TESTING CONFIGURATIONS")
        print("=" * 70)
        
        # Basic configurations (0-9)
        print("📋 BASIC TEST CASES:")
        basic_configs = [
            ("config_0", "0"), ("config_1", "1"), ("config_2", "2"), ("config_3", "3"), ("config_4", "4"),
            ("config_5", "5"), ("config_6", "6"), ("config_7", "7"), ("config_8", "8"), ("config_9", "9")
        ]
        for config, key in basic_configs:
            if config in configs_info:
                print(f"  [{key}] {config:15} : {configs_info[config]}")
        
        print("\n📋 EXTENDED TEST CASES:")
        extended_mapping = [
            ("config_10", "y"), ("config_11", "u"), ("config_12", "i"),
            ("config_13", "p"), ("config_14", "a"), ("config_15", "d")
        ]
        for config, key in extended_mapping:
            if config in configs_info:
                print(f"  [{key}] {config:15} : {configs_info[config]}")
        
        print("\n📋 RANDOM TEST CONFIGURATIONS:")
        random_mapping = [
            ("random_config_1", "f"), ("random_config_2", "h"), ("random_config_3", "j")
        ]
        for config, key in random_mapping:
            if config in configs_info:
                print(f"  [{key}] {config:15} : {configs_info[config]}")
        
        print("\n📋 VALIDATION CONFIGURATIONS:")
        validation_mapping = [
            ("validation_1", "k"), ("validation_2", "z")
        ]
        for config, key in validation_mapping:
            if config in configs_info:
                print(f"  [{key}] {config:15} : {configs_info[config]}")
        
        print("=" * 70)
        print("Usage: Press the corresponding key to load configuration")
        print("📝 Note: Numbers 0-9 for basic configs, letters for extended/special configs")
        print()

    def list_cube_presets(self):
        """List all available cube presets with descriptions (for normal mode)"""
        presets_info = {
            "default": "Original IsaacLab training positions",
            "custom_1": "Your requested custom positions",
            "wide_spread": "Cubes spread across full workspace",
            "tight_cluster": "Cubes close together in center",
            "corner_formation": "L-shaped corner arrangement",
            "stacking_ready": "Optimal positions for stacking tasks",
            "manipulation_test": "Standard manipulation testing layout",
            "reach_challenge": "Tests maximum reach capabilities",
            "pick_place_demo": "Demonstration of pick-and-place",
            "sorting_task": "Three-bin sorting scenario",
            "assembly_line": "Linear assembly sequence",
            "circular_arrangement": "Triangular/circular formation",
            "precision_test": "Close spacing for precision testing",
            "learning_progression_1": "Beginner difficulty progression",
            "learning_progression_2": "Advanced difficulty progression",
            "workspace_corners": "Extreme workspace positions"
        }
        
        print("\n📋 AVAILABLE CUBE PRESETS")
        print("=" * 60)
        for preset, description in presets_info.items():
            print(f"  {preset:<22} │ {description}")
        print("=" * 60)
        print("Usage: Press the corresponding number key or use 'p' + preset name")
        print()

    def update_status(self, status: str, additional_info: str = ""):
        """Update status display without interfering with other output"""
        # Simple status update without cursor manipulation
        print(f"\n{status}{additional_info}")
        print()  # Add spacing