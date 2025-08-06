#!/usr/bin/env python3
"""
Keyboard Handler Mixin for BC Policy Runner
Handles all keyboard input processing and command execution.
"""

import sys

class KeyboardHandlerMixin:
    """Mixin class for handling keyboard input and command processing"""
    
    def check_keyboard_input_testing(self):
        """Check for keyboard input and handle commands for testing mode"""
        try:
            key = self.keyboard.get_key()
            if key is None:
                return
            
            # Flush stdout and stderr to ensure clean output
            key = key.lower()
            sys.stdout.flush()
            sys.stderr.flush()
            
            if key == ' ':  # Space bar - start/resume testing
                if not self.is_running:
                    self.start_policy()
                    
            elif key == 's':  # S - stop testing
                if self.is_running:
                    self.stop_policy()
                    
            elif key == 'r':  # R - reset episode AND increment trial ID
                self.reset_to_home()
                self.increment_trial_id()
                
            elif key == 'o':  # O - manual gripper toggle
                self.toggle_gripper_manual()
                
            elif key == 'c':  # C - randomly spawn cubes
                self.randomly_spawn_cubes()
                self.current_config_name = "Random"

            elif key == 'g':  # G - emergency gripper reset
                self._emergency_gripper_reset()
            
            # BASIC TEST CONFIGURATIONS (0-9)
            elif key == '0':  # 0 - config_0
                self.spawn_cubes_testing("config_0")
                self.current_config_name = "Line, centered"

            elif key == '1':  # 1 - config_1
                self.spawn_cubes_testing("config_1")
                self.current_config_name = "Spread triangle"

            elif key == '2':  # 2 - config_2
                self.spawn_cubes_testing("config_2")
                self.current_config_name = "Vertical line"

            elif key == '3':  # 3 - config_3
                self.spawn_cubes_testing("config_3")
                self.current_config_name = "Random triangle"

            elif key == '4':  # 4 - config_4
                self.spawn_cubes_testing("config_4")
                self.current_config_name = "Long diagonals"

            elif key == '5':  # 5 - config_5
                self.spawn_cubes_testing("config_5")
                self.current_config_name = "Line with offset"

            elif key == '6':  # 6 - config_6
                self.spawn_cubes_testing("config_6")
                self.current_config_name = "Close cluster"

            elif key == '7':  # 7 - config_7
                self.spawn_cubes_testing("config_7")
                self.current_config_name = "Far y-range"

            elif key == '8':  # 8 - config_8
                self.spawn_cubes_testing("config_8")
                self.current_config_name = "Edge-to-edge test"

            elif key == '9':  # 9 - config_9
                self.spawn_cubes_testing("config_9")
                self.current_config_name = "Increasing height"

            # EXTENDED TEST CONFIGURATIONS (letters)
            elif key == 'y':  # Y - config_10
                self.spawn_cubes_testing("config_10")
                self.current_config_name = "Centered close"

            elif key == 'u':  # U - config_11
                self.spawn_cubes_testing("config_11")
                self.current_config_name = "Side-to-side"

            elif key == 'i':  # I - config_12
                self.spawn_cubes_testing("config_12")
                self.current_config_name = "Opposite corners"

            elif key == 'p':  # P - config_13
                self.spawn_cubes_testing("config_13")
                self.current_config_name = "Full spread triangle"

            elif key == 'a':  # A - config_14
                self.spawn_cubes_testing("config_14")
                self.current_config_name = "Short diagonal pattern"

            elif key == 'd':  # D - config_15
                self.spawn_cubes_testing("config_15")
                self.current_config_name = "Inverted triangle"

            # RANDOM TEST CONFIGURATIONS
            elif key == 'f':  # F - random_config_1
                self.spawn_cubes_testing("random_config_1")
                self.current_config_name = "Random Test 1"

            elif key == 'h':  # H - random_config_2
                self.spawn_cubes_testing("random_config_2")
                self.current_config_name = "Random Test 2"

            elif key == 'j':  # J - random_config_3
                self.spawn_cubes_testing("random_config_3")
                self.current_config_name = "Random Test 3"

            # VALIDATION CONFIGURATIONS
            elif key == 'k':  # K - validation_1
                self.spawn_cubes_testing("validation_1")
                self.current_config_name = "Validation 1"

            elif key == 'z':  # Z - validation_2
                self.spawn_cubes_testing("validation_2")
                self.current_config_name = "Validation 2"
                
            elif key == 'l':  # L - list all testing configs
                self.list_testing_configs()
            
            elif key == 'q':  # Q - quit
                self._handle_shutdown_request()
            
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"\nKeyboard input error: {e}")
            sys.stdout.flush()

    def check_keyboard_input(self):
        """Check for keyboard input and handle commands"""
        try:
            key = self.keyboard.get_key()
            if key is None:
                return
            
            # Flush stdout and stderr to ensure clean output
            key = key.lower()
            sys.stdout.flush()
            sys.stderr.flush()
            
            if key == ' ':  # Space bar - start/resume
                if not self.is_running:
                    self.start_policy()
                    
            elif key == 's':  # S - stop
                if self.is_running:
                    self.stop_policy()
                    
            elif key == 'r':  # R - reset episode AND increment trial ID
                self.reset_to_home()
                self.increment_trial_id()
                
            elif key == 'o':  # O - manual gripper toggle
                self.toggle_gripper_manual()
                
            elif key == 'c':  # C - randomly spawn cubes (config change, no trial increment)
                self.randomly_spawn_cubes()
                self.current_config_name = "Random"

            elif key == 'g':  # G - emergency gripper reset
                self._emergency_gripper_reset()
                
            elif key == '1':  # 1 - spawn cubes in line pattern
                self.spawn_cubes_in_pattern("line")
                self.current_config_name = "Line Pattern"
                
            elif key == '2':  # 2 - spawn cubes in triangle pattern
                self.spawn_cubes_in_pattern("triangle")
                self.current_config_name = "Triangle Pattern"
                
            elif key == '3':  # 3 - spawn cubes in stack-ready pattern
                self.spawn_cubes_in_pattern("stack_ready")
                self.current_config_name = "Stack Ready"
        
            # PRESET COMMANDS (Numbers 4-9, 0)
            elif key == '4':  # 4 - default preset
                self.spawn_cubes_preset("default")
                self.current_config_name = "Default"
                
            elif key == '5':  # 5 - custom_1 preset
                self.spawn_cubes_preset("custom_1")
                self.current_config_name = "Custom 1"
                
            elif key == '6':  # 6 - wide_spread preset
                self.spawn_cubes_preset("wide_spread")
                self.current_config_name = "Wide Spread"
                
            elif key == '7':  # 7 - tight_cluster preset
                self.spawn_cubes_preset("tight_cluster")
                self.current_config_name = "Tight Cluster"
                
            elif key == '8':  # 8 - corner_formation preset
                self.spawn_cubes_preset("corner_formation")
                self.current_config_name = "Corner Formation"
                
            elif key == '9':  # 9 - stacking_ready preset
                self.spawn_cubes_preset("stacking_ready")
                self.current_config_name = "Stacking Ready"
                
            elif key == '0':  # 0 - manipulation_test preset
                self.spawn_cubes_preset("manipulation_test")
                self.current_config_name = "Manipulation Test"
            
            # LETTER COMMANDS FOR REMAINING PRESETS
            elif key == 'a':  # A - reach_challenge preset
                self.spawn_cubes_preset("reach_challenge")
                self.current_config_name = "Reach Challenge"
                
            elif key == 'b':  # B - pick_place_demo preset
                self.spawn_cubes_preset("pick_place_demo")
                self.current_config_name = "Pick Place Demo"
                
            elif key == 'd':  # D - sorting_task preset
                self.spawn_cubes_preset("sorting_task")
                self.current_config_name = "Sorting Task"
                
            elif key == 'e':  # E - assembly_line preset
                self.spawn_cubes_preset("assembly_line")
                self.current_config_name = "Assembly Line"
                
            elif key == 'f':  # F - circular_arrangement preset
                self.spawn_cubes_preset("circular_arrangement")
                self.current_config_name = "Circular Arrangement"
                
            elif key == 'h':  # H - precision_test preset
                self.spawn_cubes_preset("precision_test")
                self.current_config_name = "Precision Test"
                
            elif key == 'i':  # I - learning_progression_1 preset
                self.spawn_cubes_preset("learning_progression_1")
                self.current_config_name = "Learning Progression 1"
                
            elif key == 'j':  # J - learning_progression_2 preset
                self.spawn_cubes_preset("learning_progression_2")
                self.current_config_name = "Learning Progression 2"
                
            elif key == 'k':  # K - workspace_corners preset
                self.spawn_cubes_preset("workspace_corners")
                self.current_config_name = "Workspace Corners"
                
            elif key == 'l':  # L - list all presets
                self.list_cube_presets()
            
            elif key == 'q':  # Q - quit
                self._handle_shutdown_request()
                
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"\nKeyboard input error: {e}")
            sys.stdout.flush()

    def _emergency_gripper_reset(self):
        """Helper method for emergency gripper reset"""
        with self.gripper_action_lock:
            self.gripper_action_in_progress = False
            self.gripper_last_command_time = 0.0
        print("\n🚨 Emergency gripper reset performed")

    def _handle_shutdown_request(self):
        """Helper method for shutdown request"""
        self.shutdown_requested = True
        print("\nShutdown requested...")
        sys.stdout.flush()
        raise KeyboardInterrupt("User requested shutdown")