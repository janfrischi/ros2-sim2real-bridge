import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from PyQt5.QtWidgets import (
    QApplication, QVBoxLayout, QHBoxLayout, QWidget, QLabel, 
    QLineEdit, QPushButton, QGroupBox, QSlider, QComboBox, QTextEdit,
    QTabWidget, QCheckBox
)
from PyQt5.QtCore import Qt, pyqtSignal
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R
import threading


class PoseSender(Node):
    """ROS2 Node for sending pose commands to the cartesian position controller"""
    
    def __init__(self, topic_name="/cartesian_position_controller/commands", publish_rate=20.0):
        super().__init__('pose_sender')
        
        # Store current pose [x, y, z, qx, qy, qz, qw]
        # Default pose with Franka's natural roll orientation (180°)
        self.current_pose = [0.5, 0.0, 0.4, 0.0, 1.0, 0.0, 0.0]  # Roll=180°, Pitch=0°, Yaw=0°
        
        # Create publisher
        self.publisher = self.create_publisher(
            Float64MultiArray,
            topic_name,
            10
        )
        
        # Create timer for continuous publishing if enabled
        self.timer = None
        self.continuous_publishing = False
        self.publish_rate = publish_rate
        
        self.get_logger().info(f"Pose Sender initialized, publishing to: {topic_name}")
        self.get_logger().info("Default orientation: Roll=180° (Franka natural downward pose)")
    
    def update_pose(self, pose_data):
        """Update the current pose"""
        self.current_pose = pose_data.copy()
    
    def publish_pose(self):
        """Publish the current pose"""
        msg = Float64MultiArray()
        msg.data = self.current_pose
        self.publisher.publish(msg)
        
        # Log the published pose
        self.get_logger().info(
            f"Published pose: "
            f"Pos=[{self.current_pose[0]:.3f}, {self.current_pose[1]:.3f}, {self.current_pose[2]:.3f}], "
            f"Quat=[{self.current_pose[3]:.3f}, {self.current_pose[4]:.3f}, "
            f"{self.current_pose[5]:.3f}, {self.current_pose[6]:.3f}]"
        )
    
    def start_continuous_publishing(self):
        """Start continuous publishing"""
        if not self.continuous_publishing:
            self.continuous_publishing = True
            self.timer = self.create_timer(1.0 / self.publish_rate, self.publish_pose)
            self.get_logger().info(f"Started continuous publishing at {self.publish_rate} Hz")
    
    def stop_continuous_publishing(self):
        """Stop continuous publishing"""
        if self.continuous_publishing:
            self.continuous_publishing = False
            if self.timer:
                self.timer.destroy()
                self.timer = None
            self.get_logger().info("Stopped continuous publishing")


class PoseSenderGUI(QWidget):
    """Interactive GUI for sending pose commands"""
    
    pose_updated = pyqtSignal(list)
    
    def __init__(self, node):
        super().__init__()
        self.node = node
        
        # Control mode: 'euler' or 'quaternion'
        self.orientation_mode = 'euler'
        self.updating_from_mode_change = False  # Prevent recursive updates
        
        # Predefined poses with Franka's default roll orientation (180°)
        # Format: [x, y, z, qx, qy, qz, qw]
        self.predefined_poses = {
            "Home": [0.5, 0.0, 0.4, 0.0, 1.0, 0.0, 0.0],  # Roll=180°, Pitch=0°, Yaw=0°
            "Left": [0.5, -0.3, 0.4, 0.0, 1.0, 0.0, 0.0],
            "Right": [0.5, 0.3, 0.4, 0.0, 1.0, 0.0, 0.0],
            "Forward": [0.7, 0.0, 0.4, 0.0, 1.0, 0.0, 0.0],
            "Backward": [0.3, 0.0, 0.4, 0.0, 1.0, 0.0, 0.0],
            "High": [0.5, 0.0, 0.6, 0.0, 1.0, 0.0, 0.0],
            "Low": [0.5, 0.0, 0.2, 0.0, 1.0, 0.0, 0.0],
            "Rotated_X_90": [0.5, 0.0, 0.4, 0.707, 0.707, 0.0, 0.0],   # Roll=180°+90°=270°
            "Rotated_Y_90": [0.5, 0.0, 0.4, 0.0, 0.707, 0.0, 0.707],   # Roll=180°, Pitch=90°
            "Rotated_Z_90": [0.5, 0.0, 0.4, 0.0, 0.707, -0.707, 0.0],  # Roll=180°, Yaw=90°
            "Natural_Down": [0.5, 0.0, 0.4, 0.0, 1.0, 0.0, 0.0],       # Natural downward orientation
        }
        
        self.init_ui()
        
        # Connect pose update signal
        self.pose_updated.connect(self.node.update_pose)
    
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Franka Pose Sender - Imitation Learning Mode Test")
        self.setGeometry(100, 100, 900, 900)
        
        main_layout = QVBoxLayout()
        
        # Title
        title = QLabel("Franka Pose Command Sender")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin: 10px;")
        main_layout.addWidget(title)
        
        # Current pose display
        self.create_current_pose_display(main_layout)
        
        # Position controls
        self.create_position_controls(main_layout)
        
        # Orientation controls with tabs
        self.create_orientation_controls(main_layout)
        
        # Predefined poses
        self.create_predefined_poses(main_layout)
        
        # Control buttons
        self.create_control_buttons(main_layout)
        
        # Status display
        self.create_status_display(main_layout)
        
        self.setLayout(main_layout)
        
        # Initialize with default pose
        self.update_all_displays()
    
    def create_current_pose_display(self, main_layout):
        """Create current pose display section"""
        group = QGroupBox("Current Pose")
        layout = QVBoxLayout()
        
        self.current_pose_label = QLabel()
        self.current_pose_label.setStyleSheet(
            "background-color: #f0f0f0; padding: 10px; font-family: monospace;"
        )
        layout.addWidget(self.current_pose_label)
        
        group.setLayout(layout)
        main_layout.addWidget(group)
    
    def create_position_controls(self, main_layout):
        """Create position control sliders"""
        group = QGroupBox("Position Controls (meters)")
        layout = QVBoxLayout()
        
        self.position_inputs = {}
        self.position_sliders = {}
        
        # Position limits [min, max, default]
        position_config = {
            'X': [0.1, 0.9, 0.5],
            'Y': [-0.5, 0.5, 0.0],
            'Z': [0.1, 0.8, 0.4]
        }
        
        for axis, (min_val, max_val, default) in position_config.items():
            row_layout = QHBoxLayout()
            
            # Label
            label = QLabel(f"{axis}:")
            label.setFixedWidth(20)
            row_layout.addWidget(label)
            
            # Slider
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(int(min_val * 1000))
            slider.setMaximum(int(max_val * 1000))
            slider.setValue(int(default * 1000))
            slider.valueChanged.connect(
                lambda value, a=axis.lower(): self.update_position_from_slider(a, value)
            )
            self.position_sliders[axis.lower()] = slider
            row_layout.addWidget(slider)
            
            # Input field
            input_field = QLineEdit()
            input_field.setFixedWidth(80)
            input_field.setText(f"{default:.3f}")
            input_field.editingFinished.connect(
                lambda a=axis.lower(): self.update_position_from_input(a)
            )
            self.position_inputs[axis.lower()] = input_field
            row_layout.addWidget(input_field)
            
            layout.addLayout(row_layout)
        
        group.setLayout(layout)
        main_layout.addWidget(group)
    
    def create_orientation_controls(self, main_layout):
        """Create orientation control tabs (Euler angles and Quaternion)"""
        group = QGroupBox("Orientation Controls")
        layout = QVBoxLayout()
        
        # Create tab widget
        self.orientation_tabs = QTabWidget()
        
        # Euler angles tab
        self.create_euler_tab()
        
        # Quaternion tab
        self.create_quaternion_tab()
        
        # Add tabs to widget
        self.orientation_tabs.addTab(self.euler_widget, "Euler Angles")
        self.orientation_tabs.addTab(self.quaternion_widget, "Quaternion")
        
        # Connect tab change signal
        self.orientation_tabs.currentChanged.connect(self.on_orientation_tab_changed)
        
        layout.addWidget(self.orientation_tabs)
        
        # Auto-sync checkbox
        self.auto_sync_checkbox = QCheckBox("Auto-sync between Euler and Quaternion")
        self.auto_sync_checkbox.setChecked(True)
        layout.addWidget(self.auto_sync_checkbox)
        
        group.setLayout(layout)
        main_layout.addWidget(group)
    
    def create_euler_tab(self):
        """Create Euler angles tab"""
        self.euler_widget = QWidget()
        layout = QVBoxLayout()
        
        self.orientation_inputs = {}
        self.orientation_sliders = {}
        
        # Orientation limits [min, max, default] in degrees
        orientation_config = {
            'Roll': [-180, 180, 180],    # Default to 180° (Franka's natural orientation)
            'Pitch': [-180, 180, 0],     # Default to 0°
            'Yaw': [-180, 180, 0]        # Default to 0°
        }
        
        for axis, (min_val, max_val, default) in orientation_config.items():
            row_layout = QHBoxLayout()
            
            # Label with explanation for Roll
            if axis == 'Roll':
                label = QLabel(f"{axis}: (180° = natural down)")
                label.setFixedWidth(150)
            else:
                label = QLabel(f"{axis}:")
                label.setFixedWidth(50)
            row_layout.addWidget(label)
            
            # Slider
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(min_val)
            slider.setMaximum(max_val)
            slider.setValue(default)
            slider.valueChanged.connect(
                lambda value, a=axis.lower(): self.update_orientation_from_slider(a, value)
            )
            self.orientation_sliders[axis.lower()] = slider
            row_layout.addWidget(slider)
            
            # Input field
            input_field = QLineEdit()
            input_field.setFixedWidth(80)
            input_field.setText(f"{default}")
            input_field.editingFinished.connect(
                lambda a=axis.lower(): self.update_orientation_from_input(a)
            )
            self.orientation_inputs[axis.lower()] = input_field
            row_layout.addWidget(input_field)
            
            layout.addLayout(row_layout)
        
        # Add explanation text
        explanation = QLabel("Note: Roll=180° is Franka's natural downward orientation")
        explanation.setStyleSheet("font-style: italic; color: #666; font-size: 10px;")
        layout.addWidget(explanation)
        
        self.euler_widget.setLayout(layout)
    
    def create_quaternion_tab(self):
        """Create Quaternion tab"""
        self.quaternion_widget = QWidget()
        layout = QVBoxLayout()
        
        self.quaternion_inputs = {}
        self.quaternion_sliders = {}
        
        # Quaternion components [min, max, default]
        quaternion_config = {
            'qx': [-1.0, 1.0, 0.0],
            'qy': [-1.0, 1.0, 1.0],  # Default to 1.0 for Franka's natural orientation
            'qz': [-1.0, 1.0, 0.0],
            'qw': [-1.0, 1.0, 0.0]
        }
        
        for component, (min_val, max_val, default) in quaternion_config.items():
            row_layout = QHBoxLayout()
            
            # Label
            label = QLabel(f"{component}:")
            label.setFixedWidth(50)
            row_layout.addWidget(label)
            
            # Slider
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(int(min_val * 1000))
            slider.setMaximum(int(max_val * 1000))
            slider.setValue(int(default * 1000))
            slider.valueChanged.connect(
                lambda value, c=component: self.update_quaternion_from_slider(c, value)
            )
            self.quaternion_sliders[component] = slider
            row_layout.addWidget(slider)
            
            # Input field
            input_field = QLineEdit()
            input_field.setFixedWidth(80)
            input_field.setText(f"{default:.3f}")
            input_field.editingFinished.connect(
                lambda c=component: self.update_quaternion_from_input(c)
            )
            self.quaternion_inputs[component] = input_field
            row_layout.addWidget(input_field)
            
            layout.addLayout(row_layout)
        
        # Normalize button
        normalize_layout = QHBoxLayout()
        normalize_btn = QPushButton("Normalize Quaternion")
        normalize_btn.setStyleSheet("background-color: #FF9800; color: white;")
        normalize_btn.clicked.connect(self.normalize_quaternion)
        normalize_layout.addWidget(normalize_btn)
        
        # Quaternion magnitude display
        self.quat_magnitude_label = QLabel("Magnitude: 1.000")
        self.quat_magnitude_label.setStyleSheet("font-family: monospace; color: #666;")
        normalize_layout.addWidget(self.quat_magnitude_label)
        
        layout.addLayout(normalize_layout)
        
        # Add explanation text
        explanation = QLabel("Note: Quaternions should be normalized (magnitude = 1.0)")
        explanation.setStyleSheet("font-style: italic; color: #666; font-size: 10px;")
        layout.addWidget(explanation)
        
        self.quaternion_widget.setLayout(layout)
    
    def create_predefined_poses(self, main_layout):
        """Create predefined poses section"""
        group = QGroupBox("Predefined Poses")
        layout = QVBoxLayout()
        
        # Dropdown for pose selection
        pose_layout = QHBoxLayout()
        pose_layout.addWidget(QLabel("Select Pose:"))
        
        self.pose_combo = QComboBox()
        self.pose_combo.addItems(list(self.predefined_poses.keys()))
        pose_layout.addWidget(self.pose_combo)
        
        load_pose_btn = QPushButton("Load Pose")
        load_pose_btn.clicked.connect(self.load_predefined_pose)
        pose_layout.addWidget(load_pose_btn)
        
        layout.addLayout(pose_layout)
        group.setLayout(layout)
        main_layout.addWidget(group)
    
    def create_control_buttons(self, main_layout):
        """Create control buttons"""
        group = QGroupBox("Control")
        layout = QVBoxLayout()
        
        # Single publish button
        publish_btn = QPushButton("Publish Pose Once")
        publish_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        publish_btn.clicked.connect(self.publish_single_pose)
        layout.addWidget(publish_btn)
        
        # Continuous publishing controls
        continuous_layout = QHBoxLayout()
        
        self.start_continuous_btn = QPushButton("Start Continuous Publishing")
        self.start_continuous_btn.setStyleSheet("background-color: #2196F3; color: white;")
        self.start_continuous_btn.clicked.connect(self.start_continuous_publishing)
        continuous_layout.addWidget(self.start_continuous_btn)
        
        self.stop_continuous_btn = QPushButton("Stop Continuous Publishing")
        self.stop_continuous_btn.setStyleSheet("background-color: #f44336; color: white;")
        self.stop_continuous_btn.clicked.connect(self.stop_continuous_publishing)
        self.stop_continuous_btn.setEnabled(False)
        continuous_layout.addWidget(self.stop_continuous_btn)
        
        layout.addLayout(continuous_layout)
        
        # Reset button
        reset_btn = QPushButton("Reset to Home")
        reset_btn.clicked.connect(self.reset_to_home)
        layout.addWidget(reset_btn)
        
        group.setLayout(layout)
        main_layout.addWidget(group)
    
    def create_status_display(self, main_layout):
        """Create status display area"""
        group = QGroupBox("Status")
        layout = QVBoxLayout()
        
        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(100)
        self.status_text.setReadOnly(True)
        layout.addWidget(self.status_text)
        
        group.setLayout(layout)
        main_layout.addWidget(group)
    
    def get_current_pose(self):
        """Get current pose from input fields"""
        try:
            # Get position
            x = float(self.position_inputs['x'].text())
            y = float(self.position_inputs['y'].text())
            z = float(self.position_inputs['z'].text())
            
            # Get quaternion based on current tab
            current_tab = self.orientation_tabs.currentIndex()
            
            if current_tab == 0:  # Euler angles tab
                # Get Euler angles and convert to quaternion
                roll = np.radians(float(self.orientation_inputs['roll'].text()))
                pitch = np.radians(float(self.orientation_inputs['pitch'].text()))
                yaw = np.radians(float(self.orientation_inputs['yaw'].text()))
                
                # Convert Euler angles to quaternion
                r = R.from_euler('xyz', [roll, pitch, yaw])
                quat = r.as_quat()  # Returns [qx, qy, qz, qw]
                
                return [x, y, z, quat[0], quat[1], quat[2], quat[3]]
                
            else:  # Quaternion tab
                # Get quaternion directly
                qx = float(self.quaternion_inputs['qx'].text())
                qy = float(self.quaternion_inputs['qy'].text())
                qz = float(self.quaternion_inputs['qz'].text())
                qw = float(self.quaternion_inputs['qw'].text())
                
                return [x, y, z, qx, qy, qz, qw]
        
        except ValueError as e:
            self.add_status_message(f"Error getting pose: {e}")
            return None
    
    def on_orientation_tab_changed(self, index):
        """Handle orientation tab change"""
        if self.auto_sync_checkbox.isChecked() and not self.updating_from_mode_change:
            self.sync_orientation_values()
    
    def sync_orientation_values(self):
        """Sync values between Euler and Quaternion tabs"""
        try:
            self.updating_from_mode_change = True
            current_tab = self.orientation_tabs.currentIndex()
            
            if current_tab == 0:  # Switched to Euler tab, convert from quaternion
                qx = float(self.quaternion_inputs['qx'].text())
                qy = float(self.quaternion_inputs['qy'].text())
                qz = float(self.quaternion_inputs['qz'].text())
                qw = float(self.quaternion_inputs['qw'].text())
                
                # Convert quaternion to Euler angles
                r = R.from_quat([qx, qy, qz, qw])
                euler = r.as_euler('xyz', degrees=True)
                
                # Update Euler inputs and sliders
                self.orientation_inputs['roll'].setText(f"{euler[0]:.1f}")
                self.orientation_inputs['pitch'].setText(f"{euler[1]:.1f}")
                self.orientation_inputs['yaw'].setText(f"{euler[2]:.1f}")
                
                self.orientation_sliders['roll'].setValue(int(euler[0]))
                self.orientation_sliders['pitch'].setValue(int(euler[1]))
                self.orientation_sliders['yaw'].setValue(int(euler[2]))
                
            else:  # Switched to Quaternion tab, convert from Euler
                roll = np.radians(float(self.orientation_inputs['roll'].text()))
                pitch = np.radians(float(self.orientation_inputs['pitch'].text()))
                yaw = np.radians(float(self.orientation_inputs['yaw'].text()))
                
                # Convert Euler angles to quaternion
                r = R.from_euler('xyz', [roll, pitch, yaw])
                quat = r.as_quat()  # Returns [qx, qy, qz, qw]
                
                # Update quaternion inputs and sliders
                self.quaternion_inputs['qx'].setText(f"{quat[0]:.3f}")
                self.quaternion_inputs['qy'].setText(f"{quat[1]:.3f}")
                self.quaternion_inputs['qz'].setText(f"{quat[2]:.3f}")
                self.quaternion_inputs['qw'].setText(f"{quat[3]:.3f}")
                
                self.quaternion_sliders['qx'].setValue(int(quat[0] * 1000))
                self.quaternion_sliders['qy'].setValue(int(quat[1] * 1000))
                self.quaternion_sliders['qz'].setValue(int(quat[2] * 1000))
                self.quaternion_sliders['qw'].setValue(int(quat[3] * 1000))
                
                # Update magnitude display
                self.update_quaternion_magnitude()
                
        except ValueError as e:
            self.add_status_message(f"Error syncing orientation: {e}")
        finally:
            self.updating_from_mode_change = False
    
    def update_position_from_slider(self, axis, value):
        """Update position input from slider"""
        position_val = value / 1000.0
        self.position_inputs[axis].setText(f"{position_val:.3f}")
        self.update_all_displays()
    
    def update_position_from_input(self, axis):
        """Update position slider from input"""
        try:
            value = float(self.position_inputs[axis].text())
            self.position_sliders[axis].setValue(int(value * 1000))
            self.update_all_displays()
        except ValueError:
            self.add_status_message(f"Invalid {axis} position value")
    
    def update_orientation_from_slider(self, axis, value):
        """Update orientation input from slider"""
        self.orientation_inputs[axis].setText(str(value))
        if self.auto_sync_checkbox.isChecked() and not self.updating_from_mode_change:
            self.sync_to_quaternion()
        self.update_all_displays()
    
    def update_orientation_from_input(self, axis):
        """Update orientation slider from input"""
        try:
            value = int(float(self.orientation_inputs[axis].text()))
            self.orientation_sliders[axis].setValue(value)
            if self.auto_sync_checkbox.isChecked() and not self.updating_from_mode_change:
                self.sync_to_quaternion()
            self.update_all_displays()
        except ValueError:
            self.add_status_message(f"Invalid {axis} orientation value")
    
    def update_quaternion_from_slider(self, component, value):
        """Update quaternion input from slider"""
        quat_val = value / 1000.0
        self.quaternion_inputs[component].setText(f"{quat_val:.3f}")
        self.update_quaternion_magnitude()
        if self.auto_sync_checkbox.isChecked() and not self.updating_from_mode_change:
            self.sync_to_euler()
        self.update_all_displays()
    
    def update_quaternion_from_input(self, component):
        """Update quaternion slider from input"""
        try:
            value = float(self.quaternion_inputs[component].text())
            self.quaternion_sliders[component].setValue(int(value * 1000))
            self.update_quaternion_magnitude()
            if self.auto_sync_checkbox.isChecked() and not self.updating_from_mode_change:
                self.sync_to_euler()
            self.update_all_displays()
        except ValueError:
            self.add_status_message(f"Invalid {component} quaternion value")
    
    def sync_to_quaternion(self):
        """Sync Euler angles to quaternion"""
        try:
            self.updating_from_mode_change = True
            
            roll = np.radians(float(self.orientation_inputs['roll'].text()))
            pitch = np.radians(float(self.orientation_inputs['pitch'].text()))
            yaw = np.radians(float(self.orientation_inputs['yaw'].text()))
            
            # Convert Euler angles to quaternion
            r = R.from_euler('xyz', [roll, pitch, yaw])
            quat = r.as_quat()  # Returns [qx, qy, qz, qw]
            
            # Update quaternion inputs and sliders
            self.quaternion_inputs['qx'].setText(f"{quat[0]:.3f}")
            self.quaternion_inputs['qy'].setText(f"{quat[1]:.3f}")
            self.quaternion_inputs['qz'].setText(f"{quat[2]:.3f}")
            self.quaternion_inputs['qw'].setText(f"{quat[3]:.3f}")
            
            self.quaternion_sliders['qx'].setValue(int(quat[0] * 1000))
            self.quaternion_sliders['qy'].setValue(int(quat[1] * 1000))
            self.quaternion_sliders['qz'].setValue(int(quat[2] * 1000))
            self.quaternion_sliders['qw'].setValue(int(quat[3] * 1000))
            
            self.update_quaternion_magnitude()
            
        except ValueError as e:
            self.add_status_message(f"Error syncing to quaternion: {e}")
        finally:
            self.updating_from_mode_change = False
    
    def sync_to_euler(self):
        """Sync quaternion to Euler angles"""
        try:
            self.updating_from_mode_change = True
            
            qx = float(self.quaternion_inputs['qx'].text())
            qy = float(self.quaternion_inputs['qy'].text())
            qz = float(self.quaternion_inputs['qz'].text())
            qw = float(self.quaternion_inputs['qw'].text())
            
            # Convert quaternion to Euler angles
            r = R.from_quat([qx, qy, qz, qw])
            euler = r.as_euler('xyz', degrees=True)
            
            # Update Euler inputs and sliders
            self.orientation_inputs['roll'].setText(f"{euler[0]:.1f}")
            self.orientation_inputs['pitch'].setText(f"{euler[1]:.1f}")
            self.orientation_inputs['yaw'].setText(f"{euler[2]:.1f}")
            
            self.orientation_sliders['roll'].setValue(int(euler[0]))
            self.orientation_sliders['pitch'].setValue(int(euler[1]))
            self.orientation_sliders['yaw'].setValue(int(euler[2]))
            
        except ValueError as e:
            self.add_status_message(f"Error syncing to Euler: {e}")
        finally:
            self.updating_from_mode_change = False
    
    def update_quaternion_magnitude(self):
        """Update quaternion magnitude display"""
        try:
            qx = float(self.quaternion_inputs['qx'].text())
            qy = float(self.quaternion_inputs['qy'].text())
            qz = float(self.quaternion_inputs['qz'].text())
            qw = float(self.quaternion_inputs['qw'].text())
            
            magnitude = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
            self.quat_magnitude_label.setText(f"Magnitude: {magnitude:.3f}")
            
            # Color code the magnitude
            if abs(magnitude - 1.0) < 0.01:
                self.quat_magnitude_label.setStyleSheet("font-family: monospace; color: green;")
            elif abs(magnitude - 1.0) < 0.1:
                self.quat_magnitude_label.setStyleSheet("font-family: monospace; color: orange;")
            else:
                self.quat_magnitude_label.setStyleSheet("font-family: monospace; color: red;")
                
        except ValueError:
            self.quat_magnitude_label.setText("Magnitude: N/A")
            self.quat_magnitude_label.setStyleSheet("font-family: monospace; color: red;")
    
    def normalize_quaternion(self):
        """Normalize the quaternion"""
        try:
            qx = float(self.quaternion_inputs['qx'].text())
            qy = float(self.quaternion_inputs['qy'].text())
            qz = float(self.quaternion_inputs['qz'].text())
            qw = float(self.quaternion_inputs['qw'].text())
            
            magnitude = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
            
            if magnitude > 0:
                qx_norm = qx / magnitude
                qy_norm = qy / magnitude
                qz_norm = qz / magnitude
                qw_norm = qw / magnitude
                
                # Update inputs and sliders
                self.quaternion_inputs['qx'].setText(f"{qx_norm:.3f}")
                self.quaternion_inputs['qy'].setText(f"{qy_norm:.3f}")
                self.quaternion_inputs['qz'].setText(f"{qz_norm:.3f}")
                self.quaternion_inputs['qw'].setText(f"{qw_norm:.3f}")
                
                self.quaternion_sliders['qx'].setValue(int(qx_norm * 1000))
                self.quaternion_sliders['qy'].setValue(int(qy_norm * 1000))
                self.quaternion_sliders['qz'].setValue(int(qz_norm * 1000))
                self.quaternion_sliders['qw'].setValue(int(qw_norm * 1000))
                
                self.update_quaternion_magnitude()
                
                if self.auto_sync_checkbox.isChecked():
                    self.sync_to_euler()
                
                self.update_all_displays()
                self.add_status_message("Quaternion normalized")
            else:
                self.add_status_message("Cannot normalize zero quaternion")
                
        except ValueError:
            self.add_status_message("Invalid quaternion values for normalization")
    
    def update_all_displays(self):
        """Update all display elements"""
        pose = self.get_current_pose()
        if pose:
            # Update current pose display
            self.current_pose_label.setText(
                f"Position: [{pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f}]\n"
                f"Quaternion: [{pose[3]:.3f}, {pose[4]:.3f}, {pose[5]:.3f}, {pose[6]:.3f}]"
            )
            
            # Update node pose
            self.pose_updated.emit(pose)
    
    def load_predefined_pose(self):
        """Load a predefined pose"""
        pose_name = self.pose_combo.currentText()
        pose = self.predefined_poses[pose_name]
        
        # Set position values
        self.position_inputs['x'].setText(f"{pose[0]:.3f}")
        self.position_inputs['y'].setText(f"{pose[1]:.3f}")
        self.position_inputs['z'].setText(f"{pose[2]:.3f}")
        
        # Update position sliders
        self.position_sliders['x'].setValue(int(pose[0] * 1000))
        self.position_sliders['y'].setValue(int(pose[1] * 1000))
        self.position_sliders['z'].setValue(int(pose[2] * 1000))
        
        # Set quaternion values
        self.quaternion_inputs['qx'].setText(f"{pose[3]:.3f}")
        self.quaternion_inputs['qy'].setText(f"{pose[4]:.3f}")
        self.quaternion_inputs['qz'].setText(f"{pose[5]:.3f}")
        self.quaternion_inputs['qw'].setText(f"{pose[6]:.3f}")
        
        # Update quaternion sliders
        self.quaternion_sliders['qx'].setValue(int(pose[3] * 1000))
        self.quaternion_sliders['qy'].setValue(int(pose[4] * 1000))
        self.quaternion_sliders['qz'].setValue(int(pose[5] * 1000))
        self.quaternion_sliders['qw'].setValue(int(pose[6] * 1000))
        
        # Convert quaternion to Euler angles
        quat = [pose[3], pose[4], pose[5], pose[6]]  # qx, qy, qz, qw
        r = R.from_quat(quat)
        euler = r.as_euler('xyz', degrees=True)
        
        # Set Euler values
        self.orientation_inputs['roll'].setText(f"{euler[0]:.1f}")
        self.orientation_inputs['pitch'].setText(f"{euler[1]:.1f}")
        self.orientation_inputs['yaw'].setText(f"{euler[2]:.1f}")
        
        # Update Euler sliders
        self.orientation_sliders['roll'].setValue(int(euler[0]))
        self.orientation_sliders['pitch'].setValue(int(euler[1]))
        self.orientation_sliders['yaw'].setValue(int(euler[2]))
        
        # Update quaternion magnitude
        self.update_quaternion_magnitude()
        
        self.update_all_displays()
        self.add_status_message(f"Loaded predefined pose: {pose_name}")
    
    def publish_single_pose(self):
        """Publish pose once"""
        pose = self.get_current_pose()
        if pose:
            self.node.publish_pose()
            self.add_status_message("Published pose once")
    
    def start_continuous_publishing(self):
        """Start continuous publishing"""
        self.node.start_continuous_publishing()
        self.start_continuous_btn.setEnabled(False)
        self.stop_continuous_btn.setEnabled(True)
        self.add_status_message("Started continuous publishing")
    
    def stop_continuous_publishing(self):
        """Stop continuous publishing"""
        self.node.stop_continuous_publishing()
        self.start_continuous_btn.setEnabled(True)
        self.stop_continuous_btn.setEnabled(False)
        self.add_status_message("Stopped continuous publishing")
    
    def reset_to_home(self):
        """Reset to home pose"""
        self.pose_combo.setCurrentText("Home")
        self.load_predefined_pose()
    
    def add_status_message(self, message):
        """Add message to status display"""
        self.status_text.append(f"[{self.get_timestamp()}] {message}")
        
        # Keep only last 10 messages
        text = self.status_text.toPlainText()
        lines = text.split('\n')
        if len(lines) > 10:
            self.status_text.setPlainText('\n'.join(lines[-10:]))
        
        # Scroll to bottom
        cursor = self.status_text.textCursor()
        cursor.movePosition(cursor.End)
        self.status_text.setTextCursor(cursor)
    
    def get_timestamp(self):
        """Get current timestamp"""
        import datetime
        return datetime.datetime.now().strftime("%H:%M:%S")


def main(args=None):
    """Main function"""
    rclpy.init(args=args)
    
    # Create ROS2 node
    node = PoseSender()
    
    # Create Qt application
    app = QApplication(sys.argv)
    gui = PoseSenderGUI(node)
    gui.show()
    
    # Run ROS2 node in separate thread
    def spin_node():
        rclpy.spin(node)
    
    ros_thread = threading.Thread(target=spin_node, daemon=True)
    ros_thread.start()
    
    # Run Qt application
    try:
        exit_code = app.exec_()
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down Pose Sender")
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(exit_code)


if __name__ == '__main__':
    main()