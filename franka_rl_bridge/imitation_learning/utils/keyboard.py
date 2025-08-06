import sys
import termios
import select

class KeyboardInput:
    """Handles keyboard input in a non-blocking way"""
    
    def __init__(self):
        self.old_settings = None
        self.setup_terminal()
        
    def setup_terminal(self):
        """Setup terminal for non-blocking input"""
        try:
            self.old_settings = termios.tcgetattr(sys.stdin)
            new_settings = termios.tcgetattr(sys.stdin)
            
            # Use cbreak mode instead of raw mode for better formatting
            new_settings[3] = new_settings[3] & ~(termios.ECHO | termios.ICANON)
            new_settings[6][termios.VMIN] = 0  # Non-blocking read
            new_settings[6][termios.VTIME] = 0  # No timeout
            
            termios.tcsetattr(sys.stdin, termios.TCSANOW, new_settings)
        except:
            pass
        
    def restore_terminal(self):
        """Restore terminal settings"""
        try:
            if self.old_settings:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
                # Force flush after restoring terminal
                sys.stdout.flush()
                sys.stderr.flush()
        except:
            pass
        
    def get_key(self):
        """Get a single keypress if available"""
        try:
            if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                key = sys.stdin.read(1)
                # Force flush after reading key to prevent formatting issues
                sys.stdout.flush()
                return key
            return None
        except:
            return None