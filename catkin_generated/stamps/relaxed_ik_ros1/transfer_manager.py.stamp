#!/usr/bin/env python3
import subprocess
import time
import signal
import sys
import re
import os
from typing import List
from datetime import datetime
from threading import Thread, Event
import rospy
from relaxed_ik_ros1.msg import GUIMsg
from jsk_rviz_plugins.msg import OverlayText

def read_transfer_times(filename: str) -> List[float]:
    """
    Read transfer times from file, converting hours to floats.
    
    Args:
        filename: Path to the file containing transfer times
        
    Returns:
        List of float values representing hours
    """
    try:
        with open(filename, 'r') as f:
            # Read non-empty lines and convert to float
            return [float(line.strip()) for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Transfer times file '{filename}' not found. Using default of 1 hour.")
        return [1.0]
    except ValueError as e:
        print(f"Error parsing transfer times: {e}. Using default of 1.0 hour.")
        return [1.0]

class TransferManager:
    def __init__(self, transfer_times: List[float]):
        self.transfer_times = transfer_times
        self.current_process = None
        self.package_name = 'relaxed_ik_ros1'
        self.launch_file = 'gui_v3_control.launch'
        self.max_retries = 3
        self.retry_delay = 3
        self.start_time = None
        self.current_transfer_index = 0
        self.is_running = False
        
        # State management
        self.needs_restart = False
        self.restart_lock = False
        self.current_mode = 'xbox'
        self.exit_flag = Event()
        self.transition_flag = Event()
        
        # Error patterns
        self.error_patterns = [
            r"Motion finished commanded, but the robot is still moving!.*velocity_limits_violation.*velocity_discontinuity.*acceleration_discontinuity",
            r"control_command_success_rate: 0\.95",
            r"control_command_success_rate: 0\.78",
            r"libfranka: Move command aborted: motion aborted by reflex!.*communication_constraints_violation"
        ]
        
        # Initialize ROS node and communications
        rospy.init_node('transfer_manager', anonymous=True)
        self.gui_pub = rospy.Publisher('/gui_msg', GUIMsg, queue_size=1)
        self.text_pub = rospy.Publisher("/text_overlay", OverlayText, queue_size=1)
        
        # Subscribe to GUI messages
        rospy.Subscriber("/gui_msg", GUIMsg, self.gui_callback)
        
        self.setup_signal_handlers()

    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def signal_handler(self, signum, frame):
        """Handle termination signals."""
        if signum == signal.SIGINT and self.current_mode == 'xbox':
            self.log_message("Ctrl+C received, transitioning to timer mode...")
            self.transition_flag.set()
        elif signum == signal.SIGTERM or self.current_mode == 'list':
            self.log_message("Termination signal received, cleaning up...")
            self.exit_flag.set()
            self.cleanup()
            sys.exit(0)

    def gui_callback(self, msg):
        """Handle incoming GUI messages and timer control."""
        if msg.go_flag == "start" and not self.is_running:
            self.log_message("Received start command from GUI")
            self.is_running = True
            self.start_time = datetime.now()
            self.current_mode = 'list'
            self.transition_flag.set()
            
        elif msg.go_flag == "stop":
            self.log_message("Received stop command from GUI")
            self.is_running = False
            self.current_mode = 'xbox'
            self.cleanup()
            self.current_process = self.execute_launch_file()

    def run_transfer_sequence(self):
        """Execute the transfer sequence based on timer."""
        while not self.exit_flag.is_set() and self.current_transfer_index < len(self.transfer_times):
            try:
                if not self.is_running:
                    time.sleep(0.1)
                    continue

                current_time = datetime.now()
                elapsed_hours = (current_time - self.start_time).total_seconds() / 3600
                target_time = self.transfer_times[self.current_transfer_index]
                
                if elapsed_hours >= target_time:
                    self.log_message(f"Executing transfer {self.current_transfer_index + 1}")
                    # Execute transfer
                    if self.current_process:
                        self.cleanup()
                    self.current_process = self.execute_launch_file()
                    
                    # Move to next transfer
                    self.current_transfer_index += 1
                    if self.current_transfer_index < len(self.transfer_times):
                        self.start_time = current_time  # Reset timer for next transfer
                
                # Update GUI with time remaining
                remaining_hours = target_time - elapsed_hours if elapsed_hours < target_time else 0
                self.publish_timer_update(remaining_hours)
                
                time.sleep(0.1)  # Prevent CPU overload
                
            except Exception as e:
                self.log_message(f"Error in transfer sequence: {e}", error=True)
                self.cleanup()
                break

    def format_time_remaining(self, hours_remaining: float) -> str:
        """Convert hours to HH:MM:SS format."""
        total_seconds = int(hours_remaining * 3600)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def get_elapsed_time(self) -> float:
        """Get elapsed time in hours since timer start."""
        if self.start_time is None:
            return 0.0
        elapsed = datetime.now() - self.start_time
        return elapsed.total_seconds() / 3600

    def publish_timer_update(self, hours_remaining: float):
        """Publish timer update to GUI."""
        msg = GUIMsg()
        msg.transfer_times = self.format_time_remaining(hours_remaining)
        msg.robot_mode = self.current_mode
        msg.curr_transfers = self.current_transfer_index + 1
        msg.num_transfers = len(self.transfer_times)
        self.gui_pub.publish(msg)

    def log_message(self, message: str, error: bool = False):
        """Log a message with timestamp."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        prefix = 'ERROR: ' if error else 'INFO: '
        print(f"[{timestamp}] {prefix}{message}")

    def cleanup(self):
        """Perform cleanup of processes."""
        try:
            if self.current_process:
                self.log_message("Cleaning up current process...")
                try:
                    pgid = os.getpgid(self.current_process.pid)
                    os.killpg(pgid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                
                try:
                    self.current_process.terminate()
                    self.current_process.wait(timeout=3)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    try:
                        self.current_process.kill()
                        self.current_process.wait(timeout=2)
                    except:
                        pass
                
                self.current_process = None
                
            # Final cleanup
            subprocess.run(['pkill', '-f', f'roslaunch.*{self.launch_file}'], timeout=5)
        except Exception as e:
            self.log_message(f"Error during cleanup: {e}", error=True)

    def execute_launch_file(self) -> subprocess.Popen:
        """Execute the launch file with retry logic."""
        retries = 0
        while retries < self.max_retries:
            try:
                cmd = ['roslaunch', self.package_name, self.launch_file, f'flag:={self.current_mode}']
                self.log_message(f"Starting launch with mode '{self.current_mode}' and command: {' '.join(cmd)}")
                
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=1,
                    universal_newlines=False,
                    preexec_fn=os.setsid
                )
                
                stdout_thread = Thread(target=self.stream_output, args=(process.stdout,), daemon=True)
                stderr_thread = Thread(target=self.stream_output, args=(process.stderr, 'ERROR: '), daemon=True)
                stdout_thread.start()
                stderr_thread.start()
                
                if process.poll() is None:
                    return process
                else:
                    raise subprocess.SubprocessError("Process failed to start")
                    
            except (subprocess.SubprocessError, OSError) as e:
                retries += 1
                self.log_message(f"Attempt {retries} failed: {str(e)}", error=True)
                if retries < self.max_retries:
                    self.log_message(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    self.log_message("Maximum retry attempts reached. Exiting.", error=True)
                    sys.exit(1)
        
        self.log_message("Failed to start process after all retry attempts", error=True)
        sys.exit(1)

    def stream_output(self, stream, prefix=''):
        """Stream output from a pipe to stdout."""
        for line in iter(stream.readline, b''):
            if self.exit_flag.is_set():
                break
            decoded_line = line.decode().rstrip()
            print(f"{prefix}{decoded_line}")
            
            if prefix == 'ERROR: ' and self.check_error_patterns(decoded_line):
                error_msg = f"Detected critical error: {decoded_line}"
                self.log_message(error_msg, error=True)
                if not self.restart_lock:
                    self.needs_restart = True

    def check_error_patterns(self, line: str) -> bool:
        """Check if line matches any error patterns."""
        for pattern in self.error_patterns:
            if re.search(pattern, line):
                return True
        return False

    def run(self):
        """Main execution loop with improved timer handling."""
        try:
            # Start in xbox mode
            self.current_mode = 'xbox'
            self.current_process = self.execute_launch_file()
            
            while not self.exit_flag.is_set():
                if self.transition_flag.is_set():
                    # Switch to timed transfer mode
                    self.cleanup()
                    self.current_mode = 'list'
                    self.start_time = datetime.now()
                    self.current_transfer_index = 0
                    self.run_transfer_sequence()
                    self.transition_flag.clear()
                
                if self.needs_restart:
                    self.handle_process_restart()
                
                # Always publish current status
                msg = GUIMsg()
                msg.robot_mode = self.current_mode
                if self.is_running and self.start_time:
                    current_target = self.transfer_times[self.current_transfer_index]
                    elapsed = self.get_elapsed_time()
                    remaining = max(0, current_target - elapsed)
                    msg.transfer_times = self.format_time_remaining(remaining)
                else:
                    msg.transfer_times = "00:00:00"
                self.gui_pub.publish(msg)
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            self.cleanup()
        except Exception as e:
            self.log_message(f"Unexpected error in main loop: {e}", error=True)
            self.cleanup()
            sys.exit(1)

def main():
    try:
        transfer_times = read_transfer_times('/home/caleb/robochem_steps/transfer_times.txt')
        manager = TransferManager(transfer_times)
        manager.run()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()