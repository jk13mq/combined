"""
Music Generation Pipeline

This module provides functionality to generate music using AI models,
including lyrics generation and structured music composition.
"""

import os
import sys
import json
import time
from typing import Dict, List, Tuple, Optional

import librosa
import numpy as np
import requests
from openai import OpenAI
from pydub import AudioSegment
from IPython.display import Audio, display

# Try to import serial, but make it optional
try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("⚠️  PySerial not available. Running in simulation mode only.")
    print("To install: pip install pyserial")


# Configuration
class Config:
    """Configuration settings for the music generation pipeline."""
    
    # API Keys and URLs
    OPENAI_API_KEY = ""
    UBERDUCK_API_KEY = ""
    
    # API endpoints
    UBERDUCK_BASE_URL = "https://api.uberduck.ai/generate-instrumental"
    UBERDUCK_MUSIC_ENDPOINT = "https://api.uberduck.ai/generate-song"
    
    # Audio settings
    DEFAULT_OUTPUT_FILE = "music.mp3"
    DEFAULT_COMMANDS_FILE = "jetrover_dance_commands.json"
    
    # Rover settings
    DEFAULT_BAUDRATE = 115200
    DEFAULT_MOVEMENT_DURATION = 0.5
    MIN_SPEED = 30
    MAX_SPEED = 255


# Set environment variables
os.environ["OPENAI_API_KEY"] = Config.OPENAI_API_KEY


class LyricsGenerator:
    """Handles AI-powered lyrics generation."""
    
    def __init__(self):
        self.client = OpenAI()
    
    def generate_structured_lyrics(self, prompt: str) -> str:
        """Generate structured song lyrics with multiple parts."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system", 
                        "content": """You are a songwriter. Write song lyrics with clear structure:
                        - Part 1: First verse and chorus
                        - Part 2: Second verse and chorus (or bridge and final chorus)
                        Keep each part under 100 words and make them flow as one complete song."""
                    },
                    {
                        "role": "user", 
                        "content": f"Write structured song lyrics about: {prompt}"
                    }
                ],
                max_tokens=3000
            )
            
            full_lyrics = response.choices[0].message.content
            print(full_lyrics)
            return full_lyrics
        
        except Exception as e:
            print(f"Error generating lyrics: {e}")
            return f"La la la, this song is about {prompt}"


class MusicGenerator:
    """Handles music generation using Uberduck API."""
    
    def __init__(self):
        self.headers = {
            "Authorization": f"Bearer {Config.UBERDUCK_API_KEY}",
            "Content-Type": "application/json"
        }
    
    def generate_music(self, prompt: str, lyrics: str, output_file: str = Config.DEFAULT_OUTPUT_FILE) -> Optional[str]:
        """Generate music using Uberduck's API."""
        
        style_prompt = f"""music track backed up with vocal need to balance both music tracks and vocal. 
        music tracks should be louder than the vocal.
        
        Song theme: {prompt}."""
        
        payload = {
            "lyrics": lyrics,
            "model_version": "v2",
            "style_prompt": style_prompt
        }
        
        try:
            print("Submitting generation request...")
            response = requests.post(
                Config.UBERDUCK_MUSIC_ENDPOINT,
                headers=self.headers,
                json=payload
            )
            
            response.raise_for_status()
            result = response.json()
            
            if 'output_url' in result:
                return self._download_audio(result['output_url'], output_file)
            else:
                print("❌ No audio URL in response")
                print(f"Response: {result}")
                return None
                
        except requests.exceptions.HTTPError as http_err:
            print(f"❌ HTTP error occurred: {http_err}")
            if hasattr(http_err, 'response') and hasattr(http_err.response, 'text'):
                print(f"Response: {http_err.response.text}")
        except Exception as e:
            print(f"❌ Error generating music: {str(e)}")
        
        return None
    
    def _download_audio(self, audio_url: str, output_file: str) -> Optional[str]:
        """Download audio file from URL."""
        try:
            print(f"🎧 Audio available at: {audio_url}")
            print("⬇️  Downloading audio file...")
            
            audio_response = requests.get(audio_url, stream=True)
            audio_response.raise_for_status()
            
            with open(output_file, 'wb') as f:
                for chunk in audio_response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file) / 1024
                print(f"✅ Success! Music saved to: {output_file} ({file_size:.2f} KB)")
                return output_file
            else:
                print("❌ Failed to save the audio file")
                return None
                
        except Exception as download_error:
            print(f"❌ Error downloading audio: {str(download_error)}")
            return None


class AudioAnalyzer:
    """Analyzes audio files and extracts features for robot movement."""
    
    def __init__(self, audio_file_path: str):
        self.audio_file = audio_file_path
        self.y = None
        self.sr = None
        self.tempo = None
        self.beat_frames = None
        self.beat_times = None
        self._load_audio()
        
    def _load_audio(self):
        """Load audio file."""
        try:
            self.y, self.sr = librosa.load(self.audio_file)
            print(f"🎵 Audio loaded: {len(self.y)} samples at {self.sr} Hz")
            print(f"⏱️  Duration: {len(self.y)/self.sr:.2f} seconds")
        except Exception as e:
            print(f"❌ Error loading audio: {e}")
            sys.exit(1)
    
    def extract_beats(self) -> Tuple[float, np.ndarray]:
        """Extract beats and tempo from audio."""
        try:
            tempo, self.beat_frames = librosa.beat.beat_track(
                y=self.y, sr=self.sr, hop_length=512
            )
            
            self.tempo = float(tempo) if hasattr(tempo, '__len__') else tempo
            self.beat_times = librosa.frames_to_time(
                self.beat_frames, sr=self.sr, hop_length=512
            )
            
            print(f"🎼 Detected tempo: {self.tempo:.2f} BPM")
            print(f"🥁 Number of beats: {len(self.beat_times)}")
            
            return self.tempo, self.beat_times
        except Exception as e:
            print(f"❌ Error extracting beats: {e}")
            return 120.0, np.array([])
    
    def extract_audio_features(self) -> Tuple[np.ndarray, ...]:
        """Extract audio features for movement mapping."""
        try:
            # RMS Energy for movement intensity
            rms_energy = librosa.feature.rms(y=self.y, hop_length=512)[0]
            
            # Spectral centroid for movement direction changes
            spectral_centroids = librosa.feature.spectral_centroid(
                y=self.y, sr=self.sr, hop_length=512
            )[0]
            
            # Zero crossing rate for movement type variation
            zcr = librosa.feature.zero_crossing_rate(self.y, hop_length=512)[0]
            
            # Onset envelope for rhythm detection
            onset_envelope = librosa.onset.onset_strength(
                y=self.y, sr=self.sr, hop_length=512
            )
            
            times = librosa.frames_to_time(
                np.arange(len(rms_energy)), sr=self.sr, hop_length=512
            )
            
            return times, rms_energy, spectral_centroids, zcr, onset_envelope
        except Exception as e:
            print(f"❌ Error extracting features: {e}")
            return tuple(np.array([]) for _ in range(5))
    
    @staticmethod
    def safe_normalize(arr: np.ndarray) -> np.ndarray:
        """Safely normalize array to 0-1 range."""
        if len(arr) == 0:
            return arr
        arr_min, arr_max = np.min(arr), np.max(arr)
        if arr_max == arr_min:
            return np.ones_like(arr) * 0.5
        return (arr - arr_min) / (arr_max - arr_min)


class MovementCommandGenerator:
    """Generates movement commands for robot based on audio analysis."""
    
    def __init__(self, analyzer: AudioAnalyzer):
        self.analyzer = analyzer
        # Define safer arm positions with smoother transitions
        self.arm_positions = {
            "snake_wave": {
                "forward": {
                    "sequence": [
                        {2: 550, 3: 500, 4: 550},  # Slight up
                        {2: 500, 3: 550, 4: 500},  # Slight down
                        {2: 550, 3: 500, 4: 550}   # Slight up
                    ]
                },
                "return": {2: 500, 3: 500, 4: 500}  # Home position
            },
            "energy_burst": {
                "forward": {
                    "sequence": [
                        {2: 600, 3: 600, 4: 600},  # Moderate extension
                        {2: 400, 3: 400, 4: 400},  # Moderate contraction
                        {2: 500, 3: 500, 4: 500}   # Center
                    ]
                },
                "return": {2: 500, 3: 500, 4: 500}
            },
            "gentle_sway": {
                "forward": {
                    "sequence": [
                        {2: 550, 3: 450, 4: 550},  # Very slight up
                        {2: 450, 3: 550, 4: 450},  # Very slight down
                        {2: 500, 3: 500, 4: 500}   # Center
                    ]
                },
                "return": {2: 500, 3: 500, 4: 500}
            },
            "spiral_motion": {
                "forward": {
                    "sequence": [
                        {1: 550, 2: 600, 3: 450},  # Gentle spiral up
                        {1: 450, 2: 400, 3: 550},  # Gentle spiral down
                        {1: 500, 2: 500, 3: 500}   # Center
                    ]
                },
                "return": {1: 500, 2: 500, 3: 500}
            },
            "wave_dance": {
                "forward": {
                    "sequence": [
                        {4: 600, 5: 600},          # Gentle wrist up
                        {4: 400, 5: 400},          # Gentle wrist down
                        {4: 500, 5: 500}           # Center
                    ]
                },
                "return": {4: 500, 5: 500}
            },
            "figure_eight": {
                "forward": {
                    "sequence": [
                        {1: 550, 2: 600, 3: 450, 4: 550},  # Top right
                        {1: 450, 2: 600, 3: 550, 4: 450},  # Top left
                        {1: 550, 2: 400, 3: 550, 4: 550},  # Bottom right
                        {1: 450, 2: 400, 3: 450, 4: 450},  # Bottom left
                        {1: 500, 2: 500, 3: 500, 4: 500}   # Center
                    ]
                },
                "return": {1: 500, 2: 500, 3: 500, 4: 500}
            },
            "cascade_flow": {
                "forward": {
                    "sequence": [
                        {2: 600, 3: 500, 4: 500},  # Shoulder up
                        {2: 500, 3: 600, 4: 500},  # Elbow up
                        {2: 500, 3: 500, 4: 600},  # Wrist up
                        {2: 400, 3: 500, 4: 500},  # Shoulder down
                        {2: 500, 3: 400, 4: 500},  # Elbow down
                        {2: 500, 3: 500, 4: 400}   # Wrist down
                    ]
                },
                "return": {2: 500, 3: 500, 4: 500}
            },
            "helix_spin": {
                "forward": {
                    "sequence": [
                        {1: 550, 2: 600, 3: 450, 4: 550, 5: 600},  # Spiral up right
                        {1: 450, 2: 600, 3: 550, 4: 450, 5: 400},  # Spiral up left
                        {1: 550, 2: 400, 3: 550, 4: 550, 5: 600},  # Spiral down right
                        {1: 450, 2: 400, 3: 450, 4: 450, 5: 400},  # Spiral down left
                        {1: 500, 2: 500, 3: 500, 4: 500, 5: 500}   # Center
                    ]
                },
                "return": {1: 500, 2: 500, 3: 500, 4: 500, 5: 500}
            },
            "pendulum_swing": {
                "forward": {
                    "sequence": [
                        {1: 600, 2: 550, 3: 450},  # Swing right
                        {1: 400, 2: 550, 3: 550},  # Swing left
                        {1: 600, 2: 450, 3: 450},  # Swing right low
                        {1: 400, 2: 450, 3: 550},  # Swing left low
                        {1: 500, 2: 500, 3: 500}   # Center
                    ]
                },
                "return": {1: 500, 2: 500, 3: 500}
            },
            "wave_cascade": {
                "forward": {
                    "sequence": [
                        {2: 600, 3: 500, 4: 600, 5: 600},  # Wave up
                        {2: 500, 3: 600, 4: 500, 5: 400},  # Wave middle
                        {2: 400, 3: 500, 4: 400, 5: 600},  # Wave down
                        {2: 500, 3: 400, 4: 500, 5: 400},  # Wave middle
                        {2: 500, 3: 500, 4: 500, 5: 500}   # Center
                    ]
                },
                "return": {2: 500, 3: 500, 4: 500, 5: 500}
            },
            "energy_pulse": {
                "forward": {
                    "sequence": [
                        {2: 600, 3: 600, 4: 600, 5: 600},  # Moderate extension
                        {2: 400, 3: 400, 4: 400, 5: 400},  # Moderate contraction
                        {2: 550, 3: 550, 4: 550, 5: 550},  # Slight extension
                        {2: 450, 3: 450, 4: 450, 5: 450},  # Slight contraction
                        {2: 500, 3: 500, 4: 500, 5: 500}   # Center
                    ]
                },
                "return": {2: 500, 3: 500, 4: 500, 5: 500}
            }
        }
    
    def _determine_movement_pattern(self, energy: float, brightness: float, roughness: float) -> str:
        """Determine movement pattern based on audio features with more even distribution."""
        # Create a more balanced distribution by using modulo operation on beat number
        # This ensures each pattern gets roughly equal representation
        beat_number = int(energy * 1000) % 5  # Use energy to create variation but maintain even distribution
        
        # Map beat number to movement patterns
        pattern_map = {
            0: "cascade_flow",    # ~20%
            1: "gentle_sway",     # ~20%
            2: "pendulum_swing",  # ~20%
            3: "snake_wave",      # ~20%
            4: "wave_dance"       # ~20%
        }
        
        # Add some variation based on audio features while maintaining balance
        if energy > 0.8 and roughness > 0.7:
            # High energy and roughness might trigger special patterns
            if beat_number % 2 == 0:
                return "cascade_flow"
            else:
                return "wave_dance"
        elif energy < 0.3 and brightness < 0.3:
            # Low energy and brightness might favor gentler patterns
            if beat_number % 2 == 0:
                return "gentle_sway"
            else:
                return "snake_wave"
        
        # Default to the evenly distributed pattern
        return pattern_map[beat_number]
    
    def _calculate_movement_duration(self, current_time: float, beat_times: np.ndarray, index: int, energy: float) -> float:
        """Calculate movement duration based on energy level and pattern complexity."""
        base_duration = 0.8  # Increased base duration for smoother movements
        
        if index < len(beat_times) - 1:
            beat_interval = float(beat_times[index + 1] - current_time)
            # Adjust duration based on energy level, but keep movements smooth
            if energy > 0.8:
                return min(0.6, beat_interval)  # Faster but still smooth for high energy
            elif energy < 0.3:
                return min(1.0, beat_interval)  # Slower movements for low energy
            else:
                return min(0.8, beat_interval)  # Medium speed for moderate energy
        else:
            return base_duration
    
    def generate_commands(self) -> List[Dict]:
        """Generate robot movement commands with synchronized arm movements."""
        tempo, beat_times = self.analyzer.extract_beats()
        
        if len(beat_times) == 0:
            print("❌ No beats detected in audio")
            return []
        
        times, rms_energy, spectral_centroids, zcr, onset_envelope = self.analyzer.extract_audio_features()
        
        # Normalize features
        norm_energy = self.analyzer.safe_normalize(rms_energy)
        norm_spectral = self.analyzer.safe_normalize(spectral_centroids)
        norm_zcr = self.analyzer.safe_normalize(zcr)
        
        commands = []
        
        for i, beat_time in enumerate(beat_times):
            if len(times) > 0 and len(norm_energy) > 0:
                feature_idx = min(np.argmin(np.abs(times - beat_time)), len(norm_energy) - 1)
                energy = norm_energy[feature_idx]
                brightness = norm_spectral[feature_idx]
                roughness = norm_zcr[feature_idx]
            else:
                energy = 0.5 + 0.3 * np.sin(i * 0.5)
                brightness = 0.5 + 0.2 * np.cos(i * 0.3)
                roughness = 0.5
            
            # Generate movement parameters
            movement_pattern = self._determine_movement_pattern(energy, brightness, roughness)
            duration = self._calculate_movement_duration(beat_time, beat_times, i, energy)
            
            # Get arm positions for the pattern
            arm_positions = self.arm_positions[movement_pattern]
            
            # Create sequence of movements
            sequence = arm_positions["forward"]["sequence"]
            return_pos = arm_positions["return"]
            
            # Calculate duration for each movement in sequence
            sequence_duration = duration / len(sequence)
            
            # Add forward movements
            for j, pos in enumerate(sequence):
                command = {
                    'timestamp': float(beat_time + j * sequence_duration),
                    'movement_pattern': movement_pattern,
                    'duration': float(sequence_duration),
                    'energy_level': float(energy),
                    'brightness': float(brightness),
                    'beat_number': i + 1,
                    'arm_positions': pos,
                    'movement_type': 'forward'
                }
                commands.append(command)
            
            # Add return movement
            return_command = {
                'timestamp': float(beat_time + duration),
                'movement_pattern': movement_pattern,
                'duration': float(duration),
                'energy_level': float(energy),
                'brightness': float(brightness),
                'beat_number': i + 1,
                'arm_positions': return_pos,
                'movement_type': 'return'
            }
            commands.append(return_command)
        
        return commands


class JetRoverController:
    """Controls Jet Rover hardware or simulation."""
    
    def __init__(self, port: Optional[str] = None, baudrate: int = Config.DEFAULT_BAUDRATE):
        self.serial_connection = None
        self.connected = False
        self.movement_queue = []
        
        if port and SERIAL_AVAILABLE:
            self._connect_hardware(port, baudrate)
        else:
            print("🔄 Running in simulation mode")
    
    def _connect_hardware(self, port: str, baudrate: int):
        """Connect to hardware."""
        try:
            self.serial_connection = serial.Serial(port, baudrate, timeout=1)
            self.connected = True
            print(f"🤖 Jet Rover connected on {port}")
            time.sleep(2)
            self._test_connection()
        except Exception as e:
            print(f"⚠️  Serial connection failed: {e}")
            print("🔄 Running in simulation mode")
    
    def _test_connection(self):
        """Test the connection with a simple command."""
        if self.connected:
            try:
                self.serial_connection.write(b"TEST\n")
                print("✅ Connection test sent")
            except Exception as e:
                print(f"⚠️  Connection test failed: {e}")
    
    def send_wheel_command(self, left_speed: int, right_speed: int, duration: float):
        """Send wheel speed commands to Jet Rover."""
        if self.connected:
            try:
                command = f"MOTOR_L:{left_speed},MOTOR_R:{right_speed}\n"
                self.serial_connection.write(command.encode())
                print(f"📡 Hardware: L={left_speed}, R={right_speed} for {duration:.2f}s")
            except Exception as e:
                print(f"❌ Command send error: {e}")
        else:
            print(f"🎮 Simulation: Left={left_speed}, Right={right_speed} for {duration:.2f}s")
    
    def stop_rover(self):
        """Stop all Jet Rover motors."""
        if self.connected:
            try:
                self.serial_connection.write(b"MOTOR_L:0,MOTOR_R:0\n")
                print("🛑 Rover stopped")
            except Exception as e:
                print(f"❌ Stop command error: {e}")
        else:
            print("🛑 Simulation: Rover stopped")
    
    def load_movement_commands(self, commands: List[Dict]):
        """Load movement commands."""
        self.movement_queue = commands
        print(f"📥 Loaded {len(commands)} Jet Rover commands")
    
    def execute_synchronized_dance(self, start_delay: int = 3):
        """Execute synchronized dance with audio."""
        if not self.movement_queue:
            print("❌ No commands to execute")
            return
            
        print(f"🎭 Starting Jet Rover dance in {start_delay} seconds...")
        for i in range(start_delay, 0, -1):
            print(f"    {i}...")
            time.sleep(1)
        
        print("🎵 DANCE STARTED! 🎵")
        start_time = time.time()
        
        try:
            for command in self.movement_queue:
                target_time = start_time + command['timestamp']
                current_time = time.time()
                
                if current_time < target_time:
                    time.sleep(target_time - current_time)
                
                self._execute_movement_pattern(command)
                
        except KeyboardInterrupt:
            print("\n⏹️  Dance interrupted by user")
        except Exception as e:
            print(f"\n❌ Dance error: {e}")
        finally:
            self.stop_rover()
            print("🎉 Dance complete!")
    
    def _execute_movement_pattern(self, command: Dict):
        """Execute specific movement pattern."""
        pattern = command['movement_pattern']
        duration = command['duration']
        
        print(f"⏰ Beat {command['beat_number']} ({command['timestamp']:.1f}s): {pattern.upper()}")
        
        # Movement pattern implementations
        movement_methods = {
            "snake_wave": self._snake_wave,
            "energy_burst": self._energy_burst,
            "gentle_sway": self._gentle_sway,
            "spiral_motion": self._spiral_motion,
            "wave_dance": self._wave_dance,
        }
        
        method = movement_methods.get(pattern, self._gentle_sway)
        method(duration)
    
    def _snake_wave(self, duration: float):
        """Snake wave movement."""
        print(f"    🐍 SNAKE WAVE")
        
        if duration > 0.4:
            movements = [(700, 0), (0, 700), (700, 700), (0, 0)]
            segment_time = duration / len(movements)
            for left, right in movements:
                self.send_wheel_command(left, right, segment_time)
                time.sleep(segment_time)
        else:
            self.send_wheel_command(700, 700, duration)
            time.sleep(duration)
    
    def _energy_burst(self, duration: float):
        """Energy burst movement."""
        print(f"    💥 ENERGY BURST")
        self.send_wheel_command(700, 0, duration)
        time.sleep(duration)
    
    def _gentle_sway(self, duration: float):
        """Gentle swaying movement."""
        print(f"    🌊 GENTLE SWAY")
        
        if duration > 0.4:
            half_time = duration / 2
            avg_speed = 600
            
            self.send_wheel_command(avg_speed, int(avg_speed * 0.6), half_time)
            time.sleep(half_time)
            self.send_wheel_command(int(avg_speed * 0.6), avg_speed, half_time)
            time.sleep(half_time)
        else:
            self.send_wheel_command(700, 700, duration)
            time.sleep(duration)
    
    def _spiral_motion(self, duration: float):
        """Spiral motion movement."""
        print(f"    🌀 SPIRAL MOTION")
        self.send_wheel_command(700, 0, duration)
        time.sleep(duration)
    
    def _wave_dance(self, duration: float):
        """Wave dance movement."""
        print(f"    💃 WAVE DANCE")
        
        if duration > 0.3:
            shake_time = duration / 4
            for _ in range(2):
                self.send_wheel_command(700, 0, shake_time)
                time.sleep(shake_time)
                self.send_wheel_command(0, 700, shake_time)
                time.sleep(shake_time)
        else:
            self.send_wheel_command(700, 700, duration)
            time.sleep(duration)


class Utils:
    """Utility functions for file operations and data display."""
    
    @staticmethod
    def save_commands(commands: List[Dict], filename: str = Config.DEFAULT_COMMANDS_FILE) -> bool:
        """Save commands to JSON file."""
        try:
            with open(filename, 'w') as f:
                json.dump(commands, f, indent=2)
            print(f"💾 Commands saved to {filename}")
            return True
        except Exception as e:
            print(f"❌ Save error: {e}")
            return False
    
    @staticmethod
    def print_command_summary(commands: List[Dict], tempo: float):
        """Print a summary of the generated commands."""
        if not commands:
            return
            
        print(f"\n📊 Robot Dance Summary:")
        print(f"Duration: {commands[-1]['timestamp']:.2f} seconds")
        print(f"Total movements: {len(commands)}")
        print(f"Tempo: {tempo:.2f} BPM")
        
        # Movement pattern distribution
        pattern_counts = {}
        total_energy = 0
        
        for cmd in commands:
            pattern = cmd['movement_pattern']
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            total_energy += cmd['energy_level']
        
        avg_energy = total_energy / len(commands)
        
        print(f"Average energy: {avg_energy:.2f}")
        print(f"\n🎭 Movement Pattern Distribution:")
        for pattern, count in sorted(pattern_counts.items()):
            percentage = (count / len(commands)) * 100
            print(f"  {pattern.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
        
        print(f"\n🎯 First 5 Movement Commands:")
        for i, cmd in enumerate(commands[:5]):
            print(f"  {i+1}. {cmd['timestamp']:.1f}s: {cmd['movement_pattern']} "
                  f"({cmd['movement_type']}) - Arm: {cmd['arm_positions']}")


class SynchronizedMovementExecutor:
    """Executes synchronized arm and base movements based on music analysis."""
    
    def __init__(self, robot_controller):
        self.robot_controller = robot_controller
        self.movement_queue = []
    
    def load_commands(self, commands: List[Dict]):
        """Load movement commands."""
        self.movement_queue = commands
        print(f"📥 Loaded {len(commands)} movement commands")
    
    def execute_synchronized_dance(self, start_delay: int = 3):
        """Execute synchronized dance with audio."""
        if not self.movement_queue:
            print("❌ No commands to execute")
            return
            
        print(f"🎭 Starting synchronized dance in {start_delay} seconds...")
        for i in range(start_delay, 0, -1):
            print(f"    {i}...")
            time.sleep(1)
        
        print("🎵 DANCE STARTED! 🎵")
        start_time = time.time()
        
        try:
            for command in self.movement_queue:
                target_time = start_time + command['timestamp']
                current_time = time.time()
                
                if current_time < target_time:
                    time.sleep(target_time - current_time)
                
                # Execute arm movement
                self.robot_controller.move_arm(
                    command['arm_positions'],
                    command['duration']
                )
                
                # Execute base movement
                if command['movement_type'] == 'forward':
                    self.robot_controller.safe_move_robot(
                        0.2,  # linear_speed
                        0.1,  # angular_speed
                        command['duration']
                    )
                else:  # return movement
                    self.robot_controller.safe_move_robot(
                        -0.2,  # linear_speed
                        -0.1,  # angular_speed
                        command['duration']
                    )
                
        except KeyboardInterrupt:
            print("\n⏹️  Dance interrupted by user")
        except Exception as e:
            print(f"\n❌ Dance error: {e}")
        finally:
            # Return to home position
            self.robot_controller.move_arm(self.robot_controller.home_position)
            self.robot_controller.safe_move_robot(0.0, 0.0, 0.1)
            print("🎉 Dance complete!")


class MusicDanceApplication:
    """Main application class that orchestrates the entire pipeline."""
    
    def __init__(self):
        self.lyrics_generator = LyricsGenerator()
        self.music_generator = MusicGenerator()
    
    def run(self):
        """Main application workflow."""
        print("🎵 AI Complete Song Generator 🎵")
        print("=" * 50)
        
        try:
            # Get user input
            user_prompt = self._get_user_input()
            
            # Generate lyrics and music
            audio_file = self._generate_music_content(user_prompt)
            if not audio_file:
                return
            
            # Analyze audio and generate movement commands
            commands = self._analyze_and_generate_commands(audio_file)
            if not commands:
                return
            
            # Execute dance performance
            self._execute_dance_performance(commands)
            
        except KeyboardInterrupt:
            print("\n👋 Program interrupted by user")
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    def _get_user_input(self) -> str:
        """Get and validate user input."""
        user_prompt = input("\nWhat should the song be about? ").strip()
        
        if not user_prompt:
            user_prompt = "love and happiness"
            print(f"Using default: {user_prompt}")
        
        print(f"\n🚀 Creating complete song about: '{user_prompt}'")
        return user_prompt
    
    def _generate_music_content(self, prompt: str) -> Optional[str]:
        """Generate lyrics and music."""
        print("\nStep 1: Writing structured lyrics...")
        lyrics = self.lyrics_generator.generate_structured_lyrics(prompt)
        
        print("\nStep 2: Creating song...")
        audio_file = self.music_generator.generate_music(prompt, lyrics)
        
        if not audio_file:
            print("❌ Failed to generate music")
            return None
        
        return audio_file
    
    def _analyze_and_generate_commands(self, audio_file: str) -> Optional[List[Dict]]:
        """Analyze audio and generate movement commands."""
        print("\n🎵 Jet Rover Music Dance Controller 🤖")
        print("=" * 50)
        
        print("🔍 Analyzing audio file...")
        analyzer = AudioAnalyzer(audio_file)
        command_generator = MovementCommandGenerator(analyzer)
        commands = command_generator.generate_commands()
        
        if not commands:
            print("❌ No commands generated. Check audio file.")
            return None
        
        # Save and display commands
        Utils.save_commands(commands)
        Utils.print_command_summary(commands, analyzer.tempo)
        
        return commands
    
    def _execute_dance_performance(self, commands: List[Dict]):
        """Set up and execute dance performance."""
        # Hardware connection setup
        rover = self._setup_rover_connection()
        executor = SynchronizedMovementExecutor(rover)
        executor.load_commands(commands)
        
        # Execute dance
        execute = input("\n🎵 Start the dance performance? (y/n): ").lower() == 'y'
        if execute:
            executor.execute_synchronized_dance()
    
    def _setup_rover_connection(self) -> JetRoverController:
        """Set up Jet Rover connection."""
        if SERIAL_AVAILABLE:
            use_hardware = input("\n🤖 Connect to Jet Rover hardware? (y/n): ").lower() == 'y'
            
            if use_hardware:
                port = input("Enter serial port (e.g., COM3, /dev/ttyUSB0): ").strip()
                return JetRoverController(port=port)
            else:
                return JetRoverController()
        else:
            print("\n🔄 PySerial not available - running in simulation mode")
            return JetRoverController()


def main():
    """Main entry point."""
    app = MusicDanceApplication()
    app.run()


if __name__ == "__main__":
    main()