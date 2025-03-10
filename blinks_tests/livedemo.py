import numpy as np
import pyaudio
import time
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes

# Initialize BrainFlow
params = BrainFlowInputParams()
params.serial_port = 'COM5'  # Adjust for your device
board = BoardShim(BoardIds.CYTON_BOARD.value, params)
board.prepare_session()
board.start_stream()

# Audio settings
RATE = 44100  # Sample rate
DURATION = 1  # Duration for each glide (in seconds)
CHUNK = 1024  # Audio chunk size

# Initialize PyAudio
p = pyaudio.PyAudio()

note_frequencies = {
    'C': 261.63, 'C#': 277.18, 'D': 293.66, 'D#': 311.13, 'E': 329.63, 'F': 349.23,
    'F#': 369.99, 'G': 392.00, 'G#': 415.30, 'A': 440.00, 'A#': 466.16, 'B': 493.88
}
def closest_note(frequency):
    closest_note = min(note_frequencies, key=lambda note: abs(note_frequencies[note] - frequency))
    return note_frequencies[closest_note]

chords = {
    'C_major': ['C', 'E', 'G'],
    'D_minor': ['D', 'F', 'A'],
    'E_major': ['E', 'G#', 'B'],
    'F_major': ['F', 'A', 'C'],
    'G_major': ['G', 'B', 'D'],
    'A_minor': ['A', 'C', 'E']
}

# Function to generate sine wave for a frequency
def generate_sine_wave(frequencies, duration, rate):
    t = np.linspace(0, duration, int(rate * duration), endpoint=False)
    # Generate a sine wave for each frequency and add them together (chord)
    wave = np.zeros_like(t)
    for freq in frequencies:
        wave += np.sin(2 * np.pi * freq * t) * 0.3  # Adjust amplitude per note
    return wave.astype(np.float32)

def glide_frequencies(start_freqs, end_freqs, duration, rate):
    t = np.linspace(0, duration, int(rate * duration), endpoint=False)
    # Smooth glide: linearly interpolate between the start and end frequencies
    freqs = np.array([np.linspace(start, end, len(t)) for start, end in zip(start_freqs, end_freqs)])
    wave = np.zeros_like(t)
    for f in freqs:
        wave += np.sin(2 * np.pi * f * t) * 0.3  # Apply smooth glide for each frequency in the chord
    return wave.astype(np.float32)

def play_sound(stream, wave_data):
    stream.write(wave_data)

def get_combined_exg(data):
    exg_channels = board.get_exg_channels(BoardIds.CYTON_BOARD.value)
    combined_exg = np.sum(data[exg_channels, :], axis=0)
    return combined_exg

def normalize_frequency(exg_values, min_hz=110, max_hz=440):
    min_exg, max_exg = np.min(exg_values), np.max(exg_values)
    return min_hz + (exg_values - min_exg) / (max_exg - min_exg) * (max_hz - min_hz)

try:
    print("Streaming live EEG data to sound...")
    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=RATE, output=True)
    
    last_frequencies = [220]  # Start with a single frequency (can be a chord, too)
    
    while True:
        data = board.get_current_board_data(256)  # Get latest EEG data
        if data.shape[1] > 0:
            combined_exg = get_combined_exg(data)
            frequency = normalize_frequency(combined_exg)[-1]  # Take most recent value
            print(f"Gliding frequency to: {frequency:.2f} Hz")
            
            # Map the frequency to the closest musical note
            target_note = closest_note(frequency)
            print(f"Mapped to note frequency: {target_note:.2f} Hz")
            
            # Select a chord based on the frequency
            if frequency < 200:
                chord = chords['C_major']
            elif frequency < 300:
                chord = chords['D_minor']
            elif frequency < 400:
                chord = chords['E_major']
            elif frequency < 500:
                chord = chords['F_major']
            else:
                chord = chords['G_major']
            
            target_frequencies = [note_frequencies[note] for note in chord]
            
            # Smoothly glide from the last chord to the target chord
            wave_data = glide_frequencies(last_frequencies, target_frequencies, DURATION, RATE)
            
            # Play the gliding chord sound
            play_sound(stream, wave_data)
            
            # Update the last frequencies for the next iteration
            last_frequencies = target_frequencies

except KeyboardInterrupt:
    print("Interrupted by user")
finally:
    board.stop_stream()
    board.release_session()
    p.terminate()
