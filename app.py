
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import streamlit as st
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parent.parent
AUDIO_DIR = PROJECT / "audio"
RESULTS_DIR = PROJECT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="Noise Reduction Lab", layout="wide")
st.title("Noise Cancellation and Voice Enhancement")
st.write("Analyze audio, suppress background noise, and compare results.")

files = sorted(AUDIO_DIR.glob("*.wav"))

if not files:
    st.error("Place WAV recordings inside the Noise_Project/audio folder.")
    st.stop()

selected = st.selectbox("Choose a WAV recording", files, format_func=lambda p: p.name)
audio, sr = sf.read(selected)

if audio.ndim == 2:
    audio = audio.mean(axis=1)

st.subheader("Original audio")
st.audio(str(selected))

duration = len(audio) / sr
rms = float(np.sqrt(np.mean(audio ** 2)))

c1, c2, c3 = st.columns(3)
c1.metric("Duration", f"{duration:.2f} seconds")
c2.metric("Sample rate", f"{sr:,} Hz")
c3.metric("RMS amplitude", f"{rms:.5f}")

st.subheader("Waveform")
time = np.arange(len(audio)) / sr
fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(time, audio, linewidth=0.5)
ax.set_xlabel("Time (seconds)")
ax.set_ylabel("Amplitude")
ax.grid(True)
st.pyplot(fig)
plt.close(fig)

st.subheader("Frequency spectrum")
spectrum = np.abs(np.fft.rfft(audio))
frequencies = np.fft.rfftfreq(len(audio), 1 / sr)
fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(frequencies, spectrum, linewidth=0.5)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Magnitude")
ax.set_xlim(0, sr / 2)
st.pyplot(fig)
plt.close(fig)

st.subheader("Spectrogram")
fig, ax = plt.subplots(figsize=(12, 4))
ax.specgram(audio, NFFT=1024, Fs=sr, noverlap=512)
ax.set_xlabel("Time (seconds)")
ax.set_ylabel("Frequency (Hz)")
st.pyplot(fig)
plt.close(fig)

st.subheader("Noise reduction")
st.write(
    "For this experiment, use the speech-plus-traffic mixture "
    "created by step2_mix_audio.py."
)

mixture_path = RESULTS_DIR / "speech_plus_traffic.wav"

if mixture_path.exists():
    mixture, mix_sr = sf.read(mixture_path)
    if mixture.ndim == 2:
        mixture = mixture.mean(axis=1)

    st.audio(str(mixture_path))
    strength = st.slider("Suppression strength", 1.0, 4.0, 2.0, 0.5)

    if st.button("Reduce noise"):
        if len(mixture) < int(0.5 * mix_sr):
            st.error("The recording is too short.")
        else:
            # Baseline spectral suppression. The initial segment
            # must contain mostly noise for this estimate to work.
            n_fft = 1024
            hop = 256
            D = librosa.stft(mixture, n_fft=n_fft, hop_length=hop)
            noise_frames = max(1, int(0.5 * mix_sr / hop))
            noise_profile = np.mean(
                np.abs(D[:, :noise_frames]), axis=1, keepdims=True
            )
            magnitude = np.abs(D)
            mask = np.clip(
                (magnitude - strength * noise_profile)
                / (magnitude + 1e-10),
                0.05, 1.0
            )
            enhanced = librosa.istft(
                D * mask, hop_length=hop, length=len(mixture)
            )
            out = RESULTS_DIR / "dashboard_enhanced.wav"
            sf.write(out, enhanced, mix_sr)

            st.success("Processing complete.")
            st.audio(str(out))

            fig, ax = plt.subplots(figsize=(12, 3))
            ax.plot(np.arange(len(mixture)) / mix_sr, mixture,
                    label="Noisy", linewidth=0.5)
            ax.plot(np.arange(len(enhanced)) / mix_sr, enhanced,
                    label="Enhanced", linewidth=0.5, alpha=0.8)
            ax.set_xlabel("Time (seconds)")
            ax.set_ylabel("Amplitude")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)
else:
    st.info(
        "Run step2_mix_audio.py first to create "
        "results/speech_plus_traffic.wav."
    )

# Directional Enhancement section
st.header("Directional Voice Enhancement")

st.write(
    "Simulated microphone-array beamforming "
    "to emphasize speech from a selected direction."
)

beam_pattern_path = RESULTS_DIR / "directional_beam_pattern.png"
directional_audio_path = RESULTS_DIR / "directional_enhanced.wav"

if beam_pattern_path.exists():
    st.subheader("Microphone Array Beam Pattern")
    st.image(
        str(beam_pattern_path),
        caption="Theoretical response at different sound frequencies"
    )
else:
    st.info(
        "Run step6_directional_enhancement.py "
        "to generate the beam pattern."
    )

if directional_audio_path.exists():
    st.subheader("Directionally Enhanced Audio")
    st.audio(str(directional_audio_path))

    st.download_button(
        "Download Directionally Enhanced Audio",
        data=directional_audio_path.read_bytes(),
        file_name="directional_enhanced.wav",
        mime="audio/wav"
    )
else:
    st.info("Run the directional enhancement script to generate audio.")

# Directional Enhancement Evaluation Summary
st.subheader("Directional Enhancement Evaluation")

st.write("**Simulation configuration**")
st.write("- Simulated microphones: 8")
st.write("- Target speech direction: 0°")
st.write("- Interference direction: 60°")
st.write("- Nominal input SNR: 5 dB")

st.warning(
    "Preliminary simulation result: the previous run reported "
    "+0.71 dB SNR improvement. This value has not yet been "
    "independently validated."
)

st.caption(
    "Software simulation only. Real-world directional performance "
    "requires testing with suitable multichannel microphone hardware."
)

# Directional Audio: Before and After Comparison
st.subheader("Before-and-After Audio Comparison")

noisy_audio_path = RESULTS_DIR / "speech_plus_traffic.wav"
directional_audio_path = RESULTS_DIR / "directional_enhanced.wav"

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Before: Speech + Traffic Noise**")
    if noisy_audio_path.exists():
        st.audio(str(noisy_audio_path))
    else:
        st.warning("Noisy input audio file not found.")

with col2:
    st.markdown("**After: Directional Enhancement**")
    if directional_audio_path.exists():
        st.audio(str(directional_audio_path))
    else:
        st.warning("Enhanced audio file not found.")

# Directional Enhancement Performance Results
st.subheader("Performance Results")

performance_data = {
    "Metric": [
        "Number of simulated microphones",
        "Target speech direction",
        "Interference direction",
        "Input SNR",
        "Output SNR",
        "SNR improvement",
    ],
    "Value": [
        "8",
        "0°",
        "60°",
        "5.00 dB",
        "5.71 dB",
        "+0.71 dB",
    ],
}

st.table(performance_data)

st.caption(
    "Preliminary simulation results from the previous run. "
    "Validate the measurements before making performance claims."
)
