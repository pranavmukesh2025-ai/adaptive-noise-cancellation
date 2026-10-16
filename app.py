
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import streamlit as st
import matplotlib.pyplot as plt

# ---------- Project paths ----------
APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent

# Support the repository-root audio folder
AUDIO_DIR = PROJECT_DIR / "audio"
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="Noise Reduction Lab",
    page_icon="🔊",
    layout="wide",
)

st.title("Noise Cancellation and Voice Enhancement")
st.write(
    "Analyze audio, suppress background noise, "
    "and explore simulated directional enhancement."
)

# ---------- Load available audio ----------
audio_files = sorted(AUDIO_DIR.glob("*.wav")) if AUDIO_DIR.exists() else []

st.subheader("Audio Analysis")

uploaded_file = st.file_uploader(
    "Or upload your own WAV audio",
    type=["wav"],
)

selected_audio = None

if uploaded_file is not None:
    audio, sr = sf.read(uploaded_file)
    audio_name = uploaded_file.name
elif audio_files:
    selected_audio = st.selectbox(
        "Choose a WAV recording",
        audio_files,
        format_func=lambda p: p.name,
    )
    audio, sr = sf.read(selected_audio)
    audio_name = selected_audio.name
else:
    st.warning(
        "No repository WAV files were found. "
        "Upload a WAV file above to use audio analysis."
    )
    audio = None
    sr = None
    audio_name = None

if audio is not None:
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    audio = np.asarray(audio, dtype=np.float32)
    st.caption(f"Selected recording: {audio_name}")
    st.audio(audio, sample_rate=sr)

    duration = len(audio) / sr
    rms = float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0.0

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

    st.subheader("Frequency Spectrum")
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

# ---------- Noise reduction ----------
st.header("Noise Reduction")

mixture_path = RESULTS_DIR / "speech_plus_traffic.wav"

if mixture_path.exists():
    mixture, mix_sr = sf.read(mixture_path)
    if mixture.ndim == 2:
        mixture = mixture.mean(axis=1)

    st.write("Speech mixed with traffic noise")
    st.audio(mixture, sample_rate=mix_sr)

    strength = st.slider(
        "Suppression strength",
        min_value=1.0,
        max_value=4.0,
        value=2.0,
        step=0.5,
    )

    if st.button("Reduce Noise"):
        if len(mixture) < int(0.5 * mix_sr):
            st.error("The recording is too short.")
        else:
            n_fft = 1024
            hop = 256
            D = librosa.stft(
                mixture, n_fft=n_fft, hop_length=hop
            )
            noise_frames = max(
                1, int(0.5 * mix_sr / hop)
            )
            noise_profile = np.mean(
                np.abs(D[:, :noise_frames]),
                axis=1,
                keepdims=True,
            )
            magnitude = np.abs(D)
            mask = np.clip(
                (magnitude - strength * noise_profile)
                / (magnitude + 1e-10),
                0.05,
                1.0,
            )
            enhanced = librosa.istft(
                D * mask,
                hop_length=hop,
                length=len(mixture),
            )

            output_path = RESULTS_DIR / "dashboard_enhanced.wav"
            sf.write(output_path, enhanced, mix_sr)

            st.success("Noise reduction complete.")
            st.audio(enhanced, sample_rate=mix_sr)

            st.download_button(
                "Download Noise-Reduced Audio",
                data=output_path.read_bytes(),
                file_name="dashboard_enhanced.wav",
                mime="audio/wav",
            )
else:
    st.info(
        "The saved speech-plus-traffic mixture is not available. "
        "Upload a WAV recording above for analysis, or add "
        "results/speech_plus_traffic.wav to the repository "
        "to enable this noise-reduction section."
    )

# ---------- Directional enhancement ----------
st.header("Directional Voice Enhancement")

st.write(
    "This section displays saved results from the simulated "
    "microphone-array beamforming experiment."
)

beam_pattern_path = RESULTS_DIR / "directional_beam_pattern.png"
directional_audio_path = RESULTS_DIR / "directional_enhanced.wav"

if beam_pattern_path.exists():
    st.subheader("Theoretical Microphone Array Beam Pattern")
    st.image(
        str(beam_pattern_path),
        caption="Simulated response at different sound frequencies",
    )
else:
    st.info(
        "Beam-pattern image not found. Add "
        "results/directional_beam_pattern.png to GitHub "
        "to display it."
    )

if directional_audio_path.exists():
    st.subheader("Directionally Enhanced Audio")
    st.audio(str(directional_audio_path))

    st.download_button(
        "Download Directionally Enhanced Audio",
        data=directional_audio_path.read_bytes(),
        file_name="directional_enhanced.wav",
        mime="audio/wav",
    )
else:
    st.info(
        "Directional output not found. Add "
        "results/directional_enhanced.wav to GitHub "
        "to play the saved simulation."
    )

# ---------- Evaluation summary ----------
st.subheader("Directional Enhancement Evaluation")

st.write("**Previous simulation configuration**")
st.write("- Simulated microphones: 8")
st.write("- Target speech direction: 0°")
st.write("- Interference direction: 60°")
st.write("- Nominal input SNR: 5 dB")

st.warning(
    "The previous run reported +0.71 dB SNR improvement. "
    "This is a preliminary simulation result and has not "
    "been independently validated."
)

st.caption(
    "Software simulation only. Real-world directional performance "
    "requires suitable multichannel microphone hardware."
)

# ---------- Before-and-after comparison ----------
st.subheader("Before-and-After Audio Comparison")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Before: Speech + Traffic Noise**")
    if mixture_path.exists():
        st.audio(str(mixture_path))
    else:
        st.info("Saved noisy mixture is not available.")

with col2:
    st.markdown("**After: Directional Enhancement**")
    if directional_audio_path.exists():
        st.audio(str(directional_audio_path))
    else:
        st.info("Saved directional output is not available.")

# ---------- Performance table ----------
st.subheader("Preliminary Performance Results")

st.table(
    {
        "Metric": [
            "Simulated microphones",
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
)

st.caption(
    "Performance values are from a previous simulation run, "
    "not recalculated by this dashboard."
)
