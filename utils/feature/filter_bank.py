# Reference: https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
import numpy as np


def compute_filter_bank(signal, sample_rate, frame_size=0.025, frame_stride=0.01, NFFT=512,
                        n_filt=40, gain_ctrl=0, gradient=1, MFCC=0):
    PRINT_INFO = 0

    # Initialization
    np.random.seed(2)

    sample_rate = float(sample_rate)

    # Framing
    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
    signal_length = len(signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(
        np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(signal,
                           z)  # Pad Signal to make sure that all frames have equal number of samples without
    # truncating any samples from the original signal

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(
        np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T

    frames = pad_signal[indices.astype(np.int32, copy=False)]

    # Real-Time Normalization (Volumn Control)
    processed_frames = frames

    # Windowing
    processed_frames *= np.hamming(frame_length)

    # Fourier-Transform and Power Spectrum
    mag_frames = np.absolute(np.fft.rfft(processed_frames, NFFT))  # Magnitude of the FFT
    pow_frames = ((1.0 / NFFT) * (mag_frames ** 2))  # Power Spectrum

    # Filter Banks
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, n_filt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10 ** (mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = np.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = np.zeros((n_filt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, n_filt + 1):
        f_m_minus = int(bin[m - 1])  # left
        f_m = int(bin[m])  # center
        f_m_plus = int(bin[m + 1])  # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

    energy = np.sum(pow_frames, axis=-1, keepdims=True)
    features = np.dot(pow_frames, fbank.T)
    features = np.where(features == 0, np.finfo(float).eps, features)  # Numerical Stability
    features = 20 * np.log(features)  # dB
    features = np.hstack((features, energy))
    if gradient:
        features_grad_first_order = np.gradient(features, edge_order=2, axis=0)
        features_grad_second_order = np.gradient(features_grad_first_order, edge_order=2, axis=0)
        features = np.hstack((features, features_grad_first_order, features_grad_second_order))

    return features
