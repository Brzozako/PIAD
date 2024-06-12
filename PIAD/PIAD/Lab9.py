import sounddevice as sd
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
import librosa

audioFile = 'PiadStudent.wav'
s, fs = sf.read(audioFile, dtype='float32')

print(f'Czas trwania: {len(s) / fs} s')
print(f'Częstotliwość próbkowania: {fs} Hz')
print(f'Rozdzielczość bitowa: {s.dtype}')
print(f'Liczba kanałów: {s.shape[1] if s.ndim > 1 else 1}')

sd.play(s, fs)
status = sd.wait()

if np.max(np.abs(s)) > 1:
    s = s / np.max(np.abs(s))

time = np.arange(s.shape[0]) / fs
plt.plot(time * 1000, s)
plt.xlabel('Czas [ms]')
plt.ylabel('Amplituda')
plt.title('Sygnał audio')
plt.show()

window_length_ms = 10
window_length_samples = int(window_length_ms * fs / 1000)

def frames(signal, frame_length, overlap=0):
    step = frame_length - overlap
    shape = ((signal.size - overlap) // step, frame_length)
    strides = (signal.strides[0] * step, signal.strides[0])
    return np.lib.stride_tricks.as_strided(signal, shape=shape, strides=strides)

frames = frames(s, window_length_samples)

E = np.sum(frames**2, axis=1)
Z = np.sum(np.diff(np.sign(frames), axis=1) != 0, axis=1)

E_normalized = E / np.max(E)
Z_normalized = Z / np.max(Z)

time_frames = np.arange(len(E)) * window_length_ms
plt.plot(time_frames, E_normalized, label='Energia (E)', color='red')
plt.plot(time_frames, Z_normalized, label='Przejścia przez zero (Z)', color='blue')
plt.xlabel('Czas [ms]')
plt.ylabel('Wartość znormalizowana')
plt.legend()
plt.show()

fragment_start = int(1.0 * fs)
fragment_end = fragment_start + 2048
fragment = s[fragment_start:fragment_end]

window = np.hamming(len(fragment))
masked_fragment = fragment * window

yf = scipy.fftpack.fft(masked_fragment)
xf = np.fft.fftfreq(len(yf), 1 / fs)

plt.plot(xf[:len(yf)//2], np.log(np.abs(yf[:len(yf)//2])))
plt.xlabel('Częstotliwość [Hz]')
plt.ylabel('Logarytm amplitudy')
plt.title('Logarytmiczne widmo amplitudowe')
plt.show()

p = 20
a = librosa.lpc(fragment,order=p)
a = np.pad(a, (0, len(fragment) - len(a)), 'constant')
lpc_spectrum = np.log(np.abs(np.fft.fft(a)))

plt.plot(xf[:len(yf)//2], np.log(np.abs(yf[:len(yf)//2])), label='Widmo amplitudowe')
plt.plot(xf[:len(lpc_spectrum)//2], -lpc_spectrum[:len(lpc_spectrum)//2], label='Widmo LPC')
plt.xlabel('Częstotliwość [Hz]')
plt.ylabel('Logarytm amplitudy')
plt.legend()
plt.show()

formants = np.sort(np.abs(np.fft.fftfreq(len(lpc_spectrum), 1 / fs)[np.argsort(-lpc_spectrum)[:2]]))
print(f'Formanty F1: {formants[0]} Hz, F2: {formants[1]} Hz')
