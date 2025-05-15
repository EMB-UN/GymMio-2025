#!/usr/bin/env python3
"""Plot the live microphone signal(s) with matplotlib.

Matplotlib and NumPy have to be installed.

"""
import argparse
import queue
import sys
import time
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from scipy.signal import find_peaks

def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    '-l', '--list-devices', action='store_true',
    help='show list of audio devices and exit')
args, remaining = parser.parse_known_args()
if args.list_devices:
    print(sd.query_devices())
    parser.exit(0)
parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    parents=[parser])
parser.add_argument(
    'channels', type=int, default=[1], nargs='*', metavar='CHANNEL',
    help='input channels to plot (default: the first)')
parser.add_argument(
    '-d', '--device', type=int_or_str,
    help='input device (numeric ID or substring)')
parser.add_argument(
    '-w', '--window', type=float, default=200, metavar='DURATION',
    help='visible time slot (default: %(default)s ms)')
parser.add_argument(
    '-i', '--interval', type=float, default=30,
    help='minimum time between plot updates (default: %(default)s ms)')
parser.add_argument(
    '-b', '--blocksize', type=int, help='block size (in samples)')
parser.add_argument(
    '-r', '--samplerate', type=float, help='sampling rate of audio device')
parser.add_argument(
    '-n', '--downsample', type=int, default=10, metavar='N',
    help='display every Nth sample (default: %(default)s)')
args = parser.parse_args(remaining)
if any(c < 1 for c in args.channels):
    parser.error('argument CHANNEL: must be >= 1')
mapping = [c - 1 for c in args.channels]  # Channel numbers start with 1
q = queue.Queue()


def audio_callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    # Fancy indexing with mapping creates a (necessary!) copy:
    q.put(indata[::args.downsample, mapping])

def imprimir_frecuencias_detectadas(signal, samplerate):
    """Imprime una lista de frecuencias detectadas de la más grave a la más aguda."""
    fft_data = np.fft.rfft(signal)
    magnitude = np.abs(fft_data)
    freqs = np.fft.rfftfreq(len(signal), d=1.0/samplerate)
    # Encuentra los picos en la magnitud
    peaks, _ = find_peaks(magnitude, height=np.max(magnitude)*0.1)  # Ajusta el umbral si es necesario
    frecuencias_detectadas = freqs[peaks]
    frecuencias_detectadas = np.sort(frecuencias_detectadas)
    print("Frecuencias detectadas (Hz):")
    time.sleep(0.3)
    print(frecuencias_detectadas)

def update_plot(frame):
    """Actualiza la gráfica con el espectro de frecuencias (FFT)."""
    global plotdata
    while True:
        try:
            data = q.get_nowait()
        except queue.Empty:
            break
        shift = len(data)
        plotdata = np.roll(plotdata, -shift, axis=0)
        plotdata[-shift:, :] = data

    # Tomamos solo el primer canal para la FFT (puedes cambiar esto)
    signal = plotdata[:, 0]
    # Calcula la FFT
    fft_data = np.fft.rfft(signal)
    magnitude = np.abs(fft_data)
    freqs = np.fft.rfftfreq(len(signal), d=1.0/args.samplerate)

    # Actualiza la gráfica con la magnitud de la FFT
    lines[0].set_data(freqs, magnitude)
    ax.set_xlim(0, args.samplerate / 2)
    ax.set_ylim(0, np.max(magnitude) * 1.1 if np.max(magnitude) > 0 else 1)
    signal = plotdata[:, 0]
    # Llama a la función para imprimir las frecuencias detectadas
    imprimir_frecuencias_detectadas(signal, args.samplerate)
    return lines

try:
    #print("Samplerate:", args.samplerate)
    if args.samplerate is None:
        device_info = sd.query_devices(args.device, 'input')
        args.samplerate = device_info['default_samplerate']
    length = int(args.window * args.samplerate / (1000 * args.downsample))
    plotdata = np.zeros((length, len(args.channels)))

    fig, ax = plt.subplots()
    # Inicializa la línea para la FFT
    freqs = np.fft.rfftfreq(length, d=1.0/args.samplerate)
    lines = ax.plot(freqs, np.zeros_like(freqs))
    ax.set_xlabel('Frecuencia (Hz)')
    ax.set_ylabel('Magnitud')
    ax.set_title('Espectro de Frecuencia (FFT)')
    fig.tight_layout(pad=0.5)

    stream = sd.InputStream(
        device=args.device, channels=max(args.channels),
        samplerate=args.samplerate, callback=audio_callback)
    ani = FuncAnimation(fig, update_plot, interval=args.interval, blit=True)
    with stream:
        plt.show()
except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))