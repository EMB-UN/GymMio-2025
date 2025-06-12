import serial
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from collections import deque

# === CONFIGURACIÓN SERIAL ===
puerto = 'COM7'           # Cambiar si es necesario
baudrate = 115200
ser = serial.Serial(puerto, baudrate, timeout=0.05)

# === PARÁMETROS ===
fs = 2000                 # Frecuencia de muestreo (Hz)
N = 1024                  # Tamaño de la ventana FFT
buffer = deque([0]*N, maxlen=N)

frequencies = fftfreq(N, 1/fs)[:N//2]  # Eje X para la FFT
x_time = np.linspace(0, N/fs, N)       # Eje X para señal en el tiempo

# === CONFIGURAR GRÁFICA ===
plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))

# Señal en el tiempo
line_time, = ax1.plot(x_time, np.zeros(N))
ax1.set_ylim(0, 4095)
ax1.set_xlim(0, N/fs)
ax1.set_ylabel("Valor ADC")
ax1.set_title("Señal en el tiempo (desde ESP32)")


# FFT
line_fft, = ax2.plot(frequencies, np.zeros(N//2))
marker, = ax2.plot([0], [0], 'ro')  # <-- Cambia 0,0 por [0],[0]

ax2.set_xlim(0, fs/2)
ax2.set_ylim(0, 1000)
ax2.set_xlabel("Frecuencia (Hz)")
ax2.set_ylabel("Magnitud")
ax2.set_title("FFT en vivo con frecuencia dominante")

# === BUCLE PRINCIPAL ===
try:
    while True:
        while ser.in_waiting:
            dato = ser.readline().decode().strip()
            if dato.isdigit():
                buffer.append(int(dato))

        if len(buffer) == N:
            datos = np.array(buffer)

            # Señal en el tiempo
            line_time.set_ydata(datos)

            # FFT
            datos_dc = datos - np.mean(datos)
            fft_vals = fft(datos_dc)
            magnitudes = np.abs(fft_vals[:N//2])

            line_fft.set_ydata(magnitudes)

            # Frecuencia dominante
            idx_max = np.argmax(magnitudes)
            f_dom = frequencies[idx_max]
            mag_dom = magnitudes[idx_max]
            marker.set_data([f_dom], [mag_dom])  # <-- Usa listas aquí
            ax2.set_ylim(0, mag_dom * 1.2)

            # Mostrar el nombre del punto (frecuencia dominante)
            # Elimina anotaciones anteriores antes de crear una nueva
            if hasattr(ax2, 'text_fdom'):
                ax2.text_fdom.remove()
            ax2.text_fdom = ax2.annotate(
                f"{f_dom:.1f} Hz",
                xy=(f_dom, mag_dom),
                xytext=(10, 10),
                textcoords='offset points',
                color='red',
                fontsize=10,
                weight='bold',
                arrowprops=dict(arrowstyle="->", color='red')
            )

            plt.pause(0.001)

except KeyboardInterrupt:
    print("Interrumpido por el usuario.")
    ser.close()