
""" 

Este código implementa funciones para:
- Generar una señal
- Procesar un filtro pasa bajas
- Calcular densidad espectral de potencia
- Calcular la densidad espectral por bandas de frecuencia """


import numpy as np
import matplotlib.pyplot as plt

#Función para generar la señal simulada

def generate_signal(time,low_freq,high_freq,sampling_rate):
    """
    Esta función genera una señal compuesta por la suma de dos ondas senoidales.
    
    Args:
        time : Duración total de la señal en segundos
        low_freq : Frecuencia de la primera senoidal en Hz
        high_freq : Frecuencia de la segunda senoidal en Hz
        sampling_rate : Frecuencia de muestreo en Hz
    
    Returns:
        tuple: (t, signal) donde t es el vector de tiempo y signal es la señal generada
    """
    
    t =np.linspace(0,time,int(time * sampling_rate))
    # Señal compuesta: senoidal de amplitud 1 + senoidal de amplitud 0.5
    signal = np.sin(2 * np.pi * low_freq * t) + 0.5 * np.sin(2 * np.pi * high_freq * t)
    return t, signal

# Función para filtro pasa bajas

def low_pass_filter(signal,cutoff, sampling_rate):
    """
       Esta función aplica un filtro pasa bajas ideal a la señal ingresada.
        
        Args:
            signal : Señal de entrada
            cutoff : Frecuencia de corte del filtro en Hz
            sampling_rate : Frecuencia de muestreo en Hz
        
        Returns:
            Señal filtrada
    """       
    
    fft_signal = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(len(signal), 1 / sampling_rate)
    filtered_signal = np.fft.ifft(fft_signal * (np.abs(frequencies)<= cutoff)).real
    return filtered_signal

# Función para densidad espectral completa

def get_spectrum(signal, sampling_rate):
    
    """
    Esta función calcula la densidad espectral de potencia de la señal.
    
    Args:
        signal  : Señal de entrada
        sampling_rate : Frecuencia de muestreo en Hz
    
    Returns:
        Frecuencias positivas y densidad espectral de potencia
    """
    n = len(signal)
    fft_signal = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(n, 1/sampling_rate)
    
    # Calcular la densidad espectral de potencia
    psd = (np.abs(fft_signal) ** 2) / (n * sampling_rate)
    
    # Tomar solo las frecuencias positivas
    positive_frequencies = frequencies[:n//2]
    psd = psd[:n//2]
    # Ajustar potencia para compensar las frecuencias negativas eliminadas
    psd[1:] *= 2  
    
    return positive_frequencies, psd

# Función para la densidad espectral por bandas

def spectrum_bands(frequencies, psd, band_width=10, max_bands=10):
    """
    Esta función calcula la densidad espectral por bandas de frecuencia.
    
    Args:
        frequencies : Vector de frecuencias
        psd : Densidad espectral de potencia
        band_width : Ancho de cada banda en Hz
        max_bands : Número máximo de bandas a analizar
    
    Returns:
        tuple: (bands, powers) nombres de las bandas y potencia en cada banda
    """    
    max_freq = min(frequencies[-1], band_width * max_bands)
    bands = []
    powers = []
    
    for start_freq in np.arange(0, max_freq, band_width):
        end_freq = start_freq + band_width
        # Seleccionar frecuencias dentro de la banda actual
        mask = (frequencies >= start_freq) & (frequencies < end_freq)
        # Calcular la potencia total en la banda
        band_power = np.trapz(psd[mask], frequencies[mask]) if np.any(mask) else 0
        
        bands.append(f"{start_freq}-{end_freq}Hz")
        powers.append(band_power)
    
    return bands, powers

# Parámetros

time = 5 # Duración total en segundos
low_freq = 5  # Frecuencia baja 
high_freq = 50 # Frecuencia alta 
sampling_rate = 1000 # Frecuencia de muestreo
cutoff = 10 # Frecuencia de corte del filtro

# Generar y filtrar la señal
t, signal = generate_signal(time, low_freq, high_freq, sampling_rate)
filtered_signal = low_pass_filter(signal, cutoff, sampling_rate)

# Análisis densidad espectral completa
freq, psd = get_spectrum(signal, sampling_rate)

# Análisis densidad espectral por bandas
bands, powers = spectrum_bands(freq, psd)


# Primera figura: Señal original y filtrada
plt.figure(figsize=(12, 6))

# Señal original
plt.subplot(2, 1, 1)
plt.plot(t, signal, label="Señal Original")
plt.legend(loc="upper right")
plt.title("Filtro Pasa Bajas para la Señal")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")

# Señal filtrada
plt.subplot(2, 1, 2)
plt.plot(t, filtered_signal, label="Señal Filtrada", color="red")
plt.legend(loc="upper right")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")

plt.tight_layout()
plt.show()

# Segunda figura: Análisis espectral
plt.figure(figsize=(12, 6))

# Densidad espectral
plt.subplot(2, 1, 1)
plt.plot(freq, psd, label="Densidad Espectral de Potencia")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Potencia Espectral")
plt.title("Densidad Espectral de Potencia")
plt.grid(True)
plt.legend()

# Densidad espectral por bandas
plt.subplot(2, 1, 2)
plt.bar(bands, powers, color='lightblue')
plt.xlabel("Bandas de Frecuencia")
plt.ylabel("Potencia en la banda")
plt.title("Densidad Espectral por Bandas")
plt.grid(True)

plt.tight_layout()
plt.show()