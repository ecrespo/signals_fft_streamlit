<<<<<<< HEAD
import matplotlib.pylab as plt
import numpy as np
import scipy.signal as signal
import scipy.stats as stats
import streamlit as st
from scipy import signal as signal


def u(amplitud, t):
    """Función escalón unitario

    Args:
        amplitud (int): Amplitud del escalon
        t (list): Lista de tiempo

    Returns:
        list: Lista de valores
=======
import matplotlib.pylab as plt  # type: ignore
import numpy as np  # type: ignore
import scipy.signal as signal  # type: ignore
import streamlit as st  # type: ignore
from scipy import signal as sp  # type: ignore


def u(amplitud: int, t: np.ndarray) -> np.ndarray:
    """ Función escalón unitario

    Args:
        amplitud (int): Amplitud del escalon
        t (np.ndarray): Lista de tiempo

    Returns:
        np.ndarray: Lista de valores
>>>>>>> feat/validado
    """
    return amplitud * np.piecewise(t, [t < 0.0, t >= 0.0], [0, 1])


<<<<<<< HEAD
def OndaCuadrada(amplitud, t, fs=1):
    """Función  de Onda Cuadrada
=======
def onda_cuadrada(amplitud: int, t: np.ndarray, fs: int = 1) -> np.ndarray:
    """ Función  de Onda Cuadrada
>>>>>>> feat/validado

    Args:
        amplitud (int): Amplitud de la segnal
        t (list): Lista de valores de tiempo
        fs (int, optional): Frecuencia.. Defaults to 1.

    Returns:
        list: Lista de valores
    """
    return ((sp.square(2 * fs * t)) * (amplitud / 2.0)) + (amplitud / 2.0)


<<<<<<< HEAD
def segnal_triangular(amplitud, simetria, t, fs=1):
    """Señal triangular

    Args:
        amplitud (int): Amplitud de la señal
        simetria (float): simetria de la señal
        t (list): Lista de valores que definen el tiempo.
        fs (int, optional): Frecuencia de la señal. Defaults to 1.

    Returns:
        list: Lista de valores de la señal
=======
def segnal_triangular(amplitud: int, simetria: float, t: np.ndarray, fs: int = 1) -> np.ndarray:
    """Señal triangular
    Args:
        amplitud (int): Amplitud de la señal
        simetria (float): simetria de la señal
        t (np.ndarray): Lista de valores que definen el tiempo.
        fs (int, optional): Frecuencia de la señal. Defaults to 1.

    Returns:
        np.ndarray: Lista de valores de la señal
>>>>>>> feat/validado
    """
    return amplitud * (signal.sawtooth(2 * np.pi * fs * t, simetria))


<<<<<<< HEAD
def seno(amplitud, t, fs=1):
=======
def seno(amplitud: int, t: np.ndarray, fs: int = 1) -> np.ndarray:
>>>>>>> feat/validado
    """Onda Seno

    Args:
        amplitud (int): Amplitud de la señal
<<<<<<< HEAD
        t (list): Lista de valores de tiempo
        fs (int, optional): Frecuencia de la señal. Defaults to 1.

    Returns:
        list: Lista de valores de la señal de seno
=======
        t (np.ndarray): Lista de valores de tiempo
        fs (int, optional): Frecuencia de la señal. Defaults to 1.

    Returns:
        np.ndarray: Lista de valores de la señal de seno
>>>>>>> feat/validado
    """
    return amplitud * np.sin(fs * t)


<<<<<<< HEAD
def coseno(amplitud, t, fs=1):
=======
def coseno(amplitud: int, t: np.ndarray, fs: int = 1) -> np.ndarray:
>>>>>>> feat/validado
    """Señal de coseno

    Args:
        amplitud (int): Amplitud de la señal
<<<<<<< HEAD
        t (list): lista de valores para generar la señal
        fs (int, optional): Frecuencia de la señal. Defaults to 1.

    Returns:
        list: Lista de valores de la señal
=======
        t (np.ndarray): lista de valores para generar la señal
        fs (int, optional): Frecuencia de la señal. Defaults to 1.

    Returns:
        np.ndarray: Lista de valores de la señal
>>>>>>> feat/validado
    """
    return amplitud * np.cos(fs * t)


<<<<<<< HEAD
def tiempo(lim_inf, lim_sup, n):
    """Lista de valores que definen el tiempo de la señal

=======
def tiempo(lim_inf: int, lim_sup: int, n: int) -> np.ndarray:
    """ Lista de valores que definen el tiempo de la señal
>>>>>>> feat/validado
    Args:
        lim_inf (int): Límite inferior del tiempo
        lim_sup (int): Límite superior del tiempo
        n (int): Cantidad de valores a generar del tiempo

    Returns:
<<<<<<< HEAD
        list: Lista de valores del tiemmpo
    """
    return np.linspace(lim_inf, lim_sup, n)


def plot_signal(xi, xf, yi, yf, t, titulo, etiqueta, values):
    """Generación de la gráfica de la señal.

    Args:
        xi (int): x inicial
        xf (int): x final
        yi (int): y inicial
        yf (int): y final
        t (list): lista de valores de tiempo
        titulo (str): Título de la gráfica
        etiqueta (str): Etiqueta de la señal.
        values (list): Valores de la señal
    """
    plot(t, values, "k", label=etiqueta, lw=2)
    xlim(xi, xf)
=======
        np.ndarray: Lista de valores del tiemmpo
    """

    return np.linspace(lim_inf, lim_sup, n)
>>>>>>> feat/validado


def main():
    # Definir título
    st.title("Generación de gráficas de señales")
    st.sidebar.header("Entradas:")
    segnales = [
        "Escalon Unitario",
        "Onda Cuadrada",
        "Onda triangular",
        "Seno",
        "Coseno",
    ]

    resp = st.sidebar.selectbox("Tipo de señal", segnales)

    st.sidebar.header("Definición del tiempo:")
    st.sidebar.subheader("Rango")
    # SelectBox
    t0 = int(st.sidebar.selectbox("", range(0, 10)))
    ti = 0
    tf = t0
    n = 10000
    t = tiempo(ti, tf, n)
    st.sidebar.subheader("Amplitud de la señal")
    amplitud = int(st.sidebar.selectbox("", range(1, 10)))

    # numpy.ndarray
    if resp == "Escalon Unitario":
        ti = -tf
        resultado = u(amplitud, t)
    elif resp == "Onda Cuadrada":
        st.sidebar.subheader("Frecuencia de la señal")
        fs = int(st.sidebar.selectbox("", range(1, 11)))
<<<<<<< HEAD
        resultado = OndaCuadrada(amplitud, t, fs)
=======
        resultado = onda_cuadrada(amplitud, t, fs)
>>>>>>> feat/validado
    elif resp == "Onda triangular":
        simetria = 0.5
        st.sidebar.subheader("Frecuencia de la señal")
        fs = int(st.sidebar.selectbox("", range(1, 11)))
        resultado = segnal_triangular(amplitud, simetria, t, fs)
    elif resp == "Seno":
        st.sidebar.subheader("Frecuencia de la señal")
        fs = int(st.sidebar.selectbox("", range(1, 11)))
        resultado = seno(amplitud, t, fs)
    elif resp == "Coseno":
        st.sidebar.subheader("Frecuencia de la señal")
        fs = int(st.sidebar.selectbox("", range(1, 11)))
        resultado = coseno(amplitud, t, fs)
    else:
        resultado = 0
        st.error("Error")

    fig = plt.figure()
    ax = fig.add_subplot(111)
    st.header(f"Gráfica de {resp}")
    ax.plot(t, resultado)
    ax.set_xlim(ti, tf)
    ax.set_xlabel("Tiempo")
    ax.set_ylabel("f(t)")
    ax.set_ylim(2 * amplitud * -1, 2 * amplitud)
    st.pyplot(fig)


if __name__ == "__main__":
    main()
