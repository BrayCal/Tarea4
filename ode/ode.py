#!/usr/bin/env python3
import numpy as np

def euler(f, y0, t0, tf, h):
    """Resuelve una ODE usando el método de Euler.

    Examples:
        >>> def f(t, y): return t + y
        >>> euler(f, 0, 0, 1, 0.1)
        [0, 0.1, 0.2, 0.3]

    Args:
        f (function): Función que describe la ODE (dy/dt = f(t, y))
        y0 (float): Valor inicial de y
        t0 (float): Tiempo inicial
        tf (float): Tiempo final
        h (float): Paso de tiempo

    Returns:
        list: Lista de valores de y en cada paso de tiempo.
    """
    t = t0
    y = y0
    ys = [y]
    while t < tf:
        y = y + h * f(t, y)
        t = t + h
        ys.append(y)
    return ys

def rk2(f, y0, t0, tf, h):
    """Resuelve una ODE usando el método de Runge-Kutta de segundo orden.

    Examples:
        >>> def f(t, y): return t + y
        >>> rk2(f, 0, 0, 1, 0.1)
        [0, 0.1, 0.2, 0.3]

    Args:
        f (function): Función que describe la ODE (dy/dt = f(t, y))
        y0 (float): Valor inicial de y
        t0 (float): Tiempo inicial
        tf (float): Tiempo final
        h (float): Paso de tiempo

    Returns:
        list: Lista de valores de y en cada paso de tiempo.
    """
    t = t0
    y = y0
    ys = [y]
    while t < tf:
        k1 = h * f(t, y)
        k2 = h * f(t + 0.5 * h, y + 0.5 * k1)
        y = y + k2
        t = t + h
        ys.append(y)
    return ys

def rk4(f, y0, t0, tf, h):
    """Resuelve una ODE usando el método de Runge-Kutta de cuarto orden.

    Examples:
        >>> def f(t, y): return t + y
        >>> rk4(f, 0, 0, 1, 0.1)
        [0, 0.1, 0.2, 0.3]

    Args:
        f (function): Función que describe la ODE (dy/dt = f(t, y))
        y0 (float): Valor inicial de y
        t0 (float): Tiempo inicial
        tf (float): Tiempo final
        h (float): Paso de tiempo

    Returns:
        list: Lista de valores de y en cada paso de tiempo.
    """
    t = t0
    y = y0
    ys = [y]
    while t < tf:
        k1 = h * f(t, y)
        k2 = h * f(t + 0.5 * h, y + 0.5 * k1)
        k3 = h * f(t + 0.5 * h, y + 0.5 * k2)
        k4 = h * f(t + h, y + k3)
        y = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        t = t + h
        ys.append(y)
    return ys
