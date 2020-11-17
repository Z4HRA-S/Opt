import numpy as np
import pandas as pd
import time


def gradient_descent(gradient_func, initial, iteration=10000, learning_rate=0.1, precision=0.001,
                     mute=False):
    print("Running gradient descent method")
    coef = np.array(initial)
    gradient = gradient_func(coef)
    iter = 0
    tik = time.time()
    history = [coef]
    for i in range(iteration):
        iter += 1
        gradient = np.array(gradient_func(coef))
        coef = coef - learning_rate * gradient
        history.append(coef)
        if np.max(np.abs(gradient)) < precision:
            break
    tok = time.time()
    if not mute:
        print("We reached gradient:", np.linalg.norm(gradient), "with learning_rate:", learning_rate, "in ", str(iter),
              "iteration",
              "and in",
              tok - tik, "seconds")
    return coef, history


def conjugate_gradient(gradient_func, initial, iteration=10000, learning_rate=0.1, precision=0.01,
                       mute=False):
    print("Running conjugate gradient method")
    coef = np.array(initial)
    prev = coef * 100
    gradient = gradient_func(coef)
    iter = 0
    tik = time.time()
    pk = -gradient
    history = [coef]
    for i in range(iteration):
        if np.max(np.abs(coef - prev)) < precision or np.max(np.abs(gradient)) < precision:
            break
        iter += 1
        prev = coef
        coef = coef + learning_rate * pk
        prev_gredient = gradient
        gradient = np.array(gradient_func(coef))
        pk = (np.linalg.norm(gradient) / np.linalg.norm(prev_gredient)) * pk - gradient
        history.append(coef)

    tok = time.time()
    if not mute:
        print("We reached gradient:", np.linalg.norm(gradient), "with learning_rate:", learning_rate, "in ", str(iter),
              "iteration",
              "and in",
              tok - tik, "seconds")
    return coef, history
