import numpy as np


def normalize(A):
    return A / np.linalg.norm(A)


def newton_method(gradient, hessian, initial, precision=0.01, iteration=100):
    print("Running newton Method:")
    coef = np.array(initial)
    iter = 0
    history = [coef]
    print("=" * 20)
    print("coef: ", coef)
    print("gradient: ", gradient(coef))

    for i in range(iteration):
        iter += 1
        prev = coef
        coef = prev - np.matmul(np.linalg.inv(hessian(prev)), gradient(prev))
        print("=" * 20)
        print("coef: ", coef)
        print("gradient: ", gradient(coef))
        history.append(coef)
        if np.max(np.abs(prev - coef)) < precision:
            break
    print("We reached gradient:", gradient(coef), "in ", str(iter), "iteration")
    return coef, history


def newton_gradient(gradient, hessian, initial, precision=0.01, iteration=100, learning_rate=0.1):
    print("Running newton Method with the help of gradient descent:")
    coef = np.array(initial)
    iter = 0
    history = [coef]
    print("=" * 20)
    print("coef: ", coef)
    print("gradient: ", gradient(coef))

    for i in range(iteration):
        iter += 1
        prev = coef
        coef = prev - np.matmul(np.linalg.inv(hessian(prev)), gradient(prev))

        if np.min(np.linalg.eigvals(hessian(coef))) < 0:
            coef = prev - learning_rate * np.array(gradient(prev))
            print("using gradient descent")

        print("=" * 20)
        print("coef: ", coef)
        print("gradient: ", gradient(coef))
        history.append(coef)
        if np.max(np.abs(prev - coef)) < precision:
            break
    print("We reached gradient:", gradient(coef), "in ", str(iter), "iteration")
    return coef, history


def modified_newton(gradient, hessian, initial, precision=0.01, iteration=100):
    print("Running Modified Newton Method")
    coef = np.array(initial)
    iter = 0
    tau = 1
    history = [coef]
    print("=" * 20)
    print("coef: ", coef)
    print("gradient: ", gradient(coef))

    for i in range(iteration):
        iter += 1
        prev = coef
        hess_matrix = hessian(prev)
        min_eig = np.min(np.linalg.eigvals(hess_matrix))
        if min_eig < 0:
            hess_matrix = hess_matrix + (tau - min_eig) * np.identity(hess_matrix.shape[0], dtype='float64')
            print("Modified Hessian")

        coef = prev - np.matmul(np.linalg.inv(hess_matrix), gradient(prev))
        print("=" * 20)
        print("coef: ", coef)
        print("gradient: ", gradient(coef))
        history.append(coef)
        if np.max(np.abs(prev - coef)) < precision:
            break
    print("We reached gradient:", gradient(coef), "in ", str(iter), "iteration")
    return coef, history


def SR1(gradient, initial, precision=0.001, iteration=100, learning_rate=1, beta=1):
    print("Running Symmetric Rank-1 Method")
    coef = np.array(initial)
    iter = 0
    inv_B = beta * np.identity(coef.shape[0])
    # inv_B = np.linalg.inv([[20, 0], [0, 2]])
    history = [coef]
    print("=" * 20)
    print("coef: ", coef)
    print("gradient: ", gradient(coef))

    for i in range(iteration):
        iter += 1
        coef = history[-1] - learning_rate * np.matmul(inv_B, gradient(history[-1]))
        s = coef - history[-1]
        y = gradient(coef) - gradient(history[-1])
        u = s - np.matmul(inv_B, y)
        update = np.matmul(u, u.T) / (np.matmul(u.T, y))
        if not np.isnan(update):
            inv_B = inv_B + update

        #print("=" * 20)
        #print("coef: ", coef)
        #print("gradient: ", gradient(coef))
        history.append(coef)
        if np.max(np.abs(gradient(coef))) < precision or np.all(s) == 0:
            break
    print("We reached gradient:", gradient(coef), "in ", str(iter), "iteration")
    return coef, history


def BFGS(gradient, initial, precision=0.01, iteration=100, learning_rate=1, beta=1):
    print("Running BFGS Method")
    coef = np.array(initial)
    iter = 0
    B = beta * np.identity(coef.shape[0])
    # B = np.array([[20, 0], [0, 2]])
    history = [coef]
    print("=" * 20)
    print("coef: ", coef)
    print("gradient: ", gradient(coef))

    for i in range(iteration):
        iter += 1
        coef = history[-1] - learning_rate * np.matmul(np.linalg.inv(B), gradient(history[-1]))
        s = coef - history[-1]
        y = gradient(coef) - gradient(history[-1])
        update = -1 * (np.matmul(np.matmul(B, s), np.matmul(s.T, B)) / (np.matmul(np.matmul(s.T, B), s))) + (
                np.matmul(y, y.T) / np.matmul(y.T, s))
        if not np.isnan(update):
            B = B + update

        #print("=" * 20)
        #print("coef: ", coef)
        #print("gradient: ", gradient(coef))

        history.append(coef)
        if np.max(np.abs(gradient(coef))) < precision or np.all(s) == 0:
            break
    print("We reached gradient:", gradient(coef), "in ", str(iter), "iteration")
    return coef, history


def DFP(gradient, initial, precision=0.01, iteration=100, learning_rate=1, beta=1):
    print("Running DFP Method")
    coef = np.array(initial)
    iter = 0
    inv_B = beta * np.identity(coef.shape[0])
    # B = np.array([[20, 0], [0, 2]])
    history = [coef]
    print("=" * 20)
    print("coef: ", coef)
    print("gradient: ", gradient(coef))

    for i in range(iteration):
        iter += 1
        coef = history[-1] - learning_rate * np.matmul(inv_B, gradient(history[-1]))
        s = coef - history[-1]
        y = gradient(coef) - gradient(history[-1])
        update = -1 * (
                np.matmul(np.matmul(inv_B, y), np.matmul(y.T, inv_B)) / (np.matmul(np.matmul(y.T, inv_B), y))) + (
                         np.matmul(s, s.T) / np.matmul(y.T, s))
        if not np.isnan(update):
            inv_B = inv_B + update

        print("=" * 20)
        print("coef: ", coef)
        print("gradient: ", gradient(coef))
        history.append(coef)
        if np.max(np.abs(gradient(coef))) < precision or np.all(s) == 0:
            break
    print("We reached gradient:", gradient(coef), "in ", str(iter), "iteration")
    return coef, history
