import numpy as np


def cauchy_point(objective_func, gradient_func, max_trust_radius, initial_point,
                 hessian=lambda x: x,
                 hessian_approximation=True):
    trust_radius = max_trust_radius / 10
    x = initial_point
    prev = initial_point
    g_x = gradient_func(x)
    B = np.identity(g_x.shape[0]) if hessian_approximation else hessian(x)

    history = [initial_point]

    for i in range(10000):
        print(x)
        print(B)
        g_x = gradient_func(x)
        g_prev = gradient_func(prev)
        # Approximating Hessian matrix with BFGS formula
        if not np.array_equal(x, prev):
            if hessian_approximation:
                y = g_x - g_prev
                s = x - prev
                u = y - np.matmul(B, s)
                # B = B + (np.matmul(u, u.T) / np.matmul(u.T, s)) #SR1
                B = B - (np.matmul(u, u.T) / np.matmul(s.T, u)) + (np.matmul(y, y.T) / (y.T, s))  # BFGS
            else:
                B = hessian(x)
        # calculating direction of next step with Cauchy point method
        ps_direction = -(trust_radius / np.linalg.norm(g_x)) * g_x
        tau_condition = np.matmul(g_x.T, np.matmul(B, g_x))  # we use this to check on negative definiteness!!
        tau_step = 1 if tau_condition <= 0 else min(1, np.linalg.norm(g_x) ** 3 / trust_radius * tau_condition)
        p_cauchy = tau_step * ps_direction

        qu_apx = objective_func(x) + np.matmul(g_x.T, p_cauchy) + 1 / 2 * np.matmul(p_cauchy.T, np.matmul(B, p_cauchy))
        # qu_apx is a quadratic approximation of objective function (f) which use B as approximation of Hessian of f
        # based on the Taylor-series expansion of f around x
        qu_zero = objective_func(x)  # qu_apx when p = 0

        # Evaluating Rho_k for decide on how to update trust_radius
        rho_k = (objective_func(x) - objective_func(x + p_cauchy)) / (qu_zero - qu_apx)
        if rho_k < 0.25:
            trust_radius = 0.25 * trust_radius
        else:
            if rho_k > 0.75 and np.linalg.norm(p_cauchy) - trust_radius > 0.0001:
                trust_radius = np.min(2 * trust_radius, max_trust_radius)

        if rho_k > 0.25:
            prev = x
            x = x + p_cauchy
            history.append(x)

    return x, history


def dogleg(objective_func, gradient_func, max_trust_radius, initial_point,
           hessian=lambda x: x,
           hessian_approximation=True):
    trust_radius = max_trust_radius / 10
    x = initial_point
    prev = initial_point
    g_x = gradient_func(x)
    B = np.identity(g_x.shape[0]) if hessian_approximation else hessian(x)
    B_inv = np.identity(g_x.shape[0]) if hessian_approximation else np.linalg.inv(hessian(x))
    history = [initial_point]

    for i in range(30000):
        print(x)
        print(B)
        g_x = gradient_func(x)
        g_prev = gradient_func(prev)
        # Approximating Hessian matrix with BFGS formula
        if not np.array_equal(x, prev):
            if hessian_approximation:
                y = g_x - g_prev
                s = x - prev
                u = y - np.matmul(B, s)

                #B = B + (np.matmul(u, u.T) / np.matmul(u.T, s))  # SR1
                #v = s - np.matmul(B_inv, y)
                #B_inv = B_inv + np.matmul(v, v.T) / np.matmul(v.T, y)

                B = B - (np.matmul(u, u.T) / np.matmul(s.T, u)) + (np.matmul(y, y.T) / (y.T, s))  # BFGS
                v = np.matmul(B_inv, y)
                B_inv = B_inv - (np.matmul(v, v.T) / np.matmul(y.T, v)) + (np.matmul(s, s.T) / (y.T, s))
            else:
                B = hessian(x)
                B_inv = np.linalg.inv(B)

        Pb = -1 * np.matmul(B_inv, g_x)
        Pu = -1 * (np.matmul(g_x.T, g_x) / np.matmul(np.matmul(g_x.T, B), g_x)) * g_x
        tau = trust_radius / np.linalg.norm(Pu)
        p = np.zeros(Pb.shape)
        if 0 <= tau <= 1:
            p = tau * Pu
        elif 1 < tau <= 2:
            p = Pu + (tau - 1) * (Pb - Pu)

        qu_apx = objective_func(x) + np.matmul(g_x.T, p) + 1 / 2 * np.matmul(p.T, np.matmul(B, p))
        # qu_apx is a quadratic approximation of objective function (f) which use B as approximation of Hessian of f
        # based on the Taylor-series expansion of f around x
        qu_zero = objective_func(x)  # qu_apx when p = 0

        # Evaluating Rho_k for decide on how to update trust_radius
        rho_k = (objective_func(x) - objective_func(x + p)) / (qu_zero - qu_apx)
        if rho_k < 0.25:
            trust_radius = 0.25 * trust_radius
        else:
            if rho_k > 0.75 and np.linalg.norm(p) - trust_radius > 0.0001:
                trust_radius = min(2 * trust_radius, max_trust_radius)

        if rho_k > 0.25:
            prev = x
            x = x + p
            history.append(x)

    return x, history
