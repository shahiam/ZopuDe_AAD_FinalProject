import numpy as np
import matplotlib.pyplot as plt
import time

# Set random seed for reproducibility
np.random.seed(42)

# ----- Data Generation -----
# Generate noisy data based on a nonlinear model:
def generate_data(a, b, c, d, X, noise_std=0.5):
    """
    Generates y = a*sin(b*x) + c*cos(d*x) + noise
    :param a, b, c, d: True parameters for the model
    :param X: Input array
    :param noise_std: Standard deviation of Gaussian noise
    """
    noise = np.random.normal(0, noise_std, size=X.shape)
    return a*np.sin(b*X) + c*np.cos(d*X) + noise

# ----- Model Definition -----
def model(params, X):
    """
    Computes model prediction y for params and inputs X
    :param params: Array-like (a, b, c, d)
    :param X: Input array
    """
    a, b, c, d = params
    return a * np.sin(b * X) + c * np.cos(d * X)

# ----- Loss Function -----
def loss(params, X, Y):
    """
    Mean Squared Error between model predictions and data
    """
    return np.mean((model(params, X) - Y)**2)

# ----- SPSA Gradient Estimation -----
def spsa_grad(loss_fn, params, X, Y, ck):
    """
    Estimates gradient using SPSA random perturbations
    :param loss_fn: The loss function
    :param params: Current parameter estimate
    :param X, Y: Inputs and targets
    :param ck: Perturbation magnitude (decays over iterations)
    """
    p = len(params)
    delta = np.random.choice([-1, 1], size=p)  # Random +/-1 for each parameter
    params_plus = params + ck * delta
    params_minus = params - ck * delta
    loss_plus = loss_fn(params_plus, X, Y)
    loss_minus = loss_fn(params_minus, X, Y)
    grad = (loss_plus - loss_minus) / (2 * ck * delta)
    return grad

# ----- SPSA Optimizer -----
def SPSA(loss_fn, init_params, X, Y,
         a=0.1,   # Learning rate numerator (higher = bigger steps)
         c=0.1,   # Initial perturbation magnitude (higher = noisier estimates)
         alpha=0.602, # Learning rate decay (typical values: 0.6-0.8)
         gamma=0.101, # Perturbation decay (typical values: 0.1-0.2)
         iterations=200): # Number of optimization steps
    params = init_params.copy()
    A = iterations / 10  # Stability constant for learning rate schedule
    for k in range(1, iterations + 1):
        ak = a / ((k + A)**alpha)       # Current learning rate
        ck = c / (k**gamma)             # Current perturbation size
        grad = spsa_grad(loss_fn, params, X, Y, ck)
        params -= ak * grad
    return params

# ----- Adam Optimizer -----
def adam_optimizer(loss_fn, init_params, X, Y,
                   lr=0.05,    # Initial learning rate
                   beta1=0.9,  # Exponential decay rate for first moment
                   beta2=0.999,# Exponential decay rate for second moment
                   eps=1e-8,   # Small value to avoid division by zero
                   iterations=200): # Number of optimization steps
    params = init_params.copy()
    m = np.zeros_like(params)
    v = np.zeros_like(params)
    for t in range(1, iterations + 1):
        grad = numerical_grad(loss_fn, params, X, Y)
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        params -= lr * m_hat / (np.sqrt(v_hat) + eps)
    return params

# ----- Numerical Gradient for Adam -----
def numerical_grad(loss_fn, params, X, Y, epsilon=1e-5):
    """
    Numerical finite difference gradient estimation
    :param epsilon: Small step for difference
    """
    grad = np.zeros_like(params)
    for i in range(len(params)):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[i] += epsilon
        params_minus[i] -= epsilon
        grad[i] = (loss_fn(params_plus, X, Y) - loss_fn(params_minus, X, Y)) / (2 * epsilon)
    return grad

def parameter_error(estimated, true):
    """
    Compute Euclidean distance between estimated and true parameters.
    """
    return np.linalg.norm(estimated - true)



# ----- Example Usage -----
# True model parameters (edit these for a new true curve)
true_params = np.array([2.5, 1.5, -1.0, 0.5])

# Input data
X = np.linspace(0, 2 * np.pi, 100)
# Generate noisy output data (edit noise_std for a "cleaner" problem)
Y = generate_data(*true_params, X, noise_std=0.5)

# Initial parameter guess (edit these to start closer/farther from the optimum)
init_params = np.array([2.3, 1.7, -0.8, 0.6])

# ----- Run SPSA -----
start_time = time.time()
params_spsa = SPSA(loss, init_params, X, Y,
                   a=0.1, c=0.1, alpha=0.602, gamma=0.101, iterations=300)
time_spsa = time.time() - start_time

# ----- Run Adam -----
start_time = time.time()
params_adam = adam_optimizer(loss, init_params, X, Y,
                            lr=0.05, beta1=0.9, beta2=0.999, eps=1e-8, iterations=300)
time_adam = time.time() - start_time

error_spsa = parameter_error(params_spsa, true_params)
error_adam = parameter_error(params_adam, true_params)

# ----- Results -----
print("True parameters:         ", true_params,"\n")
print("SPSA estimated params:   ", params_spsa)
print("SPSA parameter error:", error_spsa)
print("SPSA time (seconds):     ", time_spsa)
print()
print("Adam estimated params:   ", params_adam)
print("Adam parameter error:", error_adam)
print("Adam time (seconds):     ", time_adam)






# ----- Visualization -----
plt.scatter(X, Y, label='Noisy data', color='gray', alpha=0.6)
plt.plot(X, model(true_params, X), label='True model', linewidth=2)
plt.plot(X, model(params_spsa, X), label='SPSA fit', linestyle='dashed')
plt.plot(X, model(params_adam, X), label='Adam fit', linestyle='dotted')
plt.legend()
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Model Fitting with SPSA vs Adam Optimization")
plt.show()
