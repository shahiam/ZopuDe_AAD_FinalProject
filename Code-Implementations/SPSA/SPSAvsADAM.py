import numpy as np
import matplotlib.pyplot as plt
import time
np.random.seed(42)

#generate noisy data based on a nonlinear model:
def generate_data(a, b, c, d, X, noise_std=0.5):
    """
    generates y = a*sin(b*x) + c*cos(d*x) + noise
    -> a, b, c, d: True parameters for the model
    -> X: Input array
    -> noise_std: Standard deviation of Gaussian noise
    """
    noise = np.random.normal(0, noise_std, size=X.shape)
    return a*np.sin(b*X) + c*np.cos(d*X) + noise

def model(params, X):
    """
    computes model prediction y for params and inputs X
    -> params: array like (a, b, c, d)
    -> param X: input array
    """
    a, b, c, d = params
    return a * np.sin(b * X) + c * np.cos(d * X)


def loss(params, X, Y):
    #model vs prediction
    return np.mean((model(params, X) - Y)**2)

def spsa_grad(loss_fn, params, X, Y, ck):
    """
    estimates gradient using SPSA random perturbations
    ->loss_fn: The loss function
    ->params: Current parameter estimate
    ->X, Y: Inputs and targets
    ->ck: Perturbation magnitude
    """
    p = len(params)
    delta = np.random.choice([-1, 1], size=p)  #random step for each parameter
    params_plus = params + ck * delta
    params_minus = params - ck * delta
    loss_plus = loss_fn(params_plus, X, Y)
    loss_minus = loss_fn(params_minus, X, Y)
    grad = (loss_plus - loss_minus) / (2 * ck * delta)
    return grad

def SPSA(loss_fn, init_params, X, Y,
         a=0.1,   #learning rate numerator
         c=0.1,   #perturbation magnitude
         alpha=0.602, # Learning rate decay
         gamma=0.101, # Perturbation decay
         iterations=200): #steps
    params = init_params.copy()
    A = iterations / 10  #stability constant
    for k in range(1, iterations + 1):
        ak = a / ((k + A)**alpha)       #learning rate
        ck = c / (k**gamma)             #perturbation size
        grad = spsa_grad(loss_fn, params, X, Y, ck)
        params -= ak * grad
    return params

def adam_optimizer(loss_fn, init_params, X, Y,
                   lr=0.01,    #initial learning rate
                   beta1=0.9,  #decay rate for first moment
                   beta2=0.999,#decay rate for second moment
                   eps=1e-8,   #to avoid division by zero
                   iterations=200): #steps
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

#ADAM gradient
def numerical_grad(loss_fn, params, X, Y, epsilon=1e-5):

    grad = np.zeros_like(params)
    for i in range(len(params)):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[i] += epsilon
        params_minus[i] -= epsilon
        grad[i] = (loss_fn(params_plus, X, Y) - loss_fn(params_minus, X, Y)) / (2 * epsilon)
    return grad

def parameter_error(estimated, true):

    return np.linalg.norm(estimated - true)


true_params = np.array([2.5, 1.5, -1.0, 0.5])


X = np.linspace(0, 2 * np.pi, 100)

#we calculate for different values of noise_std for evaluation
Y = generate_data(*true_params, X, noise_std= 3.5)

init_params = np.array([1.3, 1.7, -0.8, 0.6])

iteration_values = np.arange(20, 471, 10)  
errors_spsa = []
errors_adam = []
times_spsa = []
times_adam = []

Kerrors_spsa = []
Ktimes_spsa = []


for iters in iteration_values:
    #Adam
    t0 = time.time()
    params_adam = adam_optimizer(loss, init_params, X, Y, iterations=iters)
    times_adam.append(time.time() - t0)
    errors_adam.append(parameter_error(params_adam, true_params))

iteration_values = np.arange(20, 1000, 30)
     
for iters in iteration_values:

    Ktimes_spsa = []    
    Kerrors_spsa = []     
    for i in np.arange(1, 20, 1):
        np.random.seed(i)
        t0 = time.time()
        params_spsa = SPSA(loss, init_params, X, Y, iterations=iters)
        Ktimes_spsa.append(time.time() - t0)
        Kerrors_spsa.append(parameter_error(params_spsa, true_params))
    times_spsa.append(np.mean(Ktimes_spsa))
    errors_spsa.append(np.mean(Kerrors_spsa))
     
    


plt.figure(figsize=(8,6))
plt.plot(times_spsa, errors_spsa, label='SPSA', marker='o')
plt.plot(times_adam, errors_adam, label='Adam', marker='o')
plt.xlabel('Time Taken (seconds)')
plt.ylabel('Parameter Error (Euclidean distance)')
plt.title('Accuracy vs Time Taken for SPSA and Adam')
plt.legend()
plt.grid(True)
ax = plt.gca()
plt.scatter([0], [0], color='black', s=30, zorder=5)

ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

plt.show()
