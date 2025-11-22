import numpy as np
import time 

def learn_rate_sched( eta_0,k ):
    '''
    Calculates a time decay learning rate schedule, using k(number of iterations as time)
    used a decay constant of 0.01 as only small variations are desired

    Inputs: eta_0(float):initial learning rate,k(int):current iteration 
    Output: eta_k(float):learning rate at iteration k
    '''
    return eta_0/(1 + 0.01*k)

def gradDesc( f,gradf,theta_0,eta_0 = 1e-3,L = 1500,tol = 1e-12 ):
    '''
    Implements gradient descent(as explained in report) with a time decay learning rate schedule

    Inputs: f(callable):objective function,gradf(callable)gradient of objective function,
            theta_0(np.array):initial guess,eta_0(float):initial learning rate,
            L(int):maximum number of iterations,tol(float):tolerance for convergence
    Output(a tuple): theta_k(np.array):estimated location of local minima,f(theta_k)(float):value of function at local minima
    '''
    theta_k = np.array( theta_0 )
    k = 1

    while k <= L:
        eta_k = learn_rate_sched( eta_0,k )
        df_k = gradf( theta_k )
        theta_k1 = theta_k - eta_k*df_k

        if np.abs( f( theta_k1 ) - f( theta_k ) ) < tol :
            break
        
        theta_k = theta_k1
        k = k + 1
    
    return theta_k,f( theta_k )

def basin_hopping( f,gradf,theta_0,eta_0 = 1e-4,T_0 = 10.0,alpha = 0.98,delta_0 = 0.8,NBH = 50,L = 300,tol = 1e-9 ):
    '''
    Implements the basin-hopping algorithm(as explained in report)
    Inputs: f(callable):objective function,gradf(callable):gradient of objective function,
            theta_0(np.array):initial guess,eta_0(float):initial learning rate,
            T_0(float):initial temperature,alpha(float):cooling rate,
            delta_0(float):initial perturbation size,NBH(int)max number of basin-hopping iterations without improvement,
            L(int):max number of gradient descent iterations,tol(float):tolerance for convergence
    Output (a tuple): x(np.array):estimated location of global minima,glmin(float):value of function at global minima
    '''
    x,val = gradDesc( f,gradf,theta_0,eta_0,L,tol )
    glmin = val
    NOimp = 0
    T = T_0
    delta = delta_0

    while NOimp < NBH:
        xpert = x + delta*np.random.uniform( -1.0,1.0,size = len( x ) )
        lomin,flomin = gradDesc( f,gradf,xpert,eta_0,L,tol )
        delta_E = flomin - val

        if delta_E < 0 :
            x = lomin
            val = flomin
            glmin = min(glmin,val)
            NOimp = 0
        elif np.random.uniform( 0.0,1.0 ) < np.exp( -delta_E/T ) :
            x = lomin
            val = flomin
            NOimp = NOimp + 1
        else :
            NOimp = NOimp + 1

        T = alpha*T
        delta = delta_0*(T/T_0)

    return x,glmin

def goldstein_price(z):
    '''
    Goldstein-Price function:
    f(x, y) = [1 + (x + y + 1)^2(19 - 14x + 3x^2 - 14y + 6xy + 3y^2)]
               * [30 + (2x - 3y)^2(18 - 32x + 12x^2 + 48y - 36xy + 27y^2)]
    Input: np.array([x, y])
    Output: float
    '''
    x, y = z
    term1 = 1 + (x + y + 1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)
    term2 = 30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2)
    return term1 * term2

def goldstein_price_grad(z, eps=1e-4):
    '''
    Numerical gradient of Goldstein-Price using central differences.
    Input: np.array([x, y])
    Output: np.array([df/dx, df/dy])
    '''
    x, y = z
    fx_plus = goldstein_price(np.array([x + eps, y]))
    fx_minus = goldstein_price(np.array([x - eps, y]))
    dlogdx = (np.log(fx_plus) - np.log(fx_minus)) / (2*eps) # was overflowing without using log 
    dfdx = dlogdx * goldstein_price(z)

    fy_plus = goldstein_price(np.array([x, y + eps]))
    fy_minus = goldstein_price(np.array([x, y - eps]))
    dlogdy = (np.log(fy_plus) - np.log(fy_minus)) / (2*eps)
    dfdy = dlogdy * goldstein_price(z)
    return np.array([dfdx, dfdy])

a = float( input("Initial guess for x coordinate: ") )
b = float( input("Initial guess for y coordinate: ") )
guess = np.array( [a,b] )

start = time.time()
best_x,best_val = basin_hopping( goldstein_price, goldstein_price_grad, guess )
end = time.time()

print( "Optimal (x,y) = ",best_x )
print( "Time taken (in seconds) = ", end - start )
