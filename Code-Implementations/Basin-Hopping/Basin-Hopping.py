import numpy as np

def learn_rate_sched( eta_0,k ):
    '''
    Calculates a time decay learning rate schedule, using k(number of iterations as time)
    used a decay constant of 0.01 as only small variations are desired

    Inputs: eta_0(float):initial learning rate,k(int):current iteration 
    Output: eta_k(float):learning rate at iteration k
    '''
    return eta_0/(1 + 0.01*k)

def gradDesc( f,gradf,theta_0,eta_0 = 1e-3,L = 500,tol = 1e-11 ):
    '''
    Implements gradient descent(as explained in report) with a time decay learning rate schedule

    Inputs: f(callable):objective function,gradf(callable):gradient of objective function,
            theta_0(np.array):initial guess,eta_0(float):initial learning rate,
            L(int):maximum number of iterations,tol(float):tolerance for convergence
    Output(a tuple): theta_k(np.array):estimated location of local minima,f(theta_k)(float):value of function at local minima
    '''
    theta_k = np.array( theta_0 )
    k = 1

    while k <= L:
        eta_k = learn_rate_sched( eta_0,k ) #compute learning rate at iteration k
        df_k = gradf( theta_k ) #compute gradient at iteration k
        theta_k1 = theta_k - eta_k*df_k #update solution using the gradient 

        if np.abs( f( theta_k1 ) - f( theta_k ) ) < tol : #convergence check to stop if tolerance met
            break
        
        theta_k = theta_k1
        k = k + 1
    
    return theta_k,f( theta_k )

def basin_hopping( f,gradf,theta_0,eta_0 = 1e-3,T_0 = 10.0,alpha = 0.9,delta_0 = 0.02,NBH = 25,L = 500,tol = 1e-11 ):
    '''
    Implements the basin-hopping algorithm(as explained in report)
    Inputs: f(callable):objective function,gradf(callable):gradient of objective function,
            theta_0(np.array):initial guess,eta_0(float):initial learning rate,
            T_0(float):initial temperature,alpha(float):cooling rate,
            delta_0(float):initial perturbation size,NBH(int)max number of basin-hopping iterations without improvement,
            L(int):max number of gradient descent iterations,tol(float):tolerance for convergence
    Output (a tuple): x(np.array):estimated location of global minima,glmin(float):value of function at global minima
    '''
    x,val = gradDesc( f,gradf,theta_0,eta_0,L,tol ) #initial local minimisation 
    glmin = val 
    NOimp = 0
    T = T_0
    delta = delta_0

    while NOimp < NBH:
        xpert = x + delta*np.random.uniform( -1.0,1.0,size = len( x ) ) #perturbing current solution randomly about 0 symmetrically 
        lomin,flomin = gradDesc( f,gradf,xpert,eta_0,L,tol ) #local minimisation of perturbed solution
        delta_E = flomin - val

        #accepting based on Metropolis criterion
        if delta_E < 0 : #better solution found
            x = lomin
            val = flomin
            glmin = min(glmin,val)
            NOimp = 0
        elif np.random.uniform( 0.0,1.0 ) < np.exp( -delta_E/T ) : #accepting worse solution according to probability
            x = lomin
            val = flomin
            NOimp = NOimp + 1
        else : #reject solution
            NOimp = NOimp + 1

        T = alpha*T #decrease T to reduce acceptance of worse solutions over time
        delta = delta_0*(T/T_0) #reduces perturbation size as temperature decreases to enable finer local search

    return x,glmin

def springmodel_energy( x ):
    '''
    Helps calculate the energy of a H2O molecule based on the spring model 
    ( uses the fact that potential energy stored in a spring is 0.5*k*(x - x0)^2 )
    Input: np.array containing [<bond length 1>,<bond length 2>,<bond angle>]
    Output: (float)Energy of the molecule based on the spring model
    '''
    r1,r2,theta = x 
    kb,r_0 = 1882.8,0.958
    E_bond = 0.5*kb*(r1 - r_0)**2 + 0.5*kb*(r2 - r_0)**2 #Harmonic bond stretching energy
    ka,theta_0 = 0.07011,104.5
    E_angle = 0.5*ka*(theta - theta_0)**2 #Harmonic bond angle bending energy

    return E_bond + E_angle

def springmodel_energygrad( x ):
    '''
    Helps calculate the gradient of the energy of a H2O molecule based on the spring model
    Input: np.array containing [<bond length 1>,<bond length 2>,<bond angle>]
    Output: np.array conataining [<Energy gradient wrt bond length 1>,
                                  <energy gradient wrt bond length 2>,
                                  <energy gradient wrt bond angle>]
    '''
    r1,r2,theta = x 
    kb,rtheta = 1882.8,0.958
    dE_dr1 = kb*(r1 - rtheta)
    dE_dr2 = kb*(r2 - rtheta)
    ka,theta_0 = 0.07011,104.5
    dE_dtheta = ka*(theta - theta_0)

    return np.array([dE_dr1,dE_dr2,dE_dtheta])

a = float( input("Initial guess for bond length-r1 of H2O(in Angstroms): ") )
b = float( input("Initial guess for bond length-r2 of H2O(in Angstroms): ") )
c = float( input("Initial guess for bond angle theta of H2O(in degrees): ") )
guess = np.array( [a,b,c] )
best_x,best_val = basin_hopping( springmodel_energy,springmodel_energygrad,guess )
print( "Optimal (r1,r2,theta) in Angstorm,degrees = ",best_x )
print( "Optimal Energy in (kJ/mol) = ",best_val )