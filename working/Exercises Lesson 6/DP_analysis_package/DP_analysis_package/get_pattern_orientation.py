import numpy as np
import matplotlib.pyplot as plt
import scipy

import emcee
import corner

from IPython.display import display, Math

from numba import njit


# useful functions for the fitting:
@njit
def findMiddle(input_list):
    middle = float(len(input_list))/2
    #if middle % 2 != 0:
    return input_list[int(middle - .5)]
    #else:
    #    return (input_list[int(middle)], input_list[int(middle-1)])

@njit
def rotation_matrix(x,y,theta):

    x_prime = x*np.cos(theta) + y*np.sin(theta)
    y_prime = -x*np.sin(theta) + y*np.cos(theta)
    
    return x_prime, y_prime

@njit
def check_parabola_cen(params,x_prime,x,y):
    
    a,theta,x_v,y_v = params
    
    y_prime = a*(x_prime)**2  # building the parabola centered in (0,0) for (x',y')
   
    y_second = (x_prime)*np.sin(theta) + (y_prime)*np.cos(theta)# rotating the parabola clockwise
    x_second = ((x_prime)*np.cos(theta) - (y_prime)*np.sin(theta))
    
    x_final = x_second + x_v # translating the parabola to the vertex
    y_final = y_second + y_v
    
    return x_final, y_final

# fitting functions:

@njit
def log_Likelihood_cen(params,x_data,y_data,x_err,y_err):
    
    a,theta,x_v,y_v = params
    
    #x_v = findMiddle(x_data)
    #y_v = findMiddle(y_data)
    
    x_trasl = x_data - x_v
    y_trasl = y_data - y_v  # translating the parabola in (0,0)
   
    x_rot = (x_trasl)*np.cos(theta) + (y_trasl)*np.sin(theta)
    y_rot = -(x_trasl)*np.sin(theta) + (y_trasl)*np.cos(theta) # rotating the parabola anti-clockwise()

    model = (y_rot - a*x_rot**2)**2 / (x_err**2 + y_err**2)
    
    return -0.5*np.sum(model)

def log_Prior_cen(params,x,y,sigmax,sigmay):
    
    a,theta,x_v,y_v = params
    
    # constraints on the parameters:
    a_constr = -np.inf < a < np.inf 
    theta_constr = -1.57 < theta < 1.57 
    
    # gaussian prior on x_v and y_v: 
    mu_x = findMiddle(x)
    mu_y = findMiddle(y)
    sx = findMiddle(sigmax)
    sy = findMiddle(sigmay)
    
    # uniform prior on a and theta:
    if ((a_constr==True) and (theta_constr==True)):
        return np.log(scipy.stats.norm.pdf(x_v,loc=mu_x,scale=sx/2)) + np.log(scipy.stats.norm.pdf(y_v,loc=mu_y,scale=sy/2))
    else: 
        return -np.inf

def log_Posterior_cen(params,x_data,y_data,x_err,y_err,sigmax,sigmay):
    
    log_prior = log_Prior_cen(params,x_data,y_data,sigmax,sigmay)
    #print(log_prior)
    if not np.isfinite(log_prior):
        return -np.inf
    
    return log_prior + log_Likelihood_cen(params,x_data,y_data,x_err,y_err)

def fit_emcee_cen(x,y,sx,sy,sigmax,sigmay,image):
    
    # fit ----------------------------------------------------------
    
    ndim=4 # number of parameters
    nwalkers = 50 # number of MCMC walkers
    burn = 500 # 'burn-in' period to let the chain stabilize
    nsteps = 5000 # number of steps for each walker

    # initialize parameters:
    a_init = -3e-5
    xv_init = findMiddle(x)
    yv_init = findMiddle(y)
    theta_init = 0.2 # initialize at the expected angle's rotation

    initial_params = np.array([a_init,theta_init,xv_init,yv_init])
    # initializing the walkers in a tiny Gaussian ball around the initial value
    starting_guesses = initial_params + 1e-4*np.random.random((nwalkers,ndim))
    
    # fit emcee:
    sampler = emcee.EnsembleSampler(nwalkers,ndim,log_Posterior_cen,args=[x,y,sx,sy,sigmax,sigmay])

    sampler.run_mcmc(starting_guesses,nsteps,progress=True)
    
    # plot ----------------------------------------------------------
    
    samples = sampler.get_chain()

    fig,axes = plt.subplots(ndim,figsize=(10,7),sharex=True)

    labels = [r'$a$',r'$\theta$',r'$x_v$',r'$y_v$']
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:,:,i],'k',alpha=0.3)
        ax.set_xlim(0,len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[0].set_title('Full chains')
    axes[-1].set_xlabel('step number')

    plt.show()
    
    emcee_trace = sampler.chain[:,burn:,:].reshape(-1,ndim) # burned and flattend chain
    x_grid = np.arange(0,emcee_trace.shape[0])

    fig,axes = plt.subplots(ndim,figsize=(10,7),sharex=True)

    for i in range(ndim):
    
        ax = axes[i]
        ax.plot(x_grid,emcee_trace[:,i])
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[0].set_title('1000-Burned chain')
    axes[-1].set_xlabel('steps')
    
    plt.show()

    fig = corner.corner(emcee_trace,labels=labels,labelpad=0.2);
    plt.show()
    
    # compute values and uncertaintes ------------------------------------
    
    values_emcee = np.zeros(4)
    sigma_emcee = np.zeros(4)

    for i in range(ndim):
        mcmc = np.percentile(emcee_trace[:,i],[16,50,84])
        q = np.diff(mcmc)
        values_emcee[i] = mcmc[1]
        sigma_emcee[i] = (q[0]+q[1])/2
        #txt = r"$\mathrm{{{3}}} = {0:.7f}_{{\,-{1:.7f}}}^{{\,+{2:.7f}}}$"
        #txt = txt.format(mcmc[1], q[0], q[1], labels[i])
        #display(Math(txt))
    
    theta_deg = values_emcee[1]*180/np.pi
    sigma_theta_deg = sigma_emcee[1]*180/np.pi

    fig,ax = plt.subplots(1,figsize=(10,7))
    
    x_plot= np.linspace(-800,800,1000)

    x_fit,y_fit = check_parabola_cen(values_emcee,x_plot,x,y)
    x_rot,y_rot = rotation_matrix(x,y,values_emcee[1])

    ax.imshow(image,origin='lower')

    ax.scatter(x,y,label='data',s=1.5)
    ax.plot(x_fit,y_fit,label='fit', linewidth=0.5)
    ax.set_title('Fit rotated parabola emcee vertex')
    ax.text(1000,1000,r'$\theta = (%f \pm %f)^\circ$'%(theta_deg,sigma_theta_deg),color='black',backgroundcolor='white')
    ax.set_xlabel('x [pixels]')
    ax.set_ylabel('y [pixels]')
    ax.legend()

    plt.savefig(r"c:\Users\Ludovico\Desktop\unimib\PhD\Scientific Computing\scientificcomputing_bicocca_2024\working\Exercises Lesson 6\DP_analysis_package\DP_analysis_package\result_images\pattern_orientation%f.jpg"%values_emcee[1], format='jpg')
    
    #fig.savefig("..\COSMOCal notes\Photogrammetry//Pico Veleta//09_30_24//set_04//Diffraction pattern//fit_rot_parabola_emcee_theta%f.jpg"%values_emcee[1], format='jpg',dpi=400)
    
    #file = open("..\COSMOCal notes\Photogrammetry//Pico Veleta//09_30_24//set_04//Diffraction pattern//Diffraction_results.txt", 'a')
    #file.write('%f \t %f \n'%(theta_deg,sigma_theta_deg))
    #file.close()
    
    return theta_deg,sigma_theta_deg