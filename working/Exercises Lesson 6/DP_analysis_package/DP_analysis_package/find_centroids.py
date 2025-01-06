import numpy as np
import matplotlib.pyplot as plt
import scipy


def twoD_Gaussian(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x,y = xy
    
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
    
    return g.ravel()

def find_centroids(data_image,sigma_P):
    
    # for the plot for control:
    x_plot = np.arange(data_image.shape[1])
    y_plot = np.arange(data_image.shape[0])
    
    ncols = 1
    nrows = 4
    
    fig,ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(10,nrows*7))
    
    ax[0].imshow(data_image,origin='lower')
    ax[0].set_title('data image')
    
    # sum over the axis:
    sum_x = np.sum(data_image,axis=0)
    sum_y = np.sum(data_image,axis=1)
    
    # find the initial values for centroids:
    threshold_x = 1/2.7*(sum_x.max())
    peak_indices_x, _ = scipy.signal.find_peaks(sum_x, height=threshold_x, threshold=None, 
                                       distance=100, prominence=None, 
                                       width=10, wlen=None, rel_height=0.5, 
                                       plateau_size=None)
    
    # check the true-peaks
    count_x = 0
    for i in range(1,len(peak_indices_x)):
        if(abs(peak_indices_x[i]-peak_indices_x[i-1])>250):
            new_x_arr = np.delete(peak_indices_x, i)
            count_x +=1
            
    if(count_x != 0):
        peak_indices_x = new_x_arr
            
    count_x2 = 0        
    for i in range(0,len(peak_indices_x)-1):
        if(abs(peak_indices_x[i]-peak_indices_x[i+1])>250):
            new_x_arr2 = np.delete(peak_indices_x, i)
            count_x2 += 1
            
    if(count_x2 != 0):
        peak_indices_x = new_x_arr2
    
    ax[1].plot(x_plot,sum_x)
    ax[1].plot(x_plot[peak_indices_x],sum_x[peak_indices_x], 'x')
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('counts [ADU]')
    ax[1].set_title('sum over x and initial x centroids')

    threshold_y = 1/2.7*(sum_y.max())
    peak_indices_y, _ = scipy.signal.find_peaks(sum_y, height=threshold_y, threshold=None, 
                                       distance=100, prominence=None, 
                                       width=10, wlen=None, rel_height=0.5, 
                                       plateau_size=None)
    
    count_y = 0
    for i in range(1,len(peak_indices_y)):
        if(abs(peak_indices_y[i]-peak_indices_y[i-1])>300):
            new_y_arr = np.delete(peak_indices_y, i)
            count_y +=1
            
    if(count_y != 0):
        peak_indices_y = new_y_arr
            
    count_y2 = 0        
    for i in range(0,len(peak_indices_y)-1):
        if(abs(peak_indices_y[i]-peak_indices_y[i+1])>300):
            new_y_arr2 = np.delete(peak_indices_y, i)
            count_y2 += 1
            
    if(count_y2 != 0):
        peak_indices_y = new_y_arr2
    
    ax[2].plot(y_plot,sum_y)
    ax[2].plot(y_plot[peak_indices_y],sum_y[peak_indices_y], 'x')
    ax[2].set_xlabel('y')
    ax[2].set_ylabel('counts [ADU]')
    ax[2].set_title('sum over y and initial y centroids')
    
    # determine the best axis:
    if(len(peak_indices_x)>=len(peak_indices_y)):
        best_ax = 'x'
    else: 
        best_ax = 'y'
    
    print('best ax: ',best_ax)
    # fit with 2D gaussians the neighbourhood of the peaks found:
    if(best_ax=='x'):
        init_cenx = np.zeros(len(peak_indices_x))
        init_ceny = np.zeros(len(peak_indices_x))
        
        for i,cen_x in zip(range(len(peak_indices_x)),peak_indices_x):
            init_cenx[i] = cen_x 
            # find the maximum along the line of the peak
            cen_y = int(np.median(np.where(data_image[:,cen_x]==np.max(data_image[:,cen_x]))))
            init_ceny[i] = cen_y
    
    if(best_ax=='y'):
        init_cenx = np.zeros(len(peak_indices_y))
        init_ceny = np.zeros(len(peak_indices_y))
        
        for i,cen_y in zip(range(len(peak_indices_y)),peak_indices_y):
            init_ceny[i] = cen_y
            # find the maximum along the line of the peak
            cen_x = int(np.median(np.where(data_image[cen_y,:]==np.max(data_image[cen_y,:]))))
            init_cenx[i] = cen_x
            
    centroids = np.empty([len(init_cenx),2])
    err_centroids = np.empty([len(init_cenx),2])
    sigma_x = np.zeros(len(init_cenx))
    sigma_y = np.zeros(len(init_cenx))
    x_maxline = np.empty([len(init_cenx)],dtype=object)
    y_maxline = np.empty([len(init_cenx)],dtype=object)
    
    fig_G,ax_G = plt.subplots(nrows=len(init_cenx),ncols=1,figsize=(10,nrows*10))
    
    for i,cx,cy in zip(range(len(init_cenx)),init_cenx,init_ceny):
    # fit with a 2D Gaussian
        print(cx,cy)
        
        # if the centroids are too close to the borders of the image
        if((cx-100<0)|(cx+100>(data_image.shape[1]))|(cy-100<0)|(cy+100>(data_image.shape[0]))):
            
            len_cutout = int(min([abs(cx),abs(data_image.shape[1]-cx),abs(cy),abs(data_image.shape[0]-cy)]))
            
            cutout = data_image[int(cy)-len_cutout:int(cy)+len_cutout,int(cx)-len_cutout:int(cx)+len_cutout]
            sigma_cutout = sigma_P[int(cy)-len_cutout:int(cy)+len_cutout,int(cx)-len_cutout:int(cx)+len_cutout]
            
            x_plot_G = np.arange(-len_cutout,len_cutout)
            y_plot_G = np.arange(-len_cutout,len_cutout)
            x_plot_G,y_plot_G = np.meshgrid(x_plot_G,y_plot_G)
        
        else:
            cutout = data_image[int(cy)-100:int(cy)+100,int(cx)-100:int(cx)+100]
            sigma_cutout = sigma_P[int(cy)-100:int(cy)+100,int(cx)-100:int(cx)+100]
            
            x_plot_G = np.arange(-100,100)
            y_plot_G = np.arange(-100,100)
            x_plot_G,y_plot_G = np.meshgrid(x_plot_G,y_plot_G)
            
            
        init_param = [data_image[int(cy),int(cx)],0,0,10,10,0.01,5]
            
        popt, pcov = scipy.optimize.curve_fit(twoD_Gaussian,(x_plot_G,y_plot_G), cutout.ravel(),p0=init_param,sigma=sigma_cutout.ravel())
            
        centroids[i] = popt[1]+cx,popt[2]+cy
        err_centroids[i] = np.sqrt(pcov[1,1]),np.sqrt(pcov[2,2]) # errors associated to the determination of the centroids
        sigma_x[i] = popt[3]
        sigma_y[i] = popt[4]
            
        gaus2d_fit = twoD_Gaussian((x_plot_G,y_plot_G),*popt)
        gaus2d_fit = gaus2d_fit.reshape(x_plot_G.shape)
        
        ax_G = ax_G.ravel()
        ax_G[i].imshow(cutout,origin='lower')
        ax_G[i].contour(gaus2d_fit,cmap='turbo')
        ax_G[i].set_xlabel('x [pixels]')
        ax_G[i].set_ylabel('y [pixels]')
        ax_G[i].set_title('2D gaus fit')

    ax[3].set_xlabel('x [pixels]')
    ax[3].set_ylabel('y [pixels]')
    ax[3].imshow(data_image,origin='lower')
    ax[3].set_title('centroids')
    for i in range(len(centroids)):
        ax[3].scatter(centroids[i,0],centroids[i,1],color='red',s=5)

    plt.show()
            
    print('centroids:','\n',centroids)
    print('err centroids: ', '\n',err_centroids)
    
    return centroids,err_centroids,sigma_x,sigma_y