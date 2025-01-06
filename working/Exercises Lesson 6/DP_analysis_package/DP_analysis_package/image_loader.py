# read the images

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from tqdm import tqdm

def read_images(path,number_meas):
    """This function reads the images and produce a first raw plot"""
    im = np.empty(number_meas,dtype=object)
    #im_tbr = np.empty(number_meas,dtype=object)
    sigma_image = np.empty(number_meas,dtype=object)
    index = np.arange(number_meas)

    for i in index:
        name = path[i]
        file = fits.open(name)
        im[i] = file[0].data
        sigma_image[i] = np.sqrt(im[i]).astype('float64')
        file.close()
    
    # plot
    '''
    ncols = 1
    nrows = int(number_meas)

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6*ncols,6*nrows))

    for i in index:
        
        ax = ax.ravel()

        ax[i].set_title('Image %i'%i)
        ax[i].imshow(im[i])
        
        fig.colorbar(ax[i].imshow(im[i]),ax=ax[i],fraction=0.03,label='Intensity [ADU]')

    fig.suptitle('Raw images')
    plt.show()
    '''
    
    return im,sigma_image

def correct_orientation(init_im,sigma_P_init):
    """This function takes the images and correct their orientation"""
    #ncols=1
    #nrows=len(init_im)
    #fig,ax = plt.subplots(ncols=ncols,nrows=nrows,figsize=(ncols*6,nrows*6))
    orientation = False # camera upside-down
    if(orientation==False):
        flipped_im = []
        sigma_flipped = []
        for i in range(len(init_im)):
            flipped_im.append(init_im[i][::-1,:])
            sigma_flipped.append(sigma_P_init[i][::-1,:])
            #ax[i].imshow(flipped_im[i],origin='lower')
            #ax[i].set_ylabel('y [pixels]')
            #ax[i].set_title('Image %d'%i)
    
    #ax[-1].set_xlabel('x [pixels]')
    #fig.suptitle('Rotated images')
    #plt.show()

    return flipped_im, sigma_flipped

def pre_process(image,sigma):
    """This funtion takes as input an images and the associated Poissonian uncertainties.
       It returns a processed image, where all the pixel with an intensity below the treshold are setted to zero."""
    
    process_image = image
    
    x_range = np.arange(0,image.shape[1])
    y_range = np.arange(0,image.shape[0])

    norm_sigma = sigma/np.max(sigma)
    unit_sigma = 1/norm_sigma
    
    for i in x_range:
        for j in y_range:      
            if not(unit_sigma[j,i]<1.5):
                process_image[j,i] = 0
                
    return process_image

def pre_process_images(images,sigmas):
    "This function pre-process the input images"

    processed_im = np.empty(len(images),dtype=object)

    # visualization of the processed image
    ncols = 2
    nrows = len(images)

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(11*ncols,7*nrows))

    for i in tqdm(range(len(images))):
        processed_im[i] = pre_process(images[i],sigmas[i])

        ax[i,1].set_title('Processed Image %i'%i)
        ax[i,1].set_xlabel('x [pixels]')
        ax[i,1].set_ylabel('y [pixels]')
    
        ax[i,0].contour(sigmas[i],cmap='turbo',origin='lower')
        norm_sigma = sigmas[i]/np.max(sigmas[i])
        unit_sigma = 1/norm_sigma
    
        ax[i,0].set_title('Poisson error image %i'%i)
        ax[i,0].set_xlabel('x [pixels]')
        ax[i,0].set_ylabel('y [pixels]')
    
        fig.colorbar(ax[i,1].imshow(processed_im[i],origin='lower'), ax=ax[i,1],fraction=0.03,label='Intensity [ADU]')
        fig.colorbar(ax[i,0].imshow(unit_sigma,cmap='turbo_r',origin='lower'), ax=ax[i,0],fraction=0.03,label=r'$\sigma$ units')

    plt.show()

    return processed_im