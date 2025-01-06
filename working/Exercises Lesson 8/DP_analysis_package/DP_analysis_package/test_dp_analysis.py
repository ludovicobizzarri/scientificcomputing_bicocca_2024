import glob
import numpy as np

import image_loader
import find_centroids
import get_pattern_orientation

# set the path
PATH = r"c:\Users\Ludovico\Desktop\Parigi\LPENS\COSMOCal notes\Photogrammetry\Pico Veleta\09_30_24\set_04\Diffraction pattern\CapObj*.FIT"
# get the images
print('Loading the images...')
data_name = glob.glob(PATH)
# load the images
init_im,sigma_P_init = image_loader.read_images(data_name[:2],len(data_name[:2])) # just the first two images
# correct the image orientation:
rot_im,sigma_rot = image_loader.correct_orientation(init_im,sigma_P_init)
print('done')
# preprocess the images:
print('Pre-processing the images...')
preprocessed_images = image_loader.pre_process_images(rot_im,sigma_rot)
print('done')

# find the centroids in all images:
print('Looking for the diffraction maxima centroids...')
centroids = np.empty([len(preprocessed_images)],dtype=object)
sigma_centroids = np.empty([len(preprocessed_images)],dtype=object)
sigma_x = np.empty([len(preprocessed_images)],dtype=object)
sigma_y = np.empty([len(preprocessed_images)],dtype=object)

for i in range(len(preprocessed_images)):
    centroids[i],sigma_centroids[i],sigma_x[i],sigma_y[i] = find_centroids.find_centroids(preprocessed_images[i],sigma_rot[i])

print('done')

print('Making sure of an odd number of centroids...')
# delete images with even centroids
idx_delete = []

for i in range(len(centroids)):
    if(centroids[i].shape[0]%2==0):
        idx_delete.append(i)
print('done')

print('# images before remove even detection:',len(centroids))
centroids = np.delete(centroids,idx_delete)
print('# images after remove even detection:',len(centroids))


print('Estimating the pattern orientation in the camera plane...')
# fit with emcee fitting the vertex:
angle_values = np.zeros(len(preprocessed_images))
sigma_angles = np.zeros(len(preprocessed_images))

for i in range(len(preprocessed_images)):

    angle_values[i],sigma_angles[i] = get_pattern_orientation.fit_emcee_cen(centroids[i].T[0],centroids[i].T[1],sigma_centroids[i].T[0],sigma_centroids[i].T[1],sigma_x[i],sigma_y[i],preprocessed_images[i])

print('done')
