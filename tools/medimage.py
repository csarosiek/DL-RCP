'''! @file medimage.py
'''
import os
import logging
import numpy as np
import cv2
import datetime
#import torchio as tio
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import generate_uid, ImplicitVRLittleEndian, PYDICOM_IMPLEMENTATION_UID
from pydicom.sequence import Sequence
from pydicom.filereader import dcmread
from dataclasses import dataclass
#from scipy import stats, signal
#from scipy.spatial import Delaunay
from tools.common import is_none


def search_left(idx=0, val=0, arr=None, threshold=0):
    """!
    @brief Searching to lower index item

    This function will search for the lower index item in @a arr which has difference with the @a val 
    within @a threshold (|val-arr[i]| < threshold).

    @param idx starting index
    @param val reference value
    @param arr searching array
    @param threshold thresholdfor value difference
    @return index of array @b arr
    """
    i = idx
    while i > -1:
        if abs(val-arr[i]) < threshold:
            break
        i = i - 1
    return i


def search_right(idx=0, val=0, arr=None, threshold=0):
    """!
    @brief Searching to higher index item

    This function will search for the higher index item in @a arr which has difference with the @a val 
    within @a threshold (|val-arr[i]| < threshold).

    @param idx starting index
    @param val reference value
    @param arr searching array
    @param threshold thresholdfor value difference
    @return index of array @b arr
    """
    i = idx
    while i < len(arr):
        if abs(val-arr[i]) < threshold:
            break
        i = i + 1
    return i


def percent_diff(arr1, arr2):
    """!
    @brief Calculating percentage difference

    This function will calculate the percentage difference between two arrays.
    
    @param arr1 first array
    @param arr2 second array
    @return percentage difference array
    """
    return [ (val2-val1)/val1*100. for val1,val2 in zip(arr1,arr2) ]
        

def dta(arr1, arr2, threshold=0, pixel_size=0.336):
    """!
    @brief Calculating DTA

    This function will calculate the distance-to-agreement (DTA) between two arrays.
    
    @param arr1 first array
    @param arr2 second array
    @param threshold threshold of agreement value (two values are considered the same if their difference is within the threshold)
    @return DTA array
    """
    arr = []
    if not threshold:
        for idx1,val1 in enumerate(arr1):
            diff = [abs(val2-val1) for val2 in arr2]
            idx2 = diff.index(min(diff))
            arr.append(abs(idx1-idx2)*pixel_size)
    else:
        n = len(arr1)
        for idx1,val1 in enumerate(arr1):
            if idx1 < n//2:
                idx2 = search_right(idx1, val1, arr2, threshold)
                idx = search_left(idx1, val1, arr2, threshold)
            else:
                idx2 = search_left(idx1, val1, arr2, threshold)
                idx = search_right(idx1, val1, arr2, threshold)
            if abs(idx2-idx1) > abs(idx-idx1):
                idx2 = idx
            arr.append(abs(idx1-idx2)*pixel_size)
    return arr


def gamma_index(arr1, arr2, d_tol=3.0, r_tol=3.0, pixel_size=0.336):
    """!
    @brief Calculating Gamma Index

    This function will calculate the Gamma index between two arrays.
    
    @param arr1 first array
    @param arr2 second array
    @param d_tol dose tolerance
    @param r_tol distance tolerance
    @param pixel_size size of pixel
    @return gamma index array
    """
    return gamma_index_1D(arr1, arr2, d_tol, r_tol, pixel_size)


def gamma_index_1D(arr1, arr2, d_tol=3.0, r_tol=3.0, pixel_size=0.336):
    """!
    @brief Calculating Gamma Index

    This function will calculate the Gamma index between two 1D arrays.
    
    @param arr1 first 1D array
    @param arr2 second 1D array
    @param d_tol dose tolerance
    @param r_tol distance tolerance
    @param pixel_size size of pixel
    @return gamma index array
    """
    arr = []
    for idx1,val1 in enumerate(arr1):
        dd = [ (val2-val1)/val1*100. for val2 in arr2 ]
        dr = [ (idx1-idx2)*pixel_size for idx2 in range(len(arr2)) ]
        g = [ dd[i]*dd[i]/d_tol/d_tol + dr[i]*dr[i]/r_tol/r_tol for i in range(len(arr2)) ]
        arr.append(math.sqrt(np.min(g)))
    return arr


def transform_matrix(ref=None, orig=None):
    """!
    @brief Calculating transformation matrix 

    This function will calculate the transformation matix to transform
    the original matrix to the reference one.

    @param ref reference matrix
    @param orig original matrix
    @return transformation matrix
    """
    logger = logging.getLogger(__name__.split('.')[-1])
    if is_none(ref) or is_none(orig):
        logger.error('Incorrect input!')
        return
    n = ref.shape[0]
    X = np.hstack([orig, np.ones((n, 1))])
    Y = np.hstack([ref, np.ones((n, 1))])
    # find transformation matrix A
    A, res, rank, s = np.linalg.lstsq(X, Y, rcond=None)
    A[np.abs(A) < 1e-10] = 0
    return A


def transform(arr=None, A=None):
    """!
    @brief Transforming matrix 

    This function will transform the original image by applying the transformation matrix.

    @param arr input matrix
    @param A transformation matrix
    @return tranformed matrix
    """
    logger = logging.getLogger(__name__.split('.')[-1])
    if not isinstance(arr, np.ndarray) or not isinstance(A, np.ndarray):
        logger.error('Incorrect input!')
        return
    # size of matrix
    xsize, ysize, zsize = arr.shape
    logger.debug('Size of matrix: (%d,%d,%d)'%(xsize, ysize, zsize))
    # get point matrix
    grid_x, grid_y, grid_z = np.mgrid[0:xsize, 0:ysize, 0:zsize]
    points = np.array([grid_x, grid_y, grid_z]).reshape(3, -1).T
    logger.debug('Size of matrix: ' + str(points.shape))
    # calculate tranformed points
    X = np.hstack([points, np.ones((points.shape[0], 1))])
    trans = (np.dot(X, A)[:,:-1]).astype(int)
    # transform arr to new_arr
    values = arr.flatten()
    new_arr = griddata(trans, values, (grid_x, grid_y, grid_z), method='nearest')
    return new_arr


def perturbate(self, arr=None, coor=[0,0,0], xsize=3, ysize=3, zsize=1, val=0):
    """!
    @brief Creating pertubated matrix 

    This function will change some pixel values in dicom matrix.

    @param arr input matrix
    @param coor coordinate [x,y,z] 
    @param xsize number of pixels on x dimension will be changed
    @param ysize number of pixels on y dimension will be changed
    @param zsize number of pixels on z dimension will be changed
    @return new arr
    """
    if not isinstance(arr, np.ndarray):
        logger.error('Incorrect input!')
        return
    # change pixel values
    for i in range(coor[0],coor[0]+xsize):
        for j in range(coor[1],coor[1]+ysize):
            for k in range(coor[2],coor[2]+zsize):
                arr[i,j,k] = val
    return arr


def bad_pixel_correction(data=None, bpm=None):
    """!
    @brief Correct image for bad pixels

    This function will correct the image using a bad pixel map (BPM).
    
    @param data imput image (2D array)
    @param bpm bad pixel map
    @return new image
    """
    logger = logging.getLogger(__name__.split('.')[-1])
    bad_pixels = np.where(bpm > 0)
    for idx in bad_pixels:
        data[idx[0],idx[1]] = (np.sum(data[idx[0]-1:idx[0]+1, idx[1]-1:idx[1]+1])- data[idx[0],idx[1]])/8
    return data


def centroid_image(data=None):
    """!
    @brief Calculating center of mass

    This function will calculate the center of mass (centroid) from a set of points or 2D array.
    
    @param data set of points or 2D array
    @return center of mass
    """
    logger = logging.getLogger(__name__.split('.')[-1])
    # check data
    if not data:
        logger.error('No input!')
        return
    if not isinstance(data, list) and not isinstance(data, np.ndarray):
        logger.error('Invalid input!')
        return
    elif isinstance(data, list):
        data = np.array(data)
    if data.ndim != 2:
        logger.error('Incorrect number of dimension: ndim = ' + str(data.ndim))
        return
    xsize, ysize = data.shape
    # calculate centroid
    sum = np.sum(data)
    if sum == 0:
        logger.error('Zero sum values!')
        return
    cx = 0.0
    cy = 0.0
    for i in range(xsize):
        for j in range(ysize):
            cx += i*data[i,j]
            cy += j*data[i,j]
    return cx/sum, cy/sum


def centroid(data=None):
    """!
    @brief Calculating center of mass

    This function will calculate the center of mass (centroid) from a set of points or 2D array.
    
    @param data set of points or 2D array
    @return center of mass
    """
    logger = logging.getLogger(__name__.split('.')[-1])
    # check data
    if is_none(data):
        logger.error('Invalid input!')
        return
    if isinstance(data, list):
        data = np.array(data)
    # calculate weights for centroid
    if data.shape[1] > 2:
        w = data[:,2].T
    else:
        w = np.ones(data.shape[0])
    # calculate centroid
    sum = 0.0
    cx = 0.0
    cy = 0.0
    #print(w)
    for i in range(len(data)):
        cx += data[i][0]*w[i]
        cy += data[i][1]*w[i]
        sum += w[i]
        #print(w[i])
    if sum == 0.0:
        logger.error('Zero sum values!')
        return float('inf'), float('inf')
    return cx/sum, cy/sum


def get_point_inside(polygon=None):
    logger = logging.getLogger(__name__.split('.')[-1])
    # check data
    if not isinstance(polygon, np.ndarray):
        logger.error('Invalid input!')
        return
    xmin, xmax = np.amin(polygon[:,0]), np.amax(polygon[:,0])
    ymin, ymax = np.amin(polygon[:,1]), np.amax(polygon[:,1])
    zmin, zmax = np.amin(polygon[:,2]), np.amax(polygon[:,2])
    if zmin == zmax:
        x, y = np.meshgrid(np.arange(xmin-1, xmax+1), np.arange(ymin-1, ymax+1))
        x, y = x.flatten(), y.flatten()
        grid = np.vstack((x,y)).T
        delaunay = Delaunay(polygon[:,:2])
        p = delaunay.find_simplex(grid)>=0
    points = []
    for i in range(len(grid)):
        if p[i]:
            points.append([grid[i,1], grid[i,0], zmin])
    #logger.debug(points)
    return points


def create_mask(img=None, threshold=0.0):
    """!
    @brief Masking an image

    This function will create a mask from an image using threshold.
    
    @param img image
    @param threshold threshold for masking
    @return mask (1 inside and 0 outside mask)
    """
    logger = logging.getLogger(__name__.split('.')[-1])
    if not isinstance(img, np.ndarray):
        logger.error('Invalid image type!')
        return
    img[img < threshold] = 0.
    return np.divide(img,img)


def create_histogram(img=None, masking=False, threshold=0.0):
    """!
    @brief Creating histogram from an image

    This function will create a histogram from an image.
    
    @param img image
    @param masking True if creating histogram with mask; else False 
    @return histogram
    """
    logger = logging.getLogger(__name__.split('.')[-1])
    if not isinstance(img, np.ndarray):
        logger.error('Invalid image type!')
        return
    if masking:
        mask = create_mask(img, threshold)   
        mask = np.uint8(mask)
    else:
        mask = None
    img = cv2.normalize(src=img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return cv2.calcHist([img], channels=[0], mask=mask, histSize=[256], ranges=[0,256])
    

def create_image(points=None, xrange=[400,400], yrange=[400,400]):
    """!
    @brief Creating image for set of points

    This function will create a image for a set of points.
    
    @param points set of points
    @param size of the image
    @return image array
    """
    logger = logging.getLogger(__name__.split('.')[-1])
    if is_none(points):
        logger.error('No input for image!')
        return
    if is_none(xrange) or is_none(yrange):
        logger.error('Invalid size of image!')
        return
    xsize = xrange[1] - xrange[0]
    ysize = yrange[1] - yrange[0]
    img = np.zeros((xsize, ysize))
    for point in points:
        i = int(point[0] - xrange[0])
        j = int(point[1] - yrange[0])
        if len(point) > 2:
            img[i,j] = point[2]
        else:
            img[i,j] = 1
    return img


def diff_image(img1=None, img2=None, threshold=0., norm=True):
    """!
    @brief Difference image

    This function will calculate the difference image between two input images within threshold.
    
    @param img1 first image
    @param img2 second image
    @param threshold threshold level
    @return difference image
    """
    logger = logging.getLogger(__name__.split('.')[-1])
    if not isinstance(img1, np.ndarray) or not isinstance(img1, np.ndarray):
        logger.error('Incorrect image type!')
        return
    if img1.ndim != 2 or img2.ndim != 2:
        logger.error('Incorrect image dimension!')
        return
    # normalize
    if norm:
        scale = np.sum(img1)/np.sum(img2)
        img2 = img2 * scale
    # calculate difference
    img = abs(img1 - img2)
    img[img < threshold] = 0
    return img


def norm_image(img=None, pct_range=[1., 99.], interpolation='midpoint'):
    """!
    @brief Normalize image

    This function will normalize the input images within the percentile range.
    
    @param img input image
    @param pct_range percentile range
    @param interpolation interpolation method
    @return normalized image
    """
    logger = logging.getLogger(__name__.split('.')[-1])
    lower = np.percentile(img, pct_range[0], interpolation=interpolation)
    upper = np.percentile(img, pct_range[1], interpolation=interpolation)
    img = np.clip(img, lower, upper)
    im_max = np.amax(img)
    norm_img = (img - lower)/(upper - lower)
    return norm_img


def distance(source1=None, source2=None, type='Manhattan'):
    """!
    @brief Distance between images or histograms

    This function will calculate the histogram distance per non-zero bin between two inputs.
    
    @param source1 first array
    @param source2 second array
    @return distance
    """
    logger = logging.getLogger(__name__.split('.')[-1])
    if not isinstance(source1, np.ndarray) or not isinstance(source2, np.ndarray):
        logger.error('Incorrect histogram type!')
        return
    # calculate distance
    if type == 'Manhattan':
        diff = abs(source1 - source2)
        dist = np.sum(diff) / np.count_nonzero(diff)
        return dist
    else:
        logger.error('Undefined distance type!')
        return float('inf')


def KStest(data1=None, data2=None):
    """!
    @brief Calculating Kolmogorov-Smirnov statistic on 2 datasets or distribution test on 1 dataset

    This function will calculate the Kolmogorov-Smirnov statistic on 2 datasets or distribution test on 1 dataset.
    
    @param data1 first dataset
    @param data2 second dataset or distrution type (string)
    @return KS statistic
    """
    if is_none(data1) or is_none(data2):
        logger.error('Not enough input for KS test!')
        return
    if isinstance(data2, str):
        D, p_value = stats.kstest(data1[0], data2)
    else:
        D, p_value = stats.ks_2samp(data1[0], data2[0])
    return D, p_value


def denoise(img=None, weight=0.1, eps=1e-3, num_iter_max=200):
    """!
    Perform total-variation denoising on a grayscale image.
    Using Rudin, Osher and Fatemi algorithm.
    
    Ref: http://www.askaswiss.com/2016/12/how-to-denoise-images-in-python.html
        https://gist.github.com/mbeyeler/d9c4cff18e8b7324cd0f319d2841e72c
    
    @param img 2-D input data to be de-noised
    @param weight denoising weight, the greater `weight`, the more de-noising (at 
        the expense of fidelity to `img`).
    @param eps relative difference of the value of the cost function that determines
        the stop criterion. The algorithm stops when:
            (E_(n-1) - E_n) < eps * E_0
    @param num_iter_max : maximal number of iterations used for the optimization
    @returns de-noised array of floats.
    """
    # prepare parameters
    u = np.zeros_like(img)
    px = np.zeros_like(img)
    py = np.zeros_like(img)    
    nm = np.prod(img.shape[:2])
    tau = 0.125
    # loop
    i = 0
    while i < num_iter_max:
        u_old = u        
        # x and y components of u's gradient
        ux = np.roll(u, -1, axis=1) - u
        uy = np.roll(u, -1, axis=0) - u        
        # update the dual variable
        px_new = px + (tau / weight) * ux
        py_new = py + (tau / weight) * uy
        norm_new = np.maximum(1, np.sqrt(px_new **2 + py_new ** 2))
        px = px_new / norm_new
        py = py_new / norm_new
        # calculate divergence
        rx = np.roll(px, 1, axis=1)
        ry = np.roll(py, 1, axis=0)
        div_p = (px - rx) + (py - ry)        
        # update image
        u = img + weight * div_p        
        # calculate error
        error = np.linalg.norm(u - u_old) / np.sqrt(nm)        
        if i == 0:
            err_init = error
            err_prev = error
        else:
            # break if error small enough
            if np.abs(err_prev - error) < eps * err_init:
                break
            else:
                e_prev = error                
        # don't forget to update iterator
        i += 1
    return u



def detect_peak(img=None):
    '''
    from scipy.ndimage.filters import maximum_filter, minimum_filter
    from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
    
    neighborhood = generate_binary_structure(2,2)
    local_max = maximum_filter(img, footprint=neighborhood)==img
    #points = [[x,y,val] for (x, y), val in np.ndenumerate(img) if local_max[x,y] == True]
    xsize, ysize = img.shape
    points = []
    for i in range(xsize):
        for j in range(ysize):
            if local_max[i,j] == True:
                points.append([i,j,img[i,j]])
    '''
    from skimage.feature import peak_local_max
    from scipy.ndimage import gaussian_filter
    
    img = gaussian_filter(img, 8, mode='constant')
    pos = peak_local_max(img, threshold_abs=0.0005)
    points = []
    for i in range(len(pos)):
        points.append([pos[i,0], pos[i,1], img[pos[i,0],pos[i,1]]])
    return np.array(points)


def point_distance(points0=None, points1=None):
    from scipy.spatial.distance import directed_hausdorff
    
    #if (len(points0) - len(points1)) > 1:
    #    return float('inf')
    return directed_hausdorff(points0, points1)[0] + directed_hausdorff(points1, points0)[0]


def matching_distance(points0=None, points1=None):
    from scipy.spatial.distance import cdist
    from scipy.linalg import orthogonal_procrustes
    
    if is_none(points0) or is_none(points1):
        return float('inf')
    size = min(len(points0), len(points1))
    points0 = points0[:size,:]
    points1 = points1[:size,:]
    '''
    A = transform_matrix(points0[:,:-1], points1[:,:-1])
    points0 = points0[:,:-1]
    points1 = np.matmul(points1[:,:-1],A[:-1])[:,:-1]
    c0 = centroid(points0)
    c1 = centroid(points1)
    for i in range(size):
        points1[i][0] -= c1[0]-c0[0]
        points1[i][1] -= c1[1]-c0[1]
    '''
    c0 = centroid(points0)
    c1 = centroid(points1)
    for i in range(size):
        points0[i][0] -= c0[0]
        points0[i][1] -= c0[1]
        points1[i][0] -= c1[0]
        points1[i][1] -= c1[1]
    R, sca = orthogonal_procrustes(points0[:,:-1], points1[:,:-1])
    points0[:,:-1] = np.matmul(points0[:,:-1],R)
    return np.trace(cdist(points0[:,:-1], points1[:,:-1]))


def polygon_area(x=None, y=None):
    if is_none(x) or is_none(y):
        return 0
    correction = x[-1] * y[0] - y[-1]* x[0]
    main_area = np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:])
    return 0.5*np.abs(main_area + correction)


def color(i=0):
    rgb = [[234, 221, 202], [128,0,128], [224,255,255], [255,255,0], [255,105,180], [0,181,226], [218, 165, 32], [0,100,0], [0,255,255], [255,0,0], [255,0,255], [0,255,0], [0,0,255], [128,0,0], [165,42,42], [0, 204, 153]]
    return rgb[i]



@dataclass
class ROIData:
    mask: str
    color: []
    number: int
    name: str
    frame_of_reference_uid: int
    description: str = ''


def create_roi_contour(roi_data, series_data):
    roi_contour = Dataset()
    roi_contour.ROIDisplayColor = roi_data.color
    # contour sequence
    contour_sequence = Sequence()
    contours_coords = get_contours_coords(roi_data, series_data)
    for series_slice, slice_contours in zip(series_data, contours_coords):
        for contour_data in slice_contours:
            contour = create_contour(series_slice, contour_data)
            contour_sequence.append(contour)
    roi_contour.ContourSequence = contour_sequence
    # roi number
    roi_contour.ReferencedROINumber = str(roi_data.number)
    return roi_contour


def get_contours_coords(roi_data, series_data):
    transformation_matrix = get_pixel_to_patient_transformation_matrix(series_data)
    series_contours = []
    for i, series_slice in enumerate(series_data):
        mask_slice = roi_data.mask[:,:,i]
        if np.sum(mask_slice) == 0:
            series_contours.append([])
            continue
        # Get contours from mask
        contours = find_mask_contours(mask_slice)
        # Format for DICOM
        formatted_contours = []
        for contour in contours:
            contour = np.concatenate((np.array(contour), np.full((len(contour), 1), i)), axis=1)
            transformed_contour = apply_transformation_to_3d_points(contour, transformation_matrix)
            dicom_formatted_contour = np.ravel(transformed_contour).tolist()
            formatted_contours.append(dicom_formatted_contour)
        series_contours.append(formatted_contours)
    return series_contours


def get_pixel_to_patient_transformation_matrix(series_data):
    first_slice = series_data[0]
    offset = np.array(first_slice.ImagePositionPatient)
    row_spacing, column_spacing = first_slice.PixelSpacing
    slice_spacing = get_spacing_between_slices(series_data)
    row_direction, column_direction, slice_direction = get_slice_directions(first_slice)
    mat = np.identity(4, dtype=np.float32)
    mat[:3,0] = row_direction * row_spacing
    mat[:3,1] = column_direction * column_spacing
    mat[:3,2] = slice_direction * slice_spacing
    mat[:3,3] = offset
    return mat


def find_mask_contours(mask):
    approximation_method = cv2.CHAIN_APPROX_SIMPLE 
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, approximation_method)
    contours = list(contours)
    for i, contour in enumerate(contours):
        contours[i] = [[pos[0][0], pos[0][1]] for pos in contour]
    return contours


def create_contour(series_slice, contour_data):
    contour_image = Dataset()
    contour_image_sequence = Sequence()
    contour_image_sequence.append(contour_image)
    contour = Dataset()
    contour.ContourImageSequence = contour_image_sequence
    contour.ContourGeometricType = 'CLOSED_PLANAR'
    contour.NumberOfContourPoints = len(contour_data) / 3 
    contour.ContourData = contour_data
    return contour


def create_contour_image_sequence(series_data):
    contour_image_sequence = Sequence()
    for series in series_data:
        contour_image = Dataset()
        contour_image_sequence.append(contour_image)
    return contour_image_sequence


class RTStruct:
    
    def __init__(self, series_data, ds):
        self.series_data = series_data
        self.ds = ds
        self.frame_of_reference_uid = ds.ReferencedFrameOfReferenceSequence[-1].FrameOfReferenceUID

    def add_contour(self, mask, color=[], name='', description=''):
        roi_number = len(self.ds.StructureSetROISequence) + 1
        roi_data = ROIData(mask, color, roi_number, name, self.frame_of_reference_uid, description)
        self.ds.ROIContourSequence.append(create_roi_contour(roi_data, self.series_data))
        # StructureSetROISequence
        structure_set_roi = Dataset()
        structure_set_roi.ROINumber = roi_data.number
        structure_set_roi.ReferencedFrameOfReferenceUID = roi_data.frame_of_reference_uid
        structure_set_roi.ROIName = roi_data.name
        structure_set_roi.ROIDescription = roi_data.description
        #structure_set_roi.ROIGenerationAlgorithm = 0
        self.ds.StructureSetROISequence.append(structure_set_roi)
        # RTROIObservationsSequence
        rtroi_observation = Dataset()
        rtroi_observation.ObservationNumber = roi_data.number
        rtroi_observation.ReferencedROINumber = roi_data.number
        rtroi_observation.ROIObservationDescription = 'Type:Soft,Range:*/*,Fill:0,Opacity:0.0,Thickness:1,LineThickness:2,read-only:false'
        rtroi_observation.RTROIInterpretedType = ''
        rtroi_observation.ROIInterpreter = ''
        self.ds.RTROIObservationsSequence.append(rtroi_observation)

    def get_roi_mask_by_name(self, name):
        for structure_roi in self.ds.StructureSetROISequence:
            if structure_roi.ROIName == name:
                for roi_contour in self.ds.ROIContourSequence:
                    if str(roi_contour.ReferencedROINumber) == str(structure_roi.ROINumber):
                        contour_sequence = roi_contour.ContourSequence
                        break
                return create_series_mask_from_contour_sequence(self.series_data, contour_sequence)

    def save(self, file_path):
        file = open(file_path, 'w')
        self.ds.save_as(file_path)
        file.close()


class Transform:
    def __init__(self, conf):
        self.conf = conf
        
    def apply(self, image, masks):
        ncp = self.conf['ncp']
        images = np.stack(tuple([image] + masks))
        transform = tio.RandomElasticDeformation(num_control_points=ncp, locked_borders=2, image_interpolation= 'linear')
        aug_image = transform(images)
        aug_mask = [aug_image[i,:,:,:] for i in range(1,aug_image.shape[0])]
        aug_image = aug_image[0]
        threshold = 0.5
        zero = np.zeros((image[:,:,0].shape[0],image[:,:,0].shape[1]))
        for i in range(3):
            aug_image[:,:,i] = zero
            aug_image[:,:,aug_image.shape[2]-i-1] = zero
        aug_masks = []
        for mask in aug_mask:
            mask[mask < threshold] = 0
            mask[mask >= threshold] = 1
            for i in range(3):
                mask[:,:,i] = zero
                mask[:,:,image.shape[2]-i-1] = zero
            aug_masks.append(mask)
        return aug_image, aug_masks


def create_series_mask_from_contour_sequence(series_data, contour_sequence):
    # mask
    ref_dicom_image = series_data[0]
    mask_dims = (int(ref_dicom_image.Rows), int(ref_dicom_image.Columns), len(series_data))
    mask = np.zeros(mask_dims).astype(bool)
    # transformation matrix
    first_slice = series_data[0]
    offset = np.array(first_slice.ImagePositionPatient)
    row_spacing, column_spacing = first_slice.PixelSpacing
    slice_spacing = get_spacing_between_slices(series_data)
    row_direction, column_direction, slice_direction = get_slice_directions(first_slice)
    linear = np.identity(3, dtype=np.float32)
    linear[0,:3] = row_direction / row_spacing
    linear[1,:3] = column_direction / column_spacing
    linear[2,:3] = slice_direction / slice_spacing
    transformation_matrix = np.identity(4, dtype=np.float32)
    transformation_matrix[:3,:3] = linear
    transformation_matrix[:3,3] = offset.dot(-linear.T)
    # loop through each slice of the series
    for i, series_slice in enumerate(series_data):
        slice_contour_data = get_slice_contour_data(series_slice, contour_sequence)
        if len(slice_contour_data):
            mask[:, :, i] = get_slice_mask_from_slice_contour_data(series_slice, slice_contour_data,
                transformation_matrix)
    return mask


def get_slice_contour_data(series_slice, contour_sequence):
    slice_contour_data = []
    for contour in contour_sequence:
        for contour_image in contour.ContourImageSequence:
            try:
                if contour_image.ReferencedSOPInstanceUID == series_slice.SOPInstanceUID:
                    slice_contour_data.append(contour.ContourData)
            except:
                if abs(series_slice.SliceLocation - contour.ContourData[2]) < 0.5:
                    slice_contour_data.append(contour.ContourData)
    return slice_contour_data


def get_slice_mask_from_slice_contour_data(series_slice, slice_contour_data, transformation_matrix):
    slice_mask = create_empty_slice_mask(series_slice)
    for contour_coords in slice_contour_data:
        fill_mask = get_contour_fill_mask(series_slice, contour_coords, transformation_matrix)
        slice_mask[fill_mask == 1] = np.invert(slice_mask[fill_mask == 1])
    return slice_mask


def get_contour_fill_mask(series_slice, contour_coords, transformation_matrix):
    reshaped_contour_data = np.reshape(contour_coords, [len(contour_coords) // 3, 3])
    translated_contour_data = apply_transformation_to_3d_points(reshaped_contour_data, transformation_matrix)
    polygon = [np.around([translated_contour_data[:,:2]]).astype(np.int32)]
    fill_mask = create_empty_slice_mask(series_slice).astype(np.uint8)
    cv2.fillPoly(img=fill_mask, pts=polygon, color=1)
    return fill_mask


def create_empty_slice_mask(series_slice):
    mask_dims = (int(series_slice.Rows), int(series_slice.Columns))
    mask = np.zeros(mask_dims).astype(bool)
    return mask


def apply_transformation_to_3d_points(points, transformation_matrix):
    vec = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)
    return vec.dot(transformation_matrix.T)[:,:3]


def create_rtstruct_dataset(series_data):
    # Meta dataset
    file_meta = FileMetaDataset()
    file_meta.FileMetaInformationGroupLength = 202
    file_meta.FileMetaInformationVersion = b'\x00\x01'
    file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.481.3'
    file_meta.MediaStorageSOPInstanceUID = generate_uid() 
    file_meta.ImplementationClassUID = PYDICOM_IMPLEMENTATION_UID
    # File dataset
    ds = FileDataset('rtstruct', {}, file_meta=file_meta, preamble=b"\0" * 128)
    # add elements
    dt = datetime.datetime.now()
    ds.InstanceCreationDate = dt.strftime('%Y%m%d')
    ds.InstanceCreationTime = dt.strftime('%H%M%S.%f')
    ds.StructureSetLabel = 'RTstruct'
    ds.StructureSetDate = dt.strftime('%Y%m%d')
    ds.StructureSetTime = dt.strftime('%H%M%S.%f')
    ds.Modality = 'RTSTRUCT'
    ds.is_little_endian = True
    ds.is_implicit_VR = True
    ds.SOPClassUID = ds.file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
    ds.ApprovalStatus = 'UNAPPROVED'
    # add sequences
    ds.StructureSetROISequence = Sequence()
    ds.ROIContourSequence = Sequence()
    ds.RTROIObservationsSequence = Sequence()
    # add study and series information
    reference_ds = series_data[0] 
    ds.StudyDate = reference_ds.StudyDate
    ds.SeriesDate = getattr(reference_ds, 'SeriesDate', '')
    ds.StudyTime = reference_ds.StudyTime
    ds.SeriesTime = getattr(reference_ds, 'SeriesTime', '')
    ds.StudyDescription = getattr(reference_ds, 'StudyDescription', '')
    ds.SeriesDescription = getattr(reference_ds, 'SeriesDescription', '')
    ds.StudyInstanceUID = reference_ds.StudyInstanceUID
    ds.SeriesInstanceUID = generate_uid() 
    ds.StudyID = reference_ds.StudyID
    ds.SeriesNumber = "1"
    # add patient information
    reference_ds = series_data[0]
    ds.PatientName = getattr(reference_ds, 'PatientName', '')
    ds.PatientID = getattr(reference_ds, 'PatientID', '')
    ds.PatientBirthDate = getattr(reference_ds, 'PatientBirthDate', '')
    ds.PatientSex = getattr(reference_ds, 'PatientSex', '')
    ds.PatientAge = getattr(reference_ds, 'PatientAge', '')
    ds.PatientSize = getattr(reference_ds, 'PatientSize', '')
    ds.PatientWeight = getattr(reference_ds, 'PatientWeight', '')
    # add frame of ref sequence
    reference_ds = series_data[0]
    refd_frame_of_ref = Dataset()
    refd_frame_of_ref.FrameOfReferenceUID = generate_uid()
    rt_refd_series = Dataset()
    rt_refd_series.SeriesInstanceUID = reference_ds.SeriesInstanceUID
    rt_refd_series.ContourImageSequence = create_contour_image_sequence(series_data)
    rt_refd_series_sequence = Sequence()
    rt_refd_series_sequence.append(rt_refd_series)
    rt_refd_study = Dataset()
    rt_refd_study.ReferencedSOPClassUID = '1.2.840.10008.3.1.2.3.1'
    rt_refd_study.ReferencedSOPInstanceUID = reference_ds.StudyInstanceUID
    rt_refd_study.RTReferencedSeriesSequence = rt_refd_series_sequence
    rt_refd_study_sequence = Sequence()
    rt_refd_study_sequence.append(rt_refd_study)
    refd_frame_of_ref.RTReferencedStudySequence = rt_refd_study_sequence
    ds.ReferencedFrameOfReferenceSequence = Sequence()
    ds.ReferencedFrameOfReferenceSequence.append(refd_frame_of_ref)
    # return file dataset
    return ds


def get_slice_directions(series_slice):
    orientation = series_slice.ImageOrientationPatient
    row_direction = np.array(orientation[:3])
    column_direction = np.array(orientation[3:])
    slice_direction = np.cross(row_direction, column_direction)
    return row_direction, column_direction, slice_direction


def get_slice_position(series_slice):
    _, _, slice_direction = get_slice_directions(series_slice)
    return np.dot(slice_direction, series_slice.ImagePositionPatient)


def get_spacing_between_slices(series_data):
    if len(series_data) > 1:
        first = get_slice_position(series_data[0])
        last = get_slice_position(series_data[-1])
        return (last - first) / (len(series_data) - 1)
    return 1.0


def load_image_series(dicom_path):
    series_data = []
    for root, _, files in os.walk(dicom_path):
        for file in files:
            try:
                ds = dcmread(os.path.join(root, file), force=True)
                if hasattr(ds, 'pixel_array'):
                    series_data.append(ds)
            except Exception:
                continue
    if len(series_data) > 1:
        series_data.sort(key=get_slice_position, reverse=False)
    return series_data


def create_rtstruct(dicom_path):
    series_data = load_image_series(dicom_path)
    ds = create_rtstruct_dataset(series_data)
    return RTStruct(series_data, ds)


def find_rt_struct(path, mark=''):
    l = get_file_list(path, fullname=True)
    for i in l:
        if mark != '' and mark not in i:
            continue
        if os.path.getsize(i) > 3e5:
            return i
    return


def get_masks(path, organs):
    inputs = get_listdir(path)
    logging.debug(str(inputs))
    masks, rtstruct = {}, None
    for d in inputs:
        if os.path.isdir(path+'/'+d):
            rtstruct = create_from(path+'/'+d, rt_struct_path=find_rt_struct(path+'/'+d))
            for organ in organs:
                masks.update({organ: rtstruct.get_roi_mask_by_name(organ)})
        elif d.endswith('.mat'):
            name = d[:-4]
            f = scipy.io.loadmat(path+'/'+d)
            masks.update({name: f['mask']})
    return masks


def save_masks(masks, folder='', filename='rtstruct.dcm'):
    rtstruct = create_new(folder)
    for i, organ in enumerate(masks.keys()):
        rtstruct.add_roi(mask=masks[organ].astype(bool), color=color(i), name=organ)
    rtstruct.save(folder+'/'+filename)
    logging.info(folder+'/'+filename)


def create_mask(series_data, contour_sequence):
    # mask
    ref_dicom_image = series_data#[0]
    mask_dims = (int(ref_dicom_image.Rows), int(ref_dicom_image.Columns), len(series_data))
    mask = np.zeros(mask_dims).astype(bool)
    # transformation matrix
    first_slice = series_data#[0]
    offset = np.array(first_slice.ImagePositionPatient)
    row_spacing, column_spacing = first_slice.PixelSpacing
    slice_spacing = get_spacing_between_slices(series_data)
    row_direction, column_direction, slice_direction = get_slice_directions(first_slice)
    linear = np.identity(3, dtype=np.float32)
    linear[0,:3] = row_direction / row_spacing
    linear[1,:3] = column_direction / column_spacing
    linear[2,:3] = slice_direction / slice_spacing
    transformation_matrix = np.identity(4, dtype=np.float32)
    transformation_matrix[:3,:3] = linear
    transformation_matrix[:3,3] = offset.dot(-linear.T)
    # loop through each slice of the series
    for i, series_slice in enumerate(series_data):
        slice_contour_data = get_slice_contour_data(series_slice, contour_sequence)
        if len(slice_contour_data):
            mask[:, :, i] = get_slice_mask_from_slice_contour_data(series_slice, slice_contour_data,
                transformation_matrix)
    return mask

