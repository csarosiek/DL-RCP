'''! @file fitting.py
'''
import logging
import os
import numpy as np
from scipy.optimize import curve_fit
from sklearn.linear_model import Lasso
from tools.common import *


def fit(data=None, target=None, method='lstsq'):
    """! @brief Fitting data array
    
    This function will fit an data array to a function or other arrays.

    @param data data array
    @param type type of fitting
    @return fitting result
    """
    logger = logging.getLogger(os.path.splitext(__file__)[0])
    data = __valid(data)
    if is_none(data):
        return
    # fit data to function
    if isinstance(target, str):
        return fit_func(data, target=target, method=method)
    # fit data to data
    target = __valid(target, 1)
    if isinstance(target, np.ndarray):
        return fit_data(data, target=target, method=method)
    # undefined fitting type
    logger.error('Undefined fitting type!')
    return
    

def fit_data(data=None, target=None, method='lstsq'):
    """! @brief Fitting data array to 1D data
    
    This function fits data array to 1D data.
    
    Data array (MxN):
        [x11 x12 . . . x1N]
        [. . . . . . . . .]
        [xM1 xM2 . . . xMN]
    
    Target array (1xM):
        [y1 y2  . . .  yM]
    
    Return coeffients (1xN):
        [c1 c2  . . .  cN]
    
    @param data data array
    @param target 1D data array
    @param method fitting methods
    @return fitting coefficients
    """
    logger = logging.getLogger(os.path.splitext(__file__)[0])
    # validate data
    data, target = __valid(data,2), __valid(target,1)
    if is_none(data) or is_none(target):
        logger.error('Unable to fit!')
        return
    nbins = min(len(data), len(target))
    if nbins < 2:
        logger.error('Not enough data points!')
        return
    data, target = data[:nbins,:], target[:nbins]
    # fit with numpy least square method
    if 'lstsq' in method:
        coeffs, r, rank, s = np.linalg.lstsq(data, target, rcond=None)
        return coeffs
    # fit with sklearn Lasso method
    elif 'Lasso' in method:
        positive = True if 'non-neg' in method else False
        lasso = Lasso(alpha=0.0001, precompute=True, max_iter=1000, positive=positive, random_state=9999, selection='random')
        return lasso.fit(data,target).coef_
    # undefined fitting method
    else:
        logger.error('Undefined fitting method!')
    return


def fit_func(data=None, target='poly5', method='lstsq'):
    """! @brief Fitting 2D array
    
    This function fits 2D array data to 1D data or poly function.
    
    @param data 2D data array
    @param target 1D data array or string ('polyN')
    @param method fitting methods
    @return fitting result
    """
    logger = logging.getLogger(os.path.splitext(__file__)[0])
    data = __valid(data, 2)
    if is_none(data):
        logger.error('Unable to fit!')
        return
    if data.shape[0] < 2 or data.shape[1] < 2:
        logger.error('Not enough data to fit!')
        return
    if data.shape[1] == 2:
        fit_1d()
    if data.shape[1] > 2: 
        if 'poly' in target:
            return fit_poly2d(data, order=int(target[-1]), method=method)
        elif 'gaus2d' in target:
            return fit_func2d(data, target=target, method=method)
        else:
            logger.error('Undefined fitting function!')
    return


def fit_1d():
    pass

    
def fit_poly2d(data=None, order=5, method='lstsq'):
    """! @brief Fitting 2D array to polynomial function
    
    This function fits 2D array data and return a 2D polynomial function.
    
    @param data 2D array of data
    @param order order of polynomial function
    @param method fitting methods (curve_fit, lstsq)
    """
    logger = logging.getLogger(os.path.splitext(__file__)[0])
    # validate data
    data = __valid(data, 2)
    if is_none(data):
        return
    if order < 1:
        logger.error('Polynomial order for fitting is smaller than 1. Unable to fit!')
        return 
    # flatten data
    xsize,ysize = np.shape(data)
    if data.shape[1] == 3:
        xdata, ydata = data[:,:-1], data[:,-1]
    else:
        xdata, ydata = __flatten(data)
    # fit with numpy least square method 
    if method == 'lstsq':
        return __fit_lstsq_poly(xdata, ydata, xsize, ysize, order).reshape(xsize, ysize)
    # fit with scipy curve_fit method
    elif method == 'curve_fit':
        return __fit_curve_poly(xdata, ydata, order).reshape(xsize, ysize)
    else:
        logger.error('Undefined fitting method!')
    return


def fit_func2d(data=None, target='gaus2d', method='lstsq'):
    """! @brief Fitting 2D array to a function
    
    This function fits 2D array data and return a 2D function.
    
    @param data 2D array of data
    @param target target function
    @param method fitting methods (curve_fit, lstsq)
    """
    logger = logging.getLogger(os.path.splitext(__file__)[0])
    # validate data
    data = __valid(data, 2)
    if is_none(data):
        return
    # flatten data
    xsize,ysize = np.shape(data)
    if data.shape[1] == 3:
        xdata, ydata = data[:,:-1], data[:,-1]
    else:
        xdata, ydata = __flatten(data)
    # fit with numpy least square method 
    if method == 'lstsq':
        return __fit_lstsq(xdata, ydata, xsize, ysize, order).reshape(xsize, ysize)
    # fit with scipy curve_fit method
    elif method == 'curve_fit':
        return __fit_curve(xdata, ydata, target).reshape(xsize, ysize)
    else:
        logger.error('Undefined fitting method!')
    return


def __fit_curve(xdata=None, ydata=None, target='gaus2d'):
    logger = logging.getLogger(os.path.splitext(__file__)[0])
    # perform fitting
    #func = __get_func(target)
    initial_guess = (400, 400, 100,100,1,5)
    coeffs, pcov = curve_fit(__lorentz2d, xdata, ydata, initial_guess)
    logger.debug('Fitted coefficients: ' + str(coeffs))
    # get fitted 2d results
    fitted_surf = __gaus2d(xdata, *coeffs)
    return fitted_surf


def __fit_curve_poly(xdata=None, ydata=None, order=2):
    """! @brief Fitting 2D array
    
    This function fits 2D array data by using scipy.curve_fit() and return a fitted array.
    
    @param data 2D array of data
    @param order order of polynomial function
    @return result array from fitted polynomial function
    """
    logger = logging.getLogger(os.path.splitext(__file__)[0])
    # create an array of coefficients
    a = np.zeros((order+1)*(order+2)//2)
    logger.debug('Number of coefficients: %d'%len(a))
    # perform fitting
    coeffs, pcov = curve_fit(__polyfunc, xdata, ydata, tuple(a))
    logger.debug('Fitted coefficients: ' + str(coeffs))
    # get fitted 2d results
    fitted_surf = __polyfunc(xdata, *coeffs)
    return fitted_surf


def __fit_lstsq_poly(xdata=None, ydata=None, xsize=0, ysize=0, order=5):
    """! @brief Fitting 2D array
    
    This function fits 2D array data by using numpy.linalg.lstsq() and return a polynomial function.
    
    @param data 2D array of data
    @param order order of polynomial function
    @return result array from fitted polynomial function
    """
    logger = logging.getLogger(os.path.splitext(__file__)[0])
    # create arrays of coefficients and x-y data
    kx, ky = order, order
    coeffs = np.ones((kx+1, ky+1))
    x = np.linspace(0, xsize-1, xsize)
    y = np.linspace(0, ysize-1, ysize)
    x, y = np.meshgrid(x, y, sparse=True)
    A = np.zeros((coeffs.size, xsize*ysize))
    for index, (j, i) in enumerate(np.ndindex(coeffs.shape)):
        if order is not None and i + j > order:
            arr = np.zeros((xsize,ysize))
        else:
            arr = coeffs[i, j] * x**i * y**j
        A[index] = arr.ravel()
    logger.debug(A.shape)
    # perform fitting        
    coeffs, r, rank, s = np.linalg.lstsq(A.T, ydata, rcond=None)
    logger.debug('Fitted coefficients: ' + str(coeffs))
    # get fitted 2d results
    fitted_surf = np.polynomial.polynomial.polygrid2d(x, y, coeffs.reshape((kx+1,ky+1)))
    return fitted_surf


def __valid(data=None, ndim=0):
    """! @brief Validate input array
    
    This function validates input data and returns False if the data is not numpy 2D array.
    
    @param data input data array
    @return False if data is not numpy 1D/2D array; otherwise True
    """
    logger = logging.getLogger(os.path.splitext(__file__)[0])
    if is_none(data):
        logger.debug('No data.')
        return
    if isinstance(data, list):
        return np.array(data)
    if not isinstance(data, np.ndarray):
        logger.debug('Data should be in numpy array format or list.')
        return
    if ndim == 1 and data.ndim != 1:
        logger.debug('Data is not 1D array.')
        return
    elif ndim == 2 and data.ndim != 2:
        logger.debug('Data is not 2D array.')
        return
    elif data.ndim != 1 and data.ndim != 2:
        logger.debug('Data is not 1D/2D array.')
        return
    return data


def __flatten(data=None):
    """! @brief Flatten 2D array for fitting
    
    This function flatten 2D array data for fitting,
    return 2 arrays (2 column xdata and 1 column ydata).
    
    @param data 2D array of data
    @return xdata, ydata
    """
    logger = logging.getLogger(os.path.splitext(__file__)[0])
    data = __valid(data, ndim=2)
    if is_none(data):
        return
    x_size,y_size = np.shape(data)
    x = np.linspace(0, x_size-1, x_size)
    y = np.linspace(0, y_size-1, y_size)
    x, y = np.meshgrid(x, y)
    x = x.ravel() 
    y = y.ravel() 
    xdata = np.vstack((x, y))
    ydata = data.ravel() 
    logger.debug('Shape of xdata: ' + str(xdata.shape))
    logger.debug('Shape of ydata: ' + str(ydata.shape))
    return xdata, ydata


def __get_func(target='gaus2d'):
    logger = logging.getLogger(os.path.splitext(__file__)[0])
    if target == 'gaus2d':
        return __gaus2d()


def __gaus2d(data, *coeff):
    x = data[0]
    y = data[1]
    x0 = 640 #coeff[0]
    y0 = 640 #coeff[1]
    sigma_x = coeff[0]
    sigma_y = coeff[1]
    theta = coeff[2]
    amplitude = coeff[3]
    offset = coeff[4]
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    return offset + amplitude*np.exp( - (a*((x-x0)**2) + 2*b*(x-x0)*(y-y0) + c*((y-y0)**2)))


def __lorentz2d(data, *coeff):
    x = data[0]
    y = data[1]
    x0 = coeff[0]
    y0 = coeff[1]
    w_x = coeff[2]
    w_y = coeff[3]
    amplitude = coeff[4]
    offset = coeff[5]
    return offset + amplitude / (1+((x-x0)/w_x)**2) / (1+((y-y0)/w_y)**2)


def __polyfunc(data, *coeff):
    x = data[0]
    y = data[1]
    f = coeff[0]
    order = int(np.sqrt(2*len(coeff))) - 1
    n = 1
    for i in range(1,order+1):
        for j in range(i, -1, -1):
            f += x**j*y**(i-j)*coeff[n]
            n += 1
    return f
