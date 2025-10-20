'''! @file data.py
    Module for data classes:
    - BaseData
    - DicomData
    - VarianData
    - MatlabData
'''
import os
import glob
import logging
#import pydicom
import struct
import numpy as np
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import generate_uid, ImplicitVRLittleEndian, PYDICOM_IMPLEMENTATION_UID
from pydicom.sequence import Sequence
from pydicom.filereader import dcmread
#from scipy.io import loadmat
from abc import ABC, abstractmethod
from tools.common import *
from tools.medimage import *


"""
########################## BaseData ##########################
"""

class BaseData(ABC):
    """! @brief Base class for data analysis

    This is a base class for other data analysis classes. It provides common tools for reading, validating
    and parsing data.
    """


    def __init__(self, source=None):
        """! 
        @brief The constructor

        This function initializes @ref _data variable and read data from the file.
        The data will be stored to @ref _data variable. It can also get data from another BaseData object.
        
        @param source input file path or another BaseData object
        """
        ## Data member
        self._data = None
        # load data from file
        if source: self.load(source, save_copy=True)

    
    @property
    def data(self):
        """!
        @brief Getting @ref _data

        This function returns the @ref _data. 

        @return @ref _data 
        """
        return self._data
    

    @data.setter
    def data(self, data=None):
        """!
        @brief Setting @ref _data value

        This function sets an array to @ref _data 

        @param data numpy array 
        """
        if isinstance(data, list):
            self._data = np.array(data)
        elif isinstance(data, np.ndarray):
            self._data = data
        else:
            self._logger.error('Cannot convert to numpy array!')


    def load(self, source=None, save_copy=False):
        """!
        @brief Getting data from input

        If the input is a string, this function will read the input file and parse information from it by 
        calling @ref read(). If the input is an BaseData object, this function will get data from this object.
        
        The data will also be stored in the @ref _data variable if the parameter @a save_copy set to True.

        @param source input file path or another BaseData object 
        @param save_copy True if you want to save a copy to @ref _data; otherwise False
        """
        if isinstance(source, BaseData):
            data = source.data
        elif isinstance(source, str):
            data = self.read(source)
        elif isinstance(source, list):
            data = [self.read(file) for file in source]
        else:
            self._logger.warning('No source info specified!')
            return
        # save copy to _data
        if save_copy: self._data = data
        return data


    @abstractmethod
    def read(self, infile=''):
        """!
        @brief Abstract method for reading data from input file

        @param infile input file path
        """
        pass


    def get_data(self):
        """!
        @brief Getting @ref _data

        This function calls data(). 

        @return @ref data() 
        """
        return self.data


    def valid(self, data=None):
        """!
        @brief Checking data

        This function checks if input data is None or not.
        If None, replace it by @ref _data

        @return data 
        """
        return self._data if is_none(data) else data 


    def is_image(self, data=None):
        """!
        @brief Checking if input data is an image

        This function checks if the input data we read in is an image or not.
        In case no input specified, the @ref _data will be checked.

        @param data input data
        @return True if the input is numpy array type, otherwise False
        """
        data = self.valid(data)
        if not isinstance(data, np.ndarray):
            self._logger.debug('Data is not numpy array type!')
            return False
        if data.ndim != 2:
            self._logger.debug('Dimension of data is not equal to 2!')
            return False
        return True



"""
########################## DicomData ##########################
"""


class DicomData(BaseData):
    """! @brief Tools for reading and analyzing DICOM data files.

    This class inherits from the BaseData class. It provides tools for reading and analyzing DICOM data files.
    The DICOM data will be read by using pydicom tool and analyzed further by the methods in this class.
    """


    def __init__(self, source=None):
        """! 
        @brief The constructor

        This function will initially get the dicom file path and read data from the file,
        the data will be stored to @ref _data variable. It can also get data from another DicomData object.

        @param source input file path or another DicomData object
        """
        ## Logger object
        self._logger = logging.getLogger('DicomData')
        self.thickness = []
        self.orig = []
        self.series_data = []
        # BaseData initialization
        super().__init__(source)


    def get_pixel_array(self, data=None):
        """!
        @brief Getting pixel array

        This function will return the pixel array of @ref _data. 

        @return pixel array 
        """
        # check data
        data = self.valid(data)
        if is_none(data):
            return
        # check different types of data
        if isinstance(data, DicomData):
            data = self.get_pixel_array(data.get_data())
        elif isinstance(data, FileDataset):
            data = data.pixel_array
        elif isinstance(data, np.ndarray):
            data = data
        elif isinstance(data, list):
            data = self.merge_image(data)
        self._logger.debug('Pixel data shape: ' + str(data.shape))
        return data


    def get_pixel_thickness(self, data=None):
        """!
        @brief Getting pixel thickness

        This function will return the pixel thickness of @ref _data. 

        @return pixel array 
        """
        return self.thickness
        # check data
        data = self.valid(data)#[0]
        if is_none(data):
            return
        xsize = float(data.PixelSpacing[0])
        ysize = float(data.PixelSpacing[1])
        zsize = float(data.SliceThickness)
        return xsize, ysize, zsize


    def get_origin(self, data=None):
        """!
        @brief Getting image origin

        This function will return the origin of @ref _data. 

        @return origin 
        """
        return self.orig

    
    def get_organ_mask(ref_image=None,organ='', struct_file=None, size=(512, 512, 10), orig=[0.,471.07,-195.], thickness=[0.9219,0.9219,3.0]):
        """!
        @brief Getting ROI structure from RTSTRUCT file

        """
        if organ == '':
            return 
        # read RTSTRUCT file
        struct_data = dcmread(struct_file)
        # get organ id
        organ_id = None
        organ_dict = self.__get_organ_dict(struct_data)
        for key in organ_dict:
            if organ.lower() in key.lower():
                organ_id = organ_dict[key]
                break
        if organ_id == None:
            return
        # get pixel thicknesses
        #xsize, ysize, zsize = self.get_pixel_thickness()
        mask = np.zeros(size)
        # loop over contour sequence
        for ROI in struct_data.ROIContourSequence:
            #self._logger.debug(ROI.ReferencedROINumber)
            if ROI.ReferencedROINumber == organ_id:
                contour_sequence = ROI.ContourSequence
                #print(contour_sequence)
                break
        return create_mask(ref_image, contour_sequence)


    def merge_image(self, dicoms=[]):
        """!
        @brief Merging list of dicom image

        This function will merge a list of dicom image. 

        @param data list of dicom image
        @return pixel array 
        """
        # get a list of arrays
        data = []
        thickness = []
        for dicom in dicoms:
            if self.thickness == []:
                self.orig = [float(dicom.ImagePositionPatient[0]), float(dicom.ImagePositionPatient[1]), float('nan')]
                self.thickness = [float(dicom.PixelSpacing[0]), float(dicom.PixelSpacing[1]), float(dicom.SliceThickness)]
            data.append([float(dicom.get('SliceLocation', 0.0)), dicom.pixel_array])
            self._logger.debug(float(dicom.get('SliceLocation', 0.0)))
            self.series_data.append(dicom)
        # sort list
        data = sorted(data)
        self.series_data.sort(key=get_slice_position, reverse=False)
        # append arrays in list to z axis
        new_data = []
        loc = []
        for i,item in enumerate(data):
            loc.append(item[0])
            if i == 0:
                new_data = item[1]
            else:
                new_data = np.dstack((new_data, item[1]))
        self._logger.debug('List of ordered slice locations: ' + str(loc))
        self._logger.debug('Shape of new array: ' + str(new_data.shape))
        self.orig = [self.orig[0], self.orig[1],loc[0]]
        return new_data
 
 
    def read(self, infile=''):
        """!
        @brief Getting dicom data from file 

        This function will read the input dicom file and return the data inside that file.
        The accepted file type is .dcm file.

        @param source input file path
        @return dicom data
        """
        # check file path
        if not check_file(infile): return
        # read data from file
        data = dcmread(infile)
        if not data:
            self._logger.error('Undefined dicom file type!')
            return
        self._logger.info('Loaded dicom file ' + infile)
        self.__print_debug(data)
        return data


    def read_list(self, dicom_files=[]):
        """!
        @brief Getting dicom data from a directory 

        This function will read all the input dicom files and return the data inside a directory.
        The accepted file type is .dcm file.

        @param source path
        @return dicom data
        """
        # check file path
        #if not check_dir(path, new_creation=False): return
        # read data from file
        #dicom_files = get_file_list(path, ext='dcm', fullname=True)
        data = []
        for file in dicom_files:
            dicom = self.read(file)
            if 'PixelData' in dicom.dir():
                data.append(dicom)        
        return self.merge_image(data)


    def save(self, data=None, outfile='dicom.dcm', outdir='Dicom/Test', options={}):
        """!
        @brief Saving dicom data from file 

        This function will save data to a dicom file.

        @param arr array of dicom image data
        @param outfile name of output file
        @param outdir path of output directory
        """
        # check data and file path
        data = self.valid(data)
        if is_none(data):
            return
        outdir = check_dir(outdir)
        filename = outdir + outfile
        # write data to file
        if isinstance(data, FileDataset):
            data.save_as(filename)
        elif isinstance(data, np.ndarray):
            if data.ndim == 2: 
                ds = self.dicom(self._data, data)
                ds.save_as(filename)
            elif data.ndim == 3:
                if isinstance(self._data, list):
                    for i in range(len(self._data)):
                        ds = self.dicom(self._data[i], data[:,:,i])
                        ds = self.set_meta(ds, options.get('meta',{}))
                        ds.save_as(outdir+'dicom_%d.dcm'%i)
            else:
                self._logger.error('Invalid data!')
                return
        elif isinstance(data, list):
            for i in range(len(data)):
                data[i] = self.set_meta(data[i], options.get('meta',{}))
                self.save(data[i], outfile='dicom_%d.dcm'%i, outdir=outdir, options=options)
        else:
            self._logger.error('Invalid dicom data to be saved!')
            return
        self._logger.info('Dicom data saved at directory ' + outdir)


    def dicom(self, ds=None, data=None):
        """!
        @brief Converting data array to dicom 

        This function will convert data array to dicom dataset type.

        @param ds dicom dataset
        @param data pixel matrix
        @return dicom dataset
        """
        if not isinstance(ds, FileDataset):
            self._logger.error('Invalid dicom FileDataset!')
            return
        if not isinstance(data, np.ndarray):
            self._logger.error('Invalid dicom pixel array!')
            return
        if data.ndim != 2 and data.ndim != 3:
            self._logger.error('Invalid pixel array dimension!')
            return
        ds.PixelData = data.tostring()#tobytes()
        return ds


    def set_meta(self, ds=None, meta={}):
        """!
        @brief Setting information for dicom file

        This function will set relevant information to a dicom file.

        @param options information of dicom data to be set
        @return dataset after setting info 
        """
        if not isinstance(meta,dict) or is_none(meta):
            return ds
        for key, val in meta.items():
            setattr(ds, key, val)
        return ds


    def add_air(self, array=None, mask=None, step=[1,1,0], threshold=500):
        points = np.where(mask==True)
        for p in range(len(points[0])):
            #self._logger.debug('%f'%(array[points[0][p], points[1][p], points[2][p]]))
            if array[points[0][p], points[1][p], points[2][p]] <= threshold:
                array[points[0][p], points[1][p], points[2][p]] = 0
        return array


    def __get_organ_dict(self, data=None):
        if not isinstance(data, FileDataset):
            return
        if data.Modality != 'RTSTRUCT':
            return
        dict = {}
        for ROI in data.StructureSetROISequence:
            dict.update({ROI.ROIName : ROI.ROINumber})
        return dict


    def __convert_coord(self, data=None, to='index', orig=[0.,471.07,-195.], thickness=[0.9219,0.9219,3.0]):
        if not isinstance(data, np.ndarray):
            return
        if data.shape[0] < 3 or data.shape[1] != 3:
            return
        # start converting
        if to == 'index':
            indices = []
            for coord in data:
                indices.append([int((coord[i]-orig[i])/thickness[i]) for i in range(3)])
        #self._logger.debug(indices)
        return np.array(indices)


    def __print_debug(self, data=None):
        """!
        @brief Printing debug information 

        This function will print debug information from dicom data.

        @param data input dicom data
        """
        if not data:
            return
        self._logger.debug('>>> Storage type......: ' + str(data.SOPClassUID))
        patient = data.PatientName
        self._logger.debug('>>> Patient\'s name....: ' + patient.family_name + ', ' + patient.given_name)
        self._logger.debug('>>> Patient ID........: ' + data.PatientID)
        self._logger.debug('>>> Modality..........: ' + data.Modality)
        self._logger.debug('>>> Study Date........: ' + data.StudyDate)
        self._logger.debug('>>> Frame of reference: ' + str(data.get('FrameOfReferenceUID', '(missing)')))
        # CT images
        if data.Modality == 'CT' or data.Modality == 'MR':
            self._logger.debug('>>> Bits stored.......: ' + str(data.BitsStored))
            self._logger.debug('>>> Image size........: {rows:d} x {cols:d}, {size:d} bytes'.format(rows=int(data.Rows), cols=int(data.Columns), size=len(data.PixelData)))
            self._logger.debug('>>> Pixel spacing.....: ' + str(data.PixelSpacing))
            self._logger.debug('>>> Image orientation.: ' + str(data.ImageOrientationPatient))
            self._logger.debug('>>> Image position....: ' + str(data.ImagePositionPatient))
            self._logger.debug('>>> Slice thickness...: ' + str(data.SliceThickness))
            self._logger.debug('>>> Slice location....: ' + str(data.SliceLocation))
        # Treatment plan
        elif data.Modality == 'RTPLAN':
            # get beam meterset
            meterset = []
            for frac in data.FractionGroupSequence:
                for beam in frac.ReferencedBeamSequence:
                    meterset.append(int(round(beam.BeamMeterset)))
            # loop through all beams
            beams = data.BeamSequence
            self._logger.debug("{name:^20s} {num:^8s} {t:^8s} {gantry:^8s} {ssd:^11s} {ncp:^8s} {mu:^8s}".format(name="Beam name", num="Number", t="Type", gantry="Gantry", ssd="SSD (cm)", ncp="NumCP", mu="Meterset"))
            for i,beam in enumerate(beams):
                btype = beam.BeamType
                cp0 = beam.ControlPointSequence[0]
                ncp = beam.NumberOfControlPoints
                SSD = float(cp0.SourceToSurfaceDistance / 10)
                self._logger.debug("{b.BeamName:^20s} {b.BeamNumber:8d} {t:8s} {gantry:8.1f} {ssd:8.1f} {ncp:8d} {mu:8d}".format(b=beam, t=btype, gantry=cp0.GantryAngle, ssd=SSD, ncp=ncp, mu=meterset[i]))
        # ROI structure
        elif data.Modality == 'RTSTRUCT':
            for ROI in data.StructureSetROISequence:
                self._logger.debug('ROI: number = ' + str(ROI.ROINumber) + ', name = '+ ROI.ROIName)
            for ROI in data.ROIContourSequence:
                self._logger.debug(ROI.ReferencedROINumber)
                for contour in ROI.ContourSequence:
                    self._logger.debug(contour.ContourGeometricType)
                    self._logger.debug(contour.NumberOfContourPoints)
                    self._logger.debug(contour.ContourData)


"""
########################## VarianData ##########################
"""

class VarianData(BaseData):
    """! @brief Tools for reading and analyzing Varian data files.

    This class inherits from the BaseData class. It provides tools for reading and analyzing Varian data files.
    Data files can be read by this class:
        - XIM images from TrueBeam linac
    """


    def __init__(self, source=None):
        """! 
        @brief The constructor

        This function will initially get the Varian file (.xim) path and read data from the file,
        the data will be stored to @ref _data variable. It can also get data from another VarianData object.

        @param source input file path or another VarianData object
        """
        ## Logger object
        self._logger = logging.getLogger('VarianData')
        # BaseData initialization
        super().__init__(source)
        

    def read(self, infile=''):
        """!
        @brief Getting data from file 

        This function reads the input file and return the data inside that file.
        The accepted file type is '.xim' file.

        @param infile input file path
        @return data
        """
        # check file path
        if not check_file(infile): return
        # read data from file
        if infile.lower().endswith('.xim'):
            return self.__read_xim(infile)
        else:
            self._logger.error('Undefined input file type!')
            return


    def __read_xim(self, infile=''):
        """!
        @brief Getting Varian XIM image data

        This function reads the XIM input file and return an image (np.ndarray).

        @param infile input file path
        @return data
        """
        # check file path
        infile = check_file(infile) 
        if not infile:
            return
        # open infile
        file = open(infile, 'rb')
        header = self.__read_ximHeader(file)
        data = self.__read_ximData(file, header)
        # print debug info
        self._logger.debug(header)
        return data
        
    
    def __read_ximHeader(self, file=None):
        """!
        @brief Getting Varian XIM image header

        This function reads the file object and return header info.

        @param file file object
        @return header info (dictionary)
        """
        if not file:
            return
        ximHeader = dict()  # Dictionary of header values
        # on Py3 we need to decode so that we can later use the replace
        ximHeader['FormatIdentifier'] = file.read(8).decode()
        ximHeader['FormatVersion'] = struct.unpack('<i', file.read(4))[0]
        ximHeader['Width'] = struct.unpack('<i', file.read(4))[0]
        ximHeader['Height'] = struct.unpack('<i', file.read(4))[0]
        ximHeader['BitsPerPixel'] = struct.unpack('<i', file.read(4))[0]
        ximHeader['BytesPerPixel'] = struct.unpack('<i', file.read(4))[0]
        ximHeader['CompressionIndicator'] = struct.unpack('<i', file.read(4))[0]
        self._logger.debug('Xim header: '+str(ximHeader))
        return ximHeader


    def __read_ximData(self, file=None, header=None):
        """!
        @brief Getting Varian XIM image data

        This function reads the file object and return data array.

        @param file file object
        @param header header info (dictionary)
        @return data XIM data
        """
        if not file or not header:
            return
        w = header['Width']
        h = header['Height']
        bpp = header['BytesPerPixel']
        # Image pixels are stored uncompressed in the xim image file.
        if not header['CompressionIndicator']:
            # Read in int4 (32 bit) image pixe values
            #uncompressedPixelBufferSize = struct.unpack('<%i', file.read(4))[0]
            uncompressedPixelBufferSize = struct.unpack('<i', file.read(4))[0]
            # Read in pixel values in 1D array
            uncompressedPixelBuffer = np.asarray(struct.unpack('<%ii' % (uncompressedPixelBufferSize / 4), file.read(uncompressedPixelBufferSize)))
        # Decompress the pixelData using HND decompression algorithm.
        else:
            LUTSize = struct.unpack('<i', file.read(4))[0]  # Lookup table size
            LUT = np.asarray(struct.unpack('<%iB' % LUTSize, file.read(LUTSize)))  # Lookup table
            compressedBufferSize = struct.unpack('<i', file.read(4))[0]  # Compressed pixel buffer size
            uncompressedPixelBuffer = self.__uncompressHnd(file, w, h, bpp, LUT)  # Uncompress the pixel data
            uncompressedBufferSize = struct.unpack('<i', file.read(4))[0]  # Uncompressed pixel image size
        # Reshape uncompressed image into 2D array
        uncompressedImage = np.reshape(uncompressedPixelBuffer, (h, w))
        return uncompressedImage
 
 
    def __uncompressHnd(self, file, w, h, bpp, lut):
        """!
        @brief Uncompressing image data
        
        This function uncompresses the xim file based on HND algorithm. The first row and the 
        first pixel of the second row are stored uncompressed. The remainders 
        of the pixels are compressed by storing only the difference between 
        neighboring pixels.
        
        E.g. consider the following hypothetical 12 pixel image:
                R11    R12    R13    R14
                R21    R22    R23    R24
                R31    R32    R33    R34
        Pixels R11 through R14 and R21 are stored uncompressed, while pixels 
        R22 through R34 are compressed by storing only the difference: 
        
        diff = R11 + R22 - R21 - R12
        
        Exploiting the fact that most images exhibit similarity in neighboring 
        pixel values, the above difference can be stored using fewer bytes, 
        e.g. 1, 2 or 4 bytes.
         
        For decompression, the algorithm needs to know the byte size of each 
        stored difference. To accomplish this, a lookup table is placed at the 
        beginning of the image. The lookup table contains a 2-bit flag for each 
        pixel which defines the byte size for each compressed pixel difference. 
        So a flag value of 0 means the difference fits into one byte while 
        1 and 2 mean a two and four byte difference respectively.
          
        @param w uncompressed image width
        @param h uncompressed image height
        @param bpp byte per pixel
        @param lut look up table
        @return uncompressed image data
        """
        self._logger.debug('uncompressHnd called, with args: w= {} h={} bpp={} {}'.format(w,h,bpp,lut))
        # Initialize uncompressed image variable
        imagePix = np.zeros((h * w), dtype='int32')
        # Read in the first row
        # ... and the first pixel of the second row
        # which is why we do w + 1
        ind = 0  # Index variable
        for i in range(w + 1):
            imagePix[ind] = struct.unpack('<i', file.read(4))[0]
            ind += 1
        # Calculate current pixel value based  on "diff"
        # and adjacent pixel values as following:
        # R22 (current pixel) = diff + R21 + R12 - R11
        for byte_size in self.__lut_reader(w, h, lut):
            # read in appropriate number of bytes
            diff = self.__char2Int(file, byte_size)
            # R22 (current pixel) = diff + R21 + R12 - R11
            if ind < 1638400: #1416100:   #### fix problem
                imagePix[ind] = diff + imagePix[ind - 1] + imagePix[ind - w] - imagePix[ind - w - 1]
            ind += 1
        self._logger.debug('processed {} pixels'.format(len(imagePix)))
        return imagePix


    def __lut_sizer(self, byte, maximum = None):
        """!
        @brief Getting lut size
        
        Each lut byte contains 4 two-bit flags, except for at the tail end,
        there may be some partial flags left over

        @param byte  the byte to be decoded into 4 windowsx
        @param maximum  the maximum number of windows to pull 

        annoyingly these suckers seem to be put in backwards for some reason
        """
        # lookup table 'bit' flag to byte conversion
        byte_conversion = {'00':1, '01':2, '10':4}
        bit_flags = '{0:08b}'.format(byte)
        for count, idx in enumerate(range(6, -1, -2)):
            #if maximum and count>=maximum:
            #    raise StopIteration
            pair = bit_flags[idx:idx+2]
            yield byte_conversion[pair]


    def __lut_reader(self, w, h, lut):
        """!
        @brief Reading lut info
        
        This function reads the lookup table and generate bite sizes for latter diff

        Assisted by @ref __lut_sizer which actually parses each lut byte, this function
        just wraps the @ref __lut_sizer and yield from it the approprate byte size for
        each diff in sequence

        @param w Uncompressed image width
        @param h Uncompressed image height
        @param lut look up table
        """
        # Determine the number of unused 2-bit flag fields
        # in the last byte of the look up table
        # was dividing by bpp, but should be by 4, as there are 4 flags per 8
        # bit byte, regardless of underlying bytes per pixel
        complete_bytes, partial_bytes = divmod((w * (h - 1) - 1), 4)
        for count, b in enumerate(lut):
            if count>= complete_bytes:
                # we have come to the end, so only yield part of the last lut
                # byte
                yield from self.__lut_sizer(b, partial_bytes)
            else:
                yield from self.__lut_sizer(b)


    def __char2Int(self, file, sz):
        """!
        @brief Converting little-endian chars to a 32 bit integer
        
        Character size can be 1 byte: signed char 
                              2 bytes : short
                              4 bytes : int4 
        
        @param sz character size
        @return value
        """
        if sz == 1:
            value = struct.unpack('<b', file.read(1))[0]  # b: signed char
        elif sz == 2:
            value = struct.unpack('<h', file.read(2))[0]  # h: short
        elif sz == 4:
            value = struct.unpack('<i', file.read(4))[0]  # i: int4
        return value
        

"""
########################## MatlabData ##########################
"""

class MatlabData(BaseData):
    """! @brief Tools for reading and analyzing Matlab data files.

    This class inherits from the BaseData class. It provides tools for reading and analyzing Matlab data files.
    """


    def __init__(self, source=None):
        """! 
        @brief The constructor

        This function will initially get the Matlab data file (.mat) path and read data from the file,
        the data will be stored to @ref _data variable.

        @param source input file path or another MatlabData object
        """
        ## Logger object
        self._logger = logging.getLogger('MatlabData')
        # BaseData initialization
        super().__init__(source)


    def read(self, infile=''):
        """!
        @brief Getting data from file 

        This function reads the input file and return the data inside that file.
        The accepted file type is '.xim' file.

        @param infile input file path
        @return data
        """
        # check file path
        if not check_file(infile): return
        # read data from file
        if infile.lower().endswith('.mat'):
            return self.__read_mat(infile)
        else:
            self._logger.error('Undefined input file type!')
            return

    
    def get(self, key='', data=None):
        """!
        @brief Getting specific Matlab data

        This function will return the data corresponding to the key value. 

        @return data
        """
        # check data
        data = self.valid(data)
        # check key
        if key == '':
            self._logger.error('No key value is specified!')
            return
        # check type of data
        if is_none(data) or not isinstance(data, dict):
            self._logger.error('Data type is not dictionary!')
            return
        return data[key]


    def get_list(self, data=None):
        """!
        @brief Getting list of keys in Matlab data

        This function will return the list of keys in @ref _data. 

        @return list of keys
        """
        # check data
        data = self.valid(data)
        # check type of data
        if is_none(data) or not isinstance(data, dict):
            self._logger.error('Data type is not dictionary!')
            return
        return list(data.keys())


    def __read_mat(self, infile=''):
        """!
        @brief Getting Matlab data

        This function reads the Matlab data file and return a dictionary.

        @param infile input file path
        @return dictionary data
        """
        # check file path
        infile = check_file(infile) 
        if not infile:
            return
        # read file
        data = loadmat(infile)
        # print debug info
        self._logger.debug(data)
        return data

