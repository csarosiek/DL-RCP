'''! @file common.py
'''
import os
import fnmatch
import shutil
import math
#import yaml
#import xmltodict
import json
import numpy as np
from copy import copy
import logging
import logging.config
#import colorama


def is_none(source=None):
    """!
    @brief Checking if input data is None or empty array

    This function will check if the input data we read in is None or empty array.

    @param source input data
    @return True if the input is None or empty array, otherwise False 
    """
    if isinstance(source, np.ndarray):
        if source.size == 0:
            return True
    elif not source:
        return True
    return False


def str2bool(string=''):
    """! String to boolean conversion
    
    This function will get a string and return True if the string is
    "yes", "true", "t" or "1".
    
    @param string input string
    @return a boolean
    """
    return string.lower() in ('yes', 'true', 't', '1')


def str2int(string=''):
    """! String to integer conversion
    
    This function will get a string and return the corresponding integer number.
    
    @param string input string
    @return an integer number (0 if incorrect string value)
    """
    try:
        val = int(string)
        return val
    except ValueError:
        return 0


def str2float(string=''):
    """! String to float conversion
    
    This function will get a string and return the corresponding float number.
    
    @param string input string
    @return a float number (NaN if incorrect string value)
    """
    try:
        val = float(string)
        return val
    except ValueError:
        return float('nan')


def add_substr(string='', pos='.', substr=''):
    """! Add substring to string at position before char
    
    This function will add a substring at position before a specific character.
    
    @param string input string
    @param pos position to be add, can be a character or an integer
    @param substr substring
    @return new string
    """
    pos = string.find(pos) if isinstance(pos, str) else pos if isinstance(pos, int) else -1
    return string[:pos] + substr + string[pos:]



def setup_logging(init_file='logging.ini'):
    """!
    @brief Setup logging

    This function will setup the logging environment.

    @param init_file initial file for logging setup
    """
    colorama.init()
    logging.ColoredLogFormatter = ColoredLogFormatter
    logging.config.fileConfig(init_file)


def get_path(filepath=''):
    """! Getting directory path
    
    This function will get path of directory which contain the input file.
    
    @param filepath input file path
    @return path of directory
    """
    dirname, filename = os.path.split(filepath)
    if dirname =='':
        dirname = '.'
    return dirname


def get_fullfilename(filepath=''):
    """! Getting filename in full (with file extension)
    
    This function will get full input file name from a given path.
    
    @param filepath input file path
    @return full input file name
    """
    dirname, filename = os.path.split(filepath)
    return filename


def get_filename(filepath=''):
    """! Getting filename without file extension
    
    This function will get input file name (without file extension) from a given path.
    
    @param filepath input file path
    @return input file name (without file extension)
    """
    dirname, filename = os.path.split(filepath)
    return os.path.splitext(filename)[0]


def get_listdir(path=''):
    """! Getting list of directory names inside a path
    
    This function will get a list of directory names from a given path.
    
    @param path input path
    @return list of directories
    """
    return os.listdir(path)


def get_extension(filepath=''):
    """! Getting filename extension
    
    This function will get extension of input file name from a given path.
    
    @param filepath input file path
    @return filename extension
    """
    dirname, filename = os.path.split(filepath)
    return os.path.splitext(filename)[1]


def get_file_list(dirpath='', ext='', pattern='', fullname=False):
    """! Getting list of filenames inside a directory
    
    This function will get a list of filenames from a given directory path.
    
    @param dirpath input directory path
    @param ext extension of files to be listed
    @param fullname if True, return the full paths of files
    @return list of filenames
    """
    logger = logging.getLogger(__name__.split('.')[-1])
    dirpath = check_dir(dirpath, new_creation=False)
    files = []
    if check_dir(dirpath) == '':
        return files
    for file in os.listdir(dirpath):
        if ext:
            if not file.endswith(ext):
                continue
        files.append(file)
    if pattern:
        files = fnmatch.filter(files, pattern)
    if fullname:
        files = [dirpath + file for file in files]
    #logger.debug('List of files: %s'%files)
    return files


def get_shape(data=None, ndim=None):
    """! Getting shape info of array
    
    This function will get the shape info (size on each dimension) of an array.
    
    @param data input array
    @return shape
    """
    if isinstance(data, list):
        data = np.array(data)
    if not isinstance(data, np.ndarray):
        data = np.array([])
    if not isinstance(ndim, int):
        return data.shape
    if ndim <= 0:
        return 0
    elif ndim == 1:
        return data.shape[0]
    elif ndim == 2:
        if data.ndim == 1:
            return data.shape[0], 0
        else:
            return data.shape[0], data.shape[1]
    elif ndim == 3:
        if data.ndim == 1:
            return data.shape[0], 0, 0
        elif data.ndim == 2:
            return data.shape[0], data.shape[1], 0
        else:
            return data.shape[0], data.shape[1], data.shape[2]


def get_fwhm(data=None):
    max_val = max(data)
    idx = [i for i in range(len(data)) if data[i] > max_val/2]
    return max(idx) - min(idx)


def normalize(image=None, norm_type='area'):
    """!
    @brief Normalizing image

    This function will normalize an image, two types of normalization:
        - 'area': normalize to image area
        - 'max': normalize to maximum pixel value
    In case no image specified, the @ref _data will be used instead.

    @param image image object
    @param norm_type type of normalization 
    @return new image after normalization
    """
    logger = logging.getLogger(__name__.split('.')[-1])
    if norm_type == 'area':
        norm = np.sum(image)
    elif norm_type == 'max':
        norm = np.amax(image)
    if norm == 0:
        logger.warning('Normalization factor = 0! Set to 1.')
        norm = 1.0
    logger.debug('Normalization factor = %d'%norm)
    return np.divide(image,norm)


def check_dir(dirpath='', new_creation=True):
    """!
    @brief Checking validation of directory

    This function will initially check the directory path and give error message if
    the path is not in string format or not exist (automatically create). It will also
    add '/' at the end of the path if the path doesn't have yet.

    @param dirpath path to directory
    @param new_creation if True, create a new directory if @a dirpath does not exist
    @return dirpath (empty string if the dirpath not valid)
    """
    logger = logging.getLogger(__name__.split('.')[-1])
    if dirpath == '':
        logger.debug('Empty directory name')
    else:
        if not isinstance(dirpath, str):
            logger.error('Invalid directory path! Return empty string.')
            return ''
        if not os.path.exists(dirpath):
            if new_creation:
                logger.warning('The directory ' + dirpath + ' does not exist, creating one.')
                os.makedirs(dirpath)
            else:
                logger.error('The directory ' + dirpath + ' does not exist! Return empty string.')
                return ''
        if not os.path.isdir(dirpath):
            logger.error('Invalid directory ' + dirpath + '! Return empty string.')
            return ''
        if dirpath[-1] != '/':
            dirpath = dirpath + '/'
    return dirpath


def check_file(filepath=''):
    """!
    @brief Checking validation of file

    This function will initially check the file path and give error message if
    the path is not in string format or the file not exist (be careful to use it 
    with the output file since it will return emtpy string if the file does not exist).

    @param filepath path to file
    @return filepath (empty string if the filepath not valid)
    """
    logger = logging.getLogger(__name__.split('.')[-1])
    if filepath == '':
        logger.debug('Empty file name')
    else:
        if not isinstance(filepath, str):
            logger.error('Invalid file path! Return empty string.')
            return ''
        if not os.path.exists(filepath):
            logger.error('The file ' + filepath + ' does not exist! Return empty string.')
            return ''
        if not os.path.isfile(filepath):
            logger.error('Invalid file ' + filepath + '! Return empty string.')
            return ''
    return filepath


def categorize_files(src='', pos=[0,1], pattern='*', dest=''):
    """!
    @brief File categorization

    This function will categorize the files based on their names and copy them to
    corresponding folders.

    @param src path to file folder
    @param pos min and max positions of the string on filenames which be used to categorized
    @param pattern pattern of selected filenames
    @param dest destination folder where the files will be copied to, there will be subfolders
    corresponding to each category created
    """
    logger = logging.getLogger(__name__.split('.')[-1])
    if src == '':
        logger.error('Empty path')
        return
    src = check_dir(src)
    logger.debug('Folder path: ' + src)
    if dest == '':
        dest = src
    else:
        dest = check_dir(dest)
    files = get_file_list(src, pattern=pattern)
    temp = []
    for file in files:
        t = file[pos[0]:pos[1]]
        if not t in temp:
            temp.append(t)
            check_dir(dest+t)
    logger.debug('List of categories: ' + str(temp))
    for file in files:
        for t in temp:
            if t in file:
                shutil.copyfile(src+file, check_dir(dest+t)+file)
                break


def remove(path=''):
    """!
    @brief Removing file or directory

    This function will remove file or directory given in the path.

    @param path path to file or directory
    """
    logger = logging.getLogger(__name__.split('.')[-1])
    if path == '':
        logger.debug('Empty path')
        return
    if os.path.isfile(path):
        os.remove(path)        
    elif os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    else:
        logger.error('Invalid path ' + path)


def setup_logging(init_file='logging.ini'):
    """!
    @brief Setup logging

    This function will setup the logging environment.

    @param init_file initial file for logging setup
    """
    colorama.init()
    logging.ColoredLogFormatter = ColoredLogFormatter
    logging.config.fileConfig(init_file)


def parse_params(args):
    """!
    @brief Parsing arguments

    This function will parse a list of key-value pairs and return a dictionary.
    
    @param args list of arguments
    @return a dictionary
    """
    logger = logging.getLogger(__name__.split('.')[-1])
    d = {}
    # check empty args
    if not args:
        logger.warning('Empty option argument list! Please check argument for --param.')
        return d
    # loop  over args
    for arg in args:
        items = arg.split('=')
        key = items[0].strip()
        if len(items) > 1:
            value = '='.join(items[1:])
            d[key] = value
    logger.debug('Argument list: '+str(d))
    return d


class ColoredLogFormatter(logging.Formatter):
    """! @brief Color format class for logging
    
    This class provides color logging output, obtained from
    https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output
    """
    
    ## Color mapping dictionary
    __MAPPING = {
        'DEBUG'   : 37, # white
        'INFO'    : 36, # cyan
        'WARNING' : 33, # yellow
        'ERROR'   : 31, # red
        'CRITICAL': 41, # white on red bg
    }
    ## Prefix string
    __PREFIX = '\033['
    ## Suffix string
    __SUFFIX = '\033[0m'
    
    def __init__(self, *args, **kwargs):
        """! 
        @brief The constructor

        This function will initialize the logging.Formatter.

        @param args, kwargs parameters for logging.Formatter
        """
        logging.Formatter.__init__(self, *args, **kwargs)

    def format(self, record):
        """! 
        @brief format logging output

        This function will add ANSI escape codes to the logging.Formatter record.

        @param record logging.Formatter record
        """
        colored_record = copy(record)
        levelname = colored_record.levelname
        seq = self.__MAPPING.get(levelname, 37) # default white
        colored_levelname = ('{0}{1}m{2}{3}') \
            .format(self.__PREFIX, seq, levelname, self.__SUFFIX)
        colored_record.levelname = colored_levelname
        return logging.Formatter.format(self, colored_record)


class Config:
    """! @brief Configuration class

    This class provides some basic tools/functions for reading configuration file; getting information 
    and checking validation of input parameters.
    """

    def __init__(self, source=None):
        """! 
        @brief The constructor

        This function will initially get the config file path and read data from the file,
        the data will be stored to @ref _data variable. It can also get data from another Config object.

        @param source input file path or another Config object
        """
        ## Logger object
        self._logger = logging.getLogger('Config')
        ## Data member
        self._data = None
        # check filepath and read data from file 
        if source:
            self.load_data(source)

    def load_data(self, source=None):
        """!
        @brief Getting data from input

        If the input is a string, this function will read the input file and parse information from it by 
        calling @ref read_data.
        
        The data will also be stored in the @ref _data variable.

        @param source input file path or another Config object 
        """
        if isinstance(source, Config):
            self._data = source.get_data()
        elif isinstance(source, str):
            self._data = self.read_data(source)
        else:
            self._logger.warning('No source info specified.')

    def get_data(self):
        """!
        @brief Getting data

        This function will return the @ref _data. 

        @return @ref _data 
        """
        return self._data
        
    def read_data(self, infile=''):
        """!
        @brief Getting configuration data from file 

        This function will read the input configuration file and return a dictionary.
        The accepted file types are XML (.xml), YAML (.yaml,.yml) and JSON (.json).

        @param source input file path
        @return dictionary
        """
        # check file path
        #filepath = check_file(filepath) 
        # read data from file        
        if infile.lower().endswith('.xml'):
            config_dict = self.__read_xml(infile)
        elif infile.lower().endswith('.yaml') or infile.lower().endswith('.yml'):
            config_dict = self.__read_yaml(infile)
        elif infile.lower().endswith('.json'):
            config_dict = self.__read_json(infile)
        else:
            self._logger.error('Undefined configuration file type!')
            return
        self._logger.debug('List of parameters: %s'%(config_dict))
        return config_dict

    def __read_xml(self, infile=''):
        """!
        @brief Getting configuration data from XML file 

        This function will read the XML configuration file and return a dictionary.

        @param infile input file path
        @return dictionary
        """
        try:
            with open(infile) as f:
                try:
                    xml = f.read()
                    xml = '<content>' + xml + '</content>'
                    config_dict = xmltodict.parse(xml)
                except xmltodict.expat.ExpatError as e:
                    if hasattr(e, 'problem_mark'):
                        mark = e.problem_mark
                        self._logger.error('Error in configuration file at position (line=%s,col=%s)' % (mark.line+1, mark.column+1))
                    return
        except IOError:
            self._logger.error('Could not open file ' + infile)
            return
        return config_dict['content']

    def __read_yaml(self, infile=''):
        """!
        @brief Getting configuration data from YAML file 

        This function will read the YAML configuration file and return a dictionary.

        @param infile input file path
        @return dictionary
        """
        try:
            with open(infile) as f:
                try:
                    config_dict = yaml.safe_load(f)
                except yaml.YAMLError as e:
                    if hasattr(e, 'problem_mark'):
                        mark = e.problem_mark
                        self._logger.error('Error in configuration file at position (line=%s,col=%s)' % (mark.line+1, mark.column+1))
                    return
        except IOError:
            self._logger.error('Could not open file ' + infile)
            return
        return config_dict

    def __read_json(self, infile=''):
        """!
        @brief Getting configuration data from JSON file 

        This function will read the JSON configuration file and return a dictionary.

        @param infile input file path
        @return dictionary
        """
        try:
            with open(infile) as f:
                try:
                    config_dict = json.load(f)
                except json.JSONDecodeError as e:
                    if hasattr(e, 'problem_mark'):
                        mark = e.problem_mark
                        self._logger.error('Error in configuration file at position (line=%s,col=%s)' % (mark.line+1, mark.column+1))
                    return
        except IOError:
            self._logger.error('Could not open file ' + infile)
            return
        return config_dict

        
