# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 10:40:04 2022
For data storage; Data is stored in a class 
@author: Philipp
"""

#imports
import pandas as pd #for data manipulation
import os #for getcwd()
import pickle
import numpy as np
import scipy.signal as sps
import scipy as sp
from glob import glob
from glob import iglob
import matplotlib.pyplot as plt
from math import e, pi
from scipy import optimize 
import warnings

'''_______________________________________________________________________________________________________________________________'''
def calc_root_lin_interpol(x1, x2, y1, y2):
    '''
    Interpolates the root btw. two points intercepting an axe 
    '''
    return x1 - y1 * (x2-x1) / (y2-y1)
'''...........................................................................................................................'''
def calc_root_discrete(x, y, x_err=0., y_err=0.):
    '''
    Return the interpolated x intersect in an discrete array of x-y data 
    Interpolates btw. the two closest points around zero. 
    If there are multiple roots, the function will return NaN 
    '''
    if type(x_err) == float:
        x_err = x_err * np.ones(x.size)
    if type(y_err) == float:
        y_err = y_err * np.ones(x.size)
        
    if not x.size == y.size == x_err.size == y_err.size:
        raise Exception('Error: All input arrays must have the same size')
    
    #First we need the zero point
    zeros = np.where(np.diff(np.sign(y)))[0]
    #Check wether only one root is present
    if not zeros.size == 1:
        return (float('NaN'), float('NaN'))
    
    #Get the index and the right x and y to interpolate
    index= zeros[0]
    x1 = x[index]
    x2 = x[index+1]
    y1 = y[index]
    y2 = y[index+1]
    d_x1 = x_err[index]
    d_x2 = x_err[index+1]
    d_y1 = y_err[index]
    d_y2 = y_err[index+1]
    
    #Now we calculate the the x0 and the error
    x0 = calc_root_lin_interpol(x1, x2, y1, y2)
    d_x0 = np.sqrt(((-y1 * (x2-x1) / (y2-y1)**2 - (x2-x1) / (y2-y1)) * d_y1)**2 + 
                   (y1 * (x2-x1) * d_y2 / (y2-y1)**2)**2 +
                   (d_x1 * (1 + y1 / (y2-y1)))**2 + 
                   (d_x2 * y2 /(y2-y1))**2) 
    
    return x0, d_x0
'''...........................................................................................................................'''

def calc_Complex_Phaseangle(real, imag, display=False):
    '''
    Calculates the phase correcton for all signal points in the complex plane. 
    If applied, real part of signal should be maximized. 
    
    linear model according to (imag = k * real + d) is used.
    In the matrix form: imag = Ap with A = [[real, 1]] and p = [[k],[d]] 
    phi is calculated according to arctan
    
    Returns the phase angle and the sum of squared residuals (important for pre selection)
    '''
    
    #First we have to set up the coefficient matrix 
    A = np.vstack([real, np.ones(len(real))]).T
    
    #Do the least sqaures fit and calc the angle
    x, residuals, _, _ = np.linalg.lstsq(A, imag, rcond=None)
    theta = np.arctan(x[0])
    
    #We can also use the coefficient of determination 
    r_squared = SS_PFM.calc_R_squared(real, imag, SS_PFM.lin_fun, x)
    
    #Optional: Display the fit in Pyplot
    if display:
        text = '\n'.join(['$\\theta$ = {theta:.2f}'.format(theta=np.degrees(theta)),
                          'k = {k:.4e}'.format(k=x[0]),
                          'd = {d:.4e}'.format(d=x[1]),
                          '$R^2$ = {r_square:.3f}'.format(r_square=r_squared)])
        plt.plot(real,imag, 'bx')
        realf = np.linspace(np.min(real), np.max(real), 1000)
        imagf = x[0]*realf + x[1]
        plt.plot(realf, imagf, 'r-', zorder=5)
        plt.annotate(text, (0.7,1.05), xycoords='axes fraction', size='x-large')
        plt.show()
    
    return theta, r_squared
'''...........................................................................................................................'''
def calc_PFM_Images(folder): 
    '''
    * Reads in the the U0 and U1 of a folder, 
    * Calculates the phase correction, 
    * Dumps the corrected Response, Imaginary, Phase and Amplitude Images in .xyz format. 
    '''
    #Read in the two files
    U0 = np.loadtxt(os.path.join(folder, 'U0.xyz'))
    U1 = np.loadtxt(os.path.join(folder, 'U1.xyz'))
    
    comp = U0[:,-1] + 1j * U1[:, -1]
    
    theta, _ = calc_Complex_Phaseangle(comp.real, comp.imag)
    
    comp = comp * e**(-1j*theta)
    
    real = comp.real
    
    arr = np.vstack((U0[:,0], U0[:,1], real)).T
    
    return 



'''_______________________________________________________________________________________________________________________________'''
'''POLDERs Data Class'''
class POLDERs_data: 
    '''...........................................................................................................................'''
    '''Init class'''
    def __init__(self, rawdata_path = 'reload'):
        #Reload old files
        if rawdata_path == 'reload':
            print('Reloading old Data...')
            self.load_Fileregister()
        #Create new file system
        else:
            print('Initializing...')
            #Init important folders
            self.rawdata_path = os.path.join(rawdata_path,'Rawdata')
            self.database_path = os.getcwd() #path of the evaluation folder (path of script)
            self.sampleRecord_path = os.path.join(rawdata_path,'Sample_record.xlsx')
            #Read in sample Record
            self.sampleRecord = self.read_SampleRecord(self.sampleRecord_path)
            #Now we create the inner register for sample organization
            self.create_FileRegister()
            #Safe all important files in a Pickle
            self.save_Fileregister()
    '''...........................................................................................................................'''
    def read_SampleRecord(self, path) -> dict():
        """
        Method to read in the data of sample Record excel file and return it
        """
        #Create an empty dictionary
        sampleRecord = {}
        
        #First we have to read in the sample overview in order to get the sample names
        sampleOverview = pd.read_excel(path, header = 1, usecols='A:P',true_values=(1.0,1),false_values=(0.0,0))
        
        #Convert some of the columns manually to bool
        boolheaders = ['Eval', 'LIMI', 'Topography', 'Domain map', 'Hysteresis ', 'Poling', 'Thermal']
        for boolheader in boolheaders:
            sampleOverview[boolheader] = sampleOverview[boolheader].astype(bool)
        
        #Delete rows that are flagged as false in EVAL
        sampleOverview = sampleOverview.loc[sampleOverview.Eval, :]
        
        #Now we have to read out the spreadsheets
        measurements = [('-LIMI','A:G',['Nr', 'Filename', 'Date', 'BF_DF', 'Magn.', 'Resolution', 'Description']), 
                        ('_Topography','I:W',['Nr', 'Filename', 'Date', 'Scan-Mode', 'Size',
                                              'Resolution', 'Used_Tip', 'k', 'f_r', 'I-Gain',
                                              'Scan speed', 'Setpoint ', 'Drive-Freq', 'Free Amp',
                                              'Description']), 
                        ('-PFM','Y:AO',['Nr', 'Filename', 'Date', 'Scan-Mode', 'Size',
                                        'Resolution', 'Used_Tip', 'k', 'Defl InvOLS',
                                        'Amp InvOLS', 'Scan speed', 'Setpoint',
                                        'Drive-Freq', 'Drive Amp', 'LIA-factor',
                                        'LIA-T_const', 'Description']), 
                        ('-Calibrations','AQ:AW',['Nr', 'Filename', 'Date', 'Type', 'Used_Tip', 'Opt_value','Description']),
                        ('-Poling','AY:BL',['Nr', 'Filename', 'Date ', 'Mode', 'Resolution', 'Size',
                                            'Shape', 'Used Tip', 'k', 'Defl InvOLS',
                                            'Pol_Vol', 'Scan speed', 'Setpoint',
                                            'Description']), 
                        ('-SS_PFM','BN:CJ',['Nr', 'Filename', 'Date', 'Reference Scan', 'Type', 'Spot',
                                            'X', 'Y', 'Temp', 'Ampl', 'Amplification',
                                            'Freq', 'Phase delay', 'Pulsetime', 'DDSAmp', 'Cycles',
                                            'Sample Rate', 'Low Pass Filter', 'Channel', 'Value', 'df',
                                            'Q', 'Description'])] 
                        #names with relevant sections in excel

        samples_names = sampleOverview.loc[:,'Nr'].astype(str) + '_' + sampleOverview.loc[:,'Name'] #Concetanate strings
        samples_names = samples_names.tolist()
        #Outer Sample for loop
        for sample_name in samples_names:
            #Inner for Loop with measurements
            for measurement in measurements: 
                sampleRecord[sample_name + measurement[0]] = pd.read_excel(path, 
                                                                           sheet_name=sample_name,
                                                                           header=None,
                                                                           usecols=measurement[1],
                                                                           names=measurement[2],
                                                                           skiprows=2).dropna() #+NaN removal
                                                                            #Key is alwas Sample name + measurement
        sampleRecord['Sample_Overview']=sampleOverview
        return sampleRecord
    '''...........................................................................................................................'''
    def create_FileRegister(self):
        '''
        Creates a dictionary structure with sub dictionaries and paths for the folders
        '''
        #Create Dictionaries
        self.samples = {}
        #Now the nested structure is generated
        sampleList = self.sampleRecord['Sample_Overview'].loc[:,'Nr'].astype(str) +'_'+ self.sampleRecord['Sample_Overview'].Name
        for sample_key in sampleList:
            self.samples[sample_key] = {}
            #Now we create the keys for the measurements 
            #SS-PFM
            self.samples[sample_key]['SS_PFM-rawDir'] = os.path.join(self.rawdata_path, sample_key, 'AFM')
            self.samples[sample_key]['SS_PFM-objDir'] = os.path.join(self.database_path,'Sample_Data',sample_key, 'SS_PFM_data.pkl')
            self.samples[sample_key]['SS_PFM-table'] = self.sampleRecord[sample_key + '-SS_PFM']
    '''...........................................................................................................................'''
    def save_Fileregister(self):
        '''
        Saves all important files for data organization for the next reload
        '''
        home_paths = {'rawdata_path':self.rawdata_path, 
                      'database_path':self.database_path, 
                      'sampleRecord_path':self.sampleRecord_path}
        POLDERs_data.save_Object(home_paths,'home_paths.pkl')
        POLDERs_data.save_Object(self.sampleRecord,'sample_Record.pkl')
        POLDERs_data.save_Object(self.samples,'samples.pkl')
        
    '''...........................................................................................................................'''
    def load_Fileregister(self):
        '''
        Loads all the important files for data organization from the previous session
        '''
        self.samples = POLDERs_data.load_Object('samples.pkl')
        self.sampleRecord = POLDERs_data.load_Object('sample_Record.pkl')
        home_paths = POLDERs_data.load_Object('home_paths.pkl')
        self.rawdata_path = home_paths['rawdata_path']
        self.database_path = home_paths['database_path']
        self.sampleRecord_path = home_paths['sampleRecord_path']
    '''...........................................................................................................................'''
    def read_SS_PFM_database(self, overwrite=False):
        '''
        Reads and updates all SS-PFM files
        '''
        for key in self.samples.keys():
            print(key)
            try:
                sample = self.get_SS_PFM_Object(key)
                sample.read_all_Files(overwrite=overwrite)
            except Exception as e:
                print('Error in: '+ key)
                print(e)
    '''...........................................................................................................................'''
    @staticmethod 
    def save_Object(obj, filename, path=''):
        '''
        Saves an Object in the path
        '''
        #Check wether path is empty == if yes, try to split up the filename
        if path == '':
            filename, path = POLDERs_data.seperate_file_folder(filename)
        
        #If the path is not empty, check wether path exists and if not, create it!
        if not path=='':
            if not os.path.exists(path):
                os.makedirs(path)
        
        #Now the file is saved!
        print('Saving file: '+filename)
        with open(os.path.join(path, filename),'wb') as f: 
            pickle.dump(obj, f)
        print('Done!')
    '''...........................................................................................................................'''
    @staticmethod
    def load_Object(filename, path='') -> object():
        '''
        Loads an object and returns it
        '''
        print('Loading file: '+filename)
        with open(os.path.join(path, filename),'rb') as f:
            obj = pickle.load(f)
        print('Done!')
        return obj
    '''...........................................................................................................................'''
    @staticmethod
    def seperate_file_folder(path):
        strings = os.path.split(path)
        file = strings[1]
        directory = strings[0]
        return file, directory
    '''______'''
    '''Getter'''
    '''...........................................................................................................................'''
    def get_Samplenames(self):
        return self.samples.keys()
    '''...........................................................................................................................'''
    def get_SampleRecord_table(self, samplename, measurement) -> pd.DataFrame():
        '''
        Returns the table of a certain measurement type of a sample
        '''
        return self.sampleRecord[samplename+'-'+measurement]
    '''...........................................................................................................................'''
    def get_SS_PFM_Object(self, samplename):
        '''
        Return the SS-PFM Object of a specific sample
        '''
        self.SS_PFM = SS_PFM(self.samples[samplename]['SS_PFM-rawDir'],
                             self.samples[samplename]['SS_PFM-table'],
                             objDir=self.samples[samplename]['SS_PFM-objDir'])
        return self.SS_PFM
    '''...........................................................................................................................'''
    def delete_SS_PFM_Object(self):
        del self.SS_PFM
    '''...........................................................................................................................'''
    def set_Column_names_SS_PFM(self, old_name, new_name):
        '''
        Renames a column in all SS-PFM files
        '''
        for samplename in self.samples.keys():
            obj = self.get_SS_PFM_Object(samplename)
            obj.set_Column_name(old_name, new_name)
            obj.save()
    '''...........................................................................................................................'''
    
'''_______________________________________________________________________________________________________________________________'''
'''SS-PFM Data Class'''
class SS_PFM(): 
    '''...........................................................................................................................'''
    def __init__(self, rawDir, table, objDir = 'empty'):
        self.files = {}
        if not objDir == 'empty':
            try:
                self.files = POLDERs_data.load_Object(objDir)
                print('Data object sucessfully loaded!')
            except:
                print('The data object could not be loaded from the given directory:')
        self.objDir = objDir
        self.rawDir = rawDir
        self.table = table
        self.objDir_dropped = os.path.join(os.path.split(objDir)[0], 'SS_PFM_dropped')

    '''...........................................................................................................................'''
    def read_Forcecurve(self, path, amplification = 1) -> pd.DataFrame():
        '''
        Reads in one single forceplot from .csv file, seperates AmpOff from AmpOn and splits the unwanted data away.
        Only applicable for Asylum Hysteresis curves with 2 cycles!
        '''
        #Read in the .csv File
        fc_raw = pd.read_csv(path,header=None, skiprows=1, index_col=0, 
                             names=('Raw','Deflection','Amplitude','Phase_1','Phase_2','Frequency','ZSnsr','Bias'))
        fc_raw = fc_raw.dropna()
        #Amplify
        fc_raw['Bias'] = fc_raw['Bias'] * amplification
        #Separate AmpOff from AmpOn
        ampOn = []
        ampOff = []
        
        ref_index = 0
        ref_bias = 0
        bias_on = False #To distinguish btw. AmpOff and on
        first = True
        
        for index, row in fc_raw.iterrows():
            #Set new value for ref_bias
            if ref_index==index: 
                ref_bias = row['Bias']
            #Check wheter we have a jump from AmpOff to AmpOn
            if np.abs(row['Bias'] - ref_bias)>1:
                #Check wether first is zero
                if first:
                    first = False
                    ampOn.append(fc_raw.loc[ref_index:(index-1), :].describe())
                    ampOff.append(fc_raw.loc[ref_index:(index-1), :].describe())
                else:
                    #Decide if Bias on or off
                    if bias_on:
                        ampOn.append(fc_raw.loc[ref_index:(index-1), :].describe())
                    else:
                        ampOff.append(fc_raw.loc[ref_index:(index-1), :].describe())
                #reset
                bias_on = not bias_on
                ref_index = index
                ref_bias = row['Bias']
        #Check if the lists have the same length
        ampOff_len = len(ampOff)
        ampOn_len = len(ampOn)
        if ampOff_len < ampOn_len:
            ampOn = ampOn [0:ampOff_len]
        #Get list of Rows and Columns and create a list for columns
        columns = ampOn[0].columns.values
        rows = ampOn[0].index.values
        headers = []
        for bias in ['AmpOn','AmpOff']:
            for column in columns: 
                for row in rows: 
                    headers.append('_'.join((bias,column,row)))
        #Create on large dataframe array
        matrix = []
        for index, on in enumerate(ampOn):
            off = ampOff[index]
            line_on = []
            line_off = []
            for column in columns: 
                for row in rows: 
                    line_on.append(on.loc[row,column])
                    line_off.append(off.loc[row,column])
            line_on.extend(line_off)
            matrix.append(line_on)
            
        matrix = pd.DataFrame(data=matrix, columns=headers)
        #Cut out the unwanted cycles of the curve
        bias = np.array(matrix.AmpOn_Bias_mean)
        maxi = sps.argrelmax(bias)[0]
        mini = sps.argrelmin(bias)[0]
        if mini[0] < maxi[0]:
            matrix = matrix.loc[mini[0]:mini[1],:]
        else:
            matrix = matrix.loc[maxi[0]:maxi[1],:]
        matrix = matrix.reset_index()
        return matrix
    '''...........................................................................................................................'''
    def split_Hysteresis(self, matrix) -> list:
        '''
        Splits up the hysteresis in two parts ==> importatnt for fitting!
        Only applicable for Asylum Hysteresis curves with 1 cycle!
        '''
        lst = []
        bias = np.array(matrix.AmpOn_Bias_mean)
        maxi = sps.argrelmax(bias)[0]
        mini = sps.argrelmin(bias)[0]
        index_max = matrix.shape[0]
        #differentiate btw. pos and neg slope
        if mini.size < maxi.size:
            lst.append(matrix.loc[0:maxi[0],:].reset_index(drop=True)) 
            lst.append(matrix.loc[maxi[0]:index_max,:].reset_index(drop=True))
        else:
            lst.append(matrix.loc[0:mini[0],:].reset_index(drop=True))
            lst.append(matrix.loc[mini[0]:index_max,:].reset_index(drop=True))
        
        return lst
    '''...........................................................................................................................'''
    def read_Forcemap(self, param_set, path) -> dict():
        '''
        Reads in a full Forcemap and saves it in a dict ==> Value of a key is list 
        '''
        #Get list of all files in the forcemap foder
        try:
            path = os.path.join(path,'*.csv')
            paths = glob(path)
            fm = []
        except Exception as e:
            print('Error reading while reading listing files in Forcemap folder:')
            print(e)
        
        for p in paths:
            try:
                name = os.path.split(p)[-1].split('_')[-1].replace('.csv', '').replace('000','-').replace('Point','_Point')
                print('Reading in: ' + name)
                fc = self.read_Forcecurve(p,amplification=param_set.Amplification)
                fm.append([name,fc])
            except Exception as e:
                print('Reading in '+ name + ' failed!')
                print(e)
        
        #Convert to Dataframe
        fm = pd.DataFrame(fm, columns=['Name','Hysteresis'])
        fm.set_index('Name',drop=True)
        fm = fm.set_index('Name',drop=True)
        #Add True/False marker columns for curve inspection
        bias = ['_AmpOn_','_AmpOff_']
        signals = ['Amplitude','Phase_1','Phase_2']
        for b in bias:
            for signal in signals: 
                fm['MD'+b+signal] = np.ones(fm.shape[0], dtype=bool)
        fm['Temp'] = np.ones(fm.shape[0]) * param_set.Temp
        return fm
    '''...........................................................................................................................'''
    def read_all_Files(self, overwrite=False):
        '''
        Reads mentioned in the table which are available
        '''
        if overwrite:
            self.files = {}
        keys = self.files.keys()
        #First we have to get the availiable folders in our homefolder
        folders = np.array(os.listdir(os.path.join(self.rawDir, 'SS_PFM')))
        #Now we iterate over this list and read in all curves

        for folder in folders: 
            if not folder in keys: #Check wheter the file already exists in the object
                try:
                    print('\nReading in the Forcemap: ' + folder)
                    try: #If error ocurrs here ==> Them file is not properly listed in Samplerecord
                        param_set = self.get_param_set(folder) #Get the matching parameter set for the curve
                    except:
                        print('Error '+folder+' is not properly listed in the sample record')
                    self.files[folder] = self.read_Forcemap(param_set, os.path.join(self.rawDir, 'SS_PFM', folder))
                except Exception as e: 
                    print('Error reading in file: '+ folder)
                    print(e)
        #Save now all the files in the object
        POLDERs_data.save_Object(self.files, self.objDir)
    
    '''...........................................................................................................................'''
    def calc_all_Responses(self, save=True, log = False):
        '''
        Calculates the responses for all forcemaps stored in the class
        '''
        #Get all the filenames in the class
        filenames = self.get_Filenames()
        
        #Iterate over all files
        for filename in filenames: 
            self.calc_Forcemap_Responses(filename, save=save, log=log)
        
        #Save the object
        if save:
            self.save()
    '''...........................................................................................................................'''
    def calc_Forcemap_Responses(self, filename, save=True, log = False):
        '''
        Calculates the Responses a forcemap
        
        Argument 'filename':
            dtype = string
            valid keywords are filnames stored in the data class
        '''
        #Get a list of the lines and points of the forcecurve
        fc_names = self.get_Forcemap_index(filename)

        if log:
            print(filename)
        
        #Now iterate over all
        fm_calc = {}
        for fc_name in fc_names:
            fm_calc[fc_name] = self.calc_Forcecurve_Response(filename, index=fc_name, save=save, log=log)
        
        return fm_calc
        
    '''...........................................................................................................................'''
    def calc_Forcecurve_Response(self, filename, line=0, point=0, index='', save = True, log = False):
        '''
        Calculates the real and imaginary Response and Error intervall of a forcecurve
        Stores the Info in the given Forcecurve
        '''
        if log:
            print(f'Line {line} - Point {point}')

        # Get the right Forcecurve
        fc = self.get_Forcemap_value(filename, 'Hysteresis', line=line, point=point, index=index)
        
        # Get the relevant data and convert them into arrays
        # AmpOn
        ampOn_amp_mean_old = np.array(fc.loc[:,'AmpOn_Amplitude_mean'])
        ampOn_p1_mean = np.radians(np.array(fc.loc[:,'AmpOn_Phase_1_mean']))
        ampOn_p2_mean = np.radians(np.array(fc.loc[:,'AmpOn_Phase_2_mean']))
        ampOn_amp_std = np.array(fc.loc[:,'AmpOn_Amplitude_std'])
        ampOn_p1_std = np.radians(np.array(fc.loc[:,'AmpOn_Phase_1_std']))
        ampOn_p2_std = np.radians(np.array(fc.loc[:,'AmpOn_Phase_2_std']))
        # AmpOff
        ampOff_amp_mean_old = np.array(fc.loc[:,'AmpOff_Amplitude_mean'])
        ampOff_p1_mean = np.radians(np.array(fc.loc[:,'AmpOff_Phase_1_mean']))
        ampOff_p2_mean = np.radians(np.array(fc.loc[:,'AmpOff_Phase_2_mean']))
        ampOff_amp_std = np.array(fc.loc[:,'AmpOff_Amplitude_std'])
        ampOff_p1_std = np.radians(np.array(fc.loc[:,'AmpOff_Phase_1_std']))
        ampOff_p2_std = np.radians(np.array(fc.loc[:,'AmpOff_Phase_2_std']))

        # Do the Q-correction and Amp correction
        # Constants
        q_factor = float(self.table.loc[self.table['Filename']==filename, 'Q'])
        df = float(self.table.loc[self.table['Filename']==filename, 'df'])
        DDSAmp = float(self.table.loc[self.table['Filename']==filename, 'DDSAmp']) * float(self.table.loc[self.table['Filename']==filename, 'Amplification'])
        w_low = 11763 * 2 * np.pi

        # AmpOn 
        ampOn_w1_mean = (np.array(fc.loc[:,'AmpOn_Frequency_mean']) - df/2) * 2 * np.pi
        ampOn_w2_mean = (np.array(fc.loc[:,'AmpOn_Frequency_mean']) + df/2) * 2 * np.pi
        ampOn_w0_mean = np.sqrt(((ampOn_w2_mean**4 - ampOn_w1_mean**4) * q_factor**2) 
                                / (2 * q_factor**2 * (ampOn_w2_mean**2 - ampOn_w1_mean**2) + ampOn_w1_mean**2 - ampOn_w2_mean**2))
        ampOn_Amax_mean = ampOn_amp_mean_old * q_factor * np.sqrt((ampOn_w0_mean**2 - ampOn_w1_mean**2)**2 + 
                                                              (ampOn_w0_mean * ampOn_w1_mean / q_factor)**2)/ampOn_w0_mean**2
        ampOn_amp_mean = (ampOn_Amax_mean * ampOn_w0_mean**2) / (np.sqrt((ampOn_w0_mean**2 - w_low**2)**2 + (ampOn_w0_mean * w_low / q_factor)**2) * q_factor * DDSAmp)
        ampOn_amp_std *= ampOn_amp_mean / ampOn_amp_mean_old

        # AmpOff 
        ampOff_w1_mean = (np.array(fc.loc[:,'AmpOff_Frequency_mean']) - df/2) * 2 * np.pi
        ampOff_w2_mean = (np.array(fc.loc[:,'AmpOff_Frequency_mean']) + df/2) * 2 * np.pi
        ampOff_w0_mean = np.sqrt(((ampOff_w2_mean**4 - ampOff_w1_mean**4) * q_factor**2) 
                                 / (2 * q_factor**2 * (ampOff_w2_mean**2 - ampOff_w1_mean**2) + ampOff_w1_mean**2 - ampOff_w2_mean**2))
        ampOff_Amax_mean = ampOff_amp_mean_old * q_factor * np.sqrt((ampOff_w0_mean**2 - ampOff_w1_mean**2)**2 + 
                                                              (ampOff_w0_mean * ampOff_w1_mean / q_factor)**2)/ampOff_w0_mean**2
        ampOff_amp_mean = (ampOff_Amax_mean * ampOff_w0_mean**2) / (np.sqrt((ampOff_w0_mean**2 - w_low**2)**2 + (ampOff_w0_mean * w_low / q_factor)**2) * q_factor * DDSAmp)
        ampOff_amp_std *= ampOff_amp_mean / ampOff_amp_mean_old
 
        # Now we calculate the responses (real + imaginary)
        # Note that the corrected amplitude is not stored in the hysteresis df, but info is in the response and imag data
        calc = {}
        #AmpOn
        calc['AmpOn_Response_1_mean'] = ampOn_amp_mean * np.cos(ampOn_p1_mean)
        calc['AmpOn_Response_2_mean'] = ampOn_amp_mean * np.cos(ampOn_p2_mean)
        calc['AmpOn_Imaginary_1_mean'] = ampOn_amp_mean * np.sin(ampOn_p1_mean)
        calc['AmpOn_Imaginary_2_mean'] = ampOn_amp_mean * np.sin(ampOn_p2_mean)
        calc['AmpOn_Frequency_0_mean'] = 0.5 * ampOn_w0_mean / np.pi
        calc['AmpOn_Frequency_1_mean'] = 0.5 * ampOn_w1_mean / np.pi
        calc['AmpOn_Frequency_2_mean'] = 0.5 * ampOn_w2_mean / np.pi
        calc['AmpOn_AmplitudeMax_mean'] = ampOn_Amax_mean
        
        # AmpOff
        calc['AmpOff_Response_1_mean'] = ampOff_amp_mean * np.cos(ampOff_p1_mean)
        calc['AmpOff_Response_2_mean'] = ampOff_amp_mean * np.cos(ampOff_p2_mean)
        calc['AmpOff_Imaginary_1_mean'] = ampOff_amp_mean * np.sin(ampOff_p1_mean)
        calc['AmpOff_Imaginary_2_mean'] = ampOff_amp_mean * np.sin(ampOff_p2_mean)
        calc['AmpOff_Frequency_0_mean'] = 0.5 * ampOff_w0_mean / np.pi
        calc['AmpOff_Frequency_1_mean'] = 0.5 * ampOff_w1_mean / np.pi
        calc['AmpOff_Frequency_2_mean'] = 0.5 * ampOff_w2_mean / np.pi
        calc['AmpOff_AmplitudeMax_mean'] = ampOff_Amax_mean
        
        # Now we calculate the error intervalls - Gauss Error Law - data correction not taken into account
        # AmpOn
        calc['AmpOn_Response_1_std'] = np.sqrt((ampOn_amp_std * np.cos(ampOn_p1_mean))**2 + 
                               (ampOn_amp_mean * np.sin(ampOn_p1_mean) * ampOn_p1_std)**2)
        calc['AmpOn_Response_2_std'] = np.sqrt((ampOn_amp_std * np.cos(ampOn_p2_mean))**2 + 
                               (ampOn_amp_mean * np.sin(ampOn_p2_mean) * ampOn_p2_std)**2)
        calc['AmpOn_Imaginary_1_std'] = np.sqrt((ampOn_amp_std * np.sin(ampOn_p1_mean))**2 + 
                               (ampOn_amp_mean * np.cos(ampOn_p1_mean) * ampOn_p1_std)**2)
        calc['AmpOn_Imaginary_2_std'] = np.sqrt((ampOn_amp_std * np.sin(ampOn_p2_mean))**2 + 
                               (ampOn_amp_mean * np.cos(ampOn_p2_mean) * ampOn_p2_std)**2)
        # AmpOff
        calc['AmpOff_Response_1_std'] = np.sqrt((ampOff_amp_std * np.cos(ampOff_p1_mean))**2 + 
                               (ampOff_amp_mean * np.sin(ampOff_p1_mean) * ampOff_p1_std)**2)
        calc['AmpOff_Response_2_std'] = np.sqrt((ampOff_amp_std * np.cos(ampOff_p2_mean))**2 + 
                               (ampOff_amp_mean * np.sin(ampOff_p2_mean) * ampOff_p2_std)**2)
        calc['AmpOff_Imaginary_1_std'] = np.sqrt((ampOff_amp_std * np.sin(ampOff_p1_mean))**2 + 
                               (ampOff_amp_mean * np.cos(ampOff_p1_mean) * ampOff_p1_std)**2)
        calc['AmpOff_Imaginary_2_std'] = np.sqrt((ampOff_amp_std * np.sin(ampOff_p2_mean))**2 + 
                               (ampOff_amp_mean * np.cos(ampOff_p2_mean) * ampOff_p2_std)**2)
        
        # Now we create a dataframe
        df = pd.DataFrame(calc)
        
        if save:
            self.add_Forcecurve_columns(filename, df, point = point, line = line, index=index)
        
        return df
    '''...........................................................................................................................'''    
    def calc_all_Statistic(self, save=True):
        '''
        Calculates the responses for all forcemaps stored in the class
        '''
        #Get all the filenames in the class
        filenames = self.get_Filenames()
        
        #Iterate over all files
        for filename in filenames: 
            self.calc_Forcemap_Statistic(filename, save=save)
        
        #Save the object
        if save:
            self.save()
    '''...........................................................................................................................'''
    def calc_Forcemap_Statistic(self, filename, save=True, display=False):
        '''
        Calculates the Jesse Auxiliary statisic of a forcemap
        
        Argument 'filename':
            dtype = string
            valid keywords are filnames stored in the data class
        '''
        #Get a list of the lines and points of the forcecurve
        fc_names = self.get_Forcemap_index(filename)
        
        #Now iterate over all
        fm_stat = {}
        for fc_name in fc_names:
            fm_stat[fc_name] = self.calc_Forcecurve_Statistic(filename, index=fc_name, display=display) 
        
        dic = pd.DataFrame(fm_stat).transpose()
        
        if save: 
            tempi=self.add_Forcemap_columns(filename, dic)
        return fm_stat
    '''...........................................................................................................................'''
    def calc_Forcecurve_Statistic(self, filename, line=0, point=0, index='', save = True, linerrmax=10, display=False):
        '''
        Applies a statistical analysis according to the paper:
        Quantitative mapping of switching behavior in piezoresponse force microscopy
        From: Stephen Jesse, Ho Nyung Lee, and Sergei V. Kalinin
        
        Includes also the complex phase correton, linerr pre-selecton, phase-evoluton selection and response fit
        Phase correcton is performed on all data, since its purpose is to maximise the real part of the signal
        '''
        #Initialize the different mode keywords for later iteration
        bias_modes = ['AmpOn', 'AmpOff']
        signals = ['1', '2'] #new definition for the str.join method
        stats = ['mean', 'std']
        branchnames = ['plus', 'minus']
        
        #Initialize the main dic and dis 
        dic = {} #For aux stat dataframe
        dis = {} #Dic for single discrete Values 
        
        #First we have to get the Forcecurve
        fc = self.get_Forcemap_value(filename, 'Hysteresis', line=line, point=point, index=index)
        
        #Now we calculate the correction angle
        dis.update(self.calc_Forcecurve_Phasecorr(fc, display=display))
        
        #Now we have to split the Hysteresis data
        branches = self.split_Hysteresis(fc)
        
        # Now we transfer the data into the complex plane and correct the data by the phase angle
        for branchindex in range(len(branches)):
            sequence = 1
            dic['_'.join(('AmpOn','Bias',branchnames[branchindex],'mean'))] = np.array(branches[branchindex].loc[:,'AmpOn_Bias_mean'])
            # Check if the arrays have to be inverted 
            if dic['_'.join(('AmpOn',
                             'Bias',
                             branchnames[branchindex],
                             'mean'))][0] > dic['_'.join(('AmpOn',
                                                          'Bias',
                                                          branchnames[branchindex],
                                                          'mean'))][-1]:
                sequence = -1
                dic['_'.join(('AmpOn',
                              'Bias',
                              branchnames[branchindex],
                              'mean'))] = dic['_'.join(('AmpOn',
                                                        'Bias',
                                                        branchnames[branchindex],
                                                        'mean'))][::sequence]
            dic['_'.join(('AmpOn',
                          'Bias',
                          branchnames[branchindex],
                          'std'))] = np.array(branches[branchindex].loc[:,'AmpOn_Bias_std'])[::sequence] 
            
            # Now the param main iteration
            for bias_mode in bias_modes: 
                for stat in stats: 
                    for signal in signals:
                        comp = np.array((branches[branchindex].loc[:,'_'.join([bias_mode, 'Response', signal, stat])] + 
                                         branches[branchindex].loc[:,'_'.join([bias_mode, 'Imaginary', signal, stat])] * 1j) * 
                                        e**(-1j*dis['_'.join([bias_mode, 'Complex', signal, 'theta'])]))[::sequence]
                        #Calc Response, Imaginary, Phase
                        dic['_'.join([bias_mode, 'Response', signal, branchnames[branchindex], stat])] = np.real(comp)
                        dic['_'.join([bias_mode, 'Imaginary', signal, branchnames[branchindex], stat])] = np.imag(comp)
                        phase = self.shift_to_equi_angle(np.angle(comp, deg=True)%360,-90,270,360)
                        dic['_'.join([bias_mode, 'Phase', signal, branchnames[branchindex], stat])] = phase
                    #Calc amplitude
                    dic['_'.join([bias_mode, 'Amplitude', branchnames[branchindex],stat])] = np.absolute(comp)

            # Resonance curve values - transforms the one branched values into the two branches 
            for bias_mode in bias_modes:
                f0 = np.array(branches[branchindex].loc[:,'_'.join([bias_mode, 'Frequency', '0', 'mean'])])[::sequence]
                amax = np.array(branches[branchindex].loc[:,'_'.join([bias_mode, 'AmplitudeMax', 'mean'])])[::sequence]
                for signal in signal:
                    fn = np.array(branches[branchindex].loc[:,'_'.join([bias_mode, 'Frequency', signal, 'mean'])])[::sequence]
                    dic['_'.join([bias_mode, 'Frequency', signal, branchnames[branchindex], 'mean'])] = fn
                dic['_'.join([bias_mode, 'Frequency', '0', branchnames[branchindex], 'mean'])] = f0
                dic['_'.join([bias_mode, 'AmplitudeMax', branchnames[branchindex], 'mean'])] = amax

        #Old code            
        '''
        # #Now we get the data for the forward branches (Maybe change it to a shorter for loop in the future, like below then)
        # #Forward = from - to +; (R+)
        # #AmpOn
        # sequence = 1
        # dic['AmpOn_Bias_plus_mean'] = np.array(branches[0].loc[:,'AmpOn_Bias_mean'])
        # #Check if the arrays have to be inverted 
        # if dic['AmpOn_Bias_plus_mean'][0] > dic['AmpOn_Bias_plus_mean'][-1]:
        #     sequence = -1
        #     dic['AmpOn_Bias_plus_mean'] = dic['AmpOn_Bias_plus_mean'][::sequence]
            
        # #Now the rest of the data
        # dic['AmpOn_Bias_plus_std'] = np.array(branches[0].loc[:,'AmpOn_Bias_std'])[::sequence] 
        # dic['AmpOn_Response_1_plus_mean'] = np.array(branches[0].loc[:,'AmpOn_Response_1_mean'])[::sequence] 
        # dic['AmpOn_Response_1_plus_std'] = np.array(branches[0].loc[:,'AmpOn_Response_1_std'])[::sequence]
        # dic['AmpOn_Response_2_plus_mean'] = np.array(branches[0].loc[:,'AmpOn_Response_2_mean'])[::sequence] 
        # dic['AmpOn_Response_2_plus_std'] = np.array(branches[0].loc[:,'AmpOn_Response_2_std'])[::sequence]
        # dic['AmpOn_Imaginary_1_plus_mean'] = np.array(branches[0].loc[:,'AmpOn_Imaginary_1_mean'])[::sequence] 
        # dic['AmpOn_Imaginary_1_plus_std'] = np.array(branches[0].loc[:,'AmpOn_Imaginary_1_std'])[::sequence]
        # dic['AmpOn_Imaginary_2_plus_mean'] = np.array(branches[0].loc[:,'AmpOn_Imaginary_2_mean'])[::sequence] 
        # dic['AmpOn_Imaginary_2_plus_std'] = np.array(branches[0].loc[:,'AmpOn_Imaginary_2_std'])[::sequence]
        # dic['AmpOn_Phase_1_plus_mean'] = np.array(branches[0].loc[:,'AmpOn_Phase_1_mean'])[::sequence] 
        # dic['AmpOn_Phase_1_plus_std'] = np.array(branches[0].loc[:,'AmpOn_Phase_1_std'])[::sequence]
        # dic['AmpOn_Phase_2_plus_mean'] = np.array(branches[0].loc[:,'AmpOn_Phase_2_mean'])[::sequence] 
        # dic['AmpOn_Phase_2_plus_std'] = np.array(branches[0].loc[:,'AmpOn_Phase_2_std'])[::sequence]
        # dic['AmpOn_Amplitude_plus_mean'] = np.array(branches[0].loc[:,'AmpOn_Amplitude_mean'])[::sequence] 
        # dic['AmpOn_Amplitude_plus_std'] = np.array(branches[0].loc[:,'AmpOn_Amplitude_std'])[::sequence]
        # #AmpOff
        # dic['AmpOff_Response_1_plus_mean'] = np.array(branches[0].loc[:,'AmpOff_Response_1_mean'])[::sequence] 
        # dic['AmpOff_Response_1_plus_std'] = np.array(branches[0].loc[:,'AmpOff_Response_1_std'])[::sequence]
        # dic['AmpOff_Response_2_plus_mean'] = np.array(branches[0].loc[:,'AmpOff_Response_2_mean'])[::sequence] 
        # dic['AmpOff_Response_2_plus_std'] = np.array(branches[0].loc[:,'AmpOff_Response_2_std'])[::sequence]
        # dic['AmpOff_Imaginary_1_plus_mean'] = np.array(branches[0].loc[:,'AmpOff_Imaginary_1_mean'])[::sequence] 
        # dic['AmpOff_Imaginary_1_plus_std'] = np.array(branches[0].loc[:,'AmpOff_Imaginary_1_std'])[::sequence]
        # dic['AmpOff_Imaginary_2_plus_mean'] = np.array(branches[0].loc[:,'AmpOff_Imaginary_2_mean'])[::sequence] 
        # dic['AmpOff_Imaginary_2_plus_std'] = np.array(branches[0].loc[:,'AmpOff_Imaginary_2_std'])[::sequence]
        # dic['AmpOff_Phase_1_plus_mean'] = np.array(branches[0].loc[:,'AmpOff_Phase_1_mean'])[::sequence] 
        # dic['AmpOff_Phase_1_plus_std'] = np.array(branches[0].loc[:,'AmpOff_Phase_1_std'])[::sequence]
        # dic['AmpOff_Phase_2_plus_mean'] = np.array(branches[0].loc[:,'AmpOff_Phase_2_mean'])[::sequence] 
        # dic['AmpOff_Phase_2_plus_std'] = np.array(branches[0].loc[:,'AmpOff_Phase_2_std'])[::sequence]
        # dic['AmpOff_Amplitude_plus_mean'] = np.array(branches[0].loc[:,'AmpOff_Amplitude_mean'])[::sequence] 
        # dic['AmpOff_Amplitude_plus_std'] = np.array(branches[0].loc[:,'AmpOff_Amplitude_std'])[::sequence]
         
        
        # #Now we get the data from the reverse branches
        # #Reverse = from + to -; (R-)
        # #AmpOn
        # sequence = 1
        # dic['AmpOn_Bias_minus_mean'] = np.array(branches[1].loc[:,'AmpOn_Bias_mean']) 
        # #Check if the arrays have to be inverted 
        # if dic['AmpOn_Bias_minus_mean'][0] > dic['AmpOn_Bias_minus_mean'][-1]:
        #     sequence = -1
        #     dic['AmpOn_Bias_minus_mean'] = dic['AmpOn_Bias_minus_mean'][::sequence]
        
        # #Now the rest of the data
        # dic['AmpOn_Bias_minus_std'] = np.array(branches[1].loc[:,'AmpOn_Bias_std'])[::sequence] 
        # dic['AmpOn_Response_1_minus_mean'] = np.array(branches[1].loc[:,'AmpOn_Response_1_mean'])[::sequence] 
        # dic['AmpOn_Response_1_minus_std'] = np.array(branches[1].loc[:,'AmpOn_Response_1_std'])[::sequence]
        # dic['AmpOn_Response_2_minus_mean'] = np.array(branches[1].loc[:,'AmpOn_Response_2_mean'])[::sequence] 
        # dic['AmpOn_Response_2_minus_std'] = np.array(branches[1].loc[:,'AmpOn_Response_2_std'])[::sequence]
        # dic['AmpOn_Imaginary_1_minus_mean'] = np.array(branches[1].loc[:,'AmpOn_Imaginary_1_mean'])[::sequence] 
        # dic['AmpOn_Imaginary_1_minus_std'] = np.array(branches[1].loc[:,'AmpOn_Imaginary_1_std'])[::sequence]
        # dic['AmpOn_Imaginary_2_minus_mean'] = np.array(branches[1].loc[:,'AmpOn_Imaginary_2_mean'])[::sequence] 
        # dic['AmpOn_Imaginary_2_minus_std'] = np.array(branches[1].loc[:,'AmpOn_Imaginary_2_std'])[::sequence]
        # dic['AmpOn_Phase_1_minus_mean'] = np.array(branches[1].loc[:,'AmpOn_Phase_1_mean'])[::sequence] 
        # dic['AmpOn_Phase_1_minus_std'] = np.array(branches[1].loc[:,'AmpOn_Phase_1_std'])[::sequence]
        # dic['AmpOn_Phase_2_minus_mean'] = np.array(branches[1].loc[:,'AmpOn_Phase_2_mean'])[::sequence] 
        # dic['AmpOn_Phase_2_minus_std'] = np.array(branches[1].loc[:,'AmpOn_Phase_2_std'])[::sequence]
        # dic['AmpOn_Amplitude_minus_mean'] = np.array(branches[1].loc[:,'AmpOn_Amplitude_mean'])[::sequence] 
        # dic['AmpOn_Amplitude_minus_std'] = np.array(branches[1].loc[:,'AmpOn_Amplitude_std'])[::sequence]
        # #AmpOff
        # dic['AmpOff_Response_1_minus_mean'] = np.array(branches[1].loc[:,'AmpOff_Response_1_mean'])[::sequence] 
        # dic['AmpOff_Response_1_minus_std'] = np.array(branches[1].loc[:,'AmpOff_Response_1_std'])[::sequence]
        # dic['AmpOff_Response_2_minus_mean'] = np.array(branches[1].loc[:,'AmpOff_Response_2_mean'])[::sequence] 
        # dic['AmpOff_Response_2_minus_std'] = np.array(branches[1].loc[:,'AmpOff_Response_2_std'])[::sequence]
        # dic['AmpOff_Imaginary_1_minus_mean'] = np.array(branches[1].loc[:,'AmpOff_Imaginary_1_mean'])[::sequence] 
        # dic['AmpOff_Imaginary_1_minus_std'] = np.array(branches[1].loc[:,'AmpOff_Imaginary_1_std'])[::sequence]
        # dic['AmpOff_Imaginary_2_minus_mean'] = np.array(branches[1].loc[:,'AmpOff_Imaginary_2_mean'])[::sequence] 
        # dic['AmpOff_Imaginary_2_minus_std'] = np.array(branches[1].loc[:,'AmpOff_Imaginary_2_std'])[::sequence]
        # dic['AmpOff_Phase_1_minus_mean'] = np.array(branches[1].loc[:,'AmpOff_Phase_1_mean'])[::sequence] 
        # dic['AmpOff_Phase_1_minus_std'] = np.array(branches[1].loc[:,'AmpOff_Phase_1_std'])[::sequence]
        # dic['AmpOff_Phase_2_minus_mean'] = np.array(branches[1].loc[:,'AmpOff_Phase_2_mean'])[::sequence] 
        # dic['AmpOff_Phase_2_minus_std'] = np.array(branches[1].loc[:,'AmpOff_Phase_2_std'])[::sequence]
        # dic['AmpOff_Amplitude_minus_mean'] = np.array(branches[1].loc[:,'AmpOff_Amplitude_mean'])[::sequence] 
        # dic['AmpOff_Amplitude_minus_std'] = np.array(branches[1].loc[:,'AmpOff_Amplitude_std'])[::sequence]
        '''
        #Chek if arrays are of equal length
        if not (dic['AmpOn_Bias_plus_mean'].size == dic['AmpOn_Bias_minus_mean'].size):
            raise Exception('Response length Error: All arrays must have same size')       
        
        #Calculation of all statistical parameters in a loop
        signals = ['_1_', '_2_'] #Redefined to the old definition which fits here
        for bias_mode in bias_modes: 
            for signal in signals:
                #Auxiliary function and errors
                dic[bias_mode+'_Auxiliary'+signal+'minus_mean'] = -(dic[bias_mode+'_Response'+signal+'plus_mean'] - 
                                                                   dic[bias_mode+'_Response'+signal+'minus_mean'])/2
                dic[bias_mode+'_Auxiliary'+signal+'plus_mean'] = (dic[bias_mode+'_Response'+signal+'plus_mean'] + 
                                                                   dic[bias_mode+'_Response'+signal+'minus_mean'])/2
                dic[bias_mode+'_Auxiliary'+signal+'minus_std'] = np.sqrt((dic[bias_mode+'_Response'+signal+'plus_std']**2 +
                                                                         dic[bias_mode+'_Response'+signal+'minus_std']**2)/4)
                dic[bias_mode+'_Auxiliary'+signal+'plus_std'] = dic[bias_mode+'_Auxiliary'+signal+'minus_std']
                #Coercive Biases
                v0_plus, d_v0_plus = calc_root_discrete(dic['AmpOn_Bias_plus_mean'], dic[bias_mode+"_Response"+signal+'plus_mean'], 
                                                        x_err=dic['AmpOn_Bias_plus_std'],
                                                        y_err=dic[bias_mode+"_Response"+signal+'plus_std'])
                v0_minus, d_v0_minus = calc_root_discrete(dic['AmpOn_Bias_minus_mean'], dic[bias_mode+"_Response"+signal+'minus_mean'], 
                                                          x_err=dic['AmpOn_Bias_plus_std'],
                                                          y_err=dic[bias_mode+"_Response"+signal+'minus_std'])
                dis[bias_mode+"_Response"+signal+'V0_plus'] = v0_plus
                dis[bias_mode+"_Response"+signal+'V0_minus'] = v0_minus
                dis[bias_mode+"_Response"+signal+'V0_plus_err'] = d_v0_plus
                dis[bias_mode+"_Response"+signal+'V0_minus_err'] = d_v0_minus
                dis[bias_mode+"_Response"+signal+'Vc'] = np.absolute((v0_plus - v0_minus)/2)
                dis[bias_mode+"_Response"+signal+'Vc_err'] = np.sqrt((d_v0_plus/2)**2 + (d_v0_minus/2)**2)
                #Saturation responses and max switchable response
                dis[bias_mode+"_Response"+signal+'Rs_plus'] = dic[bias_mode+'_Auxiliary'+signal+'plus_mean'][-1]
                dis[bias_mode+"_Response"+signal+'Rs_minus'] = dic[bias_mode+'_Auxiliary'+signal+'plus_mean'][0]
                dis[bias_mode+"_Response"+signal+'Rs'] = np.absolute(dis[bias_mode+"_Response"+signal+'Rs_plus'] -
                                                                     dis[bias_mode+"_Response"+signal+'Rs_minus'])
                #Vertical shift of hysteresis and remanent switchable response
                dis[bias_mode+"_Response"+signal+'Rv'], dis[bias_mode+"_Response"+signal+'Rv_err'] = calc_root_discrete(
                                                                        dic[bias_mode+'_Auxiliary'+signal+'plus_mean'], 
                                                                        dic['AmpOn_Bias_plus_mean'],
                                                                        x_err=dic[bias_mode+'_Auxiliary'+signal+'plus_std'],
                                                                        y_err=dic['AmpOn_Bias_plus_std']) 
                r0, r0_err = calc_root_discrete(dic[bias_mode+'_Auxiliary'+signal+'minus_mean'], 
                                                dic['AmpOn_Bias_minus_mean'],
                                                x_err=dic[bias_mode+'_Auxiliary'+signal+'minus_std'],
                                                y_err=dic['AmpOn_Bias_minus_std'])
                dis[bias_mode+"_Response"+signal+'R0'] = np.absolute(2 * r0)
                dis[bias_mode+"_Response"+signal+'R0_err'] = 2 * r0_err
                #Work of switching 
                dis[bias_mode+"_Response"+signal+'Ads'] = np.absolute(
                                                            (dic['AmpOn_Bias_plus_mean'][-1] - dic['AmpOn_Bias_plus_mean'][0]) * 
                                                            np.sum(dic[bias_mode+'_Auxiliary'+signal+'minus_mean']) / 
                                                            dic[bias_mode+'_Auxiliary'+signal+'minus_mean'].size)
                #Imprint voltage - With vertical aux correction to prevent extreme values
                maxi = np.max(dic[bias_mode+'_Auxiliary'+signal+'minus_mean'])
                mini = np.min(dic[bias_mode+'_Auxiliary'+signal+'minus_mean'])
                                                        
                if not np.sign(maxi) == np.sign(mini):
                    if np.abs(maxi)<np.abs(mini):
                        aux_offset = - np.abs(maxi)
                    else: 
                        aux_offset = np.abs(mini)
                else: 
                    aux_offset = 0.0
                
                dis[bias_mode+"_Response"+signal+'Imd'] = (np.sum(aux_offset + dic[bias_mode+'_Auxiliary'+signal+'minus_mean'] *
                                                                  dic['AmpOn_Bias_plus_mean']) / 
                                                           np.sum(aux_offset + dic[bias_mode+'_Auxiliary'+signal+'minus_mean'])
                                                           )
                #Effective width of hysteresis loop
                dis[bias_mode+"_Response"+signal+'sigma-d'] = np.sqrt(np.absolute(
                                                                        np.sum(aux_offset + dic[bias_mode+'_Auxiliary'+signal+'minus_mean'] *
                                                                        (dic['AmpOn_Bias_plus_mean'] -
                                                                        dis[bias_mode+"_Response"+signal+'Imd'])**2) / 
                                                                        np.sum(aux_offset + dic[bias_mode+'_Auxiliary'+signal+'minus_mean'])))
                #Calc the right a 
                a = 2
                while ((dis[bias_mode+"_Response"+signal+'Imd'] + a * dis[bias_mode+"_Response"+signal+'sigma-d']) > 
                       dic['AmpOn_Bias_plus_mean'][-1] or 
                       (dis[bias_mode+"_Response"+signal+'Imd'] - a * dis[bias_mode+"_Response"+signal+'sigma-d']) < 
                              dic['AmpOn_Bias_plus_mean'][0]):
                    a = a - 0.1 
                dis[bias_mode+"_Response"+signal+'a'] = a
                #saturation responses
                dis[bias_mode+"_Response"+signal+'Rds_plus'] = np.mean(dic[bias_mode+'_Auxiliary'+signal+'plus_mean']
                                                                       [dic['AmpOn_Bias_plus_mean'] > 
                                                                        (dis[bias_mode+"_Response"+signal+'Imd'] + 
                                                                         a * dis[bias_mode+"_Response"+signal+'sigma-d'])])
                dis[bias_mode+"_Response"+signal+'Rds_minus'] = np.mean(dic[bias_mode+'_Auxiliary'+signal+'plus_mean']
                                                                       [dic['AmpOn_Bias_plus_mean'] < 
                                                                        (dis[bias_mode+"_Response"+signal+'Imd'] - 
                                                                         a * dis[bias_mode+"_Response"+signal+'sigma-d'])])
                dis[bias_mode+"_Response"+signal+'Rds'] = np.absolute(dis[bias_mode+"_Response"+signal+'Rs_plus'] -
                                                                     dis[bias_mode+"_Response"+signal+'Rs_minus'])
        
        #Construct a flag matrix for the phasefit; all curves that didnt pass linfit are not fittet in phase
        signals = ['1', '2']
        skipflags = []
        for bias_mode in bias_modes:
            lst=[]
            for signal in signals:
                lst.append(not dis['_'.join((bias_mode, 'Complex', signal, 'r-squared-flag'))])
            skipflags.append(lst)
        
        #Do the phasefit for the second prefiltering
        df = pd.DataFrame(dic) 
        dis['Aux_stat'] = df
        dis.update(self.calc_Forcecurve_Phasefit(df, skipflags=skipflags, display=display))
        
        #Construct a flag matrix for the responsefit; all curves that didnt pass linfit and phasefit will not be used
        signals = ['1', '2']
        skipflags = []
        for bias_mode in bias_modes:
            lst=[]
            for signal in signals:
                lst.append(not (dis['_'.join((bias_mode, 'Complex', signal, 'r-squared-flag'))] and
                                dis['_'.join((bias_mode, 'Phase', signal, 'r-squared-flag'))] and
                                dis['_'.join((bias_mode, 'Phase', signal, 'a1-a2-flag'))]))
            skipflags.append(lst)
        
        #Do the responsefit
        dis.update(self.calc_Forcecurve_Responsefit(df, skipflags=skipflags, display=display))
        
        #Make a Series        
        if index:
            curvename = index
        else:
            curvename = 'Line-'+ str(line) +'_Point-' + str(point)
        
        sr = pd.Series(dis, name=curvename) 
        
        return sr
    '''...........................................................................................................................'''
    def calc_Complex_Phasecorr(self, real, imag, display=False):
        '''
        Calculates the phase correcton for all signal points in the complex plane. 
        If applied, real part of signal should be maximized. 
        
        linear model according to (imag = k * real + d) is used.
        In the matrix form: imag = Ap with A = [[real, 1]] and p = [[k],[d]] 
        phi is calculated according to arctan
        
        Returns the phase angle and the sum of squared residuals (important for pre selection)
        '''
        
        #First we have to set up the coefficient matrix 
        A = np.vstack([real, np.ones(len(real))]).T
        
        #Do the least sqaures fit and calc the angle
        x, residuals, _, _ = np.linalg.lstsq(A, imag, rcond=None)
        theta = np.arctan(x[0])
        
        #We can also use the coefficient of determination 
        r_squared = SS_PFM.calc_R_squared(real, imag, SS_PFM.lin_fun, x)
        
        #Optional: Display the fit in Pyplot
        if display:
            text = '\n'.join(['$\\theta$ = {theta:.2f}'.format(theta=np.degrees(theta)),
                              'k = {k:.4e}'.format(k=x[0]),
                              'd = {d:.4e}'.format(d=x[1]),
                              '$R^2$ = {r_square:.3f}'.format(r_square=r_squared)])
            plt.plot(real,imag, 'bx')
            realf = np.linspace(np.min(real), np.max(real), 1000)
            imagf = x[0]*realf + x[1]
            plt.plot(realf, imagf, 'r-', zorder=1)
            plt.annotate(text, (0.7,1.05), xycoords='axes fraction', size='x-large')
        
        return theta, r_squared
    '''...........................................................................................................................'''
    def calc_Forcecurve_Phasecorr(self, fc, display=False, r_squared_min=0.75):
        '''
        Performs the phasecorrection for a phase curve set. 
        4 complex plane fits are done. 
        Returns a dictionary with the phase angles and linerrors
        '''
        
        #Create a subplot if display is True
        if display: 
            fig, ax = plt.subplots(2,2, figsize=(14,12))
        
        signals = [[('AmpOn','1'),('AmpOn','2')],
                   [('AmpOff','1'),('AmpOff','2')]]
        
        dic = {}
        #Iterate over the signals 
        for i in range(2):
            for j in range(2):
                if display:
                    plt.sca(ax[i,j])
                    plt.title('_'.join((signals[i][j][0], 'Complex', signals[i][j][1])))   
                
                #Extract the data    
                real = fc.loc[:,'_'.join((signals[i][j][0], 'Response', signals[i][j][1], 'mean'))]
                imag = fc.loc[:,'_'.join((signals[i][j][0], 'Imaginary', signals[i][j][1], 'mean'))]
                
                #Fit the data
                theta, r_squared = self.calc_Complex_Phasecorr(real, imag, display=display)
                
                #Store the data
                dic['_'.join((signals[i][j][0], 'Complex', signals[i][j][1], 'theta'))] = theta
                dic['_'.join((signals[i][j][0], 'Complex', signals[i][j][1], 'r-squared'))] = r_squared
                dic['_'.join((signals[i][j][0], 'Complex', signals[i][j][1], 'r-squared-flag'))] = r_squared > r_squared_min
                
        if display: 
            plt.tight_layout()
        
        return pd.Series(dic)
    '''...........................................................................................................................'''
    def calc_Phasefit(self, bias_plus, bias_minus, phi_plus, phi_minus, display=False, bounds=None): 
        '''
        Performs a phasefit according to the equation: phi(V) = b1 - b2 * (1-exp((V-b3)/b4))^-1 + b5 * V
        Returns the fit parameters 
        '''
        #First we need to guess the initial fitting parameters
        a1 = (np.max(phi_plus) + np.max(phi_minus))/2
        a2 = (np.max(phi_plus) - np.min(phi_plus) + np.max(phi_minus) - np.min(phi_minus))/2
        a3 = 0 #Start in the middleIf 
        a4 = 0 #Also middle
        #This is just a rule of thumb - for first approx
        if (np.argmax(phi_plus) + np.argmax(phi_minus)) < (np.argmin(phi_plus) + np.argmin(phi_minus)): 
            a5 = -1
        else: 
            a5 = 1
        a6 = 0
        
        #Construct the x-y fields
        bias = np.hstack((bias_plus, bias_minus))
        phi = np.hstack((phi_plus, phi_minus))
        
        #Define the boundaries of the fitparameters
        if not bounds:
            #          a1           a2      a3   a4   a5       a6
            bounds = ([np.min(phi), 0,     -50, -50, -np.inf, -0.4], #lower bounds
                      [np.max(phi), np.inf, 50,  50,  np.inf,  0.4]) #upper bounds
        
        #Do the fit
        try:
            p_out, cov = optimize.curve_fit(self.sigmoid_dual, bias, phi, p0=[a1,a2,a3,a4,a5,a6], bounds=bounds)
            r_squared = self.calc_R_squared(bias, phi, self.sigmoid_dual, p_out)
        except Exception as e: 
            print(e)
            p_out = np.array([np.nan for i in range(6)])
            cov = np.array([[np.nan for i in range(6)] for j in range(6)])
            r_squared = np.nan
            
        if display:
            plt.plot(bias_plus, phi_plus, 'bx')
            plt.plot(bias_minus, phi_minus, 'bx')
            biasf = np.linspace(-75,75,1000)
            phif_plus = self.sigmoid_single(biasf, p_out[0], p_out[1], p_out[2], p_out[4], p_out[5])
            phif_minus = self.sigmoid_single(biasf, p_out[0], p_out[1], p_out[3], p_out[4], p_out[5])
            plt.plot(biasf, phif_plus, 'r-', zorder=0)
            plt.plot(biasf, phif_minus, 'r-', zorder=0)
            text1 = '\n'.join(('$a_1$ = {a1:.2f}'.format(a1=p_out[0]),
                              '$a_2$ = {a2:.2f}'.format(a2=p_out[1]),
                              '$a_3$ = {a3:.2f}'.format(a3=p_out[2]),
                              '$a_4$ = {a4:.2f}'.format(a4=p_out[3])))
            text2 = '\n'.join(('$a_5$ = {a5:.2f}'.format(a5=p_out[4]),
                              '$a_6$ = {a6:.2f}'.format(a6=p_out[5]),
                              '$R^2$ = {r:.3f}'.format(r=r_squared)))
            plt.annotate(text1, (0.1,1.05), xycoords='axes fraction', size='x-large')
            plt.annotate(text2, (0.8,1.05), xycoords='axes fraction', size='x-large')
        
        return p_out, cov, r_squared
        
    '''...........................................................................................................................'''
    def calc_Forcecurve_Phasefit(self, fc, display=False, skipflags=[[False, False],[False, False]], r_squared_min=0.93, a2_min=85):
        '''
        Calculates the Phasefit for a Forcecurve
        '''
        #Create a subplot if display is True
        if display: 
            fig, ax = plt.subplots(2,2, figsize=(14,12))
        
        #Prepare the signals
        signals = [[('AmpOn','1'),('AmpOn','2')],
                   [('AmpOff','1'),('AmpOff','2')]]
        
        #Convert skipflags to np.array
        skipflags = np.array(skipflags)
        
        dic = {}
        #Iterate over the signals 
        for i in range(2):
            for j in range(2):
                if display:
                    plt.sca(ax[i,j])
                    plt.title('_'.join((signals[i][j][0], 'Phasefit', signals[i][j][1]))) 
                #Extract te right data from forcecurve
                bias_plus = fc.loc[:,'_'.join(('AmpOn', 'Bias_plus', 'mean'))]
                bias_minus = fc.loc[:,'_'.join(('AmpOn', 'Bias_minus', 'mean'))]
                phi_plus = fc.loc[:,'_'.join((signals[i][j][0], 'Phase', signals[i][j][1], 'plus', 'mean'))]
                phi_minus = fc.loc[:,'_'.join((signals[i][j][0], 'Phase', signals[i][j][1], 'minus', 'mean'))]
                
                if skipflags[i,j]:
                    p_out = np.array([np.nan for i in range(6)])
                    cov = np.array([[np.nan for i in range(6)] for j in range(6)])
                    r_squared = np.nan
                else:
                    #Do the fit
                    p_out, cov, r_squared = self.calc_Phasefit(bias_plus, bias_minus, phi_plus, phi_minus, display=display)
                    
                #Store the data
                dic['_'.join((signals[i][j][0], 'Phase', signals[i][j][1], 'fitparam'))] = p_out
                dic['_'.join((signals[i][j][0], 'Phase', signals[i][j][1], 'fitcov'))] = cov
                dic['_'.join((signals[i][j][0], 'Phase', signals[i][j][1], 'r-squared'))] = r_squared
                dic['_'.join((signals[i][j][0], 'Phase', signals[i][j][1], 'r-squared-flag'))] = r_squared > r_squared_min
                dic['_'.join((signals[i][j][0], 
                              'Phase', signals[i][j][1], 
                              'a1-a2-flag'))] = p_out[1] > a2_min and np.cos(p_out[0]*np.pi/180) * np.cos((p_out[0] - p_out[1])*np.pi/180)<0
                # 'a1-a2-flag'))] = p_out[1] > a2_min and p_out[0] > 90 > (p_out[0] - p_out[1])
                
        if display: 
            plt.tight_layout()
            
        return pd.Series(dic)
                
    '''...........................................................................................................................'''
    def calc_Responsefit(self, bias_plus, bias_minus, r_plus, r_minus, display=False, bounds=None): 
        '''
        Performs a response fit according to the equation: phi(V) = b1 - b2 * (1-exp((V-b3)/b4))^-1 + b5 * V
        Returns the fit parameters 
        '''
        #First we need to guess the initial fitting parameters
        a1 = (np.max(r_plus) + np.max(r_minus))/2
        a2 = (np.max(r_plus) - np.min(r_plus) + np.max(r_minus) - np.min(r_minus))/2
        a3 = 0 #Start in the middle
        a4 = 0 #Also middle
        #This is just a rule of thumb - for first approx
        if (np.argmax(r_plus) + np.argmax(r_minus)) < (np.argmin(r_plus) + np.argmin(r_minus)): 
            a5 = -1
        else: 
            a5 = 1
        a6 = 0
        
        #Construct the x-y fields
        bias = np.hstack((bias_plus, bias_minus))
        r = np.hstack((r_plus, r_minus))
        
        #Define the boundaries of the fitparameters
        if not bounds:
            #          a1          a2       a3   a4   a5       a6
            bounds = ([np.min(r),  0,      -50, -50, -np.inf, -np.inf], #lower bounds
                      [np.max(r),  np.inf,  50,  50,  np.inf,  np.inf]) #upper bounds
        
        #Do the fit
        try:
            p_out, cov = optimize.curve_fit(self.sigmoid_dual, bias, r, p0=[a1,a2,a3,a4,a5,a6], bounds=bounds)
            r_squared = self.calc_R_squared(bias, r, self.sigmoid_dual, p_out)
        except Exception as e: 
            print(e)
            p_out = np.array([np.nan for i in range(6)])
            cov = np.array([[np.nan for i in range(6)] for j in range(6)])
            r_squared = np.nan
            
        if display:
            plt.plot(bias_plus, r_plus, 'bx')
            plt.plot(bias_minus, r_minus, 'bx')
            biasf = np.linspace(-75,75,1000)
            rf_plus = self.sigmoid_single(biasf, p_out[0], p_out[1], p_out[2], p_out[4], p_out[5])
            rf_minus = self.sigmoid_single(biasf, p_out[0], p_out[1], p_out[3], p_out[4], p_out[5])
            plt.plot(biasf, rf_plus, 'r-', zorder=0)
            plt.plot(biasf, rf_minus, 'r-', zorder=0)
            text1 = '\n'.join(('$a_1$ = {a1:.2f}'.format(a1=p_out[0]),
                              '$a_2$ = {a2:.2f}'.format(a2=p_out[1]),
                              '$a_3$ = {a3:.2f}'.format(a3=p_out[2]),
                              '$a_4$ = {a4:.2f}'.format(a4=p_out[3])))
            text2 = '\n'.join(('$a_5$ = {a5:.2f}'.format(a5=p_out[4]),
                              '$a_6$ = {a6:.2f}'.format(a6=p_out[5]),
                              '$R^2$ = {r:.3f}'.format(r=r_squared)))
            plt.annotate(text1, (0.1,1.05), xycoords='axes fraction', size='x-large')
            plt.annotate(text2, (0.8,1.05), xycoords='axes fraction', size='x-large')
        
        return p_out, cov, r_squared
        
    '''...........................................................................................................................'''
    def calc_Forcecurve_Responsefit(self, fc, display=False, skipflags=[[False, False],[False, False]], r_squared_min=0.90):
        '''
        Calculates the Phasefit for a Forcecurve
        '''
        #Create a subplot if display is True
        if display: 
            fig, ax = plt.subplots(2,2, figsize=(14,12))
        
        #Prepare the signals
        signals = [[('AmpOn','1'),('AmpOn','2')],
                   [('AmpOff','1'),('AmpOff','2')]]
        
        #Convert skipflags to np.array
        skipflags = np.array(skipflags)
        
        dic = {}
        #Iterate over the signals 
        for i in range(2):
            comb_flag = False # For combined values of 1 & 2
            comb_Vf = 0.
            comb_Rfs = 0.
            comb_Afs = 0.
            comb_cnt = 0
            for j in range(2):
                if display:
                    plt.sca(ax[i,j])
                    plt.title('_'.join((signals[i][j][0], 'Responsefit', signals[i][j][1]))) 
                #Extract te right data from forcecurve
                bias_plus = fc.loc[:,'_'.join(('AmpOn', 'Bias_plus', 'mean'))]
                bias_minus = fc.loc[:,'_'.join(('AmpOn', 'Bias_minus', 'mean'))]
                r_plus = fc.loc[:,'_'.join((signals[i][j][0], 'Response', signals[i][j][1], 'plus', 'mean'))]*1e12
                r_minus = fc.loc[:,'_'.join((signals[i][j][0], 'Response', signals[i][j][1], 'minus', 'mean'))]*1e12
                
                if skipflags[i,j]:
                    p_out = np.array([np.nan for i in range(6)])
                    cov = np.array([[np.nan for i in range(6)] for j in range(6)])
                    r_squared = np.nan
                else:
                    #Do the fit
                    p_out, cov, r_squared = self.calc_Responsefit(bias_plus, bias_minus, r_plus, r_minus, display=display)
                    
                #Store the data
                if r_squared > r_squared_min: #Query if valid fit, otherwise mask with NaN
                    mask = 1
                else: 
                    mask = np.nan
                pretext = '_'.join((signals[i][j][0], 'Response', signals[i][j][1]))
                dic['_'.join((pretext, 'fitparam'))] = p_out
                dic['_'.join((pretext, 'fitcov'))] = cov
                dic['_'.join((pretext, 'r-squared'))] = r_squared
                dic['_'.join((pretext, 'r-squared-flag'))] = r_squared > r_squared_min
                dic['_'.join((pretext, 'Vf_plus'))] = p_out[2] * mask
                dic['_'.join((pretext, 'Vf_minus'))] = p_out[3] * mask
                dic['_'.join((pretext, 'Vf'))] = np.abs((p_out[2] - p_out[3])/2) * mask
                dic['_'.join((pretext, 'Rfs_plus'))] = p_out[0] * 1e-12 * mask
                dic['_'.join((pretext, 'Rfs_minus'))] = (p_out[0] - p_out[1]) * 1e-12 * mask
                dic['_'.join((pretext, 'Rfs'))] = p_out[1] *1e-12 * mask
                dic['_'.join((pretext, 'Imf'))] = (p_out[2] + p_out[3])/2 * mask
                dic['_'.join((pretext, 'Afs'))] = np.abs(p_out[2] - p_out[3]) * p_out[1] * 1e-12 * mask
                if p_out[2] > p_out[3]:
                    factor = 1
                else: 
                    factor = -1
                dic['_'.join((pretext, 'Vfc_plus'))] = p_out[2] - p_out[4] * factor * mask
                dic['_'.join((pretext, 'Vfc_minus'))] = p_out[3] + p_out[4] * factor * mask
                dic['_'.join((pretext, 'Vsurf'))] = p_out[0]/p_out[5] * mask

                # Combined data
                if r_squared > r_squared_min: 
                    comb_flag = True
                    comb_Vf += np.abs((p_out[2] - p_out[3])/2)
                    comb_Rfs += p_out[1] *1e-12
                    comb_Afs += np.abs(p_out[2] - p_out[3]) * p_out[1] * 1e-12 
                    comb_cnt += 1
            
            pretext = '_'.join((signals[i][0][0], 'Response', 'C'))
            #Now we generate the combined data
            if comb_flag: 
                dic['_'.join((pretext, 'r-squared-flag'))] = True
                dic['_'.join((pretext, 'Vf'))] = comb_Vf/comb_cnt
                dic['_'.join((pretext, 'Rfs'))] = comb_Rfs/comb_cnt
                dic['_'.join((pretext, 'Afs'))] = comb_Afs/comb_cnt
            else: 
                dic['_'.join((pretext, 'r-squared-flag'))] = False
                dic['_'.join((pretext, 'Vf'))] = np.nan
                dic['_'.join((pretext, 'Rfs'))] = np.nan
                dic['_'.join((pretext, 'Afs'))] = np.nan

                
        if display: 
            plt.tight_layout()
            
        return pd.Series(dic)
                
    '''...........................................................................................................................'''
    @staticmethod 
    def calc_SS_tot(y_i): 
        '''
        Calculates the sum of total sum of squares of an array of numbers
        '''
        y_m = np.mean(y_i)
        
        return np.sum((y_i - y_m)**2)
    '''...........................................................................................................................'''
    @staticmethod
    def calc_SS_ref(y_i, f_i):
        '''
        Calculates the residual sum of squares
        '''
        return np.sum((y_i - f_i)**2)
    '''...........................................................................................................................'''
    @staticmethod
    def calc_R_squared(x_i ,y_i, fun, param):
        '''
        Calculates the coefficient of determiation
        '''
        ss_tot = SS_PFM.calc_SS_tot(y_i)
        ss_ref = SS_PFM.calc_SS_ref(y_i, fun(x_i, *param))
        
        return 1 - ss_ref/ss_tot
    '''...........................................................................................................................'''
    @staticmethod
    def lin_fun(x_i, k, d): 
        '''
        Returns the y_i values corresponding to x_i of y=k*x+d
        '''
        return k * x_i + d 
    '''...........................................................................................................................'''
    @staticmethod 
    def sigmoid_single(bias, b1, b2, b3, b4, b5):
        '''
        Returns the sigmoid single function for a SS-PFM Phase evolution of one branch
        '''
        with warnings.catch_warnings(): #ignore the warnings that come from overflow
            warnings.simplefilter('ignore')
            return b1 - b2/(1 + np.exp((bias - b3)/b4)) + b5*bias
    '''...........................................................................................................................'''
    @staticmethod
    def sigmoid_dual(bias, a1, a2, a3, a4, a5, a6):
        '''
        Returns the sigmoid function of a single branch
        Bias must be an np.array with the individual datarrays stacked as two columns
        '''
        with warnings.catch_warnings(): #ignore the warnings that come from overflow
            warnings.simplefilter('ignore')
            return np.hstack((SS_PFM.sigmoid_single(bias[:int(len(bias)/2)], a1, a2, a3, a5, a6), 
                              SS_PFM.sigmoid_single(bias[int(len(bias)/2):], a1, a2, a4, a5, a6)))
    '''...........................................................................................................................'''
    @staticmethod
    def shift_to_equi_angle(angles, lower, upper, summand):
        '''
        Shifting to equivalent angles under upper and lower condition
        if angle < lower ==> +summand
        if angle > upper ==> -summand
        if in between ==> no action
        '''
        for i in range(len(angles)): 
            if angles[i] <lower: 
                angles[i] += summand
            if angles[i] >upper: 
                angles[i] -= summand
        
        return angles
    '''...........................................................................................................................'''
    
    def save(self):
        '''
        Saves the object
        '''
        POLDERs_data.save_Object(self.files, self.objDir)
    '''______'''
    '''Getter'''
    def get_param_set(self, filename) -> pd.DataFrame:
        '''
        Returns a parameter set for a file
        '''
        names = self.table.loc[:,'Filename'] #Get the series of sample names
        index = names[names==filename].index[0] #Get the index of the questioned sample
        
        return self.table.loc[index,:]
    '''...........................................................................................................................'''
    def get_table(self):
        '''
        Return the parameter table
        '''
        return self.table
    '''...........................................................................................................................'''
    def get_All_data(self, marked_only=False) -> dict:
        '''
        Return the whole data set; Also marked only possible
        '''
        if marked_only:
            print('To be done!')
        else:
            return self.files
    '''...........................................................................................................................'''
    def get_Forcemap_data(self, filename, marked_only=False) -> pd.DataFrame:
        '''
        Return one forcemap; Also marked only possible
        '''
        if marked_only:
            print('To be done!')
        else:
            return self.files[filename]
    '''...........................................................................................................................'''
    def get_Forcemap_index(self, filename, marked_only=False) -> pd.DataFrame:
        '''
        Return the index column of the Forcemap
        '''
        return self.files[filename].index.values
    '''...........................................................................................................................'''
    def get_Forcemap_row(self, filename, line, point, curvename='') -> pd.Series:
        '''
        Returns a single forcecurve, line and point must be integers
        '''
        if curvename == '':
            curvename = 'Line-'+ str(line) +'_Point-' + str(point)
            
        return self.files[filename].loc[curvename,:]
    '''...........................................................................................................................'''
    def get_Forcemap_column(self, filename, columnname, dropna = True) -> pd.Series:
        '''
        Returns a single column
        '''
        ser = self.files[filename].loc[:,columnname]
        
        if dropna: 
            ser = ser.dropna()
        
        return ser
    
    def get_Forcemap_column_stat(self, filename, columnname, stat):
        '''
        Returns a statistical parameter of a forcemap column 
        
        Argument stat: 
        dtype: String 
        Allowed keywords: 
            *mean
            *std
            *flag
        '''
        if stat == 'mean':
            return np.mean(self.get_Forcemap_column(filename, columnname))
        if stat == 'std':
            return np.std(self.get_Forcemap_column(filename, columnname))
        if stat == 'flag':
            return np.count_nonzero(np.array(self.get_Forcemap_column(filename, columnname)))/np.array(self.get_Forcemap_column(filename, columnname)).shape[0]
    
    def get_sample_value_evol(self, columnname, stat):
        '''
        Returns the evolution of a value of all files
        Argument columnname: 
        dtype: string 
        Usually same names like columnname in the other getter functions
        
        Argument stat: 
        dtype: String 
        Allowed keywords: 
            *mean
            *std
            *flag
        '''
        #Iterate over all Files and add the values to the dic
        dic = {}
        for filename in self.files.keys():
            dic[filename] = self.get_Forcemap_column_stat(filename, columnname, stat)
        
        return pd.Series(dic, name=columnname+'_'+stat)
    
            
    '''...........................................................................................................................'''
    def get_Forcemap_value(self, filename, columnname, line=0, point=0, index=''):
        if index:
            curvename = index
        else: 
            curvename = 'Line-'+ str(line) +'_Point-' + str(point)
        
        return self.files[filename].loc[curvename,columnname]
    '''...........................................................................................................................'''
    def get_Filenames(self):
        return self.files.keys()
    '''...........................................................................................................................'''
    def get_objDir(self):
        return self.objDir
    '''...........................................................................................................................'''
    def get_objDir_path(self):
        '''
        returns the path of the SS_PFM object
        '''
        return os.path.split(self.objDir)[0]
    '''...........................................................................................................................'''
    '''______'''
    '''Setter'''
    def set_All_data(self, files):
        '''
        Overwrites all data by the files given
        '''
        self.files = files
    '''...........................................................................................................................'''
    def set_Forcemap_data(self, file, filename):
        '''
        Adds an entry in the collection, existing data is overwritten
        '''
        self.files[filename] = file
    '''...........................................................................................................................'''
    def set_Forcemap_row(self, file, filename, point, line):
        '''
        Adds a row in the forcecurve; existing data is overwritten
        '''
        curvename = 'Line-'+ str(line) +'_Point-' + str(point)
        self.files[filename].loc[curvename] = file
    '''...........................................................................................................................'''
    def set_Forcemap_column(self, column, filename, columnname):
        '''
        Adds a columns in the forcecurve; existing data is overwritten
        '''
        self.files[filename].loc[:,columnname] = column
    '''...........................................................................................................................'''
    def set_Column_name(self, old_name, new_name):
        '''
        Renames the a column in all files
        '''
        for key in self.files.keys():
            self.files[key].rename(columns={old_name: new_name}, inplace=True)
    '''...........................................................................................................................'''
    '''______'''
    '''Adder'''
    def add_Forcecurve_columns(self, filename, cols, point=0, line=0, index=''):
        '''
        Adds columns to a forcecurve
        
        Argument 'cols':
            dtype = pandas.DataFrame
        '''
        #Construct the curvename
        if index:
            curvename = index
        else: 
            curvename = 'Line-'+ str(line) +'_Point-' + str(point)
        
        #Iterate over the columns and delete duplicates
        for colname in cols.columns: 
            try:
                del self.files[filename].at[curvename, 'Hysteresis'][colname]
            except:
                pass
                
        self.files[filename].at[curvename, 'Hysteresis'] = self.files[filename].loc[curvename, 'Hysteresis'].join(cols)
        
        return self.files[filename].loc[curvename, 'Hysteresis']
    '''...........................................................................................................................'''
    def add_Forcemap_columns(self, filename, cols):
        '''
        Adds columns to a forcecurve
        
        Argument 'cols':
            dtype = pandas.DataFrame
        '''
        
        #Iterate over the columns and delete duplicates
        for colname in cols.columns: 
            try:
                del self.files[filename][colname]
            except:
                pass
                
        self.files[filename] = self.files[filename].join(cols)
        
        return self.files[filename]
    '''...........................................................................................................................'''
    '''______'''
    '''Deleter'''
    def drop_Forcemap(self, filename): 
        '''
        Deletes a forcemap
        '''
        drop = {filename: self.files[filename]}
        POLDERs_data.save_Object(drop, filename + '.pkl', path=self.objDir_dropped)
        self.files.pop(filename)