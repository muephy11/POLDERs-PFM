# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 10:40:37 2022
Evaluation methods for the database
@author: Philipp
Here are the methods which are used for the graph evaluation 
"""
import data
import os #for getcwd()
import matplotlib as mpl
import matplotlib.pyplot as plt 
import numpy as np
from data import POLDERs_data
import pandas as pd
from collections import namedtuple
import copy
import tkinter as tk
from tkinter import filedialog

def directory_querry(mode=None):
    '''
    Opens the tkinter dialog for one file
    Allowed modes: 
    * None: standard mode for choosing a single file
    * files: choose multiple files
    * dir: choose a folder 
    '''
    root = tk.Tk()
    root.attributes('-topmost', True)
    root.iconify()
    if mode == 'dir':
        filename = filedialog.askdirectory()
    elif mode == 'files': 
        filename = filedialog.askopenfilenames()
    elif mode == 'save': 
        filename = filedialog.asksaveasfile()
    else: 
        filename = filedialog.askopenfilename()

    root.destroy()

    return filename

def graph_hist(x, *args, database=None, bins=None, range=None, density=False, weights=None, cumulative=False, bottom=None,
               histtype='bar', align='mid', orientation='vertical', rwidth=None, log=False, label = None, stacked=False, 
               kwargs_plot = {}, kwargs_grid={}, grid_on=True, legend=False, kwargs_legend={}, minorticks=True, tick_params={}, 
               title='', kwargs_title={}, xlabel='', ylabel='', kwargs_xlabel={}, kwargs_ylabel={},xscale='linear', 
               yscale='linear', kwargs_xscale={}, kwargs_yscale={}, kwargs_xticks={}, kwargs_yticks={},args_xlim=(), 
               args_ylim=(),kwargs_xlim={}, kwargs_ylim={}, tight_layout = False, show = False, xerr=None, yerr=None, color=None, 
               xscale_ext=1):
    '''
    Plots a histogram of the data
    '''
    #Calc number of bins with Sturge’s Rule
    if not bins:
        bins = int(np.round(1 + 3.322 * np.log10(len(x))))
    #Plot the curve 
    histObjects = plt.hist(x*xscale_ext, data=database, bins=bins, range=range, density=density, weights=weights, cumulative=cumulative, 
                          bottom=bottom, histtype=histtype, align=align, orientation=orientation, rwidth=rwidth, log=log, 
                          label=label, stacked=stacked, **kwargs_plot, color=color)
    
    #Plot the grid
    plt.grid(visible=grid_on, **kwargs_grid)
    
    #Place a legend if wanted
    if legend: 
        plt.legend(**kwargs_legend)
        
    #Minorticks
    if minorticks:
        plt.minorticks_on()
    else:
        plt.minorticks_off()
    #Tick_params
    plt.tick_params(**tick_params)
    
    #Title
    plt.title(title, **kwargs_title)
    
    #X and Y Label
    plt.xlabel(xlabel, **kwargs_xlabel)
    plt.ylabel(ylabel, **kwargs_ylabel)
    #X and Y scale
    plt.xscale(xscale, **kwargs_xscale)
    plt.yscale(yscale, **kwargs_yscale)
    #X and Y ticks
    plt.xticks(**kwargs_xticks)
    plt.yticks(**kwargs_yticks)
    #X and Y lim
    plt.xlim(*args_xlim, **kwargs_xlim)
    plt.ylim(*args_ylim, **kwargs_ylim)
    
    #Check if show() is true
    if show:
        plt.show()
    return histObjects

def graph_None(*args, **kwargs):
    '''
    Places blanc white space, no content 
    '''
    lineObject = plt.plot(np.array([]),np.array([]))
    plt.axis('off')
    
    return lineObject

def graph_textbox(x, y, title='', kwargs_title={}, *args, **kwargs):
    '''
    Textbox on blanc space
    Formated text is stored in y
    '''
    
    #Create a graph
    lineObject = graph_None()
    
    #Title
    plt.title(title, **kwargs_title)
    
    if not type(y) == dict: 
        y = {'sdsd':y}
    cnt = 0.0
    length = float(len(y))
    for header in y.keys():
        plt.annotate(header, xy=(cnt/length, 1), xytext=(-15, -15), fontsize=15,
        xycoords='axes fraction', textcoords='offset points',
        bbox=dict(facecolor='white', alpha=0.8),
        horizontalalignment='left', verticalalignment='top')
        
        plt.annotate(y[header], xy=(cnt/length, 0.90), xytext=(-15, -15), fontsize=10,
        xycoords='axes fraction', textcoords='offset points',
        bbox=dict(facecolor='white', alpha=0.8),
        horizontalalignment='left', verticalalignment='top')
        
        cnt+=1.0

def graph_2D(x, y, fmt='', database=None, kwargs_plot = {}, kwargs_grid={}, grid_on=True, legend=False, kwargs_legend={},
             minorticks=True, tick_params={}, title='', kwargs_title={}, xlabel='', ylabel='', kwargs_xlabel={}, kwargs_ylabel={},
             xscale='linear', yscale='linear', kwargs_xscale={}, kwargs_yscale={}, kwargs_xticks={}, kwargs_yticks={},
             args_xlim=(), args_ylim=(),kwargs_xlim={}, kwargs_ylim={}, tight_layout = False, show = False, presort=True, 
             xerr=None, yerr=None, linestyle=None, marker=None, markersize=None, color=None, mask_nan=True, 
             xscale_ext=1, yscale_ext=1, label_ext=None):
    '''
    Generates a basic graph with the data provided 
    '''
    
    #Generate the plot
    if not type(x) == dict:
        x = {'Line': x}
    if not type(y) == dict:
        y = {'Line': y}
    if not x.keys() == y.keys():
        temp = copy.deepcopy(x[list(x.keys())[0]])
        x = {}
        for key in y.keys():
            x[key] = temp
            
    if not type(xerr) == dict:
        xerr = {}
        for key in x.keys():
            xerr[key] = None
    if not type(yerr) == dict:
        yerr = {}
        for key in y.keys():
            yerr[key] = None
    
    if not x.keys() == kwargs_plot.keys(): 
        temp = copy.deepcopy(kwargs_plot)
        kwargs_plot = {}
        for key in x.keys():
            kwargs_plot[key] = temp
    
    if not type(linestyle) == dict: 
        temp = copy.deepcopy(linestyle)
        linestyle = {}
        for key in x.keys():
            linestyle[key] = temp
            
    if not type(marker) == dict: 
        temp = copy.deepcopy(marker)
        marker= {}
        for key in x.keys():
            marker[key] = temp
    
    if not type(markersize) == dict: 
        temp = copy.deepcopy(markersize)
        markersize= {}
        for key in x.keys():
            markersize[key] = temp
    
    if not type(color) == dict: 
        temp = copy.deepcopy(color)
        color= {}
        for key in x.keys():
            color[key] = temp
        
    lineObjects = []
    
    for key in x.keys():
        #Change x and y and error to array
        x[key] = np.array(x[key])
        y[key] = np.array(y[key])
        if not type(xerr[key]) == type(None):
            xerr[key] = np.array(xerr[key])
        if not type(yerr[key]) == type(None):
            yerr[key] = np.array(yerr[key])

        if presort:
            args = np.argsort(x[key])
            x[key] = x[key][args]
            y[key] = y[key][args]
            if not type(xerr[key]) == type(None):
                xerr[key] = xerr[key][args]
            if not type(yerr[key]) == type(None):
                yerr[key] = yerr[key][args]
        
        #Remove all nan values
        mask = True
        if mask_nan:
            mask = ~np.isnan(x[key]) * ~np.isnan(y[key])
            x[key] = x[key][mask]
            y[key] = y[key][mask]
            if not type(xerr[key]) == type(None):
                xerr[key] = xerr[key][mask]
            if not type(yerr[key]) == type(None):
                yerr[key] = yerr[key][mask]
        
        #External Labels
        label = label_ext[key] if label_ext else key
            
        #Plot
        if fmt:
            lineObjects.append(plt.errorbar(x[key]*xscale_ext, y[key]*yscale_ext, fmt, data=database, label=label, xerr=xerr[key], 
                                            yerr=yerr[key], linestyle=linestyle[key], marker=marker[key], 
                                            markersize=markersize[key], color = color[key], **kwargs_plot[key]))
        else: 
            lineObjects.append(plt.errorbar(x[key]*xscale_ext, y[key]*yscale_ext, data=database, label=label, xerr=xerr[key], 
                                            yerr=yerr[key], linestyle=linestyle[key], marker=marker[key], 
                                            markersize=markersize[key], color = color[key], **kwargs_plot[key]))
    #Plot the grid
    plt.grid(visible=grid_on, **kwargs_grid)
    
    #Place a legend if wanted
    if legend: 
        plt.legend(**kwargs_legend)
    
    #Minorticks
    if minorticks:
        plt.minorticks_on()
    else:
        plt.minorticks_off()
    #Tick_params
    plt.tick_params(**tick_params)
    
    #Title
    plt.title(title, **kwargs_title)
    
    #X and Y Label
    plt.xlabel(xlabel, **kwargs_xlabel)
    plt.ylabel(ylabel, **kwargs_ylabel)
    #X and Y scale
    plt.xscale(xscale, **kwargs_xscale)
    plt.yscale(yscale, **kwargs_yscale)
    #X and Y ticks
    plt.xticks(**kwargs_xticks)
    plt.yticks(**kwargs_yticks)
    #X and Y lim
    plt.xlim(*args_xlim, **kwargs_xlim)
    plt.ylim(*args_ylim, **kwargs_ylim)
    
    #Check if show() is true
    if show:
        plt.show()
    return lineObjects
    '''...........................................................................................................................'''

def flex_subplot(functions, functions_args, functions_kwargs, subplots_kwargs={}, tight_layout=True, show=False, title='', 
                 title_kwargs={}, style='default'):
    '''
    Creates a flexible Subplots, made out of varius different graph functions
    Hand the arguments as numpy array!
    Returns the figure and ax object!
    functions must be an array of function objects
    function_args must be an object that can return the args list through a get() method!
    function_kwargs must be an object that can return the args collection through a get() method!
    '''
    #Style
    plt.style.use(style)
    
    #Check if the input is an nd
    if not isinstance(functions, np.ndarray):
        raise Exception('Argument function must be of type: numpy.ndarray')
    if not isinstance(functions_args, np.ndarray):
        raise Exception('Argument args must be of type: numpy.ndarray')
    if not isinstance(functions_kwargs, np.ndarray):
        raise Exception('Argument kwargs must be of type: numpy.ndarray')
    
    #We have to check if the 3 argument fields have the same dimension
    if not functions.shape == functions_args.shape == functions_kwargs.shape:
        raise Exception('Input Arrays must have the same shape')
    
    rows, columns = functions.shape
    
    #Create subplot
    fig, ax = plt.subplots(rows, columns, **subplots_kwargs)
    plt.suptitle(title, **title_kwargs)
    
    #Iterate over all subplots and generate them
    lineObjects = []
    for row in range(rows):
        lo_line = []
        for column in range(columns):
            plt.sca(ax[row, column])
            #fkw = functions_kwargs[row, column].get()
            lo_line.append(functions[row, column](*functions_args[row, column].get(), **functions_kwargs[row, column].get()))#right function is called
        lineObjects.append(lo_line)
    #lineObjects = np.array(lineObjects)
    if tight_layout:
        plt.tight_layout()
    #Check if show() is true
    if show:
        plt.show()
    
    return fig, ax, lineObjects
    '''...........................................................................................................................'''

def save_pyplot_figure(fig, filename, path='', file_extension='.png', kwargs_savefig={}):
    '''
    Saves the pyplot in the given path. 
    Standard format is .png
    '''
    #Check wether path is empty == if yes, try to split up the filename
    if path == '':
        filename, path = POLDERs_data.seperate_file_folder(filename)
    
    #If the path is not empty, check wether path exists and if not, create it! 
    if not path == '':
        if not os.path.exists(path):
            os.makedirs(path)
    
    #Now the file is saved!
    print('Saving file: '+filename)
    plt.figure(fig)
    plt.savefig(os.path.join(path, filename),**kwargs_savefig)
    print('Done!')

'''_______________________________________________________________________________________________________________________________'''
class args():
    '''
    Wraps arg lists in an object to enable storing in numpy array
    Can also manipulate and order Data
    '''
    def __init__(self, *args):
        merged= []
        for arg in args:
            if isinstance(arg,(list, tuple)):
                merged.extend(arg)
            else: 
                merged.append(arg)
        self.merged = merged
    def get(self): 
        return self.merged
'''_______________________________________________________________________________________________________________________________'''
class kwargs():
    '''
    Wraps the kwargs collection in an object to enable storing in numpy array
    Can also manipulate and order Data
    Second version of it with dynamic sub dic extender
    Maybe consider a recursive approach in the third verson.
    '''

    def __init__(self, *kwargs):
        kwargs_copy = copy.deepcopy(kwargs)
        self.merged = {}
        merged = self.merged
        for kwarg in kwargs_copy:
            #Seperate the duplicate keys from the new ones
            intersect = set(kwarg.keys()).intersection(set(self.merged.keys()))
            difference = set(kwarg.keys()) - set(self.merged.keys())
            #Extend the duplicates
            for inter in intersect: 
                #Check wether intesect is dict; otherwise overwrite and not update
                if type(kwarg[inter]) == dict:    
                    self.merged[inter].update(kwarg[inter])
                else:
                    self.merged[inter] = kwarg[inter]
            #Add new kwargs
            for diff in difference: 
               self.merged[diff] = kwarg[diff]
 
    def get(self):
        return self.merged
'''_______________________________________________________________________________________________________________________________'''
class SS_PFM_manager():
    '''
    Runs the data manager routine for a sample
    '''    
    '''...........................................................................................................................'''
    #signal kwargs for the function kwargs
    Signal_params = namedtuple('Signal_paramset', 'scaling_factor label args_lim') #Contains general signal kwargs
    Arrangement_params = namedtuple('Arrangement_paramset', 'x_signals y_signals subplot_kwargs plot_kwargs subplot_title') #Contains subplot arrag kwargs
    Mode_params = namedtuple('Mode_paramset', 'plot_kwargs')
    Title_params = namedtuple('Title_paramset', 'texts param_order suptitle_kwargs')
    
    global_kwargs = {'AmpOn_Phase_1': Signal_params(1, r'$\phi$ [°]', (None,None)), 
                     'AmpOn_Phase_2': Signal_params(1, r'$\phi$ [°]', (None,None)), 
                     'AmpOff_Phase_1': Signal_params(1, r'$\phi$ [°]', (None,None)), 
                     'AmpOff_Phase_2': Signal_params(1, r'$\phi$ [°]', (None,None)),
                     
                     'AmpOn_Amplitude': Signal_params(1e12, 'A [pm]', None),
                     'AmpOff_Amplitude': Signal_params(1e12, 'A [pm]', None),
                     
                     'AmpOff_Response_1': Signal_params(1e12, r'$A \; cos(\phi) \; [pm]$', (None,None)),
                     'AmpOff_Response_2': Signal_params(1e12, r'$A \; cos(\phi) \; [pm]$', (None,None)),
                     'AmpOn_Response_1': Signal_params(1e12, r'$A \; cos(\phi) \; [pm]$', (None,None)),
                     'AmpOn_Response_2': Signal_params(1e12, r'$A \; cos(\phi) \; [pm]$', (None,None)),
                     
                     'AmpOff_Response_1_plus': Signal_params(1e12, r'$A \; cos(\phi) \; [pm]$', (None,None)),
                     'AmpOff_Response_2_plus': Signal_params(1e12, r'$A \; cos(\phi) \; [pm]$', (None,None)),
                     'AmpOn_Response_1_plus': Signal_params(1e12, r'$A \; cos(\phi) \; [pm]$', (None,None)),
                     'AmpOn_Response_2_plus': Signal_params(1e12, r'$A \; cos(\phi) \; [pm]$', (None,None)),
                     
                     'AmpOff_Response_1_minus': Signal_params(1e12, r'$A \; cos(\phi) \; [pm]$', (None,None)),
                     'AmpOff_Response_2_minus': Signal_params(1e12, r'$A \; cos(\phi) \; [pm]$', (None,None)),
                     'AmpOn_Response_1_minus': Signal_params(1e12, r'$A \; cos(\phi) \; [pm]$', (None,None)),
                     'AmpOn_Response_2_minus': Signal_params(1e12, r'$A \; cos(\phi) \; [pm]$', (None,None)),
                     
                     'AmpOff_Auxiliary_1_plus': Signal_params(1e12, r'$\Delta^+ \; [pm]$', (None,None)),
                     'AmpOff_Auxiliary_2_plus': Signal_params(1e12, r'$\Delta^+ \; [pm]$', (None,None)),
                     'AmpOn_Auxiliary_1_plus': Signal_params(1e12, r'$\Delta^+ \; [pm]$', (None,None)),
                     'AmpOn_Auxiliary_2_plus': Signal_params(1e12, r'$\Delta^+ \; [pm]$', (None,None)),
                     
                     'AmpOff_Auxiliary_1_minus': Signal_params(1e12, r'$\Delta^- \; [pm]$', (None,None)),
                     'AmpOff_Auxiliary_2_minus': Signal_params(1e12, r'$\Delta^- \; [pm]$', (None,None)),
                     'AmpOn_Auxiliary_1_minus': Signal_params(1e12, r'$\Delta^- \; [pm]$', (None,None)),
                     'AmpOn_Auxiliary_2_minus': Signal_params(1e12, r'$\Delta^- \; [pm]$', (None,None)),
                     
                     'AmpOff_Imaginary_1': Signal_params(1e12, r'$A \; sin(\phi) \; [pm]$', (None,None)),
                     'AmpOff_Imaginary_2': Signal_params(1e12, r'$A \; sin(\phi) \; [pm]$', (None,None)),
                     'AmpOn_Imaginary_1': Signal_params(1e12, r'$A \; sin(\phi) \; [pm]$', (None,None)),
                     'AmpOn_Imaginary_2': Signal_params(1e12, r'$A \; sin(\phi) \; [pm]$', (None,None)),
                     
                     'AmpOn_Bias': Signal_params(1, 'U [V]', (-50,50)),
                     'AmpOn_Bias_plus': Signal_params(1, 'U [V]', (-50,50)),
                     'Temp': Signal_params(1, '$T \; [^{\circ}C\, ]$', (None,None)),
                     'None': Signal_params(1, '', (None,None)),
                     'Params_AmpOn': Signal_params(1, '', (None,None)),
                     'Params_AmpOff': Signal_params(1, '', (None,None)),
                     
                     'AmpOn_Response_1_flag': Signal_params(100, r'$N \; [\,\%\,]$', (0,100)),
                     'AmpOn_Response_2_flag': Signal_params(100, r'$N \; [\,\%\,]$', (0,100)),
                     'AmpOff_Response_1_flag': Signal_params(100, r'$N \; [\,\%\,]$', (0,100)),
                     'AmpOff_Response_2_flag': Signal_params(100, r'$N \; [\,\%\,]$', (0,100)),
                     'AmpOn_Response_C_flag': Signal_params(100, r'$N \; [\,\%\,]$', (0,100)),
                     'AmpOff_Response_C_flag': Signal_params(100, r'$N \; [\,\%\,]$', (0,100)),
                     'response_flag': Arrangement_params([['Temp','Temp','Temp'],
                                                        ['Temp','Temp', 'Temp']], 
                                                       [['AmpOn_Response_1_flag', 'AmpOn_Response_2_flag', 'AmpOn_Response_C_flag'],
                                                        ['AmpOff_Response_1_flag', 'AmpOff_Response_2_flag', 'AmpOff_Response_C_flag']], 
                                                       {'figsize':(12.8, 8), 'sharex':True},
                                                       [[{'xlabel':''}, {'xlabel':''}, {'xlabel':''}],
                                                        [{},{},{}]],
                                                       Title_params({'text1':': Number of accepted curves from filter'}, 
                                                                    ['samplename', 'text1'],
                                                                    {'weight':'bold', 'size':'x-large'})
                                                        ),
                     
                     'AmpOn_Response_1_Vc': Signal_params(1, r'$V_c \; [V]$', (0,None)),
                     'AmpOn_Response_2_Vc': Signal_params(1, r'$V_c \; [V]$', (0,None)),
                     'AmpOff_Response_1_Vc': Signal_params(1, r'$V_c \; [V]$', (0,None)),
                     'AmpOff_Response_2_Vc': Signal_params(1, r'$V_c \; [V]$', (0,None)),
                     'response_Vc': Arrangement_params([['Temp','Temp'],
                                                        ['Temp','Temp']], 
                                                       [['AmpOn_Response_1_Vc', 'AmpOn_Response_2_Vc'],
                                                        ['AmpOff_Response_1_Vc', 'AmpOff_Response_2_Vc']], 
                                                       {'figsize':(13.5, 12), 'sharex':True},
                                                       [[{'xlabel':''},{'xlabel':''}],
                                                        [{},{}]],
                                                       Title_params({'text1':'Temperature dependent coercive bias of the responses'}, 
                                                                    ['text1'],
                                                                    {'weight':'bold', 'size':'x-large'})
                                                       ),
                     'AmpOn_Response_1_Vf': Signal_params(1, r'$V_f \; [V\,]$', (0,None)),
                     'AmpOn_Response_2_Vf': Signal_params(1, r'$V_f \; [V\,]$', (0,None)),
                     'AmpOff_Response_1_Vf': Signal_params(1, r'$V_f \; [V\,]$', (0,None)),
                     'AmpOff_Response_2_Vf': Signal_params(1, r'$V_f \; [V\,]$', (0,None)),
                     'AmpOn_Response_C_Vf': Signal_params(1, r'$V_f \; [V\;]$', (0,None)),
                     'AmpOff_Response_C_Vf': Signal_params(1, r'$V_f \; [V\;]$', (0,None)),
                     'response_Vf': Arrangement_params([['Temp','Temp'],
                                                        ['Temp','Temp']], 
                                                       [['AmpOn_Response_1_Vf', 'AmpOn_Response_2_Vf'],
                                                        ['AmpOff_Response_1_Vf', 'AmpOff_Response_2_Vf']], 
                                                       {'figsize':(13.5, 12), 'sharex':True},
                                                       [[{'xlabel':''},{'xlabel':''}],
                                                        [{},{}]],
                                                       Title_params({'text1':'Temperature dependent coercive bias of the responses'}, 
                                                                    ['text1'],
                                                                    {'weight':'bold', 'size':'x-large'})
                                                       ),
                     
                     'AmpOn_Response_1_Vfc_plus': Signal_params(1, r'$V_{fc}^+ \; [V]$', (None,None)),
                     'AmpOn_Response_2_Vfc_plus': Signal_params(1, r'$V_{fc}^+ \; [V]$', (None,None)),
                     'AmpOff_Response_1_Vfc_plus': Signal_params(1, r'$V_{fc}^+ \; [V]$', (None,None)),
                     'AmpOff_Response_2_Vfc_plus': Signal_params(1, r'$V_{fc}^+ \; [V]$', (None,None)),
                     'response_Vfc_plus': Arrangement_params([['Temp','Temp'],
                                                        ['Temp','Temp']], 
                                                       [['AmpOn_Response_1_Vfc_plus', 'AmpOn_Response_2_Vfc_plus'],
                                                        ['AmpOff_Response_1_Vfc_plus', 'AmpOff_Response_2_Vfc_plus']], 
                                                       {'figsize':(13.5, 12), 'sharex':True},
                                                       [[{'xlabel':''},{'xlabel':''}],
                                                        [{},{}]],
                                                       Title_params({'text1':'Temperature dependent positive nucleation bias of the responses'}, 
                                                                    ['text1'],
                                                                    {'weight':'bold', 'size':'x-large'})
                                                       ),
                     
                     'AmpOn_Response_1_Vfc_minus': Signal_params(1, r'$V_{fc}^- \; [V]$', (None,None)),
                     'AmpOn_Response_2_Vfc_minus': Signal_params(1, r'$V_{fc}^- \; [V]$', (None,None)),
                     'AmpOff_Response_1_Vfc_minus': Signal_params(1, r'$V_{fc}^- \; [V]$', (None,None)),
                     'AmpOff_Response_2_Vfc_minus': Signal_params(1, r'$V_{fc}^- \; [V]$', (None,None)),
                     'response_Vfc_minus': Arrangement_params([['Temp','Temp'],
                                                        ['Temp','Temp']], 
                                                       [['AmpOn_Response_1_Vfc_minus', 'AmpOn_Response_2_Vfc_minus'],
                                                        ['AmpOff_Response_1_Vfc_minus', 'AmpOff_Response_2_Vfc_minus']], 
                                                       {'figsize':(13.5, 12), 'sharex':True},
                                                       [[{'xlabel':''},{'xlabel':''}],
                                                        [{},{}]],
                                                       Title_params({'text1':'Temperature dependent negative nucleation bias of the responses'}, 
                                                                    ['text1'],
                                                                    {'weight':'bold', 'size':'x-large'})
                                                       ),
                     
                     'AmpOn_Response_1_Vsurf': Signal_params(1, r'$V_{surf} \; [V]$', (None,None)),
                     'AmpOn_Response_2_Vsurf': Signal_params(1, r'$V_{surf} \; [V]$', (None,None)),
                     'AmpOff_Response_1_Vsurf': Signal_params(1, r'$V_{surf} \; [V]$', (None,None)),
                     'AmpOff_Response_2_Vsurf': Signal_params(1, r'$V_{surf} \; [V]$', (None,None)),
                     'response_Vsurf': Arrangement_params([['Temp','Temp'],
                                                        ['Temp','Temp']], 
                                                       [['AmpOn_Response_1_Vsurf', 'AmpOn_Response_2_Vsurf'],
                                                        ['AmpOff_Response_1_Vsurf', 'AmpOff_Response_2_Vsurf']], 
                                                       {'figsize':(13.5, 12), 'sharex':True},
                                                       [[{'xlabel':''},{'xlabel':''}],
                                                        [{},{}]],
                                                       Title_params({'text1':'Temperature dependent surface potential of the responses'}, 
                                                                    ['text1'],
                                                                    {'weight':'bold', 'size':'x-large'})
                                                       ),
                     
                     'AmpOn_Response_1_Rs': Signal_params(1e12, r'$R_s \; [pm]$', (0,None)),
                     'AmpOn_Response_2_Rs': Signal_params(1e12, r'$R_s \; [pm]$', (0,None)),
                     'AmpOff_Response_1_Rs': Signal_params(1e12, r'$R_s \; [pm]$', (0,None)),
                     'AmpOff_Response_2_Rs': Signal_params(1e12, r'$R_s \; [pm]$', (0,None)),
                     'response_Rs': Arrangement_params([['Temp','Temp'],
                                                        ['Temp','Temp']], 
                                                       [['AmpOn_Response_1_Rs', 'AmpOn_Response_2_Rs'],
                                                        ['AmpOff_Response_1_Rs', 'AmpOff_Response_2_Rs']], 
                                                       {'figsize':(13.5, 12), 'sharex':True},
                                                       [[{'xlabel':''},{'xlabel':''}],
                                                        [{},{}]],
                                                       Title_params({'text1':'Temperature dependent maximum switchable response'}, 
                                                                    ['text1'],
                                                                    {'weight':'bold', 'size':'x-large'})
                                                       ),
                     
                     'AmpOn_Response_1_Rfs': Signal_params(1e12, r'$R_{fs} \; [\, pm]$', (0,None)),
                     'AmpOn_Response_2_Rfs': Signal_params(1e12, r'$R_{fs} \; [\, pm]$', (0,None)),
                     'AmpOff_Response_1_Rfs': Signal_params(1e12, r'$R_{fs} \; [\, pm]$', (0,None)),
                     'AmpOff_Response_2_Rfs': Signal_params(1e12, r'$R_{fs} \; [\, pm]$', (0,None)),
                     'AmpOn_Response_C_Rfs': Signal_params(1e12, r'$R_{fs} \; [\, pm]$', (0,None)),
                     'AmpOff_Response_C_Rfs': Signal_params(1e12, r'$R_{fs} \; [\, pm]$', (0,None)),
                     'response_Rfs': Arrangement_params([['Temp','Temp'],
                                                        ['Temp','Temp']], 
                                                       [['AmpOn_Response_1_Rfs', 'AmpOn_Response_2_Rfs'],
                                                        ['AmpOff_Response_1_Rfs', 'AmpOff_Response_2_Rfs']], 
                                                       {'figsize':(13.5, 12), 'sharex':True},
                                                       [[{'xlabel':''},{'xlabel':''}],
                                                        [{},{}]],
                                                       Title_params({'text1':'Temperature dependent saturation response'}, 
                                                                    ['text1'],
                                                                    {'weight':'bold', 'size':'x-large'})
                                                       ),
                     
                     'AmpOn_Response_1_Rv': Signal_params(1e12, r'$R_v \; [pm]$', (None,None)),
                     'AmpOn_Response_2_Rv': Signal_params(1e12, r'$R_v \; [pm]$', (None,None)),
                     'AmpOff_Response_1_Rv': Signal_params(1e12, r'$R_v \; [pm]$', (None,None)),
                     'AmpOff_Response_2_Rv': Signal_params(1e12, r'$R_v \; [pm]$', (None,None)),
                     'response_Rv': Arrangement_params([['Temp','Temp'],
                                                        ['Temp','Temp']], 
                                                       [['AmpOn_Response_1_Rv', 'AmpOn_Response_2_Rv'],
                                                        ['AmpOff_Response_1_Rv', 'AmpOff_Response_2_Rv']], 
                                                       {'figsize':(13.5, 12), 'sharex':True},
                                                       [[{'xlabel':''},{'xlabel':''}],
                                                        [{},{}]],
                                                       Title_params({'text1':'Temperature dependent vertical shift of the hysteresis'}, 
                                                                    ['text1'],
                                                                    {'weight':'bold', 'size':'x-large'})
                                                       ),
                     
                     'AmpOn_Response_1_R0': Signal_params(1e12, r'$R_0 \; [pm]$', (0,None)),
                     'AmpOn_Response_2_R0': Signal_params(1e12, r'$R_0 \; [pm]$', (0,None)),
                     'AmpOff_Response_1_R0': Signal_params(1e12, r'$R_0 \; [pm]$', (0,None)),
                     'AmpOff_Response_2_R0': Signal_params(1e12, r'$R_0 \; [pm]$', (0,None)),
                     'response_R0': Arrangement_params([['Temp','Temp'],
                                                        ['Temp','Temp']], 
                                                       [['AmpOn_Response_1_R0', 'AmpOn_Response_2_R0'],
                                                        ['AmpOff_Response_1_R0', 'AmpOff_Response_2_R0']], 
                                                       {'figsize':(13.5, 12), 'sharex':True},
                                                       [[{'xlabel':''},{'xlabel':''}],
                                                        [{},{}]],
                                                       Title_params({'text1':'Temperature dependent remanent switchable response'}, 
                                                                    ['text1'],
                                                                    {'weight':'bold', 'size':'x-large'})
                                                       ),
                     
                     'AmpOn_Response_1_Ads': Signal_params(1, r'$A_{ds} \; [Vm]$', (0,None)),
                     'AmpOn_Response_2_Ads': Signal_params(1, r'$A_{ds} \; [Vm]$', (0,None)),
                     'AmpOff_Response_1_Ads': Signal_params(1, r'$A_{ds} \; [Vm]$', (0,None)),
                     'AmpOff_Response_2_Ads': Signal_params(1, r'$A_{ds} \; [Vm]$', (0,None)),
                     'response_Ads': Arrangement_params([['Temp','Temp'],
                                                         ['Temp','Temp']], 
                                                        [['AmpOn_Response_1_Ads', 'AmpOn_Response_2_Ads'],
                                                         ['AmpOff_Response_1_Ads', 'AmpOff_Response_2_Ads']], 
                                                        {'figsize':(13.5, 12), 'sharex':True},
                                                        [[{'xlabel':''},{'xlabel':''}],
                                                         [{},{}]],
                                                        Title_params({'text1':'Temperature dependent work of switching'}, 
                                                                     ['text1'],
                                                                     {'weight':'bold', 'size':'x-large'})
                                                       ),
                     
                     'AmpOn_Response_1_Afs': Signal_params(1, r'$A_{fs} \; [Vm]$', (0,None)),
                     'AmpOn_Response_2_Afs': Signal_params(1, r'$A_{fs} \; [Vm]$', (0,None)),
                     'AmpOff_Response_1_Afs': Signal_params(1, r'$A_{fs} \; [Vm]$', (0,None)),
                     'AmpOff_Response_2_Afs': Signal_params(1, r'$A_{fs} \; [Vm]$', (0,None)),
                     'AmpOn_Response_C_Afs': Signal_params(1e12, r'$A_{fs} \; [V\, pm]$', (0,None)),
                     'AmpOff_Response_C_Afs': Signal_params(1e12, r'$A_{fs} \; [V\, pm]$', (0,None)),
                     'response_Afs': Arrangement_params([['Temp','Temp'],
                                                         ['Temp','Temp']], 
                                                        [['AmpOn_Response_1_Afs', 'AmpOn_Response_2_Afs'],
                                                         ['AmpOff_Response_1_Afs', 'AmpOff_Response_2_Afs']], 
                                                        {'figsize':(13.5, 12), 'sharex':True},
                                                        [[{'xlabel':''},{'xlabel':''}],
                                                         [{},{}]],
                                                        Title_params({'text1':'Temperature dependent work of switching'}, 
                                                                     ['text1'],
                                                                     {'weight':'bold', 'size':'x-large'})
                                                       ),
                     
                     'AmpOn_Response_1_Imd': Signal_params(1, r'$Im_d \; [V]$', (None,None)),
                     'AmpOn_Response_2_Imd': Signal_params(1, r'$Im_d \; [V]$', (None,None)),
                     'AmpOff_Response_1_Imd': Signal_params(1, r'$Im_d \; [V]$', (None,None)),
                     'AmpOff_Response_2_Imd': Signal_params(1, r'$Im_d \; [V]$', (None,None)),
                     'response_Imd': Arrangement_params([['Temp','Temp'],
                                                         ['Temp','Temp']], 
                                                        [['AmpOn_Response_1_Imd', 'AmpOn_Response_2_Imd'],
                                                         ['AmpOff_Response_1_Imd', 'AmpOff_Response_2_Imd']], 
                                                        {'figsize':(13.5, 12), 'sharex':True},
                                                        [[{'xlabel':''},{'xlabel':''}],
                                                         [{},{}]],
                                                        Title_params({'text1':'Temperature dependent imprint voltage'}, 
                                                                     ['text1'],
                                                                     {'weight':'bold', 'size':'x-large'})
                                                       ),
                     
                     'AmpOn_Response_1_Imf': Signal_params(1, r'$Im_f \; [V]$', (None,None)),
                     'AmpOn_Response_2_Imf': Signal_params(1, r'$Im_f \; [V]$', (None,None)),
                     'AmpOff_Response_1_Imf': Signal_params(1, r'$Im_f \; [V]$', (None,None)),
                     'AmpOff_Response_2_Imf': Signal_params(1, r'$Im_f \; [V]$', (None,None)),
                     'response_Imf': Arrangement_params([['Temp','Temp'],
                                                         ['Temp','Temp']], 
                                                        [['AmpOn_Response_1_Imf', 'AmpOn_Response_2_Imf'],
                                                         ['AmpOff_Response_1_Imf', 'AmpOff_Response_2_Imf']], 
                                                        {'figsize':(13.5, 12), 'sharex':True},
                                                        [[{'xlabel':''},{'xlabel':''}],
                                                         [{},{}]],
                                                        Title_params({'text1':'Temperature dependent imprint voltage'}, 
                                                                     ['text1'],
                                                                     {'weight':'bold', 'size':'x-large'})
                                                       ),
                     
                     'AmpOn_Response_1_sigma-d': Signal_params(1, r'$\sigma _d \; [V]$', (0,None)),
                     'AmpOn_Response_2_sigma-d': Signal_params(1, r'$\sigma _d \; [V]$', (0,None)),
                     'AmpOff_Response_1_sigma-d': Signal_params(1, r'$\sigma _d \; [V]$', (0,None)),
                     'AmpOff_Response_2_sigma-d': Signal_params(1, r'$\sigma _d \; [V]$', (0,None)),
                     'response_sigma-d': Arrangement_params([['Temp','Temp'],
                                                             ['Temp','Temp']], 
                                                            [['AmpOn_Response_1_sigma-d', 'AmpOn_Response_2_sigma-d'],
                                                             ['AmpOff_Response_1_sigma-d', 'AmpOff_Response_2_sigma-d']], 
                                                            {'figsize':(13.5, 12), 'sharex':True},
                                                            [[{'xlabel':''},{'xlabel':''}],
                                                             [{},{}]],
                                                            Title_params({'text1':'Temperature dependent effective width of hysteresis loop'}, 
                                                                         ['text1'],
                                                                         {'weight':'bold', 'size':'x-large'})
                                                       ),
                     
                     'AmpOn_Response_1_Rds': Signal_params(1e12, r'$R_{ds} \; [pm]$', (0,None)),
                     'AmpOn_Response_2_Rds': Signal_params(1e12, r'$R_{ds} \; [pm]$', (0,None)),
                     'AmpOff_Response_1_Rds': Signal_params(1e12, r'$R_{ds} \; [pm]$', (0,None)),
                     'AmpOff_Response_2_Rds': Signal_params(1e12, r'$R_{ds} \; [pm]$', (0,None)),
                     'response_Rds': Arrangement_params([['Temp','Temp'],
                                                         ['Temp','Temp']], 
                                                        [['AmpOn_Response_1_Rds', 'AmpOn_Response_2_Rds'],
                                                         ['AmpOff_Response_1_Rds', 'AmpOff_Response_2_Rds']], 
                                                        {'figsize':(13.5, 12), 'sharex':True},
                                                        [[{'xlabel':''},{'xlabel':''}],
                                                         [{},{}]],
                                                        Title_params({'text1':'Temperature dependent discrete remanent switchable response'}, 
                                                                     ['text1'],
                                                                     {'weight':'bold', 'size':'x-large'})
                                                       ),
                     'response_1_Vc_R0_Ads': Arrangement_params(
                                                         [['Temp','Temp','Temp'],
                                                          ['Temp','Temp', 'Temp']], 
                                                         [['AmpOn_Response_1_Vc', 'AmpOn_Response_1_R0', 'AmpOn_Response_1_Ads'],
                                                          ['AmpOff_Response_1_Vc', 'AmpOff_Response_1_R0','AmpOff_Response_1_Ads']], 
                                                         {'figsize':(15, 10), 'sharex':True},
                                                         [[{'xlabel':''},{'xlabel':''},{'xlabel':''}],
                                                          [{},{},{}]],
                                                         Title_params({'text1':''}, 
                                                                     ['text1'],
                                                                     {'weight':'bold', 'size':'x-large'})
                                                       ),
                    'response_1_Vf_Rfs_Afs': Arrangement_params(
                                                         [['Temp','Temp','Temp'],
                                                          ['Temp','Temp', 'Temp']], 
                                                         [['AmpOn_Response_1_Vf', 'AmpOn_Response_1_Rfs', 'AmpOn_Response_1_Afs'],
                                                          ['AmpOff_Response_1_Vf', 'AmpOff_Response_1_Rfs','AmpOff_Response_1_Afs']], 
                                                         {'figsize':(15, 10), 'sharex':True},
                                                         [[{'xlabel':''},{'xlabel':''},{'xlabel':''}],
                                                          [{},{},{}]],
                                                         Title_params({'text1':''}, 
                                                                     ['text1'],
                                                                     {'weight':'bold', 'size':'x-large'})
                                                       ),
                    'response_C_Vf_Rfs_Afs': Arrangement_params(
                                                         [['Temp','Temp','Temp'],
                                                          ['Temp','Temp', 'Temp']], 
                                                         [['AmpOn_Response_C_Vf', 'AmpOn_Response_C_Rfs', 'AmpOn_Response_C_Afs'],
                                                          ['AmpOff_Response_C_Vf', 'AmpOff_Response_C_Rfs','AmpOff_Response_C_Afs']], 
                                                         {'figsize':(15, 10), 'sharex':True},
                                                         [[{'xlabel':''},{'xlabel':''},{'xlabel':''}],
                                                          [{},{},{}]],
                                                         Title_params({'text1':''}, 
                                                                     ['text1'],
                                                                     {'weight':'bold', 'size':'x-large'})
                                                       ),
                    'response_C_Vf_Rfs_Afs_flag': Arrangement_params(
                                                         [['Temp','Temp','Temp', 'Temp'],
                                                          ['Temp','Temp', 'Temp', 'Temp']], 
                                                         [['AmpOn_Response_C_Vf', 'AmpOn_Response_C_Rfs', 'AmpOn_Response_C_Afs', 'AmpOn_Response_C_flag'],
                                                          ['AmpOff_Response_C_Vf', 'AmpOff_Response_C_Rfs','AmpOff_Response_C_Afs', 'AmpOff_Response_C_flag']], 
                                                         {'figsize':(20, 10), 'sharex':True},
                                                         [[{'xlabel':''},{'xlabel':''},{'xlabel':''}, {'xlabel':''}],
                                                          [{},{},{},{}]],
                                                         Title_params({'text1':''}, 
                                                                     ['text1'],
                                                                     {'weight':'bold', 'size':'x-large'})
                                                       ),
                    'response_C_Vf_Rfs_Afs_flag_T': Arrangement_params(
                                                         [['Temp','Temp'],
                                                          ['Temp', 'Temp'],
                                                          ['Temp','Temp'], 
                                                          ['Temp', 'Temp']], 
                                                         [['AmpOn_Response_C_Vf', 'AmpOff_Response_C_Vf'],
                                                          ['AmpOn_Response_C_Rfs', 'AmpOff_Response_C_Rfs'],
                                                          ['AmpOn_Response_C_Afs', 'AmpOff_Response_C_Afs'],
                                                          ['AmpOn_Response_C_flag', 'AmpOff_Response_C_flag']], 
                                                         {'figsize':(14*0.8, 19*0.8), 'sharex':True},
                                                         [[{'xlabel':''}, {'xlabel':''}],
                                                          [{'xlabel':''}, {'xlabel':''}],
                                                          [{'xlabel':''}, {'xlabel':''}],
                                                          [{}, {}]],
                                                         Title_params({'text1':''}, 
                                                                     ['text1'],
                                                                     {'weight':'bold', 'size':'x-large'})
                                                       ),
                     
                     'response': Arrangement_params([['AmpOn_Bias','AmpOn_Bias'],
                                                     ['AmpOn_Bias','AmpOn_Bias']], 
                                                    [['AmpOn_Response_1', 'AmpOn_Response_2'],
                                                     ['AmpOff_Response_1', 'AmpOff_Response_2']], 
                                                    {'figsize':(9, 8), 'sharex':True},
                                                    [[{'xlabel':''},{'xlabel':''}],
                                                     [{},{}]],
                                                    Title_params({'text1':': Response forcecurves '}, 
                                                                 ['samplename','mode','text1','temp'],
                                                                 {'weight':'bold', 'size':'x-large'})
                                                    ),
                     'imaginary': Arrangement_params([['AmpOn_Bias','AmpOn_Bias'],
                                                     ['AmpOn_Bias','AmpOn_Bias']], 
                                                    [['AmpOn_Imaginary_1', 'AmpOn_Imaginary_2'],
                                                     ['AmpOff_Imaginary_1', 'AmpOff_Imaginary_2']], 
                                                    {'figsize':(9, 8), 'sharex':True},
                                                    [[{'xlabel':''},{'xlabel':''}],
                                                     [{},{}]],
                                                    Title_params({'text1':': Imaginary forcecurves '}, 
                                                                 ['samplename','mode','text1','temp'],
                                                                 {'weight':'bold', 'size':'x-large'})
                                                    ),
                     'complex': Arrangement_params([['AmpOn_Response_1','AmpOn_Response_2'],
                                                     ['AmpOff_Response_1','AmpOff_Response_2']], 
                                                    [['AmpOn_Imaginary_1', 'AmpOn_Imaginary_2'],
                                                     ['AmpOff_Imaginary_1', 'AmpOff_Imaginary_2']], 
                                                    {'figsize':(9, 8)},
                                                    [[{},{}],
                                                     [{},{}]],
                                                    Title_params({'text1':': Signals in complex plane '}, 
                                                                 ['samplename','mode','text1','temp'],
                                                                 {'weight':'bold', 'size':'x-large'})
                                                    ),
                     
                     'auxiliary': Arrangement_params([['AmpOn_Bias_plus','AmpOn_Bias_plus','AmpOn_Bias_plus','AmpOn_Bias_plus'],
                                                      ['AmpOn_Bias_plus','AmpOn_Bias_plus','AmpOn_Bias_plus','AmpOn_Bias_plus']], 
                                                    [['AmpOn_Auxiliary_1_plus', 'AmpOn_Auxiliary_2_plus', 
                                                      'AmpOn_Auxiliary_1_minus', 'AmpOn_Auxiliary_2_minus'],
                                                     ['AmpOff_Auxiliary_1_plus', 'AmpOff_Auxiliary_2_plus', 
                                                       'AmpOff_Auxiliary_1_minus', 'AmpOff_Auxiliary_2_minus']], 
                                                    {'figsize':(18, 8), 'sharex':True},
                                                    [[{'xlabel':''},{'xlabel':''},{'xlabel':''},{'xlabel':''},
                                                      {'xlabel':''},{'xlabel':''},{'xlabel':''},{'xlabel':''}],
                                                     [{},{},{},{},{},{},{},{}]],
                                                    Title_params({'text1':': Auxiliary functions '}, 
                                                                 ['samplename','mode','text1','temp'],
                                                                 {'weight':'bold', 'size':'x-large'})
                                                    ),
                     
                     'raw': Arrangement_params([['AmpOn_Bias','AmpOn_Bias','AmpOn_Bias'],
                                                ['AmpOn_Bias','AmpOn_Bias','AmpOn_Bias']], 
                                               [['AmpOn_Phase_1', 'AmpOn_Phase_2', 'AmpOn_Amplitude'],
                                                ['AmpOff_Phase_1', 'AmpOff_Phase_2', 'AmpOff_Amplitude']], 
                                               {'figsize':(12.8,8), 'sharex':True},
                                               [[{'xlabel':''},{'xlabel':''},{'xlabel':''}],
                                                [{},{},{}]],
                                               Title_params({'text1':': Raw forcecurves '}, 
                                                            ['samplename','mode','text1','temp'],
                                                            {'weight':'bold', 'size':'x-large'})
                                               ),

                    'AmpOff_P1_A': Arrangement_params([['AmpOn_Bias', 'AmpOn_Bias'],
                                                       ['None', 'None']], 
                                               [['AmpOff_Phase_1', 'AmpOff_Amplitude'], 
                                                ['None', 'None']], 
                                               {'figsize':(10.2,9), 'sharex':False},
                                               [[{},{}],[{},{}]],
                                               Title_params({'text1':''}, 
                                                            ['text1'],
                                                            {'weight':'bold', 'size':'x-large'})
                                               ),
                     
                     'hist': Arrangement_params([['AmpOn_Response_1_Vc', 'AmpOn_Response_2_Vc',
                                                  'AmpOn_Response_1_Rs', 'AmpOn_Response_2_Rs',
                                                  'AmpOn_Response_1_Rv', 'AmpOn_Response_2_Rv'],
                                                 ['AmpOff_Response_1_Vc', 'AmpOff_Response_2_Vc',
                                                  'AmpOff_Response_1_Rs', 'AmpOff_Response_2_Rs',
                                                  'AmpOff_Response_1_Rv', 'AmpOff_Response_2_Rv'],
                                                 ['AmpOn_Response_1_R0', 'AmpOn_Response_2_R0',
                                                  'AmpOn_Response_1_Ads', 'AmpOn_Response_2_Ads',
                                                  'AmpOn_Response_1_Imd', 'AmpOn_Response_2_Imd'],
                                                 ['AmpOff_Response_1_R0', 'AmpOff_Response_2_R0',
                                                  'AmpOff_Response_1_Ads', 'AmpOff_Response_2_Ads',
                                                  'AmpOff_Response_1_Imd', 'AmpOff_Response_2_Imd'],
                                                 ['AmpOn_Response_1_sigma-d', 'AmpOn_Response_2_sigma-d',
                                                  'AmpOn_Response_1_Rds', 'AmpOn_Response_2_Rds',
                                                  'None', 'None'],
                                                 ['AmpOff_Response_1_sigma-d', 'AmpOff_Response_2_sigma-d',
                                                  'AmpOff_Response_1_Rds', 'AmpOff_Response_2_Rds',
                                                  'None', 'None']],
                                                [['None', 'None', 'None', 'None', 'None', 'None'],
                                                 ['None', 'None', 'None', 'None', 'None', 'None'],
                                                 ['None', 'None', 'None', 'None', 'None', 'None'],
                                                 ['None', 'None', 'None', 'None', 'None', 'None'],
                                                 ['None', 'None', 'None', 'None', 'Params_AmpOn', 'None'],
                                                 ['None', 'None', 'None', 'None', 'Params_AmpOff', 'None']],
                                                {'figsize':(30, 29)},
                                                [[{},{},{},{},{},{}],
                                                 [{},{},{},{},{},{}],
                                                 [{},{},{},{},{},{}],
                                                 [{},{},{},{},{},{}],
                                                 [{},{},{},{},{},{}],
                                                 [{},{},{},{},{},{}]],
                                                Title_params({'text1':': Histograms of hysteresis\' char. values '}, 
                                                             ['samplename','text1','temp'],
                                                             {'weight':'bold', 'size':'xx-large', 'y':0.995})
                                                ),
                     
                     'overview': Arrangement_params([['AmpOn_Bias','AmpOn_Bias','AmpOn_Bias','AmpOn_Bias_plus',
                                                      'AmpOn_Bias_plus','AmpOn_Bias_plus','AmpOn_Bias_plus'],
                                                     ['AmpOn_Bias','AmpOn_Bias','AmpOn_Bias','AmpOn_Bias_plus',
                                                      'AmpOn_Bias_plus','AmpOn_Bias_plus','AmpOn_Bias_plus'],
                                                     ['AmpOn_Bias','AmpOn_Bias','AmpOn_Bias','AmpOn_Bias',
                                                      'AmpOn_Response_1','AmpOn_Response_2','None'],
                                                     ['AmpOn_Bias','AmpOn_Bias','AmpOn_Bias','AmpOn_Bias',
                                                      'AmpOff_Response_1','AmpOff_Response_2','None']], 
                                                    [['AmpOn_Phase_1', 'AmpOn_Phase_2', 'AmpOn_Amplitude',
                                                      'AmpOn_Auxiliary_1_plus', 'AmpOn_Auxiliary_2_plus',
                                                      'AmpOn_Auxiliary_1_minus', 'AmpOn_Auxiliary_2_minus'],
                                                     ['AmpOff_Phase_1', 'AmpOff_Phase_2', 'AmpOff_Amplitude',
                                                      'AmpOff_Auxiliary_1_plus', 'AmpOff_Auxiliary_2_plus',
                                                      'AmpOff_Auxiliary_1_minus', 'AmpOff_Auxiliary_2_minus'],
                                                     ['AmpOn_Response_1', 'AmpOn_Response_2','AmpOn_Imaginary_1', 
                                                      'AmpOn_Imaginary_2','AmpOn_Imaginary_1', 'AmpOn_Imaginary_2',
                                                      'Params_AmpOn'],
                                                     ['AmpOff_Response_1', 'AmpOff_Response_2','AmpOff_Imaginary_1',
                                                      'AmpOff_Imaginary_2','AmpOff_Imaginary_1', 'AmpOff_Imaginary_2',
                                                      'Params_AmpOff']], 
                                                    {'figsize':(35,18)},
                                                    [[{},{},{},{},{},{},{}],
                                                     [{},{},{},{},{},{},{}],
                                                     [{},{},{},{},{},{},{}],
                                                     [{},{},{},{},{},{},{}]],
                                                    Title_params({'text1':': Overview of all signals '}, 
                                                                 ['samplename','mode','text1','temp'],
                                                                 {'weight':'bold', 'size':'xx-large', 'y':0.99})
                                                    ),
                     
                     'custom': Arrangement_params([['None' for i in range(10)] for j in range(10)], 
                                                  [['None' for i in range(10)] for j in range(10)], 
                                                  {}, 
                                                  [[{} for i in range(10)] for j in range(10)], 
                                                  Title_params({'text1':''}, ['text1'],{})
                                                       ),
                     'single': Mode_params({'kwargs_plot':{}, 'linestyle':'-'}),
                     'super': Mode_params({'kwargs_plot':{},'markersize':0.4, 'linestyle':'', 'legend':False}),
                     'evol': Mode_params({'kwargs_plot':{}, 'marker':'o', 'linestyle':'-'}),
                     'single_dist': Mode_params({'kwargs_plot':{},})
                     }
    
    #Definition of combined function kwargs
    Function_params = namedtuple('Function_paramset', 'function function_kwargs')
    function_kwargs = pd.DataFrame(
                        {'AmpOn_Bias': pd.Series({
                            'AmpOn_Phase_1': Function_params(graph_2D, {'kwargs_grid':{'ls':'--'}, 'kwargs_plot':{},
                                                                        'xlabel': global_kwargs['AmpOn_Bias'].label,
                                                                        'ylabel': global_kwargs['AmpOn_Phase_1'].label, 
                                                                        'args_xlim': global_kwargs['AmpOn_Bias'].args_lim,
                                                                        'presort':False,
                                                                        'legend':True,
                                                                        'color':'blue',
                                                                        'marker':{'Forward branch':'>', 'Reverse branch':'<'},
                                                                        'title': 'AmpOn: Phase 1'}), 
                            'AmpOn_Phase_2': Function_params(graph_2D, {'kwargs_grid':{'ls':'--'}, 'kwargs_plot':{},
                                                                        'xlabel': global_kwargs['AmpOn_Bias'].label,
                                                                        'ylabel': global_kwargs['AmpOn_Phase_2'].label, 
                                                                        'args_xlim': global_kwargs['AmpOn_Bias'].args_lim,
                                                                        'presort':False,
                                                                        'legend':True,
                                                                        'color':'red',
                                                                        'marker':{'Forward branch':'>', 'Reverse branch':'<'},
                                                                        'title': 'AmpOn: Phase 2'}), 
                            'AmpOn_Amplitude': Function_params(graph_2D, {'kwargs_grid':{'ls':'--'}, 'kwargs_plot':{},
                                                                          'xlabel': global_kwargs['AmpOn_Bias'].label,
                                                                          'ylabel': global_kwargs['AmpOn_Amplitude'].label, 
                                                                          'args_xlim': global_kwargs['AmpOn_Bias'].args_lim,
                                                                          'presort':False,
                                                                          'legend':True,
                                                                          'color':'green',
                                                                          'marker':{'Forward branch':'>', 'Reverse branch':'<'},
                                                                          'title': 'AmpOn: Amplitude'}), 
                            'AmpOff_Phase_1': Function_params(graph_2D, {'kwargs_grid':{'ls':'--'}, 'kwargs_plot':{},
                                                                         'xlabel': global_kwargs['AmpOn_Bias'].label,
                                                                         'ylabel': global_kwargs['AmpOff_Phase_1'].label, 
                                                                         'args_xlim': global_kwargs['AmpOn_Bias'].args_lim,
                                                                         'presort':False,
                                                                         'legend':True,
                                                                         'color':'blue',
                                                                         'marker':{'Forward branch':'>', 'Reverse branch':'<'},
                                                                         'title': 'AmpOff: Phase 1'}), 
                            'AmpOff_Phase_2': Function_params(graph_2D, {'kwargs_grid':{'ls':'--'}, 'kwargs_plot':{},
                                                                         'xlabel': global_kwargs['AmpOn_Bias'].label,
                                                                         'ylabel': global_kwargs['AmpOff_Phase_2'].label, 
                                                                         'args_xlim': global_kwargs['AmpOn_Bias'].args_lim,
                                                                         'presort':False,
                                                                         'legend':True,
                                                                         'color':'red',
                                                                         'marker':{'Forward branch':'>', 'Reverse branch':'<'},
                                                                         'title': 'AmpOff: Phase 2'}), 
                            'AmpOff_Amplitude': Function_params(graph_2D, {'kwargs_grid':{'ls':'--'}, 'kwargs_plot':{},
                                                                           'xlabel': global_kwargs['AmpOn_Bias'].label,
                                                                           'ylabel': global_kwargs['AmpOff_Amplitude'].label, 
                                                                           'args_xlim': global_kwargs['AmpOn_Bias'].args_lim,
                                                                           'presort':False,
                                                                           'legend':True,
                                                                           'color':'green',
                                                                           'marker':{'Forward branch':'>', 'Reverse branch':'<'},
                                                                           'title': 'AmpOff: Amplitude'}),
                            'AmpOn_Response_1': Function_params(graph_2D, {'kwargs_grid':{'ls':'--'}, 
                                                                           'kwargs_plot':{},
                                                                            'xlabel': global_kwargs['AmpOn_Bias'].label,
                                                                            'ylabel': global_kwargs['AmpOn_Response_1'].label, 
                                                                            'args_xlim': global_kwargs['AmpOn_Bias'].args_lim,
                                                                            'presort':False,
                                                                            'legend':True,
                                                                            'color':'blue',
                                                                            'marker':{'Forward branch':'>', 'Reverse branch':'<'},
                                                                            'title': 'AmpOn: Response 1'}), 
                            'AmpOn_Response_2': Function_params(graph_2D, {'kwargs_grid':{'ls':'--'}, 'kwargs_plot':{},
                                                                            'xlabel': global_kwargs['AmpOn_Bias'].label,
                                                                            'ylabel': global_kwargs['AmpOn_Response_2'].label, 
                                                                            'args_xlim': global_kwargs['AmpOn_Bias'].args_lim,
                                                                            'presort':False,
                                                                            'legend':True,
                                                                            'color':'red',
                                                                            'marker':{'Forward branch':'>', 'Reverse branch':'<'},
                                                                            'title': 'AmpOn: Response 2'}),
                            'AmpOff_Response_1': Function_params(graph_2D, {'kwargs_grid':{'ls':'--'}, 'kwargs_plot':{},
                                                                           'xlabel': global_kwargs['AmpOn_Bias'].label,
                                                                           'ylabel': global_kwargs['AmpOff_Response_1'].label, 
                                                                           'args_xlim': global_kwargs['AmpOn_Bias'].args_lim,
                                                                           'presort':False,
                                                                           'legend':True,
                                                                           'color':'blue',
                                                                           'marker':{'Forward branch':'>', 'Reverse branch':'<'},
                                                                           'title': 'AmpOff: Response 1'}), 
                            'AmpOff_Response_2': Function_params(graph_2D, {'kwargs_grid':{'ls':'--'}, 'kwargs_plot':{},
                                                                           'xlabel': global_kwargs['AmpOn_Bias'].label,
                                                                           'ylabel': global_kwargs['AmpOff_Response_2'].label, 
                                                                           'args_xlim': global_kwargs['AmpOn_Bias'].args_lim,
                                                                           'presort':False,
                                                                           'legend':True,
                                                                           'color':'red',
                                                                           'marker':{'Forward branch':'>', 'Reverse branch':'<'},
                                                                           'title': 'AmpOff: Response 2'}),
                            'AmpOn_Imaginary_1': Function_params(graph_2D, {'kwargs_grid':{'ls':'--'}, 'kwargs_plot':{},
                                                                            'xlabel': global_kwargs['AmpOn_Bias'].label,
                                                                            'ylabel': global_kwargs['AmpOn_Imaginary_1'].label, 
                                                                            'args_xlim': global_kwargs['AmpOn_Bias'].args_lim,
                                                                            'presort':False,
                                                                            'legend':True,
                                                                            'color':'blue',
                                                                            'marker':{'Forward branch':'>', 'Reverse branch':'<'},
                                                                            'title': 'AmpOn: Imaginary 1'}), 
                            'AmpOn_Imaginary_2': Function_params(graph_2D, {'kwargs_grid':{'ls':'--'}, 'kwargs_plot':{},
                                                                            'xlabel': global_kwargs['AmpOn_Bias'].label,
                                                                            'ylabel': global_kwargs['AmpOn_Imaginary_2'].label, 
                                                                            'args_xlim': global_kwargs['AmpOn_Bias'].args_lim,
                                                                            'presort':False,
                                                                            'legend':True,
                                                                            'color':'red',
                                                                            'marker':{'Forward branch':'>', 'Reverse branch':'<'},
                                                                            'title': 'AmpOn: Imaginary 2'}),
                            'AmpOff_Imaginary_1': Function_params(graph_2D, {'kwargs_grid':{'ls':'--'}, 'kwargs_plot':{},
                                                                           'xlabel': global_kwargs['AmpOn_Bias'].label,
                                                                           'ylabel': global_kwargs['AmpOff_Imaginary_1'].label, 
                                                                           'args_xlim': global_kwargs['AmpOn_Bias'].args_lim,
                                                                           'presort':False,
                                                                           'legend':True,
                                                                           'color':'blue',
                                                                           'marker':{'Forward branch':'>', 'Reverse branch':'<'},
                                                                           'title': 'AmpOff: Imaginary 1'}), 
                            'AmpOff_Imaginary_2': Function_params(graph_2D, {'kwargs_grid':{'ls':'--'}, 'kwargs_plot':{},
                                                                           'xlabel': global_kwargs['AmpOn_Bias'].label,
                                                                           'ylabel': global_kwargs['AmpOff_Imaginary_2'].label, 
                                                                           'args_xlim': global_kwargs['AmpOn_Bias'].args_lim,
                                                                           'presort':False,
                                                                           'legend':True,
                                                                           'color':'red',
                                                                           'marker':{'Forward branch':'>', 'Reverse branch':'<'},
                                                                           'title': 'AmpOff: Imaginary 2'})}
                            
                            
                            ), 
                            'Temp': pd.Series({
                                'AmpOn_Response_C_flag': Function_params(
                                                                graph_2D, 
                                                                 {'kwargs_grid':{'ls':'--'}, 
                                                                  'kwargs_plot':{},
                                                                  'xlabel': global_kwargs['Temp'].label,
                                                                  'ylabel': global_kwargs['AmpOn_Response_C_flag'].label, 
                                                                  'args_xlim': global_kwargs['Temp'].args_lim,
                                                                  'args_ylim': global_kwargs['AmpOn_Response_C_flag'].args_lim,
                                                                  'title': 'AmpOn: Acceptance Rate', 
                                                                  'legend': True}
                                                                 ),
                                'AmpOff_Response_C_flag': Function_params(
                                                                graph_2D, 
                                                                 {'kwargs_grid':{'ls':'--'}, 
                                                                  'kwargs_plot':{},
                                                                  'xlabel': global_kwargs['Temp'].label,
                                                                  'ylabel': global_kwargs['AmpOff_Response_C_flag'].label, 
                                                                  'args_xlim': global_kwargs['Temp'].args_lim,
                                                                  'args_ylim': global_kwargs['AmpOff_Response_C_flag'].args_lim,
                                                                  'title': 'AmpOff: Acceptance Rate', 
                                                                  'legend': True}
                                                                 ),                                 
                                'AmpOn_Response_1_flag': Function_params(
                                                                graph_2D, 
                                                                 {'kwargs_grid':{'ls':'--'}, 
                                                                  'kwargs_plot':{},
                                                                  'xlabel': global_kwargs['Temp'].label,
                                                                  'ylabel': global_kwargs['AmpOn_Response_1_flag'].label, 
                                                                  'args_xlim': global_kwargs['Temp'].args_lim,
                                                                  'args_ylim': global_kwargs['AmpOn_Response_1_flag'].args_lim,
                                                                  'title': 'AmpOn: Response 1', 
                                                                  'legend': True}
                                                                 ),
                                'AmpOn_Response_2_flag': Function_params(
                                                                graph_2D, 
                                                                 {'kwargs_grid':{'ls':'--'}, 
                                                                  'kwargs_plot':{},
                                                                  'xlabel': global_kwargs['Temp'].label,
                                                                  'ylabel': global_kwargs['AmpOn_Response_2_flag'].label, 
                                                                  'args_xlim': global_kwargs['Temp'].args_lim,
                                                                  'args_ylim': global_kwargs['AmpOn_Response_2_flag'].args_lim,
                                                                  'title': 'AmpOn: Response 2', 
                                                                  'legend': True}
                                                                 ),
                                'AmpOff_Response_1_flag': Function_params(
                                                                graph_2D, 
                                                                 {'kwargs_grid':{'ls':'--'}, 
                                                                  'kwargs_plot':{},
                                                                  'xlabel': global_kwargs['Temp'].label,
                                                                  'ylabel': global_kwargs['AmpOff_Response_1_flag'].label, 
                                                                  'args_xlim': global_kwargs['Temp'].args_lim,
                                                                  'args_ylim': global_kwargs['AmpOff_Response_1_flag'].args_lim,
                                                                  'title': 'AmpOff: Response 1', 
                                                                  'legend': True}
                                                                 ),
                                'AmpOff_Response_2_flag': Function_params(
                                                                graph_2D, 
                                                                 {'kwargs_grid':{'ls':'--'}, 
                                                                  'kwargs_plot':{},
                                                                  'xlabel': global_kwargs['Temp'].label,
                                                                  'ylabel': global_kwargs['AmpOff_Response_2_flag'].label, 
                                                                  'args_xlim': global_kwargs['Temp'].args_lim,
                                                                  'args_ylim': global_kwargs['AmpOff_Response_2_flag'].args_lim,
                                                                  'title': 'AmpOff: Response 2', 
                                                                  'legend': True}
                                                                 ),
                                'AmpOn_Response_1_Vc': Function_params(
                                                                graph_2D, 
                                                                 {'kwargs_grid':{'ls':'--'}, 
                                                                  'kwargs_plot':{},
                                                                  'xlabel': global_kwargs['Temp'].label,
                                                                  'ylabel': global_kwargs['AmpOn_Response_1_Vc'].label, 
                                                                  'args_xlim': global_kwargs['Temp'].args_lim,
                                                                  'args_ylim': global_kwargs['AmpOn_Response_1_Vc'].args_lim,
                                                                  'title': 'AmpOn: Response 1', 
                                                                  'legend': True}
                                                                 ),
                                'AmpOn_Response_2_Vc': Function_params(
                                                                graph_2D, 
                                                                 {'kwargs_grid':{'ls':'--'}, 
                                                                  'kwargs_plot':{},
                                                                  'xlabel': global_kwargs['Temp'].label,
                                                                  'ylabel': global_kwargs['AmpOn_Response_2_Vc'].label, 
                                                                  'args_xlim': global_kwargs['Temp'].args_lim,
                                                                  'args_ylim': global_kwargs['AmpOn_Response_2_Vc'].args_lim,
                                                                  'title': 'AmpOn: Response 2', 
                                                                  'legend': True}
                                                                 ),
                                'AmpOff_Response_1_Vc': Function_params(
                                                                graph_2D, 
                                                                 {'kwargs_grid':{'ls':'--'}, 
                                                                  'kwargs_plot':{},
                                                                  'xlabel': global_kwargs['Temp'].label,
                                                                  'ylabel': global_kwargs['AmpOff_Response_1_Vc'].label, 
                                                                  'args_xlim': global_kwargs['Temp'].args_lim,
                                                                  'args_ylim': global_kwargs['AmpOff_Response_1_Vc'].args_lim,
                                                                  'title': 'AmpOff: Response 1', 
                                                                  'legend': True}
                                                                 ),
                                'AmpOff_Response_2_Vc': Function_params(
                                                                graph_2D, 
                                                                 {'kwargs_grid':{'ls':'--'}, 
                                                                  'kwargs_plot':{},
                                                                  'xlabel': global_kwargs['Temp'].label,
                                                                  'ylabel': global_kwargs['AmpOff_Response_2_Vc'].label, 
                                                                  'args_xlim': global_kwargs['Temp'].args_lim,
                                                                  'args_ylim': global_kwargs['AmpOff_Response_2_Vc'].args_lim,
                                                                  'title': 'AmpOff: Response 2', 
                                                                  'legend': True}
                                                                 ),
                                'AmpOn_Response_C_Vf': Function_params(
                                                                graph_2D, 
                                                                 {'kwargs_grid':{'ls':'--'}, 
                                                                  'kwargs_plot':{},
                                                                  'xlabel': global_kwargs['Temp'].label,
                                                                  'ylabel': global_kwargs['AmpOn_Response_C_Vf'].label, 
                                                                  'args_xlim': global_kwargs['Temp'].args_lim,
                                                                  'args_ylim': global_kwargs['AmpOn_Response_C_Vf'].args_lim,
                                                                  'title': 'AmpOn: Coercive Bias', 
                                                                  'legend': True}
                                                                 ),
                                'AmpOff_Response_C_Vf': Function_params(
                                                                graph_2D, 
                                                                 {'kwargs_grid':{'ls':'--'}, 
                                                                  'kwargs_plot':{},
                                                                  'xlabel': global_kwargs['Temp'].label,
                                                                  'ylabel': global_kwargs['AmpOff_Response_C_Vf'].label, 
                                                                  'args_xlim': global_kwargs['Temp'].args_lim,
                                                                  'args_ylim': global_kwargs['AmpOff_Response_C_Vf'].args_lim,
                                                                  'title': 'AmpOff: Coercive Bias', 
                                                                  'legend': True}
                                                                 ),
                                'AmpOn_Response_1_Vf': Function_params(
                                                                graph_2D, 
                                                                 {'kwargs_grid':{'ls':'--'}, 
                                                                  'kwargs_plot':{},
                                                                  'xlabel': global_kwargs['Temp'].label,
                                                                  'ylabel': global_kwargs['AmpOn_Response_1_Vf'].label, 
                                                                  'args_xlim': global_kwargs['Temp'].args_lim,
                                                                  'args_ylim': global_kwargs['AmpOn_Response_1_Vf'].args_lim,
                                                                  'title': 'AmpOn: Response 1', 
                                                                  'legend': True}
                                                                 ),
                                'AmpOn_Response_2_Vf': Function_params(
                                                                graph_2D, 
                                                                 {'kwargs_grid':{'ls':'--'}, 
                                                                  'kwargs_plot':{},
                                                                  'xlabel': global_kwargs['Temp'].label,
                                                                  'ylabel': global_kwargs['AmpOn_Response_2_Vf'].label, 
                                                                  'args_xlim': global_kwargs['Temp'].args_lim,
                                                                  'args_ylim': global_kwargs['AmpOn_Response_2_Vf'].args_lim,
                                                                  'title': 'AmpOn: Response 2', 
                                                                  'legend': True}
                                                                 ),
                                'AmpOff_Response_1_Vf': Function_params(
                                                                graph_2D, 
                                                                 {'kwargs_grid':{'ls':'--'}, 
                                                                  'kwargs_plot':{},
                                                                  'xlabel': global_kwargs['Temp'].label,
                                                                  'ylabel': global_kwargs['AmpOff_Response_1_Vf'].label, 
                                                                  'args_xlim': global_kwargs['Temp'].args_lim,
                                                                  'args_ylim': global_kwargs['AmpOff_Response_1_Vf'].args_lim,
                                                                  'title': 'AmpOff: Response 1', 
                                                                  'legend': True}
                                                                 ),
                                'AmpOff_Response_2_Vf': Function_params(
                                                                graph_2D, 
                                                                 {'kwargs_grid':{'ls':'--'}, 
                                                                  'kwargs_plot':{},
                                                                  'xlabel': global_kwargs['Temp'].label,
                                                                  'ylabel': global_kwargs['AmpOff_Response_2_Vf'].label, 
                                                                  'args_xlim': global_kwargs['Temp'].args_lim,
                                                                  'args_ylim': global_kwargs['AmpOff_Response_2_Vf'].args_lim,
                                                                  'title': 'AmpOff: Response 2', 
                                                                  'legend': True}
                                                                 ),
                                'AmpOn_Response_1_Vfc_plus': Function_params(
                                                                graph_2D, 
                                                                 {'kwargs_grid':{'ls':'--'}, 
                                                                  'kwargs_plot':{},
                                                                  'xlabel': global_kwargs['Temp'].label,
                                                                  'ylabel': global_kwargs['AmpOn_Response_1_Vfc_plus'].label, 
                                                                  'args_xlim': global_kwargs['Temp'].args_lim,
                                                                  'args_ylim': global_kwargs['AmpOn_Response_1_Vfc_plus'].args_lim,
                                                                  'title': 'AmpOn: Response 1', 
                                                                  'legend': True}
                                                                 ),
                                'AmpOn_Response_2_Vfc_plus': Function_params(
                                                                graph_2D, 
                                                                 {'kwargs_grid':{'ls':'--'}, 
                                                                  'kwargs_plot':{},
                                                                  'xlabel': global_kwargs['Temp'].label,
                                                                  'ylabel': global_kwargs['AmpOn_Response_2_Vfc_plus'].label, 
                                                                  'args_xlim': global_kwargs['Temp'].args_lim,
                                                                  'args_ylim': global_kwargs['AmpOn_Response_2_Vfc_plus'].args_lim,
                                                                  'title': 'AmpOn: Response 2', 
                                                                  'legend': True}
                                                                 ),
                                'AmpOff_Response_1_Vfc_plus': Function_params(
                                                                graph_2D, 
                                                                 {'kwargs_grid':{'ls':'--'}, 
                                                                  'kwargs_plot':{},
                                                                  'xlabel': global_kwargs['Temp'].label,
                                                                  'ylabel': global_kwargs['AmpOff_Response_1_Vfc_plus'].label, 
                                                                  'args_xlim': global_kwargs['Temp'].args_lim,
                                                                  'args_ylim': global_kwargs['AmpOff_Response_1_Vfc_plus'].args_lim,
                                                                  'title': 'AmpOff: Response 1', 
                                                                  'legend': True}
                                                                 ),
                                'AmpOff_Response_2_Vfc_plus': Function_params(
                                                                graph_2D, 
                                                                 {'kwargs_grid':{'ls':'--'}, 
                                                                  'kwargs_plot':{},
                                                                  'xlabel': global_kwargs['Temp'].label,
                                                                  'ylabel': global_kwargs['AmpOff_Response_2_Vfc_plus'].label, 
                                                                  'args_xlim': global_kwargs['Temp'].args_lim,
                                                                  'args_ylim': global_kwargs['AmpOff_Response_2_Vfc_plus'].args_lim,
                                                                  'title': 'AmpOff: Response 2', 
                                                                  'legend': True}
                                                                 ),
                                'AmpOn_Response_1_Vfc_minus': Function_params(
                                                                graph_2D, 
                                                                 {'kwargs_grid':{'ls':'--'}, 
                                                                  'kwargs_plot':{},
                                                                  'xlabel': global_kwargs['Temp'].label,
                                                                  'ylabel': global_kwargs['AmpOn_Response_1_Vfc_minus'].label, 
                                                                  'args_xlim': global_kwargs['Temp'].args_lim,
                                                                  'args_ylim': global_kwargs['AmpOn_Response_1_Vfc_minus'].args_lim,
                                                                  'title': 'AmpOn: Response 1', 
                                                                  'legend': True}
                                                                 ),
                                'AmpOn_Response_2_Vfc_minus': Function_params(
                                                                graph_2D, 
                                                                 {'kwargs_grid':{'ls':'--'}, 
                                                                  'kwargs_plot':{},
                                                                  'xlabel': global_kwargs['Temp'].label,
                                                                  'ylabel': global_kwargs['AmpOn_Response_2_Vfc_minus'].label, 
                                                                  'args_xlim': global_kwargs['Temp'].args_lim,
                                                                  'args_ylim': global_kwargs['AmpOn_Response_2_Vfc_minus'].args_lim,
                                                                  'title': 'AmpOn: Response 2', 
                                                                  'legend': True}
                                                                 ),
                                'AmpOff_Response_1_Vfc_minus': Function_params(
                                                                graph_2D, 
                                                                 {'kwargs_grid':{'ls':'--'}, 
                                                                  'kwargs_plot':{},
                                                                  'xlabel': global_kwargs['Temp'].label,
                                                                  'ylabel': global_kwargs['AmpOff_Response_1_Vfc_minus'].label, 
                                                                  'args_xlim': global_kwargs['Temp'].args_lim,
                                                                  'args_ylim': global_kwargs['AmpOff_Response_1_Vfc_minus'].args_lim,
                                                                  'title': 'AmpOff: Response 1', 
                                                                  'legend': True}
                                                                 ),
                                'AmpOff_Response_2_Vfc_minus': Function_params(
                                                                graph_2D, 
                                                                 {'kwargs_grid':{'ls':'--'}, 
                                                                  'kwargs_plot':{},
                                                                  'xlabel': global_kwargs['Temp'].label,
                                                                  'ylabel': global_kwargs['AmpOff_Response_2_Vfc_minus'].label, 
                                                                  'args_xlim': global_kwargs['Temp'].args_lim,
                                                                  'args_ylim': global_kwargs['AmpOff_Response_2_Vfc_minus'].args_lim,
                                                                  'title': 'AmpOff: Response 2', 
                                                                  'legend': True}
                                                                 ),
                                'AmpOn_Response_1_Vsurf': Function_params(
                                                                graph_2D, 
                                                                 {'kwargs_grid':{'ls':'--'}, 
                                                                  'kwargs_plot':{},
                                                                  'xlabel': global_kwargs['Temp'].label,
                                                                  'ylabel': global_kwargs['AmpOn_Response_1_Vsurf'].label, 
                                                                  'args_xlim': global_kwargs['Temp'].args_lim,
                                                                  'args_ylim': global_kwargs['AmpOn_Response_1_Vsurf'].args_lim,
                                                                  'title': 'AmpOn: Response 1', 
                                                                  'legend': True}
                                                                 ),
                                'AmpOn_Response_2_Vsurf': Function_params(
                                                                graph_2D, 
                                                                 {'kwargs_grid':{'ls':'--'}, 
                                                                  'kwargs_plot':{},
                                                                  'xlabel': global_kwargs['Temp'].label,
                                                                  'ylabel': global_kwargs['AmpOn_Response_2_Vsurf'].label, 
                                                                  'args_xlim': global_kwargs['Temp'].args_lim,
                                                                  'args_ylim': global_kwargs['AmpOn_Response_2_Vsurf'].args_lim,
                                                                  'title': 'AmpOn: Response 2', 
                                                                  'legend': True}
                                                                 ),
                                'AmpOff_Response_1_Vsurf': Function_params(
                                                                graph_2D, 
                                                                 {'kwargs_grid':{'ls':'--'}, 
                                                                  'kwargs_plot':{},
                                                                  'xlabel': global_kwargs['Temp'].label,
                                                                  'ylabel': global_kwargs['AmpOff_Response_1_Vsurf'].label, 
                                                                  'args_xlim': global_kwargs['Temp'].args_lim,
                                                                  'args_ylim': global_kwargs['AmpOff_Response_1_Vsurf'].args_lim,
                                                                  'title': 'AmpOff: Response 1', 
                                                                  'legend': True}
                                                                 ),
                                'AmpOff_Response_2_Vsurf': Function_params(
                                                                graph_2D, 
                                                                 {'kwargs_grid':{'ls':'--'}, 
                                                                  'kwargs_plot':{},
                                                                  'xlabel': global_kwargs['Temp'].label,
                                                                  'ylabel': global_kwargs['AmpOff_Response_2_Vsurf'].label, 
                                                                  'args_xlim': global_kwargs['Temp'].args_lim,
                                                                  'args_ylim': global_kwargs['AmpOff_Response_2_Vsurf'].args_lim,
                                                                  'title': 'AmpOff: Response 2', 
                                                                  'legend': True}
                                                                 ),
                                'AmpOn_Response_1_Rs': Function_params(
                                                                graph_2D, 
                                                                 {'kwargs_grid':{'ls':'--'}, 
                                                                  'kwargs_plot':{},
                                                                  'xlabel': global_kwargs['Temp'].label,
                                                                  'ylabel': global_kwargs['AmpOn_Response_1_Rs'].label, 
                                                                  'args_xlim': global_kwargs['Temp'].args_lim,
                                                                  'args_ylim': global_kwargs['AmpOn_Response_1_Rs'].args_lim,
                                                                  'title': 'AmpOn: Response 1', 
                                                                  'legend': True}
                                                                 ),
                                'AmpOn_Response_2_Rs': Function_params(
                                                                graph_2D, 
                                                                 {'kwargs_grid':{'ls':'--'}, 
                                                                  'kwargs_plot':{},
                                                                  'xlabel': global_kwargs['Temp'].label,
                                                                  'ylabel': global_kwargs['AmpOn_Response_2_Rs'].label, 
                                                                  'args_xlim': global_kwargs['Temp'].args_lim,
                                                                  'args_ylim': global_kwargs['AmpOn_Response_2_Rs'].args_lim,
                                                                  'title': 'AmpOn: Response 2', 
                                                                  'legend': True}
                                                                 ),
                                'AmpOff_Response_1_Rs': Function_params(
                                                                graph_2D, 
                                                                 {'kwargs_grid':{'ls':'--'}, 
                                                                  'kwargs_plot':{},
                                                                  'xlabel': global_kwargs['Temp'].label,
                                                                  'ylabel': global_kwargs['AmpOff_Response_1_Rs'].label, 
                                                                  'args_xlim': global_kwargs['Temp'].args_lim,
                                                                  'args_ylim': global_kwargs['AmpOff_Response_1_Rs'].args_lim,
                                                                  'title': 'AmpOff: Response 1', 
                                                                  'legend': True}
                                                                 ),
                                'AmpOff_Response_2_Rs': Function_params(
                                                                graph_2D, 
                                                                 {'kwargs_grid':{'ls':'--'}, 
                                                                  'kwargs_plot':{},
                                                                  'xlabel': global_kwargs['Temp'].label,
                                                                  'ylabel': global_kwargs['AmpOff_Response_2_Rs'].label, 
                                                                  'args_xlim': global_kwargs['Temp'].args_lim,
                                                                  'args_ylim': global_kwargs['AmpOff_Response_2_Rs'].args_lim,
                                                                  'title': 'AmpOff: Response 2', 
                                                                  'legend': True}
                                                                 ),
                                'AmpOn_Response_C_Rfs': Function_params(
                                                                graph_2D, 
                                                                 {'kwargs_grid':{'ls':'--'}, 
                                                                  'kwargs_plot':{},
                                                                  'xlabel': global_kwargs['Temp'].label,
                                                                  'ylabel': global_kwargs['AmpOn_Response_C_Rfs'].label, 
                                                                  'args_xlim': global_kwargs['Temp'].args_lim,
                                                                  'args_ylim': global_kwargs['AmpOn_Response_C_Rfs'].args_lim,
                                                                  'title': 'AmpOn: Saturation Response', 
                                                                  'legend': True}
                                                                 ),
                                'AmpOff_Response_C_Rfs': Function_params(
                                                                graph_2D, 
                                                                 {'kwargs_grid':{'ls':'--'}, 
                                                                  'kwargs_plot':{},
                                                                  'xlabel': global_kwargs['Temp'].label,
                                                                  'ylabel': global_kwargs['AmpOff_Response_C_Rfs'].label, 
                                                                  'args_xlim': global_kwargs['Temp'].args_lim,
                                                                  'args_ylim': global_kwargs['AmpOff_Response_C_Rfs'].args_lim,
                                                                  'title': 'AmpOff: Saturation Response', 
                                                                  'legend': True}
                                                                 ),
                                'AmpOn_Response_1_Rfs': Function_params(
                                                                graph_2D, 
                                                                 {'kwargs_grid':{'ls':'--'}, 
                                                                  'kwargs_plot':{},
                                                                  'xlabel': global_kwargs['Temp'].label,
                                                                  'ylabel': global_kwargs['AmpOn_Response_1_Rfs'].label, 
                                                                  'args_xlim': global_kwargs['Temp'].args_lim,
                                                                  'args_ylim': global_kwargs['AmpOn_Response_1_Rfs'].args_lim,
                                                                  'title': 'AmpOn: Response 1', 
                                                                  'legend': True}
                                                                 ),
                                'AmpOn_Response_2_Rfs': Function_params(
                                                                graph_2D, 
                                                                 {'kwargs_grid':{'ls':'--'}, 
                                                                  'kwargs_plot':{},
                                                                  'xlabel': global_kwargs['Temp'].label,
                                                                  'ylabel': global_kwargs['AmpOn_Response_2_Rfs'].label, 
                                                                  'args_xlim': global_kwargs['Temp'].args_lim,
                                                                  'args_ylim': global_kwargs['AmpOn_Response_2_Rfs'].args_lim,
                                                                  'title': 'AmpOn: Response 2', 
                                                                  'legend': True}
                                                                 ),
                                'AmpOff_Response_1_Rfs': Function_params(
                                                                graph_2D, 
                                                                 {'kwargs_grid':{'ls':'--'}, 
                                                                  'kwargs_plot':{},
                                                                  'xlabel': global_kwargs['Temp'].label,
                                                                  'ylabel': global_kwargs['AmpOff_Response_1_Rfs'].label, 
                                                                  'args_xlim': global_kwargs['Temp'].args_lim,
                                                                  'args_ylim': global_kwargs['AmpOff_Response_1_Rfs'].args_lim,
                                                                  'title': 'AmpOff: Response 1', 
                                                                  'legend': True}
                                                                 ),
                                'AmpOff_Response_2_Rfs': Function_params(
                                                                graph_2D, 
                                                                 {'kwargs_grid':{'ls':'--'}, 
                                                                  'kwargs_plot':{},
                                                                  'xlabel': global_kwargs['Temp'].label,
                                                                  'ylabel': global_kwargs['AmpOff_Response_2_Rfs'].label, 
                                                                  'args_xlim': global_kwargs['Temp'].args_lim,
                                                                  'args_ylim': global_kwargs['AmpOff_Response_2_Rfs'].args_lim,
                                                                  'title': 'AmpOff: Response 2', 
                                                                  'legend': True}
                                                                 ),
                                'AmpOn_Response_1_Rv': Function_params(
                                                                graph_2D, 
                                                                 {'kwargs_grid':{'ls':'--'}, 
                                                                  'kwargs_plot':{},
                                                                  'xlabel': global_kwargs['Temp'].label,
                                                                  'ylabel': global_kwargs['AmpOn_Response_1_Rv'].label, 
                                                                  'args_xlim': global_kwargs['Temp'].args_lim,
                                                                  'args_ylim': global_kwargs['AmpOn_Response_1_Rv'].args_lim,
                                                                  'title': 'AmpOn: Response 1', 
                                                                  'legend': True}
                                                                 ),
                                'AmpOn_Response_2_Rv': Function_params(
                                                                graph_2D, 
                                                                 {'kwargs_grid':{'ls':'--'}, 
                                                                  'kwargs_plot':{},
                                                                  'xlabel': global_kwargs['Temp'].label,
                                                                  'ylabel': global_kwargs['AmpOn_Response_2_Rv'].label, 
                                                                  'args_xlim': global_kwargs['Temp'].args_lim,
                                                                  'args_ylim': global_kwargs['AmpOn_Response_2_Rv'].args_lim,
                                                                  'title': 'AmpOn: Response 2', 
                                                                  'legend': True}
                                                                 ),
                                'AmpOff_Response_1_Rv': Function_params(
                                                                graph_2D, 
                                                                 {'kwargs_grid':{'ls':'--'}, 
                                                                  'kwargs_plot':{},
                                                                  'xlabel': global_kwargs['Temp'].label,
                                                                  'ylabel': global_kwargs['AmpOff_Response_1_Rv'].label, 
                                                                  'args_xlim': global_kwargs['Temp'].args_lim,
                                                                  'args_ylim': global_kwargs['AmpOff_Response_1_Rv'].args_lim,
                                                                  'title': 'AmpOff: Response 1', 
                                                                  'legend': True}
                                                                 ),
                                'AmpOff_Response_2_Rv': Function_params(
                                                                graph_2D, 
                                                                 {'kwargs_grid':{'ls':'--'}, 
                                                                  'kwargs_plot':{},
                                                                  'xlabel': global_kwargs['Temp'].label,
                                                                  'ylabel': global_kwargs['AmpOff_Response_2_Rv'].label, 
                                                                  'args_xlim': global_kwargs['Temp'].args_lim,
                                                                  'args_ylim': global_kwargs['AmpOff_Response_2_Rv'].args_lim,
                                                                  'title': 'AmpOff: Response 2', 
                                                                  'legend': True}
                                                                 ),
                                'AmpOn_Response_1_R0': Function_params(
                                                                graph_2D, 
                                                                 {'kwargs_grid':{'ls':'--'}, 
                                                                  'kwargs_plot':{},
                                                                  'xlabel': global_kwargs['Temp'].label,
                                                                  'ylabel': global_kwargs['AmpOn_Response_1_R0'].label, 
                                                                  'args_xlim': global_kwargs['Temp'].args_lim,
                                                                  'args_ylim': global_kwargs['AmpOn_Response_1_R0'].args_lim,
                                                                  'title': 'AmpOn: Response 1', 
                                                                  'legend': True}
                                                                 ),
                                'AmpOn_Response_2_R0': Function_params(
                                                                graph_2D, 
                                                                 {'kwargs_grid':{'ls':'--'}, 
                                                                  'kwargs_plot':{},
                                                                  'xlabel': global_kwargs['Temp'].label,
                                                                  'ylabel': global_kwargs['AmpOn_Response_2_R0'].label, 
                                                                  'args_xlim': global_kwargs['Temp'].args_lim,
                                                                  'args_ylim': global_kwargs['AmpOn_Response_2_R0'].args_lim,
                                                                  'title': 'AmpOn: Response 2', 
                                                                  'legend': True}
                                                                 ),
                                'AmpOff_Response_1_R0': Function_params(
                                                                graph_2D, 
                                                                 {'kwargs_grid':{'ls':'--'}, 
                                                                  'kwargs_plot':{},
                                                                  'xlabel': global_kwargs['Temp'].label,
                                                                  'ylabel': global_kwargs['AmpOff_Response_1_R0'].label, 
                                                                  'args_xlim': global_kwargs['Temp'].args_lim,
                                                                  'args_ylim': global_kwargs['AmpOff_Response_1_R0'].args_lim,
                                                                  'title': 'AmpOff: Response 1', 
                                                                  'legend': True}
                                                                 ),
                                'AmpOff_Response_2_R0': Function_params(
                                                                graph_2D, 
                                                                 {'kwargs_grid':{'ls':'--'}, 
                                                                  'kwargs_plot':{},
                                                                  'xlabel': global_kwargs['Temp'].label,
                                                                  'ylabel': global_kwargs['AmpOff_Response_2_R0'].label, 
                                                                  'args_xlim': global_kwargs['Temp'].args_lim,
                                                                  'args_ylim': global_kwargs['AmpOff_Response_2_R0'].args_lim,
                                                                  'title': 'AmpOff: Response 2', 
                                                                  'legend': True}
                                                                 ),
                                'AmpOn_Response_1_Ads': Function_params(
                                                                graph_2D, 
                                                                 {'kwargs_grid':{'ls':'--'}, 
                                                                  'kwargs_plot':{},
                                                                  'xlabel': global_kwargs['Temp'].label,
                                                                  'ylabel': global_kwargs['AmpOn_Response_1_Ads'].label, 
                                                                  'args_xlim': global_kwargs['Temp'].args_lim,
                                                                  'args_ylim': global_kwargs['AmpOn_Response_1_Ads'].args_lim,
                                                                  'title': 'AmpOn: Response 1', 
                                                                  'legend': True}
                                                                 ),
                                'AmpOn_Response_2_Ads': Function_params(
                                                                graph_2D, 
                                                                 {'kwargs_grid':{'ls':'--'}, 
                                                                  'kwargs_plot':{},
                                                                  'xlabel': global_kwargs['Temp'].label,
                                                                  'ylabel': global_kwargs['AmpOn_Response_2_Ads'].label, 
                                                                  'args_xlim': global_kwargs['Temp'].args_lim,
                                                                  'args_ylim': global_kwargs['AmpOn_Response_2_Ads'].args_lim,
                                                                  'title': 'AmpOn: Response 2', 
                                                                  'legend': True}
                                                                 ),
                                'AmpOff_Response_1_Ads': Function_params(
                                                                graph_2D, 
                                                                 {'kwargs_grid':{'ls':'--'}, 
                                                                  'kwargs_plot':{},
                                                                  'xlabel': global_kwargs['Temp'].label,
                                                                  'ylabel': global_kwargs['AmpOff_Response_1_Ads'].label, 
                                                                  'args_xlim': global_kwargs['Temp'].args_lim,
                                                                  'args_ylim': global_kwargs['AmpOff_Response_1_Ads'].args_lim,
                                                                  'title': 'AmpOff: Response 1', 
                                                                  'legend': True}
                                                                 ),
                                'AmpOff_Response_2_Ads': Function_params(
                                                                graph_2D, 
                                                                 {'kwargs_grid':{'ls':'--'}, 
                                                                  'kwargs_plot':{},
                                                                  'xlabel': global_kwargs['Temp'].label,
                                                                  'ylabel': global_kwargs['AmpOff_Response_2_Ads'].label, 
                                                                  'args_xlim': global_kwargs['Temp'].args_lim,
                                                                  'args_ylim': global_kwargs['AmpOff_Response_2_Ads'].args_lim,
                                                                  'title': 'AmpOff: Response 2', 
                                                                  'legend': True}
                                                                 ),
                                'AmpOn_Response_C_Afs': Function_params(
                                                                graph_2D, 
                                                                 {'kwargs_grid':{'ls':'--'}, 
                                                                  'kwargs_plot':{},
                                                                  'xlabel': global_kwargs['Temp'].label,
                                                                  'ylabel': global_kwargs['AmpOn_Response_C_Afs'].label, 
                                                                  'args_xlim': global_kwargs['Temp'].args_lim,
                                                                  'args_ylim': global_kwargs['AmpOn_Response_C_Afs'].args_lim,
                                                                  'title': 'AmpOn: Work of Switching', 
                                                                  'legend': True}
                                                                 ),
                                'AmpOff_Response_C_Afs': Function_params(
                                                                graph_2D, 
                                                                 {'kwargs_grid':{'ls':'--'}, 
                                                                  'kwargs_plot':{},
                                                                  'xlabel': global_kwargs['Temp'].label,
                                                                  'ylabel': global_kwargs['AmpOff_Response_C_Afs'].label, 
                                                                  'args_xlim': global_kwargs['Temp'].args_lim,
                                                                  'args_ylim': global_kwargs['AmpOff_Response_C_Afs'].args_lim,
                                                                  'title': 'AmpOff: Work of Switching', 
                                                                  'legend': True}
                                                                 ),
                                'AmpOn_Response_1_Afs': Function_params(
                                                                graph_2D, 
                                                                 {'kwargs_grid':{'ls':'--'}, 
                                                                  'kwargs_plot':{},
                                                                  'xlabel': global_kwargs['Temp'].label,
                                                                  'ylabel': global_kwargs['AmpOn_Response_1_Afs'].label, 
                                                                  'args_xlim': global_kwargs['Temp'].args_lim,
                                                                  'args_ylim': global_kwargs['AmpOn_Response_1_Afs'].args_lim,
                                                                  'title': 'AmpOn: Response 1', 
                                                                  'legend': True}
                                                                 ),
                                'AmpOn_Response_2_Afs': Function_params(
                                                                graph_2D, 
                                                                 {'kwargs_grid':{'ls':'--'}, 
                                                                  'kwargs_plot':{},
                                                                  'xlabel': global_kwargs['Temp'].label,
                                                                  'ylabel': global_kwargs['AmpOn_Response_2_Afs'].label, 
                                                                  'args_xlim': global_kwargs['Temp'].args_lim,
                                                                  'args_ylim': global_kwargs['AmpOn_Response_2_Afs'].args_lim,
                                                                  'title': 'AmpOn: Response 2', 
                                                                  'legend': True}
                                                                 ),
                                'AmpOff_Response_1_Afs': Function_params(
                                                                graph_2D, 
                                                                 {'kwargs_grid':{'ls':'--'}, 
                                                                  'kwargs_plot':{},
                                                                  'xlabel': global_kwargs['Temp'].label,
                                                                  'ylabel': global_kwargs['AmpOff_Response_1_Afs'].label, 
                                                                  'args_xlim': global_kwargs['Temp'].args_lim,
                                                                  'args_ylim': global_kwargs['AmpOff_Response_1_Afs'].args_lim,
                                                                  'title': 'AmpOff: Response 1', 
                                                                  'legend': True}
                                                                 ),
                                'AmpOff_Response_2_Afs': Function_params(
                                                                graph_2D, 
                                                                 {'kwargs_grid':{'ls':'--'}, 
                                                                  'kwargs_plot':{},
                                                                  'xlabel': global_kwargs['Temp'].label,
                                                                  'ylabel': global_kwargs['AmpOff_Response_2_Afs'].label, 
                                                                  'args_xlim': global_kwargs['Temp'].args_lim,
                                                                  'args_ylim': global_kwargs['AmpOff_Response_2_Afs'].args_lim,
                                                                  'title': 'AmpOff: Response 2', 
                                                                  'legend': True}
                                                                 ),
                                'AmpOn_Response_1_Imd': Function_params(
                                                                graph_2D, 
                                                                 {'kwargs_grid':{'ls':'--'}, 
                                                                  'kwargs_plot':{},
                                                                  'xlabel': global_kwargs['Temp'].label,
                                                                  'ylabel': global_kwargs['AmpOn_Response_1_Imd'].label, 
                                                                  'args_xlim': global_kwargs['Temp'].args_lim,
                                                                  'args_ylim': global_kwargs['AmpOn_Response_1_Imd'].args_lim,
                                                                  'title': 'AmpOn: Response 1', 
                                                                  'legend': True}
                                                                 ),
                                'AmpOn_Response_2_Imd': Function_params(
                                                                graph_2D, 
                                                                 {'kwargs_grid':{'ls':'--'}, 
                                                                  'kwargs_plot':{},
                                                                  'xlabel': global_kwargs['Temp'].label,
                                                                  'ylabel': global_kwargs['AmpOn_Response_2_Imd'].label, 
                                                                  'args_xlim': global_kwargs['Temp'].args_lim,
                                                                  'args_ylim': global_kwargs['AmpOn_Response_2_Imd'].args_lim,
                                                                  'title': 'AmpOn: Response 2', 
                                                                  'legend': True}
                                                                 ),
                                'AmpOff_Response_1_Imd': Function_params(
                                                                graph_2D, 
                                                                 {'kwargs_grid':{'ls':'--'}, 
                                                                  'kwargs_plot':{},
                                                                  'xlabel': global_kwargs['Temp'].label,
                                                                  'ylabel': global_kwargs['AmpOff_Response_1_Imd'].label, 
                                                                  'args_xlim': global_kwargs['Temp'].args_lim,
                                                                  'args_ylim': global_kwargs['AmpOff_Response_1_Imd'].args_lim,
                                                                  'title': 'AmpOff: Response 1', 
                                                                  'legend': True}
                                                                 ),
                                'AmpOff_Response_2_Imd': Function_params(
                                                                graph_2D, 
                                                                 {'kwargs_grid':{'ls':'--'}, 
                                                                  'kwargs_plot':{},
                                                                  'xlabel': global_kwargs['Temp'].label,
                                                                  'ylabel': global_kwargs['AmpOff_Response_2_Imd'].label, 
                                                                  'args_xlim': global_kwargs['Temp'].args_lim,
                                                                  'args_ylim': global_kwargs['AmpOff_Response_2_Imd'].args_lim,
                                                                  'title': 'AmpOff: Response 2', 
                                                                  'legend': True}
                                                                 ),
                                'AmpOn_Response_1_Imf': Function_params(
                                                                graph_2D, 
                                                                 {'kwargs_grid':{'ls':'--'}, 
                                                                  'kwargs_plot':{},
                                                                  'xlabel': global_kwargs['Temp'].label,
                                                                  'ylabel': global_kwargs['AmpOn_Response_1_Imf'].label, 
                                                                  'args_xlim': global_kwargs['Temp'].args_lim,
                                                                  'args_ylim': global_kwargs['AmpOn_Response_1_Imf'].args_lim,
                                                                  'title': 'AmpOn: Response 1', 
                                                                  'legend': True}
                                                                 ),
                                'AmpOn_Response_2_Imf': Function_params(
                                                                graph_2D, 
                                                                 {'kwargs_grid':{'ls':'--'}, 
                                                                  'kwargs_plot':{},
                                                                  'xlabel': global_kwargs['Temp'].label,
                                                                  'ylabel': global_kwargs['AmpOn_Response_2_Imf'].label, 
                                                                  'args_xlim': global_kwargs['Temp'].args_lim,
                                                                  'args_ylim': global_kwargs['AmpOn_Response_2_Imf'].args_lim,
                                                                  'title': 'AmpOn: Response 2', 
                                                                  'legend': True}
                                                                 ),
                                'AmpOff_Response_1_Imf': Function_params(
                                                                graph_2D, 
                                                                 {'kwargs_grid':{'ls':'--'}, 
                                                                  'kwargs_plot':{},
                                                                  'xlabel': global_kwargs['Temp'].label,
                                                                  'ylabel': global_kwargs['AmpOff_Response_1_Imf'].label, 
                                                                  'args_xlim': global_kwargs['Temp'].args_lim,
                                                                  'args_ylim': global_kwargs['AmpOff_Response_1_Imf'].args_lim,
                                                                  'title': 'AmpOff: Response 1', 
                                                                  'legend': True}
                                                                 ),
                                'AmpOff_Response_2_Imf': Function_params(
                                                                graph_2D, 
                                                                 {'kwargs_grid':{'ls':'--'}, 
                                                                  'kwargs_plot':{},
                                                                  'xlabel': global_kwargs['Temp'].label,
                                                                  'ylabel': global_kwargs['AmpOff_Response_2_Imf'].label, 
                                                                  'args_xlim': global_kwargs['Temp'].args_lim,
                                                                  'args_ylim': global_kwargs['AmpOff_Response_2_Imf'].args_lim,
                                                                  'title': 'AmpOff: Response 2', 
                                                                  'legend': True}
                                                                 ),
                                'AmpOn_Response_1_sigma-d': Function_params(
                                                                graph_2D, 
                                                                 {'kwargs_grid':{'ls':'--'}, 
                                                                  'kwargs_plot':{},
                                                                  'xlabel': global_kwargs['Temp'].label,
                                                                  'ylabel': global_kwargs['AmpOn_Response_1_sigma-d'].label, 
                                                                  'args_xlim': global_kwargs['Temp'].args_lim,
                                                                  'args_ylim': global_kwargs['AmpOn_Response_1_sigma-d'].args_lim,
                                                                  'title': 'AmpOn: Response 1', 
                                                                  'legend': True}
                                                                 ),
                                'AmpOn_Response_2_sigma-d': Function_params(
                                                                graph_2D, 
                                                                 {'kwargs_grid':{'ls':'--'}, 
                                                                  'kwargs_plot':{},
                                                                  'xlabel': global_kwargs['Temp'].label,
                                                                  'ylabel': global_kwargs['AmpOn_Response_2_sigma-d'].label, 
                                                                  'args_xlim': global_kwargs['Temp'].args_lim,
                                                                  'args_ylim': global_kwargs['AmpOn_Response_2_sigma-d'].args_lim,
                                                                  'title': 'AmpOn: Response 2', 
                                                                  'legend': True}
                                                                 ),
                                'AmpOff_Response_1_sigma-d': Function_params(
                                                                graph_2D, 
                                                                 {'kwargs_grid':{'ls':'--'}, 
                                                                  'kwargs_plot':{},
                                                                  'xlabel': global_kwargs['Temp'].label,
                                                                  'ylabel': global_kwargs['AmpOff_Response_1_sigma-d'].label, 
                                                                  'args_xlim': global_kwargs['Temp'].args_lim,
                                                                  'args_ylim': global_kwargs['AmpOff_Response_1_sigma-d'].args_lim,
                                                                  'title': 'AmpOff: Response 1', 
                                                                  'legend': True}
                                                                 ),
                                'AmpOff_Response_2_sigma-d': Function_params(
                                                                graph_2D, 
                                                                 {'kwargs_grid':{'ls':'--'}, 
                                                                  'kwargs_plot':{},
                                                                  'xlabel': global_kwargs['Temp'].label,
                                                                  'ylabel': global_kwargs['AmpOff_Response_2_sigma-d'].label, 
                                                                  'args_xlim': global_kwargs['Temp'].args_lim,
                                                                  'args_ylim': global_kwargs['AmpOff_Response_2_sigma-d'].args_lim,
                                                                  'title': 'AmpOff: Response 2', 
                                                                  'legend': True}
                                                                 ),
                                'AmpOn_Response_1_Rds': Function_params(
                                                                graph_2D, 
                                                                 {'kwargs_grid':{'ls':'--'}, 
                                                                  'kwargs_plot':{},
                                                                  'xlabel': global_kwargs['Temp'].label,
                                                                  'ylabel': global_kwargs['AmpOn_Response_1_Rds'].label, 
                                                                  'args_xlim': global_kwargs['Temp'].args_lim,
                                                                  'args_ylim': global_kwargs['AmpOn_Response_1_Rds'].args_lim,
                                                                  'title': 'AmpOn: Response 1', 
                                                                  'legend': True}
                                                                 ),
                                'AmpOn_Response_2_Rds': Function_params(
                                                                graph_2D, 
                                                                 {'kwargs_grid':{'ls':'--'}, 
                                                                  'kwargs_plot':{},
                                                                  'xlabel': global_kwargs['Temp'].label,
                                                                  'ylabel': global_kwargs['AmpOn_Response_2_Rds'].label, 
                                                                  'args_xlim': global_kwargs['Temp'].args_lim,
                                                                  'args_ylim': global_kwargs['AmpOn_Response_2_Rds'].args_lim,
                                                                  'title': 'AmpOn: Response 2', 
                                                                  'legend': True}
                                                                 ),
                                'AmpOff_Response_1_Rds': Function_params(
                                                                graph_2D, 
                                                                 {'kwargs_grid':{'ls':'--'}, 
                                                                  'kwargs_plot':{},
                                                                  'xlabel': global_kwargs['Temp'].label,
                                                                  'ylabel': global_kwargs['AmpOff_Response_1_Rds'].label, 
                                                                  'args_xlim': global_kwargs['Temp'].args_lim,
                                                                  'args_ylim': global_kwargs['AmpOff_Response_1_Rds'].args_lim,
                                                                  'title': 'AmpOff: Response 1', 
                                                                  'legend': True}
                                                                 ),
                                'AmpOff_Response_2_Rds': Function_params(
                                                                graph_2D, 
                                                                 {'kwargs_grid':{'ls':'--'}, 
                                                                  'kwargs_plot':{},
                                                                  'xlabel': global_kwargs['Temp'].label,
                                                                  'ylabel': global_kwargs['AmpOff_Response_2_Rds'].label, 
                                                                  'args_xlim': global_kwargs['Temp'].args_lim,
                                                                  'args_ylim': global_kwargs['AmpOff_Response_2_Rds'].args_lim,
                                                                  'title': 'AmpOff: Response 2', 
                                                                  'legend': True}
                                                                 )
                                
                                }),
                            'None': pd.Series({
                                'None': Function_params(graph_None, {}),
                                'Params_AmpOn': Function_params(graph_textbox, {'title': 'AmpOn: Characteristic Values'}),
                                'Params_AmpOff': Function_params(graph_textbox, {'title': 'AmpOff: Characteristic Values'})
                                }),
                            'AmpOn_Bias_plus': pd.Series({
                                'AmpOff_Auxiliary_1_plus': Function_params(
                                                            graph_2D, 
                                                            {'kwargs_grid':{'ls':'--'}, 'kwargs_plot':{},
                                                             'xlabel': global_kwargs['AmpOn_Bias_plus'].label,
                                                             'ylabel': global_kwargs['AmpOff_Auxiliary_1_plus'].label, 
                                                             'args_xlim': global_kwargs['AmpOn_Bias_plus'].args_lim,
                                                             'presort':False,
                                                             'marker':'x',
                                                             'color':'blue',
                                                             'title': 'AmpOff: Auxiliary(+) 1'}),
                                'AmpOff_Auxiliary_2_plus': Function_params(
                                                            graph_2D, 
                                                            {'kwargs_grid':{'ls':'--'}, 'kwargs_plot':{},
                                                             'xlabel': global_kwargs['AmpOn_Bias_plus'].label,
                                                             'ylabel': global_kwargs['AmpOff_Auxiliary_2_plus'].label, 
                                                             'args_xlim': global_kwargs['AmpOn_Bias_plus'].args_lim,
                                                             'presort':False,
                                                             'marker':'x',
                                                             'color':'red',
                                                             'title': 'AmpOff: Auxiliary(+) 2'}),
                                'AmpOn_Auxiliary_1_plus': Function_params(
                                                            graph_2D, 
                                                            {'kwargs_grid':{'ls':'--'}, 'kwargs_plot':{},
                                                             'xlabel': global_kwargs['AmpOn_Bias_plus'].label,
                                                             'ylabel': global_kwargs['AmpOn_Auxiliary_1_plus'].label, 
                                                             'args_xlim': global_kwargs['AmpOn_Bias_plus'].args_lim,
                                                             'presort':False,
                                                             'marker':'x',
                                                             'color':'blue',
                                                             'title': 'AmpOn: Auxiliary(+) 1'}),
                                'AmpOn_Auxiliary_2_plus': Function_params(
                                                            graph_2D, 
                                                            {'kwargs_grid':{'ls':'--'}, 'kwargs_plot':{},
                                                             'xlabel': global_kwargs['AmpOn_Bias_plus'].label,
                                                             'ylabel': global_kwargs['AmpOn_Auxiliary_2_plus'].label, 
                                                             'args_xlim': global_kwargs['AmpOn_Bias_plus'].args_lim,
                                                             'presort':False,
                                                             'marker':'x',
                                                             'color':'red',
                                                             'title': 'AmpOn: Auxiliary(+) 2'}),
                                'AmpOff_Auxiliary_1_minus': Function_params(
                                                            graph_2D, 
                                                            {'kwargs_grid':{'ls':'--'}, 'kwargs_plot':{},
                                                             'xlabel': global_kwargs['AmpOn_Bias_plus'].label,
                                                             'ylabel': global_kwargs['AmpOff_Auxiliary_1_minus'].label, 
                                                             'args_xlim': global_kwargs['AmpOn_Bias_plus'].args_lim,
                                                             'presort':False,
                                                             'marker':'x',
                                                             'color':'blue',
                                                             'title': 'AmpOff: Auxiliary(-) 1'}),
                                'AmpOff_Auxiliary_2_minus': Function_params(
                                                            graph_2D, 
                                                            {'kwargs_grid':{'ls':'--'}, 'kwargs_plot':{},
                                                             'xlabel': global_kwargs['AmpOn_Bias_plus'].label,
                                                             'ylabel': global_kwargs['AmpOff_Auxiliary_2_minus'].label, 
                                                             'args_xlim': global_kwargs['AmpOn_Bias_plus'].args_lim,
                                                             'presort':False,
                                                             'marker':'x',
                                                             'color':'red',
                                                             'title': 'AmpOff: Auxiliary(-) 2'}),
                                'AmpOn_Auxiliary_1_minus': Function_params(
                                                            graph_2D, 
                                                            {'kwargs_grid':{'ls':'--'}, 'kwargs_plot':{},
                                                             'xlabel': global_kwargs['AmpOn_Bias_plus'].label,
                                                             'ylabel': global_kwargs['AmpOn_Auxiliary_1_minus'].label, 
                                                             'args_xlim': global_kwargs['AmpOn_Bias_plus'].args_lim,
                                                             'presort':False,
                                                             'marker':'x',
                                                             'color':'blue',
                                                             'title': 'AmpOn: Auxiliary(-) 1'}),
                                'AmpOn_Auxiliary_2_minus': Function_params(
                                                            graph_2D, 
                                                            {'kwargs_grid':{'ls':'--'}, 'kwargs_plot':{},
                                                             'xlabel': global_kwargs['AmpOn_Bias_plus'].label,
                                                             'ylabel': global_kwargs['AmpOn_Auxiliary_2_minus'].label, 
                                                             'args_xlim': global_kwargs['AmpOn_Bias_plus'].args_lim,
                                                             'presort':False,
                                                             'marker':'x',
                                                             'color':'red',
                                                             'title': 'AmpOn: Auxiliary(-) 2'}),
                                }),
                            'AmpOn_Response_1': pd.Series({
                                'AmpOn_Imaginary_1': Function_params(
                                                            graph_2D, 
                                                            {'kwargs_grid':{'ls':'--'}, 'kwargs_plot':{},
                                                             'xlabel': global_kwargs['AmpOn_Response_1'].label,
                                                             'ylabel': global_kwargs['AmpOn_Imaginary_1'].label, 
                                                             'presort':False,
                                                             'legend':True,
                                                             'marker':'+',
                                                             'color':{'Forward branch':'blue', 'Reverse branch':'green'},
                                                             'title': 'AmpOn: Signal 1 in complex plane'})
                                }),
                            'AmpOn_Response_2': pd.Series({
                                'AmpOn_Imaginary_2': Function_params(
                                                            graph_2D, 
                                                            {'kwargs_grid':{'ls':'--'}, 'kwargs_plot':{},
                                                             'xlabel': global_kwargs['AmpOn_Response_2'].label,
                                                             'ylabel': global_kwargs['AmpOn_Imaginary_2'].label, 
                                                             'presort':False,
                                                             'legend':True,
                                                             'marker':'+',
                                                             'color':{'Forward branch':'red', 'Reverse branch':'green'},
                                                             'title': 'AmpOn: Signal 2 in complex plane'})
                                }),
                            'AmpOff_Response_1': pd.Series({
                                'AmpOff_Imaginary_1': Function_params(
                                                            graph_2D, 
                                                            {'kwargs_grid':{'ls':'--'}, 'kwargs_plot':{},
                                                             'xlabel': global_kwargs['AmpOff_Response_1'].label,
                                                             'ylabel': global_kwargs['AmpOff_Imaginary_1'].label, 
                                                             'presort':False,
                                                             'legend':True,
                                                             'marker':'+',
                                                             'color':{'Forward branch':'blue', 'Reverse branch':'green'},
                                                             'title': 'AmpOff: Signal 1 in complex plane'})
                                }),
                            'AmpOff_Response_2': pd.Series({
                                'AmpOff_Imaginary_2': Function_params(
                                                            graph_2D, 
                                                            {'kwargs_grid':{'ls':'--'}, 'kwargs_plot':{},
                                                             'xlabel': global_kwargs['AmpOff_Response_2'].label,
                                                             'ylabel': global_kwargs['AmpOff_Imaginary_2'].label, 
                                                             'presort':False,
                                                             'legend':True,
                                                             'marker':'+',
                                                             'color':{'Forward branch':'red', 'Reverse branch':'green'},
                                                             'title': 'AmpOff: Signal 2 in complex plane'})
                                }),
                                'AmpOn_Response_1_Vc': pd.Series({
                                    'None': Function_params(graph_hist, 
                                                            {'kwargs_grid':{'ls':'--'}, 
                                                             'kwargs_plot':{},
                                                             'xlabel': global_kwargs['AmpOn_Response_1_Vc'].label,
                                                             'ylabel': 'Frequency [-]', 
                                                             'legend':False,
                                                             'color': 'blue',
                                                             'title': r'AmpOn Response 1: Distribution of $V_c$'})
                                }),
                                'AmpOn_Response_2_Vc': pd.Series({
                                    'None': Function_params(graph_hist, 
                                                            {'kwargs_grid':{'ls':'--'}, 
                                                             'kwargs_plot':{},
                                                             'xlabel': global_kwargs['AmpOn_Response_2_Vc'].label,
                                                             'ylabel': 'Frequency [-]', 
                                                             'legend':False,
                                                             'color': 'red',
                                                             'title': r'AmpOn Response 2: Distribution of $V_c$'})
                                }),
                                'AmpOff_Response_1_Vc': pd.Series({
                                    'None': Function_params(graph_hist, 
                                                            {'kwargs_grid':{'ls':'--'}, 
                                                             'kwargs_plot':{},
                                                             'xlabel': global_kwargs['AmpOff_Response_1_Vc'].label,
                                                             'ylabel': 'Frequency [-]', 
                                                             'legend':False,
                                                             'color': 'blue',
                                                             'title': r'AmpOff Response 1: Distribution of $V_c$'})
                                }),
                                'AmpOff_Response_2_Vc': pd.Series({
                                    'None': Function_params(graph_hist, 
                                                            {'kwargs_grid':{'ls':'--'}, 
                                                             'kwargs_plot':{},
                                                             'xlabel': global_kwargs['AmpOff_Response_2_Vc'].label,
                                                             'ylabel': 'Frequency [-]', 
                                                             'legend':False,
                                                             'color': 'red',
                                                             'title': r'AmpOff Response 2: Distribution of $V_c$'})
                                }),
                                'AmpOn_Response_1_Rs': pd.Series({
                                    'None': Function_params(graph_hist, 
                                                            {'kwargs_grid':{'ls':'--'}, 
                                                             'kwargs_plot':{},
                                                             'xlabel': global_kwargs['AmpOn_Response_1_Rs'].label,
                                                             'ylabel': 'Frequency [-]', 
                                                             'legend':False,
                                                             'color': 'blue',
                                                             'title': r'AmpOn Response 1: Distribution of $R_s$'})
                                }),
                                'AmpOn_Response_2_Rs': pd.Series({
                                    'None': Function_params(graph_hist, 
                                                            {'kwargs_grid':{'ls':'--'}, 
                                                             'kwargs_plot':{},
                                                             'xlabel': global_kwargs['AmpOn_Response_2_Rs'].label,
                                                             'ylabel': 'Frequency [-]', 
                                                             'legend':False,
                                                             'color': 'red',
                                                             'title': r'AmpOn Response 2: Distribution of $R_s$'})
                                }),
                                'AmpOff_Response_1_Rs': pd.Series({
                                    'None': Function_params(graph_hist, 
                                                            {'kwargs_grid':{'ls':'--'}, 
                                                             'kwargs_plot':{},
                                                             'xlabel': global_kwargs['AmpOff_Response_1_Rs'].label,
                                                             'ylabel': 'Frequency [-]', 
                                                             'legend':False,
                                                             'color': 'blue',
                                                             'title': r'AmpOff Response 1: Distribution of $R_s$'})
                                }),
                                'AmpOff_Response_2_Rs': pd.Series({
                                    'None': Function_params(graph_hist, 
                                                            {'kwargs_grid':{'ls':'--'}, 
                                                             'kwargs_plot':{},
                                                             'xlabel': global_kwargs['AmpOff_Response_2_Rs'].label,
                                                             'ylabel': 'Frequency [-]', 
                                                             'legend':False,
                                                             'color': 'red',
                                                             'title': r'AmpOff Response 2: Distribution of $R_s$'})
                                }),
                                'AmpOn_Response_1_Rv': pd.Series({
                                    'None': Function_params(graph_hist, 
                                                            {'kwargs_grid':{'ls':'--'}, 
                                                             'kwargs_plot':{},
                                                             'xlabel': global_kwargs['AmpOn_Response_1_Rv'].label,
                                                             'ylabel': 'Frequency [-]', 
                                                             'legend':False,
                                                             'color': 'blue',
                                                             'title': r'AmpOn Response 1: Distribution of $R_v$'})
                                }),
                                'AmpOn_Response_2_Rv': pd.Series({
                                    'None': Function_params(graph_hist, 
                                                            {'kwargs_grid':{'ls':'--'}, 
                                                             'kwargs_plot':{},
                                                             'xlabel': global_kwargs['AmpOn_Response_2_Rv'].label,
                                                             'ylabel': 'Frequency [-]', 
                                                             'legend':False,
                                                             'color': 'red',
                                                             'title': r'AmpOn Response 2: Distribution of $R_v$'})
                                }),
                                'AmpOff_Response_1_Rv': pd.Series({
                                    'None': Function_params(graph_hist, 
                                                            {'kwargs_grid':{'ls':'--'}, 
                                                             'kwargs_plot':{},
                                                             'xlabel': global_kwargs['AmpOff_Response_1_Rv'].label,
                                                             'ylabel': 'Frequency [-]', 
                                                             'legend':False,
                                                             'color': 'blue',
                                                             'title': r'AmpOff Response 1: Distribution of $R_v$'})
                                }),
                                'AmpOff_Response_2_Rv': pd.Series({
                                    'None': Function_params(graph_hist, 
                                                            {'kwargs_grid':{'ls':'--'}, 
                                                             'kwargs_plot':{},
                                                             'xlabel': global_kwargs['AmpOff_Response_2_Rv'].label,
                                                             'ylabel': 'Frequency [-]', 
                                                             'legend':False,
                                                             'color': 'red',
                                                             'title': r'AmpOff Response 2: Distribution of $R_v$'})
                                }),
                                'AmpOn_Response_1_R0': pd.Series({
                                    'None': Function_params(graph_hist, 
                                                            {'kwargs_grid':{'ls':'--'}, 
                                                             'kwargs_plot':{},
                                                             'xlabel': global_kwargs['AmpOn_Response_1_R0'].label,
                                                             'ylabel': 'Frequency [-]', 
                                                             'legend':False,
                                                             'color': 'blue',
                                                             'title': r'AmpOn Response 1: Distribution of $R_0$'})
                                }),
                                'AmpOn_Response_2_R0': pd.Series({
                                    'None': Function_params(graph_hist, 
                                                            {'kwargs_grid':{'ls':'--'}, 
                                                             'kwargs_plot':{},
                                                             'xlabel': global_kwargs['AmpOn_Response_2_R0'].label,
                                                             'ylabel': 'Frequency [-]', 
                                                             'legend':False,
                                                             'color': 'red',
                                                             'title': r'AmpOn Response 2: Distribution of $R_0$'})
                                }),
                                'AmpOff_Response_1_R0': pd.Series({
                                    'None': Function_params(graph_hist, 
                                                            {'kwargs_grid':{'ls':'--'}, 
                                                             'kwargs_plot':{},
                                                             'xlabel': global_kwargs['AmpOff_Response_1_R0'].label,
                                                             'ylabel': 'Frequency [-]', 
                                                             'legend':False,
                                                             'color': 'blue',
                                                             'title': r'AmpOff Response 1: Distribution of $R_0$'})
                                }),
                                'AmpOff_Response_2_R0': pd.Series({
                                    'None': Function_params(graph_hist, 
                                                            {'kwargs_grid':{'ls':'--'}, 
                                                             'kwargs_plot':{},
                                                             'xlabel': global_kwargs['AmpOff_Response_2_R0'].label,
                                                             'ylabel': 'Frequency [-]', 
                                                             'legend':False,
                                                             'color': 'red',
                                                             'title': r'AmpOff Response 2: Distribution of $R_0$'})
                                }),
                                'AmpOn_Response_1_Ads': pd.Series({
                                    'None': Function_params(graph_hist, 
                                                            {'kwargs_grid':{'ls':'--'}, 
                                                             'kwargs_plot':{},
                                                             'xlabel': global_kwargs['AmpOn_Response_1_Ads'].label,
                                                             'ylabel': 'Frequency [-]', 
                                                             'legend':False,
                                                             'color': 'blue',
                                                             'title': r'AmpOn Response 1: Distribution of $A_ds$'})
                                }),
                                'AmpOn_Response_2_Ads': pd.Series({
                                    'None': Function_params(graph_hist, 
                                                            {'kwargs_grid':{'ls':'--'}, 
                                                             'kwargs_plot':{},
                                                             'xlabel': global_kwargs['AmpOn_Response_2_Ads'].label,
                                                             'ylabel': 'Frequency [-]', 
                                                             'legend':False,
                                                             'color': 'red',
                                                             'title': r'AmpOn Response 2: Distribution of $A_ds$'})
                                }),
                                'AmpOff_Response_1_Ads': pd.Series({
                                    'None': Function_params(graph_hist, 
                                                            {'kwargs_grid':{'ls':'--'}, 
                                                             'kwargs_plot':{},
                                                             'xlabel': global_kwargs['AmpOff_Response_1_Ads'].label,
                                                             'ylabel': 'Frequency [-]', 
                                                             'legend':False,
                                                             'color': 'blue',
                                                             'title': r'AmpOff Response 1: Distribution of $A_ds$'})
                                }),
                                'AmpOff_Response_2_Ads': pd.Series({
                                    'None': Function_params(graph_hist, 
                                                            {'kwargs_grid':{'ls':'--'}, 
                                                             'kwargs_plot':{},
                                                             'xlabel': global_kwargs['AmpOff_Response_2_Ads'].label,
                                                             'ylabel': 'Frequency [-]', 
                                                             'legend':False,
                                                             'color': 'red',
                                                             'title': r'AmpOff Response 2: Distribution of $A_ds$'})
                                }),
                                'AmpOn_Response_1_Imd': pd.Series({
                                    'None': Function_params(graph_hist, 
                                                            {'kwargs_grid':{'ls':'--'}, 
                                                             'kwargs_plot':{},
                                                             'xlabel': global_kwargs['AmpOn_Response_1_Imd'].label,
                                                             'ylabel': 'Frequency [-]', 
                                                             'legend':False,
                                                             'color': 'blue',
                                                             'title': r'AmpOn Response 1: Distribution of $Im_d$'})
                                }),
                                'AmpOn_Response_2_Imd': pd.Series({
                                    'None': Function_params(graph_hist, 
                                                            {'kwargs_grid':{'ls':'--'}, 
                                                             'kwargs_plot':{},
                                                             'xlabel': global_kwargs['AmpOn_Response_2_Imd'].label,
                                                             'ylabel': 'Frequency [-]', 
                                                             'legend':False,
                                                             'color': 'red',
                                                             'title': r'AmpOn Response 2: Distribution of $Im_d$'})
                                }),
                                'AmpOff_Response_1_Imd': pd.Series({
                                    'None': Function_params(graph_hist, 
                                                            {'kwargs_grid':{'ls':'--'}, 
                                                             'kwargs_plot':{},
                                                             'xlabel': global_kwargs['AmpOff_Response_1_Imd'].label,
                                                             'ylabel': 'Frequency [-]', 
                                                             'legend':False,
                                                             'color': 'blue',
                                                             'title': r'AmpOff Response 1: Distribution of $Im_d$'})
                                }),
                                'AmpOff_Response_2_Imd': pd.Series({
                                    'None': Function_params(graph_hist, 
                                                            {'kwargs_grid':{'ls':'--'}, 
                                                             'kwargs_plot':{},
                                                             'xlabel': global_kwargs['AmpOff_Response_2_Imd'].label,
                                                             'ylabel': 'Frequency [-]', 
                                                             'legend':False,
                                                             'color': 'red',
                                                             'title': r'AmpOff Response 2: Distribution of $Im_d$'})
                                }),
                                'AmpOn_Response_1_sigma-d': pd.Series({
                                    'None': Function_params(graph_hist, 
                                                            {'kwargs_grid':{'ls':'--'}, 
                                                             'kwargs_plot':{},
                                                             'xlabel': global_kwargs['AmpOn_Response_1_sigma-d'].label,
                                                             'ylabel': 'Frequency [-]', 
                                                             'legend':False,
                                                             'color': 'blue',
                                                             'title': r'AmpOn Response 1: Distribution of $\sigma _d$'})
                                }),
                                'AmpOn_Response_2_sigma-d': pd.Series({
                                    'None': Function_params(graph_hist, 
                                                            {'kwargs_grid':{'ls':'--'}, 
                                                             'kwargs_plot':{},
                                                             'xlabel': global_kwargs['AmpOn_Response_2_sigma-d'].label,
                                                             'ylabel': 'Frequency [-]', 
                                                             'legend':False,
                                                             'color': 'red',
                                                             'title': r'AmpOn Response 2: Distribution of $\sigma _d$'})
                                }),
                                'AmpOff_Response_1_sigma-d': pd.Series({
                                    'None': Function_params(graph_hist, 
                                                            {'kwargs_grid':{'ls':'--'}, 
                                                             'kwargs_plot':{},
                                                             'xlabel': global_kwargs['AmpOff_Response_1_sigma-d'].label,
                                                             'ylabel': 'Frequency [-]', 
                                                             'legend':False,
                                                             'color': 'blue',
                                                             'title': r'AmpOff Response 1: Distribution of $\sigma _d$'})
                                }),
                                'AmpOff_Response_2_sigma-d': pd.Series({
                                    'None': Function_params(graph_hist, 
                                                            {'kwargs_grid':{'ls':'--'}, 
                                                             'kwargs_plot':{},
                                                             'xlabel': global_kwargs['AmpOff_Response_2_sigma-d'].label,
                                                             'ylabel': 'Frequency [-]', 
                                                             'legend':False,
                                                             'color': 'red',
                                                             'title': r'AmpOff Response 2: Distribution of $\sigma _d$'})
                                }),
                                'AmpOn_Response_1_Rds': pd.Series({
                                    'None': Function_params(graph_hist, 
                                                            {'kwargs_grid':{'ls':'--'}, 
                                                             'kwargs_plot':{},
                                                             'xlabel': global_kwargs['AmpOn_Response_1_Rds'].label,
                                                             'ylabel': 'Frequency [-]', 
                                                             'legend':False,
                                                             'color': 'blue',
                                                             'title': r'AmpOn Response 1: Distribution of $R_ds$'})
                                }),
                                'AmpOn_Response_2_Rds': pd.Series({
                                    'None': Function_params(graph_hist, 
                                                            {'kwargs_grid':{'ls':'--'}, 
                                                             'kwargs_plot':{},
                                                             'xlabel': global_kwargs['AmpOn_Response_2_Rds'].label,
                                                             'ylabel': 'Frequency [-]', 
                                                             'legend':False,
                                                             'color': 'red',
                                                             'title': r'AmpOn Response 2: Distribution of $R_ds$'})
                                }),
                                'AmpOff_Response_1_Rds': pd.Series({
                                    'None': Function_params(graph_hist, 
                                                            {'kwargs_grid':{'ls':'--'}, 
                                                             'kwargs_plot':{},
                                                             'xlabel': global_kwargs['AmpOff_Response_1_Rds'].label,
                                                             'ylabel': 'Frequency [-]', 
                                                             'legend':False,
                                                             'color': 'blue',
                                                             'title': r'AmpOff Response 1: Distribution of $R_ds$'})
                                }),
                                'AmpOff_Response_2_Rds': pd.Series({
                                    'None': Function_params(graph_hist, 
                                                            {'kwargs_grid':{'ls':'--'}, 
                                                             'kwargs_plot':{},
                                                             'xlabel': global_kwargs['AmpOff_Response_2_Rds'].label,
                                                             'ylabel': 'Frequency [-]', 
                                                             'legend':False,
                                                             'color': 'red',
                                                             'title': r'AmpOff Response 2: Distribution of $R_ds$'})
                                }),
                            }).transpose()
    '''...........................................................................................................................'''
    def __init__(self, obj_dict):
        self.obj_dict = obj_dict
    '''...........................................................................................................................'''
    def set_active_obj(self, obj_name):
        self.samplename = obj_name
        self.obj = self.obj_dict[obj_name]
    '''...........................................................................................................................'''
    def plot_Forcecurve(self, filename, arrangement, mode, line=0, point=0, x_signals=None, y_signals=None, 
                        functions_kwargs_ext=None, subplots_kwargs_ext=None, save_fig=False, xerr=None, yerr=None, style='default'):
        '''
        New generalised Version of the Forcecurve data displayer. 
        Replaces all preceeding functions.
        
        Argument 'arrangement':
            dtype = string
            valid keywords =  
                * 'raw' ... Plots the data in rawdata mode (2x3 Graphs)
                * 'response' ... Plots the data in response mode (2x2 Graphs)
                * 'overview' ... Plots all possible Graph types of a Forcecurve (curently 2x5 Graphs)
                * 'custom' ... Custom arrangement of Graphs (use kwargs x_signals and y_signals to specify) 
            
        Argument 'mode'
            dtype = string
            valid keywords = 
                * 'single' - Default line and point is 0;0. Use line and point kwargs to choose another hyst.  
                * 'super' - Superimposes all points and lines in a single Graph
        '''
        #First initialize the signal matrix for x and y
        if arrangement == 'custom': 
            x_signals = np.array(x_signals)
            y_signals = np.array(y_signals)
            #Check if shape is equal
            if x_signals.shape == y_signals.shape: 
                rows, columns = x_signals.shape
            else: 
                raise Exception('x_signals and y_signals are not the same shape!')
        else: 
            x_signals = np.array(self.global_kwargs[arrangement].x_signals)
            y_signals = np.array(self.global_kwargs[arrangement].y_signals)
            rows, columns = x_signals.shape
        
        #Check wether functions_kwargs_ext must be processed
        if functions_kwargs_ext is not None:
            try: 
                functions_kwargs_ext = np.array(functions_kwargs_ext)
            except: 
                print("Error while converting variable functions_kwargs_ext, make sure it is iterable. kwargs are not passed")
                functions_kwargs_ext = np.array([[{} for column in range(columns)] for row in range(rows)])
        else: 
            functions_kwargs_ext = np.array([[{} for column in range(columns)] for row in range(rows)])
        
        #Check wether subplots_kwargs_ext must to be processed
        if subplots_kwargs_ext is None:
            subplots_kwargs_ext = {}
            '''try: 
                subplots_kwargs_ext = np.ndarray(functions_kwargs_ext)
            except: 
                print("Error while converting variable functions_kwargs_ext, make sure it is iterable. kwargs are not passed")
                subplots_kwargs_ext = np.array([[{} for column in range(columns)] for row in range(rows)])
        else: 
            subplots_kwargs_ext = np.array([[{} for column in range(columns)] for row in range(rows)])'''
            
        #Initializing: function_kwargs, subplots_kwargs, functions, scaling_factors, functions_args
        functions_kwargs = [[None for column in range(columns)] for row in range(rows)]
        subplots_kwargs = kwargs(self.global_kwargs[arrangement].subplot_kwargs, subplots_kwargs_ext)
        functions = [[None for column in range(columns)] for row in range(rows)]
        x_scaling_factors = [[None for column in range(columns)] for row in range(rows)]
        y_scaling_factors = [[None for column in range(columns)] for row in range(rows)]
        functions_args = [[None for column in range(columns)] for row in range(rows)]
        err_kwargs = [[{'xerr':None, 'yerr':None} for column in range(columns)] for row in range(rows)]

        
        #Filling them with the right params from the reference param sets
        for row in range(rows): 
            for column in range(columns):
                x_scaling_factors[row][column] = self.global_kwargs[x_signals[row,column]].scaling_factor
                y_scaling_factors[row][column] = self.global_kwargs[y_signals[row,column]].scaling_factor
                if not xerr == None:
                    err_kwargs[row][column]['xerr'] = self.get_FC_data(x_signals[row,column]+'_std', 
                                                                                 filename, 
                                                                                 mode, 
                                                                                 line, 
                                                                                 point, 
                                                                                 scaling=x_scaling_factors[row][column])
                if not yerr == None:
                    err_kwargs[row][column]['yerr'] = self.get_FC_data(y_signals[row,column]+'_std', 
                                                                                 filename, 
                                                                                 mode, 
                                                                                 line, 
                                                                                 point, 
                                                                                 scaling=y_scaling_factors[row][column])
                    
                functions_kwargs[row][column] = kwargs(self.function_kwargs.loc[x_signals[row,column],
                                                                                y_signals[row,column]].function_kwargs,
                                                       self.global_kwargs[mode].plot_kwargs,
                                                       self.global_kwargs[arrangement].plot_kwargs[row][column], #list of lists
                                                       functions_kwargs_ext[row,column],
                                                       err_kwargs[row][column]
                                                      )

                functions[row][column] = self.function_kwargs.loc[x_signals[row,column],
                                                                  y_signals[row,column]].function
                x = self.get_FC_data(x_signals[row,column], filename, mode, line, point, scaling=x_scaling_factors[row][column])
                y = self.get_FC_data(y_signals[row,column], filename, mode, line, point, scaling=y_scaling_factors[row][column])
                functions_args[row][column] = args(x,y)
        
        #Converting the lists into arrays
        functions_kwargs = np.array(functions_kwargs)
        functions = np.array(functions)
        x_scaling_factors = np.array(x_scaling_factors)
        y_scaling_factors = np.array(y_scaling_factors)
        functions_args = np.array(functions_args)
        
        #Arrange the subplot_title
        title_text = copy.deepcopy(self.global_kwargs[arrangement].subplot_title.texts)
        title_order = self.global_kwargs[arrangement].subplot_title.param_order
        
        if 'samplename' in title_order:
            title_text['samplename'] = self.samplename
        if 'mode' in title_order: 
            if mode == 'super': 
                title_text['mode'] = ' (superimposed)'
            else: 
                title_text['mode'] = ' (Line ' + str(line) + ' , Point '+ str (point) + ')'
        if 'temp' in title_order: 
            temp = self.obj.get_Forcemap_value(filename, 'Temp', line=line, point=point)
            title_text['temp'] = '[T= ' + str(temp) + '°C]'
            
        title = ''
        for title_part in title_order: 
            title = title + title_text[title_part]
        
        title_kwargs = self.global_kwargs[arrangement].subplot_title.suptitle_kwargs
            
        #Plot the figure
        fig, ax, lineObjects = flex_subplot(functions, functions_args, functions_kwargs, subplots_kwargs=subplots_kwargs.get(), 
                                            title=title, title_kwargs=title_kwargs, style=style)
        #Savefig
        if save_fig:
            path = os.path.join(self.obj.get_objDir_path(),'Forcecurve_'+arrangement+'_'+mode)
            if mode == 'single': 
                path = os.path.join(path,filename)
                temp_filename = filename + '_L' + str(line) + '_P' + str(point) + '_' + arrangement + '_' + mode
            else: 
                temp_filename = filename + '_' + arrangement + '_' + mode
            
            save_pyplot_figure(fig, temp_filename, path=path, kwargs_savefig={'dpi':100})
                
        return fig, ax, lineObjects
            
              
    '''...........................................................................................................................'''        
    def get_FC_data(self, signal_name, filename, mode, line, point,
                    delta_1=0, delta_2=0, scaling=1):
        '''
        A refined version of extract_Forcecurve_data compatible with the newer plot functions. 
        '''
        #Split the string 
        signal_code = str.split(signal_name,sep='_')
        
        #Declare the right function to extract the desired data
        hyst_var = ['Vc', 'Rs', 'Rv', 'R0', 'Ads', 'Imd', 'sigma-d', 'Rds', 
                    'Vf_plus','Vf_minus', 'Vf', 'Rfs_plus', 'Rfs_minus', 'Rfs', 
                    'Imf', 'Afs', 'Vfc_plus', 'Vfc_minus', 'Vsurf']
        if mode == 'evol':
            dic = {}
            #Chose the right stat param
            if signal_code[-1] == 'flag':
                signal = signal_code[-2]
                mode = signal_code[0]
                stat = 'flag'
                if signal =='C': 
                    signal_name = '_'.join([mode, 'Response', signal, 'r-squared-flag'])
                    
                    for samplename in self.obj_dict.keys():
                        self.set_active_obj(samplename)
                        dic[samplename] = self.obj.get_sample_value_evol(signal_name, stat)
                    
                    return dic

                else: 
                    signal_names = ['_'.join([mode, 'Complex', signal, 'r-squared-flag']),
                                    '_'.join([mode, 'Phase', signal, 'r-squared-flag']),
                                    '_'.join([mode, 'Phase', signal, 'a1-a2-flag']),
                                    '_'.join([mode, 'Response', signal, 'r-squared-flag'])]
                    label_names = [' '.join(['Complex', 'r-squared-flag']),
                                   ' '.join(['Phase', 'r-squared-flag']),
                                   ' '.join(['Phase', 'a1-a2-flag']),
                                   ' '.join(['Response', 'r-squared-flag'])]
                
                    for signal_name, label_name in zip(signal_names, label_names): 
                        dic[label_name] = self.obj.get_sample_value_evol(signal_name, stat)
                
                    return dic
            
            elif signal_code[-1] == 'std':
                stat = 'std'
                signal_name = '_'.join(signal_code[:-1])

            else: 
                stat = 'mean'
            
            for samplename in self.obj_dict.keys():
                self.set_active_obj(samplename)
                dic[samplename] = self.obj.get_sample_value_evol(signal_name,stat) * scaling
            
            return dic
        
        elif signal_code[0] == 'None':
            fun = lambda fc, signal_code, scaling: np.array([])
        elif signal_code[0] == 'Params':
            return self.extrac_params(filename, signal_code[1], mode, line=line, point=point)
        elif signal_code[-1] in hyst_var or signal_code[-2] in hyst_var:
            fun = lambda fc, signal_code, scaling: scaling*self.obj.get_Forcemap_column(filename, signal_name)
        elif signal_code[-1] in ['plus','minus']:
            fun = lambda fc, signal_code, scaling: scaling*(fc.loc['Aux_stat'].loc[:,'_'.join(signal_code)+'_mean'] + delta_1)
        elif signal_code[1] == 'Bias': #For single branch bias signal
            fun = lambda fc, signal_code, scaling: scaling*(fc.loc['Aux_stat'].loc[:,'_'.join(signal_code)+'_plus_mean'] + delta_1)
        elif signal_code[1] in ['Phase','Amplitude','Response','Imaginary']: #for two branch signals 
            def fun(fc, signal_code, scaling): 
                return {'Forward branch': scaling*(fc.loc['Aux_stat'].loc[:,'_'.join(signal_code)+'_plus_mean'] + delta_1),
                        'Reverse branch': scaling*(fc.loc['Aux_stat'].loc[:,'_'.join(signal_code)+'_minus_mean'] + delta_1)}
        else: 
            raise Exception('Wrong signal code')
            #fun = lambda fc, signal_code: fc.loc['Hysteresis'].loc[:,'_'.join([signal_code[0], 'Amplitude_mean'])]*np.cos(np.radians(fc.loc['Hysteresis'].loc[:,'_'.join([signal_code[0],'Phase', signal_code[2]])+'_mean']+delta_2))+delta_1
        
        #Destinguish btw. different modes
        if mode in ['single', 'single_dist']: 
            fc = self.obj.get_Forcemap_row(filename, line, point)
            return fun(fc,signal_code,scaling)
        
        elif mode == 'super': #Change it with the new index function 
            data_col = {'Line':[]}
            for index in self.obj.get_Forcemap_index(filename):
                fc = self.obj.get_Forcemap_row(filename, 0, 0, curvename=index)
                dataset = fun(fc, signal_code, scaling)
                if type(dataset) == dict:
                    for key in dataset.keys():
                        if key in data_col.keys(): 
                            data_col[key].extend(dataset[key])
                        else:
                            data_col[key] = list(dataset[key])
                else: 
                    data_col['Line'].extend(dataset)
            
            for key in data_col.keys():
                data_col[key] = np.array(data_col[key])
            
            if len(data_col['Line'])==0:
                data_col.pop('Line')
                    
            return data_col 
    '''...........................................................................................................................'''    
    def extrac_params(self, filename, bias_mode, mode, line=0, point=0):
        '''
        Returns a formated text of the char. values of the hysteresis loop
        '''
        signals = ['1','2']
        param = namedtuple('Parametervalue', 'name symbol factor error unit order')
        paramvals = [param('V0_plus', r'$V_0^+$', 1, {'single':True, 'super':True, 'single_dist':True}, 
                            '[V]', ('bias_mode', 'Response_txt', 'signal', 'key')),
                     param('V0_minus', r'$V_0^-$', 1, {'single':True, 'super':True, 'single_dist':True}, 
                           '[V]', ('bias_mode', 'Response_txt', 'signal', 'key')),
                     param('Vc', r'$V_c$', 1, {'single':True, 'super':True, 'single_dist':True}, 
                           '[V]', ('bias_mode', 'Response_txt', 'signal', 'key')),
                     param('Rs_plus', r'$R_s^+$', 1e12, {'single':False, 'super':True, 'single_dist':True}, 
                           '[pm]', ('bias_mode', 'Response_txt', 'signal', 'key')),
                     param('Rs_minus', r'$R_s^-$', 1e12, {'single':False, 'super':True, 'single_dist':True}, 
                           '[pm]', ('bias_mode', 'Response_txt', 'signal', 'key')),
                     param('Rs', r'$R_s$', 1e12, {'single':False, 'super':True, 'single_dist':True}, 
                           '[pm]', ('bias_mode', 'Response_txt', 'signal', 'key')),
                     param('Rv', r'$R_v$', 1e12, {'single':True, 'super':True, 'single_dist':True}, 
                           '[pm]', ('bias_mode', 'Response_txt', 'signal', 'key')),
                     param('R0', r'$R_0$', 1e12, {'single':True, 'super':True, 'single_dist':True}, 
                           '[pm]', ('bias_mode', 'Response_txt', 'signal', 'key')),
                     param('Ads', r'$A_{ds}$', 1e9, {'single':False, 'super':True, 'single_dist':True}, 
                           '[nVm]', ('bias_mode', 'Response_txt', 'signal', 'key')),
                     param('Imd', r'$Im_d$', 1, {'single':False, 'super':True, 'single_dist':True}, 
                           '[V]', ('bias_mode', 'Response_txt', 'signal', 'key')),
                     param('sigma-d', r'$\sigma _d$', 1, {'single':False, 'super':True, 'single_dist':True}, 
                           '[V]', ('bias_mode', 'Response_txt', 'signal', 'key')),
                     param('Rds_plus', r'$R_{ds}^+$', 1e12, {'single':False, 'super':True, 'single_dist':True}, 
                           '[pm]', ('bias_mode', 'Response_txt', 'signal', 'key')),
                     param('Rds_minus', r'$R_{ds}^-$', 1e12, {'single':False, 'super':True, 'single_dist':True}, 
                           '[pm]', ('bias_mode', 'Response_txt', 'signal', 'key')),
                     param('Rds', r'$R_{ds}$', 1e12, {'single':False, 'super':True, 'single_dist':True}, 
                           '[pm]', ('bias_mode', 'Response_txt', 'signal', 'key')),
                     param('a', r'$a$', 1, {'single':False, 'super':True, 'single_dist':True}, 
                           '', ('bias_mode', 'Response_txt', 'signal', 'key')),]
                     
        stat_kw = {'bias_mode':bias_mode,
                   'Response_txt':'Response',}
        dic = {}
        #Iterate over both signals 
        for signal in signals: 
            stat_kw['signal'] = signal
            texts = []
            #iterate over all params
            for paramval in paramvals:
                stat_kw['key'] = paramval.name
                columnname = '_'.join([stat_kw[x] for x in paramval.order])
                
                if mode == 'single': 
                    val = self.obj.get_Forcemap_value(filename, columnname, line=line, point=point) * paramval.factor
                    if paramval.error[mode] == True:
                        err = self.obj.get_Forcemap_value(filename, columnname+'_err', line=line, point=point) * paramval.factor
                        err = ' ± {err:.2g} '.format(err=err)
                    else: 
                        err = ' '
                elif mode in ['super', 'single_dist']: 
                    val = self.obj.get_Forcemap_column_stat(filename, columnname, 'mean') * paramval.factor
                    if paramval.error[mode] == True:
                        err = self.obj.get_Forcemap_column_stat(filename, columnname, 'std') * paramval.factor
                        err = ' ± {err:.3g} '.format(err=err)
                    else: 
                        err = ' '
                        
                
                texts.append('{symbol} = {val:.4g}{err}{unit}'.format(symbol=paramval.symbol, val=val, 
                                                                              err=err, unit=paramval.unit))
            #Add text to dic
            dic['Response ' + signal] = '\n'.join(texts)
        
        return dic
                
                
                
    '''...........................................................................................................................'''    
    def get_filenames(self): 
        return self.obj.get_Filenames()
    '''...........................................................................................................................'''
    
    #Old code
    #################################################################################################################################
    # def plot_Forcecurve_raw_single(self, filename, line, point, functions_kwargs_ext=None, subplots_kwargs_ext=None, save_fig=False):
    #     '''
    #     Plots the raw forcecurve; Good for separated inspection.
    #     Only for displaying on single forcecurve
    #     '''
    #     #Define the arrangment of the signals
    #     y_signals= [['AmpOn_Phase_1', 'AmpOn_Phase_2', 'AmpOn_Amplitude'],
    #                 ['AmpOff_Phase_1', 'AmpOff_Phase_2', 'AmpOff_Amplitude']]
    #     x_signals= [['AmpOn_Bias','AmpOn_Bias','AmpOn_Bias'],
    #                 ['AmpOn_Bias','AmpOn_Bias','AmpOn_Bias']]
        
    #     #Check wether functions_kwargs has to be processed
    #     if functions_kwargs_ext is not None:
    #         try: 
    #             functions_kwargs_ext = np.ndarray(functions_kwargs_ext)
    #         except: 
    #             print("Error while converting variable functions_kwargs_ext, make sure it is iterable. kwargs are not passed")
    #             functions_kwargs_ext = np.array([[{},{},{}],[{},{},{}]])
    #     else: 
    #         functions_kwargs_ext = np.array([[{},{},{}],[{},{},{}]])
        
    #     #Set the functions functions_kwargs
    #     functions_kwargs = [[kwargs({'fmt':'bx-'},functions_kwargs_ext[0,0]).get(),
    #                          kwargs({'fmt':'rx-'},functions_kwargs_ext[0,1]).get(),
    #                          kwargs({'fmt':'gx-'},functions_kwargs_ext[0,2]).get()],
    #                         [kwargs({'fmt':'bx-'},functions_kwargs_ext[1,0]).get(),
    #                          kwargs({'fmt':'rx-'},functions_kwargs_ext[1,1]).get(),
    #                          kwargs({'fmt':'gx-'},functions_kwargs_ext[1,2]).get()]]
        
    #     #Plot figure
    #     fig, ax = self.plot_Forcecurve_raw(filename, line, point, x_signals, y_signals, functions_kwargs_ext=functions_kwargs)
        
    #     #Savefig
    #     if save_fig:
    #         path = os.path.join(self.obj.get_objDir_path(),'Forcecurve_raw_single')
    #         filename_temp = filename + '_Line-'+ str(line) + '_Point-' + str(point)
    #         save_pyplot_figure(fig, filename_temp, path=path, kwargs_savefig={'dpi':100})
        
    #     return fig, ax
    # '''...........................................................................................................................'''
    # def plot_Forcecurve_raw_super(self, filename, functions_kwargs_ext=None, subplots_kwargs_ext=None, save_fig=False):
    #     '''
    #     Plots the raw forcecurve; Good for separated inspection.
    #     Superimposed version
    #     '''
    #     #Signal matrix
    #     y_signals= [['AmpOn_Phase_1_super', 'AmpOn_Phase_2_super', 'AmpOn_Amplitude_super'],
    #                 ['AmpOff_Phase_1_super', 'AmpOff_Phase_2_super','AmpOff_Amplitude_super']]
    #     x_signals= [['AmpOn_Bias_super','AmpOn_Bias_super', 'AmpOn_Bias_super'],
    #                 ['AmpOn_Bias_super','AmpOn_Bias_super', 'AmpOn_Bias_super']]
        
    #     #Check wether functions_kwargs has to be processed
    #     if functions_kwargs_ext is not None:
    #         try: 
    #             if type(functions_kwargs_ext) is not np.ndarray:
    #                 functions_kwargs_ext = np.ndarray(functions_kwargs_ext)
    #         except: 
    #             print("Error while converting variable functions_kwargs_ext, make sure it is iterable. kwargs are not passed")
    #             functions_kwargs_ext = np.array([[{},{},{}],[{},{},{}]])
    #     else: 
    #         functions_kwargs_ext = np.array([[{},{},{}],[{},{},{}]])
        
    #     #Global params
    #     ms = 0.4
        
    #     #Set the functions functions_kwargs
    #     functions_kwargs = [[kwargs({'fmt':'bo', 'kwargs_plot':{'ms':ms}},functions_kwargs_ext[0,0]).get(),
    #                          kwargs({'fmt':'ro', 'kwargs_plot':{'ms':ms}},functions_kwargs_ext[0,1]).get(),
    #                          kwargs({'fmt':'go', 'kwargs_plot':{'ms':ms}},functions_kwargs_ext[0,2]).get()],
    #                         [kwargs({'fmt':'bo', 'kwargs_plot':{'ms':ms}},functions_kwargs_ext[1,0]).get(),
    #                          kwargs({'fmt':'ro', 'kwargs_plot':{'ms':ms}},functions_kwargs_ext[1,1]).get(),
    #                          kwargs({'fmt':'go', 'kwargs_plot':{'ms':ms}},functions_kwargs_ext[1,2]).get()]]
       
    #     #Plot figure
    #     fig, ax = self.plot_Forcecurve_raw(filename, 'super', 'super', x_signals, y_signals,
    #                                        functions_kwargs_ext=functions_kwargs)
        
    #     #Savefig
    #     if save_fig:
    #         path = os.path.join(self.obj.get_objDir_path(),'Forcecurve_raw_super')
    #         save_pyplot_figure(fig, filename, path=path, kwargs_savefig={'dpi':100})
                
    #     return fig, ax
        
    # '''...........................................................................................................................'''
    # def plot_Forcecurve_raw(self, filename, line, point, x_signals, y_signals, 
    #                         functions_kwargs_ext=None, subplots_kwargs_ext=None):
    #     '''
    #     Plots the raw forcecurve; Good for separated inspection.
    #     Most general version
    #     '''
    #     #Check wether line or point are super
    #     if line == point == 'super':
    #         line_temp = 0
    #         point_temp = 0
    #     else:
    #         line_temp = line
    #         point_temp = point
    #     #Get the temperature
    #     temp = self.obj.get_Forcemap_value(filename, 'Temp', line=line_temp, point=point_temp)
    #     temp_string = '(T= ' + str(temp) + '°C)'
       
    #     #Check wheter functions_kwargs_ext has the right format
    #     if functions_kwargs_ext is not None:
    #         try: 
    #             if type(functions_kwargs_ext) is not np.ndarray:
    #                 functions_kwargs_ext = np.array(functions_kwargs_ext)
    #         except: 
    #             print("Error while converting variable functions_kwargs_ext, make sure it is iterable. kwargs are not passed")
    #             functions_kwargs_ext = np.array([[{},{},{}],[{},{},{}]])
    #     else: 
    #         functions_kwargs_ext = np.array([[{},{},{}],[{},{},{}]])
       
    #    #Now the functions_args
    #     a1_kwargs = kwargs({'title':'AmpOn: Phase 1 '+temp_string,'args_xlim':(-50,50),'kwargs_grid':{'ls':'--'}, 
    #                         'ylabel':r'$\phi$ [°]'}, functions_kwargs_ext[0,0])
    #     a2_kwargs = kwargs({'title':'AmpOn: Phase 2 '+temp_string,'args_xlim':(-50,50),'kwargs_grid':{'ls':'--'}, 
    #                         'ylabel':r'$\phi$ [°]'}, functions_kwargs_ext[0,1])
    #     a3_kwargs = kwargs({'title':'AmpOn: Amplitude '+temp_string,'args_xlim':(-50,50),'kwargs_grid':{'ls':'--'},
    #                         'kwargs_ylim':{'bottom':0}, 'ylabel':'A [pm]'}, functions_kwargs_ext[0,2])
    #     a4_kwargs = kwargs({'title':'AmpOff: Phase 1 '+temp_string,'xlabel':'Bias [V]','args_xlim':(-50,50),
    #                         'kwargs_grid':{'ls':'--'}, 'ylabel':r'$\phi$ [°]'}, functions_kwargs_ext[1,0])
    #     a5_kwargs = kwargs({'title':'AmpOff: Phase 2 '+temp_string,'xlabel':'Bias [V]','args_xlim':(-50,50),
    #                         'kwargs_grid':{'ls':'--'}, 'ylabel':r'$\phi$ [°]'}, functions_kwargs_ext[1,1])
    #     a6_kwargs = kwargs({'title':'AmpOff: Amplitude '+temp_string,'xlabel':'Bias [V]','args_xlim':(-50,50),
    #                         'kwargs_grid':{'ls':'--'}, 'kwargs_ylim':{'bottom':0}, 'ylabel':'A [pm]'}, functions_kwargs_ext[1,2])
        
    #     subplot_kwargs = {'figsize':(12.8,8), 'sharex':True}
        
    #     functions_kwargs = np.array([[a1_kwargs, a2_kwargs, a3_kwargs],[a4_kwargs, a5_kwargs, a6_kwargs]])
        
    #     #Plot figure
    #     return self.plot_SS_PFM_data(filename, line, point, x_signals, y_signals, functions_kwargs, subplot_kwargs)
    #     '''...........................................................................................................................'''
    
    # def plot_Forcecurve_response_super(self, filename, functions_kwargs_ext=None, subplots_kwargs_ext=None, save_fig=False):
    #     '''
    #     Plots a convolution of Amplitude and Phase for AmpOn, AmpOff, Phase 1/2
    #     Superimposed version.
    #     '''
    #     #Signal matrix
    #     y_signals= [['AmpOn_Response_1_super', 'AmpOn_Response_2_super'],
    #                 ['AmpOff_Response_1_super', 'AmpOff_Response_2_super']]
    #     x_signals= [['AmpOn_Bias_super','AmpOn_Bias_super'],
    #                 ['AmpOn_Bias_super','AmpOn_Bias_super']]
        
    #     #Check wether functions_kwargs has to be processed
    #     if functions_kwargs_ext is not None:
    #         try: 
    #             if type(functions_kwargs_ext) is not np.ndarray:
    #                 functions_kwargs_ext = np.ndarray(functions_kwargs_ext)
    #         except: 
    #             print("Error while converting variable functions_kwargs_ext, make sure it is iterable. kwargs are not passed")
    #             functions_kwargs_ext = np.array([[{},{}],[{},{}]])
    #     else: 
    #         functions_kwargs_ext = np.array([[{},{}],[{},{}]])
        
    #     #Global params
    #     ms = 0.4
        
    #     #Set the functions functions_kwargs
    #     functions_kwargs = [[kwargs({'fmt':'bo', 'kwargs_plot':{'ms':ms}},functions_kwargs_ext[0,0]).get(),
    #                          kwargs({'fmt':'ro', 'kwargs_plot':{'ms':ms}},functions_kwargs_ext[0,1]).get()],
    #                         [kwargs({'fmt':'bo', 'kwargs_plot':{'ms':ms}},functions_kwargs_ext[1,0]).get(),
    #                          kwargs({'fmt':'ro', 'kwargs_plot':{'ms':ms}},functions_kwargs_ext[1,1]).get()]]
        
        
    #     #Plot figure
    #     fig, ax = self.plot_Forcecurve_response(filename, 'super', 'super', x_signals, y_signals, 
    #                                             functions_kwargs_ext=functions_kwargs)
    #     #Savefig
    #     if save_fig:
    #         path = os.path.join(self.obj.get_objDir_path(),'Forcecurve_response_super')
    #         save_pyplot_figure(fig, filename, path=path, kwargs_savefig={'dpi':100})
                
    #     return fig, ax
    # '''...........................................................................................................................'''        
    # def plot_Forcecurve_response_single(self, filename, line, point, functions_kwargs_ext=None, 
    #                                     subplots_kwargs_ext=None, save_fig=False):
    #     '''
    #     Plots a convolution of Amplitude and Phase for AmpOn, AmpOff, Phase 1/2
    #     Single forceplot version.
    #     '''
    #     #Signal matrix
    #     y_signals= [['AmpOn_Response_1', 'AmpOn_Response_2'],
    #                 ['AmpOff_Response_1', 'AmpOff_Response_2']]
    #     x_signals= [['AmpOn_Bias','AmpOn_Bias'],
    #                 ['AmpOn_Bias','AmpOn_Bias']]
       
    #     #Check wether functions_kwargs_ext has to be processed
    #     if functions_kwargs_ext is not None:
    #         try: 
    #             if type(functions_kwargs_ext) is not np.ndarray:
    #                 functions_kwargs_ext = np.ndarray(functions_kwargs_ext)
    #         except: 
    #             print("Error while converting variable functions_kwargs_ext, make sure it is iterable. kwargs are not passed")
    #             functions_kwargs_ext = np.array([[{},{}],[{},{}]])
    #     else: 
    #         functions_kwargs_ext = np.array([[{},{}],[{},{}]])
        
    #     #Set the functions functions_kwargs
    #     functions_kwargs = [[kwargs({'fmt':'bx-'},functions_kwargs_ext[0,0]).get(),
    #                          kwargs({'fmt':'rx-'},functions_kwargs_ext[0,1]).get()],
    #                         [kwargs({'fmt':'bx-'},functions_kwargs_ext[1,0]).get(),
    #                          kwargs({'fmt':'rx-'},functions_kwargs_ext[1,1]).get()]]
        
    #     #Plot figure
    #     fig, ax = self.plot_Forcecurve_response(filename, line, point, x_signals, y_signals, functions_kwargs_ext=functions_kwargs)
        
    #     #Savefig
    #     if save_fig:
    #         path = os.path.join(self.obj.get_objDir_path(),'Forcecurve_response_single')
    #         filename_temp = filename + '_Line-'+ str(line) + '_Point-' + str(point)
    #         save_pyplot_figure(fig, filename_temp, path=path, kwargs_savefig={'dpi':100})
        
    #     return fig, ax
    # '''...........................................................................................................................'''    
    # def plot_Forcecurve_response(self, filename, line, point, x_signals, y_signals, 
    #                              functions_kwargs_ext=None, subplots_kwargs_ext=None):
    #     '''
    #     Plots a convolution of Amplitude and Phase for AmpOn, AmpOff, Phase 1/2
    #     Most general version.
    #     '''
        
    #     #Check wether line or point are super
    #     if line == point == 'super':
    #         line_temp = 0
    #         point_temp = 0
    #     else:
    #         line_temp = line
    #         point_temp = point
    #     #Get the temperature
    #     temp = self.obj.get_Forcemap_value(filename, 'Temp', line=line_temp, point=point_temp)
    #     temp_string = '(T= ' + str(temp) + '°C)'
        
    #     #Check wheter functions_kwargs_ext has the right format
    #     if functions_kwargs_ext is not None:
    #         try: 
    #             if type(functions_kwargs_ext) is not np.ndarray:
    #                 functions_kwargs_ext = np.array(functions_kwargs_ext)
    #         except: 
    #             print("Error while converting variable functions_kwargs_ext, make sure it is iterable. kwargs are not passed")
    #             functions_kwargs_ext = np.array([[{},{}],[{},{}]])
    #     else: 
    #         functions_kwargs_ext = np.array([[{},{}],[{},{}]])
            
        
    #     #Now the functions_args ==> AmpOn Green, AmpOff blue
    #     a1_kwargs = kwargs({'title':'AmpOn: Response 1 '+temp_string,'args_xlim':(-50,50),'kwargs_grid':{'ls':'--'},
    #                         'ylabel':r'$A \; cos(\phi) \; [pm]$'}, functions_kwargs_ext[0,0])
    #     a2_kwargs = kwargs({'title':'AmpOn: Response 2 '+temp_string,'args_xlim':(-50,50),'kwargs_grid':{'ls':'--'},
    #                         'ylabel':r'$A \; cos(\phi) \; [pm]$'}, functions_kwargs_ext[0,1])
    #     a3_kwargs = kwargs({'title':'AmpOff: Response 1 '+temp_string,'args_xlim':(-50,50),'kwargs_grid':{'ls':'--'},
    #                         'ylabel':r'$A \; cos(\phi) \; [pm]$'}, functions_kwargs_ext[1,0])
    #     a4_kwargs = kwargs({'title':'AmpOff: Response 2 '+temp_string,'args_xlim':(-50,50),'kwargs_grid':{'ls':'--'},
    #                         'ylabel':r'$A \; cos(\phi) \; [pm]$'}, functions_kwargs_ext[1,1])

    #     subplot_kwargs = {'figsize':(9,8), 'sharex':True}
        
    #     functions_kwargs = np.array([[a1_kwargs, a2_kwargs],[a3_kwargs, a4_kwargs]])
        
        
    #     return self.plot_SS_PFM_data(filename, line, point, x_signals, y_signals, functions_kwargs, subplot_kwargs)  
    # '''...........................................................................................................................'''    
    # def plot_SS_PFM_data(self, filename, line, point, x_signals, y_signals, functions_kwargs, subplots_kwargs, 
    #                      tight_layout=True, show=True):
    #     '''
    #     Plots the forcemap in a shape according to the spape of x_signals and y signals
    #     returns the figure and ax element 
    #     functions_kwargs and subplots_kwargs can be ndarray of dicts or kwargs objects
    #     '''
    #     #Convert the signals to np.array
    #     x_signals = np.array(x_signals)
    #     y_signals = np.array(y_signals)

    #     #Assign the proper functions for the signals
    #     functions_index = pd.DataFrame({'AmpOn_Bias': pd.Series({'AmpOn_Phase_1': graph_2D, 
    #                                                              'AmpOn_Phase_2': graph_2D, 
    #                                                              'AmpOn_Amplitude': graph_2D, 
    #                                                              'AmpOff_Phase_1': graph_2D, 
    #                                                              'AmpOff_Phase_2': graph_2D, 
    #                                                              'AmpOff_Amplitude': graph_2D,
    #                                                              'AmpOn_Phase_1_super': graph_2D, 
    #                                                              'AmpOn_Phase_2_super': graph_2D, 
    #                                                              'AmpOn_Amplitude_super': graph_2D, 
    #                                                              'AmpOff_Phase_1_super': graph_2D, 
    #                                                              'AmpOff_Phase_2_super': graph_2D, 
    #                                                              'AmpOff_Amplitude_super': graph_2D,
    #                                                              'AmpOff_Response_1': graph_2D, 
    #                                                              'AmpOff_Response_2': graph_2D,
    #                                                              'AmpOn_Response_1': graph_2D, 
    #                                                              'AmpOn_Response_2': graph_2D,
    #                                                              'AmpOff_Response_1_super': graph_2D, 
    #                                                              'AmpOff_Response_2_super': graph_2D,
    #                                                              'AmpOn_Response_1_super': graph_2D, 
    #                                                              'AmpOn_Response_2_super': graph_2D}
    #                                                             ),
    #                                     'AmpOn_Bias_super': pd.Series({'AmpOn_Phase_1': graph_2D, 
    #                                                                    'AmpOn_Phase_2': graph_2D, 
    #                                                                    'AmpOn_Amplitude': graph_2D, 
    #                                                                    'AmpOff_Phase_1': graph_2D, 
    #                                                                    'AmpOff_Phase_2': graph_2D, 
    #                                                                    'AmpOff_Amplitude': graph_2D,
    #                                                                    'AmpOn_Phase_1_super': graph_2D, 
    #                                                                    'AmpOn_Phase_2_super': graph_2D, 
    #                                                                    'AmpOn_Amplitude_super': graph_2D, 
    #                                                                    'AmpOff_Phase_1_super': graph_2D, 
    #                                                                    'AmpOff_Phase_2_super': graph_2D, 
    #                                                                    'AmpOff_Amplitude_super': graph_2D,
    #                                                                    'AmpOff_Response_1': graph_2D, 
    #                                                                    'AmpOff_Response_2': graph_2D,                                                                                                     
    #                                                                    'AmpOn_Response_1': graph_2D, 
    #                                                                    'AmpOn_Response_2': graph_2D,                                                                                                    
    #                                                                    'AmpOff_Response_1_super': graph_2D,                                                                                                      
    #                                                                    'AmpOff_Response_2_super': graph_2D,                                                                                                     
    #                                                                    'AmpOn_Response_1_super': graph_2D,                                                                                                      
    #                                                                    'AmpOn_Response_2_super': graph_2D}
    #                                                                   )
    #                                     })
    #     #Linear Factors for different Signals types (eg. pm for Amplitude)
    #     factors = {'AmpOn_Phase_1': 1, 'AmpOn_Phase_2': 1, 'AmpOn_Amplitude': 1e12, 
    #                'AmpOff_Phase_1': 1, 'AmpOff_Phase_2': 1, 'AmpOff_Amplitude': 1e12,
    #                'AmpOn_Phase_1_super': 1, 'AmpOn_Phase_2_super': 1, 'AmpOn_Amplitude_super': 1e12, 
    #                'AmpOff_Phase_1_super': 1, 'AmpOff_Phase_2_super': 1, 'AmpOff_Amplitude_super': 1e12,
    #                'AmpOff_Response_1': 1e12, 'AmpOff_Response_2': 1e12,
    #                'AmpOn_Response_1': 1e12, 'AmpOn_Response_2': 1e12, 'AmpOn_Bias': 1,
    #                'AmpOff_Response_1_super': 1e12, 'AmpOff_Response_2_super': 1e12,
    #                'AmpOn_Response_1_super': 1e12, 'AmpOn_Response_2_super': 1e12, 'AmpOn_Bias_super': 1}
        
    #     functions_index = functions_index.transpose() #transpose to have x axis in row

    #     #Get the data from the forcecurve ==> destinguish btw. single and super!
    #     if line == point == 'super':
    #         fc = self.obj.get_Forcemap_data(filename)
    #     else:
    #         fc = self.obj.get_Forcemap_row(filename, line, point)
        
    #     #Check wheter x_signals and y_signals have same shape and extract rows and columns
    #     if not x_signals.shape == y_signals.shape: 
    #         raise Exception('Error: x and y signals must have same shape')
    #     rows, columns = x_signals.shape
        
    #     #Prepare the function matrix ==> assigned as nested list and transformed into np.array
    #     lst = []
    #     for row in range(rows):
    #         sub_lst = []
    #         for column in range(columns):
    #             sub_lst.append(functions_index.loc[x_signals[row, column],y_signals[row, column]])
    #         lst.append(sub_lst)
    #     functions = np.array(lst)
        
    #     #Now we have to prepare functions_args matrix ==> basically x,y values wrapped in a list
    #     lst = []
    #     for row in range(rows):
    #         sub_lst = []
    #         for column in range(columns):
    #             x = self.extract_Forcecurve_data(fc, x_signals[row,column])*factors[x_signals[row,column]]
    #             y = self.extract_Forcecurve_data(fc, y_signals[row,column])*factors[y_signals[row,column]]
    #             sub_lst.append(args((x,y)))
    #         lst.append(sub_lst)
    #     functions_args = np.array(lst)
        
    #     return flex_subplot(functions, functions_args, functions_kwargs, subplots_kwargs= subplots_kwargs)
    # def extract_Forcecurve_data(self, fc, signal_name):
    #     """
    #     Extrac the matching data from the forcecurve according to the signal_name keyword
    #     """
    #     if signal_name == 'AmpOff_Response_1':
    #         return fc.loc['Hysteresis'].loc[:,'AmpOff_Amplitude_mean'] * np.cos(np.radians(
    #             fc.loc['Hysteresis'].loc[:,'AmpOff_Phase_1_mean']))
    #     #------------------------------------------------------------------------------------------------------------------------
    #     elif signal_name == 'AmpOff_Response_2':
    #         return fc.loc['Hysteresis'].loc[:,'AmpOff_Amplitude_mean'] * np.cos(np.radians(
    #             fc.loc['Hysteresis'].loc[:,'AmpOff_Phase_2_mean']))
    #     #------------------------------------------------------------------------------------------------------------------------
    #     elif signal_name == 'AmpOn_Response_1':
    #         return fc.loc['Hysteresis'].loc[:,'AmpOn_Amplitude_mean'] * np.cos(np.radians(
    #             fc.loc['Hysteresis'].loc[:,'AmpOn_Phase_1_mean']))
    #     #------------------------------------------------------------------------------------------------------------------------
    #     elif signal_name == 'AmpOn_Response_2':
    #         return fc.loc['Hysteresis'].loc[:,'AmpOn_Amplitude_mean'] * np.cos(np.radians(
    #             fc.loc['Hysteresis'].loc[:,'AmpOn_Phase_2_mean']))
    #     #------------------------------------------------------------------------------------------------------------------------
    #     elif signal_name == 'AmpOn_Phase_1_super':
    #         lst = []
    #         for index in fc.index.values:
    #             lst.extend(fc.loc[index,'Hysteresis'].loc[:,'AmpOn_Phase_1_mean'])
    #         return np.array(lst)
    #     #------------------------------------------------------------------------------------------------------------------------
    #     elif signal_name == 'AmpOn_Phase_2_super':
    #         lst = []
    #         for index in fc.index.values:
    #             lst.extend(fc.loc[index,'Hysteresis'].loc[:,'AmpOn_Phase_2_mean'])
    #         return np.array(lst)
    #     #------------------------------------------------------------------------------------------------------------------------
    #     elif signal_name == 'AmpOn_Amplitude_super':
    #         lst = []
    #         for index in fc.index.values:
    #             lst.extend(fc.loc[index,'Hysteresis'].loc[:,'AmpOn_Amplitude_mean'])
    #         return np.array(lst)
    #     #------------------------------------------------------------------------------------------------------------------------
    #     elif signal_name == 'AmpOff_Phase_1_super':
    #         lst = []
    #         for index in fc.index.values:
    #             lst.extend(fc.loc[index,'Hysteresis'].loc[:,'AmpOff_Phase_1_mean'])
    #         return np.array(lst)
    #     #------------------------------------------------------------------------------------------------------------------------
    #     elif signal_name == 'AmpOff_Phase_2_super':
    #         lst = []
    #         for index in fc.index.values:
    #             lst.extend(fc.loc[index,'Hysteresis'].loc[:,'AmpOff_Phase_2_mean'])
    #         return np.array(lst)
    #     #------------------------------------------------------------------------------------------------------------------------
    #     elif signal_name == 'AmpOff_Amplitude_super':
    #         lst = []
    #         for index in fc.index.values:
    #             lst.extend(fc.loc[index,'Hysteresis'].loc[:,'AmpOff_Amplitude_mean'])
    #         return np.array(lst)
    #     #------------------------------------------------------------------------------------------------------------------------
    #     elif signal_name == 'AmpOn_Response_1_super':
    #         lst = []
    #         for index in fc.index.values:
    #             lst.extend(fc.loc[index,'Hysteresis'].loc[:,'AmpOn_Amplitude_mean'] * np.cos(np.radians(
    #                 fc.loc[index,'Hysteresis'].loc[:,'AmpOn_Phase_1_mean'])))
    #         return np.array(lst)
    #     #------------------------------------------------------------------------------------------------------------------------
    #     elif signal_name == 'AmpOn_Response_2_super':
    #         lst = []
    #         for index in fc.index.values:
    #             lst.extend(fc.loc[index,'Hysteresis'].loc[:,'AmpOn_Amplitude_mean'] * np.cos(np.radians(
    #                 fc.loc[index,'Hysteresis'].loc[:,'AmpOn_Phase_2_mean'])))
    #         return np.array(lst)
    #     #------------------------------------------------------------------------------------------------------------------------
    #     elif signal_name == 'AmpOff_Response_1_super':
    #         lst = []
    #         for index in fc.index.values:
    #             lst.extend(fc.loc[index,'Hysteresis'].loc[:,'AmpOff_Amplitude_mean'] * np.cos(np.radians(
    #                 fc.loc[index,'Hysteresis'].loc[:,'AmpOff_Phase_1_mean'])))
    #         return np.array(lst)
    #     #------------------------------------------------------------------------------------------------------------------------
    #     elif signal_name == 'AmpOff_Response_2_super':
    #         lst = []
    #         for index in fc.index.values:
    #             lst.extend(fc.loc[index,'Hysteresis'].loc[:,'AmpOff_Amplitude_mean'] * np.cos(np.radians(
    #                 fc.loc[index,'Hysteresis'].loc[:,'AmpOff_Phase_2_mean'])))
    #         return np.array(lst)
    #     #------------------------------------------------------------------------------------------------------------------------
    #     elif signal_name == 'AmpOn_Bias_super':
    #         lst = []
    #         for index in fc.index.values:
    #             lst.extend(fc.loc[index,'Hysteresis'].loc[:,'AmpOn_Bias_mean'])
    #         return np.array(lst)
    #     #------------------------------------------------------------------------------------------------------------------------
    #     else: 
    #         return fc.loc['Hysteresis'].loc[:,signal_name + '_mean']
    # '''...........................................................................................................................'''