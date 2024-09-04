#+++import modules
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, Button, filedialog
from nptdms import TdmsFile
plt.rcParams['agg.path.chunksize'] = 1000 #for plotting optimization purposes
from lib import *
from scipy import interpolate
from sys import exit
from os.path import basename

#+++define user functions
def sifting_iteration_s(time, wfm, s_number, tolerance, min_iter, max_iter):
    k = 0
    error = 5000000
    s_iter = 10
    while (error > tolerance or s_iter < s_number):
        print('+++++++++++++++++++++++++++++++++++iteration ', k)
        h, extrema = sifting_spline(time, wfm)
        
        if k > min_iter:
            error = dif_extrema_xzeros(h)
            print('error = ', error)
            if error <= tolerance:
                s_iter += 1
            else:
                s_iter = 0
            print('conseq. it = ', s_iter)
        wfm = h
        k += 1
        if k > max_iter:
            break
        if s_iter > 15:
            break
    return h


def sifting_spline(t, x):
    t_up, x_up = env_up(t, x)
    t_down, x_down = env_down(t, x)
    extrema_x = len(x_up) + len(x_down)
    
    tck = interpolate.splrep(t_up, x_up)
    x_up = interpolate.splev(t, tck)
    tck = interpolate.splrep(t_down, x_down)
    x_down = interpolate.splev(t, tck)

    x_mean = (x_up + x_down)/2
    h = x - x_mean
    return h, extrema_x


def dif_extrema_zeros(x):
    n = len(x)
    n_zeros = 0
    n_extrema = 0    
    for i in range(n-2):
        if (x[i+1] < x[i] and x[i+2] > x[i+1]) or (x[i+1] > x[i] and x[i+2] < x[i+1]):
            n_extrema = n_extrema + 1            
        if (x[i] > 0 and x[i+1] < 0) or (x[i] < 0 and x[i+1] > 0):
            n_zeros = n_zeros + 1
    return np.absolute(n_extrema - n_zeros)

def dif_extrema_xzeros(x):
    n = len(x)
    n_xzeros = 0
    n_extrema = 0
    
    for i in range(n-2):
        if (x[i+1] < x[i] and x[i+2] > x[i+1]) or (x[i+1] > x[i] and x[i+2] < x[i+1]):
            n_extrema = n_extrema + 1
            
        if (x[i] > 0 and x[i+1] < 0) or (x[i] < 0 and x[i+1] > 0):
            n_xzeros = n_xzeros + 1


    return np.absolute(n_extrema - n_xzeros)


def env_down(t, x):
    n = len(x)
    x_down = []
    t_down = []
    x_down.append(x[0])
    t_down.append(t[0])
    for i in range(n-2):
        if (x[i+1] < x[i] and x[i+2] > x[i+1]):
            x_down.append(x[i+1])
            t_down.append(t[i+1])
    x_down.append(x[n-1])
    t_down.append(t[n-1])
    x_down = np.array(x_down)
    t_down = np.array(t_down)

    return t_down, x_down


def env_up(t, x):
    n = len(x)
    x_up = []
    t_up = []
    x_up.append(x[0])
    t_up.append(t[0])
    for i in range(n-2):
        if (x[i+1] > x[i] and x[i+2] < x[i+1]):
            x_up.append(x[i+1])
            t_up.append(t[i+1])
    x_up.append(x[n-1])
    t_up.append(t[n-1])
    x_up = np.array(x_up)
    t_up = np.array(t_up)
    
    return t_up, x_up


#+++define parameters
fs = 25.6e3 #sampling frequency
channel = 'AC1' #channel name
n_imf = 2 #number of imf to calculate
s_number = 2 #number of iterations in which zero crossings <= tolerance
tolerance = 1 #tolerance for zero crossings
min_iter = 0 #min of iterations
max_iter = 100 #max of iterations


#+++load data
print('select signal file: ')
root = Tk()
root.withdraw()
root.update()
filepath = filedialog.askopenfilename()
root.destroy()
filename = basename(filepath)

wfm = f_open_tdms(filepath, channel)
time = np.arange(len(wfm))/fs



# #+++data processing
for count in range(n_imf):        
    count += 1                
    print('To calculate: h' + str(count))
    name_out = 'imf' + str(count) + '_'            
    
    h = sifting_iteration_s(time, wfm, s_number, tolerance, min_iter, max_iter)

    print('Saving...')
    myname = name_out + filename[:-4] + 'pkl'
    save_pickle(myname, h)

    wfm = wfm - h
