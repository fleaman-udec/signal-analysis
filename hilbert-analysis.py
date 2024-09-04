#+++import modules
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, Button, filedialog
from decimal import Decimal
from scipy.signal import hilbert
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm, Normalize
from os import chdir
from os.path import basename
from lib import *


plt.rcParams['agg.path.chunksize'] = 1000 #for plotting optimization purposes
plt.rcParams['savefig.dpi'] = 1000
plt.rcParams['savefig.format'] = 'jpeg'


#+++define user functions
def hilbert_spectrum(time, frec, amp, colormap='inferno', step_cbar=10, linewidth=3):
	x = time
	y = frec
	z = amp

	n = len(x)
	cmap = plt.get_cmap(colormap)

	# norm = BoundaryNorm(np.linspace(z.min(), z.max(), 1000), cmap.N)
	points = np.array([x, y]).T.reshape(-1, 1, 2)
	segments = np.concatenate([points[:-1], points[1:]], axis=1)
	norm = Normalize(vmin=0, vmax=1300)
	# norm = None
	
	lc = LineCollection(segments, cmap=cmap, norm=None)
	lc.set_array(z)
	lc.set_linewidth(linewidth)
	
	fig1, ax1 = plt.subplots()
	plt.gca().add_collection(lc)
	plt.xlim(x.min(), x.max())
	plt.ylim(y.min(), y.max())
	
	line = ax1.add_collection(lc)
	cbar = fig1.colorbar(line, ax=ax1, format='%1.0f')
	
	cbar.set_label('zlabel', size=13)
	cbar.ax.tick_params(labelsize=12)	
	cbar.set_ticks(list(np.arange(np.min(z), np.max(z), step_cbar)))	
	ax1.set_xlabel('xlabel', fontsize=13)
	ax1.set_ylabel('ylabel', fontsize=13)
	# ax1.set_ylim(config['ylim'][0], config['ylim'][1])	
	ax1.tick_params(axis='both', labelsize=12)

	return fig1, ax1

def line_hilbert_spectrum(time, frec, amp, config):
	x = time
	y = frec
	z = amp

	n = len(x)
	cmap = plt.get_cmap(config['color_map'])

	# norm = BoundaryNorm(np.linspace(z.min(), z.max(), 1000), cmap.N)
	points = np.array([x, y]).T.reshape(-1, 1, 2)
	segments = np.concatenate([points[:-1], points[1:]], axis=1)
	# norm = Normalize(vmin=config['cbarlim'][0], vmax=config['cbarlim'][1])
	norm = None
	
	lc = LineCollection(segments, cmap=cmap, norm=norm)
	lc.set_array(z)
	lc.set_linewidth(config['linewidth'])


	return lc


def diff_signal(x, length_diff):
	#Differentiation/Derivative
	n = len(x)
	dx = np.zeros(n-length_diff)
	n_diff = len(dx)
	for i in range(n_diff):
		dx[i] = x[i+length_diff] - x[i]

	return dx


#+++define parameters
fs = 25.e3 #sampling frequency
channel = 'AC1' #channel name
hp_freq = None #highpass cutoff frequency



#+++load data
print('select signal file: ')
root = Tk()
root.withdraw()
root.update()
filepath = filedialog.askopenfilename()
root.destroy()
filename = basename(filepath)

wfm = read_pickle(filepath)
time = np.arange(len(wfm))/fs

plt.plot(time, wfm)
plt.show()


#+++data processing
wfm = hilbert(wfm)
amplitude = np.absolute(wfm)
frequency = diff_signal(x=np.unwrap(np.angle(wfm)), length_diff=1)
frequency = frequency * fs
frequency = frequency / (2*np.pi)
frequency = np.absolute(frequency)
frequency = np.array(list(frequency) + [frequency[len(frequency)-1]])

fig, ax = hilbert_spectrum(time, frequency, amplitude)
plt.show()


	# elif config['mode'] == 'hilbert_marginal':
		# root = Tk()
		# root.withdraw()
		# root.update()
		# filepath = filedialog.askopenfilename()
		# filename = os.path.basename(filepath)
		# root.destroy()
		# signal = load_signal(filepath, channel=config['channel'])
		# print(filename)
		
		# n_points = len(signal)
		# dt = 1./config['fs']
		# time = np.array([i*dt for i in range(n_points)])
		# df = 1. / (len(signal)*dt)
		# marg_frequency = np.array([i*df for i in range(int(n_points/2))])
		# marg_amplitude = np.zeros(int(n_points/2))
		
		# if config['filter'][0] != 'OFF':
			# signal = butter_filter(signal, config['fs'], config['filter'])
		
		# signal = hilbert(signal)
		# amplitude = np.absolute(signal)
		# frequency = diff_signal(x=np.unwrap(np.angle(signal)), length_diff=1)
		# frequency = frequency / dt
		# frequency = frequency / (2*np.pi)
		# frequency = np.absolute(frequency)
		# frequency = np.array(list(frequency) + [frequency[len(frequency)-1]])
		
		# if config['filter2'][0] != 'OFF':
			# frequency = butter_filter(frequency, config['fs'], config['filter2'])
		
		
		# frequency = frequency[2000:]
		# time = time[2000:]
		# amplitude = amplitude[2000:]
		
		# for i in range(n_points-2000):
			# sum_index = int(round(frequency[i]/df))
			# marg_amplitude[sum_index] += amplitude[i]
		
		# fig, ax = plt.subplots()
		# if config['logscale'] == 'ON':
			# marg_amplitude = np.log(1+marg_amplitude)
		# if config['kHz'] == 'ON':
			# marg_frequency = marg_frequency/1000.

		
		# ax.plot(marg_frequency, marg_amplitude)
			
		# ax.set_xlabel(config['xlabel'], fontsize=13)
		# ax.set_ylabel(config['ylabel'], fontsize=13)
		# ax.tick_params(axis='both', labelsize=12)
		# ax.set_title(config['title'], fontsize=13)
		# ax.set_ylim(config['ylim'][0], config['ylim'][1])
		# ax.set_xlim(config['xlim'][0], config['xlim'][1])	
		# plt.show()
	
	# elif config['mode'] == 'hilbert_spectrum_2':
		# root = Tk()
		# root.withdraw()
		# root.update()
		# filepath = filedialog.askopenfilename()
		# # filename = os.path.basename(filepath)
		# root.destroy()
		
		# # print(filename)
		
		# # Frequency = []
		# # Time = []
		# # Amplitude = []
		

		# signal = load_signal(filepath, channel=config['channel'])		
		# n_points = len(signal)
		# dt = 1./config['fs']
		# time = np.array([i*dt for i in range(n_points)])		
		
		# if config['filter'][0] != 'OFF':
			# signal = butter_filter(signal, config['fs'], config['filter'])
		
		# signal = hilbert(signal)
		# amplitude = np.absolute(signal)
		# frequency = diff_signal(x=np.unwrap(np.angle(signal)), length_diff=1)
		# frequency = frequency / dt
		# frequency = frequency / (2*np.pi)
		# frequency = np.absolute(frequency)
		# frequency = np.array(list(frequency) + [frequency[len(frequency)-1]])
		
		# if config['filter2'][0] != 'OFF':
			# frequency = butter_filter(frequency, config['fs'], config['filter2'])
		
		# if config['kHz'] == 'ON':
			# frequency = frequency /1000.
		
		# # Frequency.append(frequency)
		# # Time.append(time)
		# # Amplitude.append(Amplitude)
	

		# points = np.array([time, frequency]).T.reshape(-1, 1, 2)
		# segments = np.concatenate([points[:-1], points[1:]], axis=1)
		
		# cmap = plt.get_cmap(config['color_map'])
		# # norm = BoundaryNorm(np.linspace(z.min(), z.max(), 1000), cmap.N)
		# # norm = Normalize(vmin=0, vmax=1300)
		# norm = Normalize(vmin=config['cbarlim'][0], vmax=config['cbarlim'][1])
		# # norm = None
		
		# lc = LineCollection(segments, cmap=cmap, norm=norm)
		# lc.set_array(amplitude)
		# lc.set_linewidth(config['linewidth'])
	
		# fig, ax = plt.subplots()
	
		# plt.gca().add_collection(lc)
		# plt.xlim(time.min(), time.max())
		# plt.ylim(frequency.min(), frequency.max())
		
		# line = ax.add_collection(lc)
		# cbar = fig.colorbar(line, ax=ax, format='%1.0f')
	
		# cbar.ax.tick_params(labelsize=12)	
		# cbar.set_ticks(list(np.arange(config['cbarlim'][0], config['cbarlim'][1], config['step_cbar'])))	
		
		# cbar.set_label(config['zlabel'], size=13)		
		# ax.set_xlabel(config['xlabel'], fontsize=13)
		# ax.set_ylabel(config['ylabel'], fontsize=13)
		# ax.set_ylim(config['ylim'][0], config['ylim'][1])	
		# ax.tick_params(axis='both', labelsize=12)
		# ax.set_title(config['title'], fontsize=13)
		# plt.show()
	
	
	# elif config['mode'] == 'asynchro_obtain':
		# print('a')
	
	# else:
		# print('Mode Unknown 98359')
		
		
		
	# return

# def hilbert_spectrum(time, frec, amp, config):
	# x = time
	# y = frec
	# z = amp

	# n = len(x)
	# cmap = plt.get_cmap(config['color_map'])

	# # norm = BoundaryNorm(np.linspace(z.min(), z.max(), 1000), cmap.N)
	# points = np.array([x, y]).T.reshape(-1, 1, 2)
	# segments = np.concatenate([points[:-1], points[1:]], axis=1)
	# norm = Normalize(vmin=0, vmax=1300)
	# # norm = None
	
	# lc = LineCollection(segments, cmap=cmap, norm=norm)
	# lc.set_array(z)
	# lc.set_linewidth(config['linewidth'])
	
	# fig1, ax1 = plt.subplots()
	# plt.gca().add_collection(lc)
	# plt.xlim(x.min(), x.max())
	# plt.ylim(y.min(), y.max())
	
	# line = ax1.add_collection(lc)
	# cbar = fig1.colorbar(line, ax=ax1, format='%1.0f')
	
	# cbar.set_label(config['zlabel'], size=13)
	# cbar.ax.tick_params(labelsize=12)	
	# cbar.set_ticks(list(np.arange(np.min(z), np.max(z), config['step_cbar'])))	
	# ax1.set_xlabel(config['xlabel'], fontsize=13)
	# ax1.set_ylabel(config['ylabel'], fontsize=13)
	# # ax1.set_ylim(config['ylim'][0], config['ylim'][1])	
	# ax1.tick_params(axis='both', labelsize=12)

	# return fig1, ax1

# def line_hilbert_spectrum(time, frec, amp, config):
	# x = time
	# y = frec
	# z = amp

	# n = len(x)
	# cmap = plt.get_cmap(config['color_map'])

	# # norm = BoundaryNorm(np.linspace(z.min(), z.max(), 1000), cmap.N)
	# points = np.array([x, y]).T.reshape(-1, 1, 2)
	# segments = np.concatenate([points[:-1], points[1:]], axis=1)
	# # norm = Normalize(vmin=config['cbarlim'][0], vmax=config['cbarlim'][1])
	# norm = None
	
	# lc = LineCollection(segments, cmap=cmap, norm=norm)
	# lc.set_array(z)
	# lc.set_linewidth(config['linewidth'])
	
	
	
	

	# return lc


# # plt.show()
# def read_parser(argv, Inputs, InputsOpt_Defaults):
	# Inputs_opt = [key for key in InputsOpt_Defaults]
	# Defaults = [InputsOpt_Defaults[key] for key in InputsOpt_Defaults]
	# parser = ArgumentParser()
	# for element in (Inputs + Inputs_opt):
		# print(element)
		# if element == 'filter':
			# parser.add_argument('--' + element, nargs='+')
		# else:
			# parser.add_argument('--' + element, nargs='?')
	
	# args = parser.parse_args()
	# config = {}
	# for element in Inputs:
		# if getattr(args, element) != None:
			# config[element] = getattr(args, element)
		# else:
			# print('Required:', element)
			# sys.exit()

	# for element, value in zip(Inputs_opt, Defaults):
		# if getattr(args, element) != None:
			# config[element] = getattr(args, element)
		# else:
			# print('Default ' + element + ' = ', value)
			# config[element] = value
	
	# #Type conversion to float
	# # if config['power2'] != 'auto' and config['power2'] != 'OFF':
		# # config['power2'] = int(config['power2'])
	# config['fs'] = float(config['fs'])
	# config['linewidth'] = float(config['linewidth'])
	# config['step_cbar'] = float(config['step_cbar'])
	
	# # config['fscore_min'] = float(config['fscore_min'])
	# #Type conversion to int	
	# # Variable conversion
	# print(config['filter'][0])
	# if config['filter'][0] != 'OFF':
		# if config['filter'][0] == 'bandpass':
			# config['filter'] = [config['filter'][0], [float(config['filter'][1]), float(config['filter'][2])], float(config['filter'][3])]
		# elif config['filter'][0] == 'highpass':
			# config['filter'] = [config['filter'][0], float(config['filter'][1]), float(config['filter'][2])]
		# elif config['filter'][0] == 'lowpass':
			# config['filter'] = [config['filter'][0], float(config['filter'][1]), float(config['filter'][2])]
		# else:
			# print('error filter 87965')
			# sys.exit()
	
	# if config['filter2'][0] != 'OFF':
		# if config['filter2'][0] == 'bandpass':
			# config['filter2'] = [config['filter2'][0], [float(config['filter2'][1]), float(config['filter2'][2])], float(config['filter2'][3])]
		# elif config['filter2'][0] == 'highpass':
			# config['filter2'] = [config['filter2'][0], float(config['filter2'][1]), float(config['filter2'][2])]
		# elif config['filter2'][0] == 'lowpass':
			# config['filter2'] = [config['filter2'][0], float(config['filter2'][1]), float(config['filter2'][2])]
		# else:
			# print('error filter2 87465')
			# sys.exit()
	
	
	# return config


# if __name__ == '__main__':
	# main(sys.argv)
