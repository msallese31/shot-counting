import numpy as np
from sklearn import preprocessing
from scipy import stats
import pandas as pd
import sys
import matplotlib.pyplot as plt
import glob
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning) 

# pattern  = pd.read_csv('5shots.txt')
# patt_start = 297
# patt_end = 331

cond_show=False
plot_all=False

# X pattern
pattern_file = pd.read_csv('5shots2.txt')
patt_start = 465
patt_end = 511

# Z pattern
# pattern_file = pd.read_csv('5shots4.txt')
# patt_start = 494
# patt_end = 634

correlation_threshold = 50
marker_distance_threshold = 200
counted_shots = 0
expected_shots_count = 32

def main():
	global counted_shots
	if False:
		experiment()
	else:
		# X pattern with moving average
		pattern_file = pd.read_csv('5shots2.txt')
		patt_start = 120
		patt_end = 215
		counted_shots = 0
		predict_axis_ma('x')
		print("Axis X_ma; Counted %d shots; Expected: %d" % (counted_shots, expected_shots_count))

		# Y pattern with moving average
		pattern_file = pd.read_csv('5shots2.txt')
		# patt_start = 1993
		# patt_end = 2050
		counted_shots = 0
		predict_axis_ma('y')
		y_counted_shots = counted_shots
		print("Axis: Y; Counted %d shots; Expected: %d" % (counted_shots, expected_shots_count))

		# Z pattern with moving average
		pattern_file = pd.read_csv('5shots2.txt')
		# patt_start = 494
		# patt_end = 634
		counted_shots = 0
		predict_axis_ma('z')
		z_counted_shots = counted_shots
		print("Axis: Z; Counted %d shots; Expected: %d" % (counted_shots, expected_shots_count))


		# pattern_file = pd.read_csv('5shots2.txt')
		# patt_start = 465
		# patt_end = 511
		counted_shots = 0
		predict_axis('x')
		x_counted_shots = counted_shots
		print("Axis X; Counted %d shots; Expected: %d" % (counted_shots, expected_shots_count))

		# Y pattern
		pattern_file = pd.read_csv('5shots4.txt')
		patt_start = 1993
		patt_end = 2050
		counted_shots = 0
		predict_axis('y')
		y_counted_shots = counted_shots
		print("Axis: Y; Counted %d shots; Expected: %d" % (counted_shots, expected_shots_count))

		# Z pattern
		pattern_file = pd.read_csv('5shots4.txt')
		patt_start = 494
		patt_end = 634
		counted_shots = 0
		predict_axis('z')
		z_counted_shots = counted_shots
		print("Axis: Z; Counted %d shots; Expected: %d" % (counted_shots, expected_shots_count))




def predict_axis(axis):
	global pattern_file
	global counted_shots

	pattern_data = get_data(pattern_file, axis)
	pattern_data = pattern_data[patt_start:patt_end]

	for filename in glob.iglob('*.txt'):
		csv_data = pd.read_csv(filename)
		axis_data = get_data(csv_data, axis)
		name = filename.split('.')[0]
		counted_shots = counted_shots + count_shots(axis_data, pattern_data, name)

def predict_axis_ma(axis):
	global pattern_file
	global counted_shots

	pattern_data = get_data(pattern_file, axis)
	pattern_data = moving_average(pattern_data, 15)
	pattern_data = pattern_data[patt_start:patt_end]
	pattern_data = pd.DataFrame(np.array(pattern_data).reshape(-1,1))


	for filename in glob.iglob('*.txt'):
		csv_data = pd.read_csv(filename)
		axis_data = get_data(csv_data, axis)
		axis_data = moving_average(axis_data, 15)
		axis_data = pd.DataFrame(np.array(axis_data).reshape(-1,1))
		name = filename.split('.')[0]
		counted_shots = counted_shots + count_shots(axis_data, pattern_data, name)


def experiment():



	filename = '5shots2.txt'
	data = pd.read_csv(filename)
	x_data = get_data(data, 'x')
	x_data = x_data[0:1000]

	# plt.plot(x_data)
	# plt.gcf().canvas.set_window_title(filename + " raw data 3 shots")
	# plt.show()	

	# standardized_data = preprocessing.scale(x_data)
	# plt.plot(standardized_data)
	# plt.gcf().canvas.set_window_title(filename + " standardized data 3 shots")
	# plt.show()	

	# # Normalize
	# reshaped_x_data = x_data.reshape(-1, 1)
	# normalize_data = preprocessing.normalize(reshaped_x_data)
	# plt.plot(normalize_data)
	# plt.gcf().canvas.set_window_title(filename + " new normalized 3 shots")
	# plt.show()	

	

	# plt.plot(x_data)
	# plt.gcf().canvas.set_window_title(filename + " raw data 3 shots")
	# plt.show()


	moving_x_data = moving_average(x_data, 15)
	plt.plot(moving_x_data)
	plt.gcf().canvas.set_window_title(filename + " moving average 3 shots")
	plt.show()


def conditional_show():
	if cond_show:
		plt.show()

def moving_average(df_to_average, N):
	cumsum, moving_aves = [0], []

	for i, x in enumerate(df_to_average, 1):
		cumsum.append(cumsum[i-1] + x)
		if i>=N:
			moving_ave = (cumsum[i] - cumsum[i-N])/N
			#can do stuff with moving_ave here
			moving_aves.append(moving_ave)
	return moving_aves


def count_shots(data, pattern, filename):
	data_df = pd.DataFrame(data)
	pattern_df = pd.DataFrame(pattern)
	# pattern_mean = np.mean(pattern[patt_start:patt_end])

	data_outliers = remove_outliers(data_df)
	data_normalized = normalize_df(data_df)

	pattern_outliers = remove_outliers(pattern_df)
	pattern_normalized = normalize_df(pattern_df)

	if plot_all:
		plot_all_data(data, pattern, data_normalized, pattern_normalized, data_outliers, pattern_outliers, filename)


	auto_patt = flatten_df(pattern)
	auto_data = data.values.flatten()

	autocorr = np.correlate(auto_data, auto_patt)

	plt.plot(data)
	plt.gcf().canvas.set_window_title(filename + " raw data")
	conditional_show()
	plt.clf()	

	plt.plot(autocorr)
	plt.gcf().canvas.set_window_title(filename + " autocorrelation")
	# conditional_show()
	plt.clf()

	autocorr = autocorr.reshape(-1, 1)
	norm_autocorr = normalize_df(autocorr)
	plt.plot(norm_autocorr)
	plt.gcf().canvas.set_window_title(filename + " normalized autocorrelation")
	# conditional_show()
	plt.clf()

	norm_autocorr = flatten_df(norm_autocorr)

	markers = []
	# Changed to normalized autocorrelation
	for index, value in enumerate(norm_autocorr):
		value = value * 100
		# print("value: %d vs thresh: %d" % (value, correlation_threshold))
		if value >= correlation_threshold:
			markers.append(index)

	# Cleanse markers of values that are too close
	# print("before cleanse")
	# print(markers)

	index = 0
	while (index + 1) < len(markers):
		# print("Iteration #%d" % index)
		# print("Len markers: %d" % len(markers))
		# print("comparing index %d to index %d" % (markers[index], markers[index + 1]))
		if (abs(markers[index] - markers[index + 1]) < marker_distance_threshold):
			next_marker = markers[index + 1]
			current_marker = markers[index]
			# print("Comparing index %d value %d to index %d value %d because indexes are too close" % (current_marker, autocorr[current_marker], next_marker, autocorr[next_marker]))
			if (norm_autocorr[current_marker] < norm_autocorr[next_marker]):
				# print("Removing %d from list because it was too close to %d" % (current_marker, (markers[index + 1])))	
				markers.remove(current_marker)
				# print("printing new list")
				# print(markers)
				index = 0				
			else:
				# print("Removing %d from list because it was too close to %d" % ((markers[index + 1]), current_marker))			
				markers.remove(markers[index + 1])
				# print("printing new list")
				# print(markers)
				index = 0
		else:
			# print("Incrementing index")
			index += 1


	# print("after cleanse")
	# print(markers)

	
	plt.plot(norm_autocorr, marker='p', markevery=markers)
	plt.gcf().canvas.set_window_title(filename + " shots")
	conditional_show()
	plt.clf()
	print("Predicting %d shots for %s" % (len(markers), filename))
	return len(markers)

def get_data(data, axis):
	ret_data = data[axis]	
	return ret_data

def normalize_df(df_to_normalize):
	min_max_scaler = preprocessing.MinMaxScaler()
	np_scaled = min_max_scaler.fit_transform(df_to_normalize)
	df_normalized = pd.DataFrame(np_scaled)
	return df_normalized

def new_normalize_df(df_to_normalize):
	return preprocessing.normalize(df_to_normalize)

def remove_outliers(df_to_remove_outliers):
	df_outliers = df_to_remove_outliers[(np.abs(stats.zscore(df_to_remove_outliers)) < 3).all(axis=1)]
	return df_outliers

def flatten_df(df):
	return df.values.flatten()

def plot_all_data(data, pattern, data_normalized, pattern_normalized, data_outliers, pattern_outliers, filename):
	plt.plot(data)
	plt.gcf().canvas.set_window_title(filename + " raw data")
	# plt.show()
	# plt.savefig("raw data_%s" % filename, ext="png", close=False, verbose=True)
	plt.clf()

	norm_outliers = normalize_df(data_outliers)
	plt.plot(norm_outliers)
	plt.gcf().canvas.set_window_title(filename + " normalized outliers")
	# plt.show()
	# plt.savefig("x_norm_out_%s" % filename, ext="png", close=False, verbose=True)
	plt.clf()	


	plt.plot(pattern)
	# plt.savefig("pattern_%s" % filename, ext="png", close=False, verbose=True)
	plt.clf()	

	plt.plot(data_normalized)
	plt.gcf().canvas.set_window_title(filename + " normalized")
	# plt.show()
	# plt.savefig("x_norm_%s" % filename, ext="png", close=False, verbose=True)
	plt.clf()

	plt.plot(pattern_normalized)
	# plt.savefig("xp_norm_%s" % filename, ext="png", close=False, verbose=True)
	plt.clf()

	plt.plot(data_outliers)
	# plt.savefig("x_out_%s" % filename, ext="png", close=False, verbose=True)
	plt.clf()

	plt.plot(pattern_outliers)
	# plt.savefig("xp_out_%s" % filename, ext="png", close=False, verbose=True)
	plt.clf()

def smooth(x,window_len=11,window='hanning'):
	"""smooth the data using a window with requested size.
	
	This method is based on the convolution of a scaled window with the signal.
	The signal is prepared by introducing reflected copies of the signal 
	(with the window size) in both ends so that transient parts are minimized
	in the begining and end part of the output signal.
	
	input:
		x: the input signal 
		window_len: the dimension of the smoothing window; should be an odd integer
		window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
			flat window will produce a moving average smoothing.

	output:
		the smoothed signal
		
	example:

	t=linspace(-2,2,0.1)
	x=sin(t)+randn(len(t))*0.1
	y=smooth(x)
	
	see also: 
	
	numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
	scipy.signal.lfilter
 
	TODO: the window parameter could be the window itself if an array instead of a string
	NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
	"""

	if x.ndim != 1:
		raise ValueError, "smooth only accepts 1 dimension arrays."

	if x.size < window_len:
		raise ValueError, "Input vector needs to be bigger than window size."


	if window_len<3:
		return x


	if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
		raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


	s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
	#print(len(s))
	if window == 'flat': #moving average
		w=np.ones(window_len,'d')
	else:
		w=eval('np.'+window+'(window_len)')

	y=np.convolve(w/w.sum(),s,mode='valid')
	return y

if __name__ == '__main__':
		main()
