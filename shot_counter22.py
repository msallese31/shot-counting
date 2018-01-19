import numpy as np
from sklearn import preprocessing
from scipy import stats
import pandas as pd
import sys
import matplotlib.pyplot as plt
import glob
import warnings
import math
from skimage.filter import denoise_tv_chambolle
from scipy import signal
from scipy.signal import savgol_filter
from scipy.signal import butter, lfilter, freqz
from sklearn import linear_model, datasets
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier





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

correlation_threshold = 60
marker_distance_threshold = 70
counted_shots = 0
expected_shots_count = 32

def main():
	global counted_shots
	if True:
		log_reg()
		# experiment()
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

def log_reg():
	# numpy_play()
	label_data()

def numpy_play():
	iris = datasets.load_iris()
	X = iris.data[:, :2]  # we only take the first two features.
	Y = iris.target
	print(X.shape)
	print(Y.shape)
	print(Y)
	a = [1,2,3]
	b = [3,4,5]
	c = [6,7,8]
	e = 1
	a = np.array(a)
	b = np.array(b)
	c = np.array(c)

	d = np.vstack((a, b))
	d = np.vstack((d, e))
	print(d)

def label_data():
	pattern_file = '3shots2.txt'
	mag = mag_from_file(pattern_file)
	plt.plot(mag, 'r')
	# plt.show()
	plt.clf()
	p1 = mag[101:201]
	p2 = mag[450:550]
	p3 = mag[860:960]
	patterns = np.vstack((p1, p2, p3))
	# print(patterns.shape)

	np1 = mag[0:100]
	np2 = mag[210:310]
	np3 = mag[643:743]
	np4 = mag[1000:1100]
	np5 = mag[1140:1240]
	non_patterns = np.vstack((np1, np2, np3, np4, np5))
	# print(non_patterns.shape)

	pattern_file1 = '3shots3.txt'
	mag1 = mag_from_file(pattern_file1)
	plt.plot(mag1, 'r')
	# plt.show()
	plt.clf()
	p4 = mag1[80:180]
	p5 = mag1[300:400]
	p6 = mag1[590:690]

	patterns = np.vstack((patterns, p4, p5, p6))
	# print(patterns.shape)
	
	np6 = mag1[190:290]
	np7 = mag1[420:520]
	np8 = mag1[700:800]
	np9 = mag1[800:900]

	non_patterns = np.vstack((non_patterns, np6, np7, np8, np9))
	# print(non_patterns.shape)

	pattern_file2 = '3shots4.txt'

	mag2 = mag_from_file(pattern_file2)
	plt.plot(mag2, 'r')
	# plt.show()
	plt.clf()
	p7 = mag2[80:180]
	p8 = mag2[530:630]
	p9 = mag2[930:1030]

	patterns = np.vstack((patterns, p7, p8, p9))
	# print(patterns.shape)
	
	np10 = mag2[200:300]
	np11 = mag2[301:401]
	np12 = mag2[402:502]
	np13 = mag2[700:800]
	np14 = mag2[801:901]
	np15 = mag2[1100:1200]

	non_patterns = np.vstack((non_patterns, np10, np11, np12, np13, np14, np15))
	# print(non_patterns.shape)

	pattern_file3 = '3shots.txt'
	mag3 = mag_from_file(pattern_file3)

	plt.plot(mag3, 'r')
	# plt.show()
	plt.clf()

	p10 = mag3[120:220]
	p11 = mag3[435:535]
	p12 = mag3[870:970]

	patterns = np.vstack((patterns, p10, p11, p12))
	# print(patterns.shape)

	np16 = mag3[1:101]
	np17 = mag3[200:300]
	np18 = mag3[301:401]
	np19 = mag3[750:850]

	non_patterns = np.vstack((non_patterns, np16, np17, np18, np19))
	# print(non_patterns.shape)

	pattern_file4 = '5shots2.txt'
	mag4 = mag_from_file(pattern_file4)

	plt.plot(mag4, 'r')
	# plt.show()
	plt.clf()

	p13 = mag4[115:215]
	p14 = mag4[430:530]
	p15 = mag4[815:915]
	p16 = mag4[1330:1430]
	p17 = mag4[1530:1630]

	patterns = np.vstack((patterns, p13, p14, p15, p16, p17))
	# print(patterns.shape)

	np20 = mag4[245:345]
	np21 = mag4[580:680]
	np22 = mag4[940:1040]
	np23 = mag4[1041:1141]
	np24 = mag4[1152:1252]
	np25 = mag4[1700:1800]

	non_patterns = np.vstack((non_patterns, np20, np21, np22, np23, np24, np25))
	# print(non_patterns.shape)

	pattern_file5 = '5shots3.txt'
	mag5 = mag_from_file(pattern_file5)
	mag5[mag5 > 90] = 90

	plt.plot(mag5, 'r')
	# plt.show()
	plt.clf()

	p18 = mag5[60:160]
	p19 = mag5[594:694]
	p20 = mag5[1070:1170]
	p21 = mag5[1312:1412]
	p22 = mag5[1580:1680]

	patterns = np.vstack((patterns, p18, p19, p20, p21, p22))
	# test_patterns = np.vstack((p18, p19, p20, p21, p22))
	# print(patterns.shape)

	np26 = mag5[200:300]
	np27 = mag5[301:401]
	np28 = mag5[730:830]
	np29 = mag5[900:1000]
	np30 = mag5[1740:1840]
	np31 = mag5[1900:2000]

	non_patterns = np.vstack((non_patterns, np26, np27, np28, np29, np30, np31))
	# print(non_patterns.shape)

	pattern_file6 = '5shots4.txt'
	mag6 = mag_from_file(pattern_file6)
	# mag6[mag6 > 90] = 90

	plt.plot(mag6, 'r')
	# plt.show()
	plt.clf()

	p23 = mag6[66:166]
	p24 = mag6[370:470]
	p25 = mag6[540:640]
	p26 = mag6[2015:2115]
	p27 = mag6[2195:2295]

	patterns = np.vstack((patterns, p23, p24, p25, p26, p27))
	# print(patterns.shape)

	np32 = mag6[734:834]
	np33 = mag6[900:1000]
	np34 = mag6[1001:1101]
	np35 = mag6[1102:1202]
	np36 = mag6[1203:1303]
	np37 = mag6[1304:1404]
	np38 = mag6[1405:1505]
	np39 = mag6[1506:1606]
	np40 = mag6[1607:1707]
	np41 = mag6[1708:1808]
	np42 = mag6[1809:1909]

	non_patterns = np.vstack((non_patterns, np32, np33, np34, np35, np36, np37, np38, np39, np40, np41, np42))
	# test_non_patterns = np.vstack((np32, np33, np34, np35, np36, np37, np38, np39, np40, np41, np42))
	# print(non_patterns.shape)

	pattern_file7 = '5shots.txt'
	mag7 = mag_from_file(pattern_file7)
	mag7[mag7 > 90] = 90

	pattern_file = pd.read_csv('5shots.txt')
	xs = get_data(pattern_file, 'x')
	ys = get_data(pattern_file, 'y')
	zs = get_data(pattern_file, 'z')

	plt.plot(xs, 'b')
	plt.plot(zs, 'g')
	plt.plot(ys, 'y')

	plt.plot(mag7, 'r')
	# plt.show()
	plt.clf()

	p28 = mag7[202:302]
	p29 = mag7[585:685]
	p30 = mag7[803:903]
	p31 = mag7[1015:1115]
	p32 = mag7[1425:1525]

	# Held out for test data!
	# patterns = np.vstack((patterns, p28, p29, p30, p31, p32))
	test_patterns = np.vstack((p28, p29, p30, p31, p32))
	print("test patterns shape")
	print(test_patterns.shape)
	
	np43 = mag7[1:101]
	np44 = mag7[355:455]
	np45 = mag7[1151:1251]
	np46 = mag7[1252:1352]
	np47 = mag7[1550:1650]

	# Held out for test data!
	# non_patterns = np.vstack((non_patterns, np43, np44, np45, np46, np47))
	test_non_patterns = np.vstack((np43, np44, np45, np46, np47))
	print("non test patterns shape")
	print(test_non_patterns.shape)

	X = np.vstack((patterns,non_patterns))
	print(X.shape)

	Y = np.zeros(shape=(69,))
	# Y = np.zeros(shape=(79,))
	print(Y.shape)
	Y[0:26] = 1
	print(Y)

	LogReg = LogisticRegression()
	LogReg.fit(X, Y)

	supp_vec = svm.SVC(gamma=0.001, C=1.)
	supp_vec.fit(X, Y)

	gnb = GaussianNB()
	gnb.fit(X, Y)

	rand_for = RandomForestClassifier()
	rand_for.fit(X, Y)


	print("##### SUPPORT VECTOR MACHINE #####")
	supp_vec_negative = supp_vec.predict(test_non_patterns)
	print("Negatives:")
	print(supp_vec_negative)
	supp_vec_positive = supp_vec.predict(test_patterns)  
	print("Positives:")
	print(supp_vec_positive)

	print("##### GAUSSIAN NAIVE BAYES #####")
	y_pred_negative = gnb.predict(test_non_patterns)
	print("Negatives:")
	print(y_pred_negative)
	y_pred_positive = gnb.predict(test_patterns)
	print("Positives:")
	print(y_pred_positive)

	print("##### RANDOM FOREST #####")
	y_pred_negative = rand_for.predict(test_non_patterns)
	print("Negatives:")
	print(y_pred_negative)
	y_pred_positive = rand_for.predict(test_patterns)
	print("Positives:")
	print(y_pred_positive)


	print("##### LOGISTIC REGRESSION #####")
	y_pred_negative = LogReg.predict(test_non_patterns)
	print("Negatives:")
	print(y_pred_negative)
	y_pred_positive = LogReg.predict(test_patterns)
	print("Positives:")
	print(y_pred_positive)





	

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

def predict_from_mag():
	global counted_shots
	pattern_file = '5shots2.txt'
	patt_start = 379
	patt_end = 516
	counted_shots = 0
	
	mag_pattern = mag_from_file(pattern_file)
	moving_mag_pattern = moving_average(mag_pattern, 50)
	pattern = moving_mag_pattern[patt_start:patt_end]
	pattern = pd.DataFrame(np.array(pattern).reshape(-1,1))

	for filename in glob.iglob('*.txt'):
		mag = mag_from_file(filename)
		moving_mag_data = moving_average(mag, 50)
		moving_mag_data = pd.DataFrame(np.array(moving_mag_data).reshape(-1,1))
		name = filename.split('.')[0]
		counted_shots = counted_shots + count_shots(moving_mag_data, pattern, name)

	print("Axis magma; Counted %d shots; Expected: %d" % (counted_shots, expected_shots_count))

def mag_from_file(filename):
	data = pd.read_csv(filename)
	x_data = get_data(data, 'x')
	y_data = get_data(data, 'y')
	z_data = get_data(data, 'z')

	return magnitude(x_data, y_data, z_data)

def magnitude(x_data, y_data, z_data):
	x2 = x_data * x_data
	y2 = y_data * y_data
	z2 = z_data * z_data
	m2 = x2 + y2 + z2
	m = m2.apply(lambda x: math.sqrt(x))
	return m

def experiment():
	global counted_shots
	counted_shots = 0
	correlation_threshold = 80
	marker_distance_threshold = 70

	pattern_file = '5shots.txt'
	mag = mag_from_file(pattern_file)
	ma = moving_average(mag, 50)

	plt.gcf().canvas.set_window_title(pattern_file + " ma")
	plt.plot(ma)
	# plt.show()
	plt.clf()


	pattern_file = '5shots2.txt'
	patt_start = 82
	patt_end = 229
	mag_data = mag_from_file(pattern_file)
	mag_data_flat = flatten_df(mag_data)
	mag_data_smooth = smooth(mag_data_flat)
	mag_data_norm = normalize_df(mag_data_smooth)


	plt.gcf().canvas.set_window_title(pattern_file + " mag for pattern")
	plt.plot(mag_data_norm)
	# plt.show()
	plt.clf()

	pattern_data = mag_data_norm[patt_start:patt_end]

	for filename in glob.iglob('*.txt'):
		csv_data = pd.read_csv(filename)
		mag_data = mag_from_file(filename)
		mag_data = reject_outliers(mag_data)
		mag_data_flat = flatten_df(mag_data)
		mag_data_smooth = smooth(mag_data_flat)
		mag_data_norm = normalize_df(mag_data_smooth)
		name = filename.split('.')[0]
		counted_shots = counted_shots + count_shots(mag_data_norm, pattern_data, name, correlation_threshold, marker_distance_threshold)
	print("Axis mag; Counted %d shots; Expected: %d" % (counted_shots, expected_shots_count))

def old_research():
	mag = mag_from_file(filename)

	plt.gcf().canvas.set_window_title(filename + " mag")
	plt.plot(mag)
	plt.show()
	plt.clf()

	lowpass_mag = butter_lowpass_filter(mag, cutoff, fs, order)

	plt.gcf().canvas.set_window_title(filename + " lowpass filter mag")
	plt.plot(lowpass_mag)
	plt.show()
	plt.clf()

	sig_order = signal.order_filter(mag, domain, 0)

	moving_mag_data = moving_average(mag, 50)
	savgol_moving = savgol_filter(moving_mag_data, 29, 12)
	savgol = savgol_filter(mag, 5, 2)



	# plt.gcf().canvas.set_window_title(filename + " mag")
	# plt.plot(mag)
	# plt.show()
	# plt.clf()



	plt.gcf().canvas.set_window_title(filename + " sig order")
	plt.plot(sig_order)
	plt.show()
	plt.clf()

	plt.gcf().canvas.set_window_title(filename + " savgol moving")
	plt.plot(savgol_moving)
	# plt.show()
	plt.clf()	

	for filename in glob.iglob('*.txt'):
		mag = mag_from_file(filename)
		plt.gcf().canvas.set_window_title(filename + " magnitude")
		plt.plot(mag)
		# plt.show()
		plt.clf()


	filename = '5shots2.txt'
	data = pd.read_csv(filename)
	x_data = get_data(data, 'x')
	y_data = get_data(data, 'y')
	z_data = get_data(data, 'z')

	mag52 = magnitude(x_data, y_data, z_data)

	# plt.plot(mag)
	# plt.gcf().canvas.set_window_title(filename + " magnitude")
	# plt.show()	

	x_data = mag52

	# x_data = x_data[0:1000]

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


	moving_x_data = moving_average(x_data, 50)
	plt.plot(moving_x_data)
	plt.gcf().canvas.set_window_title(filename + " moving average 3 shots")
	# plt.show()

def reject_outliers(data, m=3):
	return data[abs(data - np.mean(data)) < m * np.std(data)]

def butter_lowpass(cutoff, fs, order=5):
	nyq = 0.5 * fs
	normal_cutoff = cutoff / nyq
	b, a = butter(order, normal_cutoff, btype='low', analog=False)
	return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
	b, a = butter_lowpass(cutoff, fs, order=order)
	y = lfilter(b, a, data)
	return y

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

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

def count_shots(data, pattern, filename, correlation_threshold, marker_distance_threshold):
	data_df = pd.DataFrame(data)
	pattern_df = pd.DataFrame(pattern)

	auto_patt = flatten_df(pattern)
	auto_data = data.values.flatten()

	autocorr = np.correlate(auto_data, auto_patt)

	plt.plot(data)
	plt.gcf().canvas.set_window_title(filename + " raw data")
	conditional_show()
	plt.clf()	

	plt.plot(autocorr)
	plt.gcf().canvas.set_window_title(filename + " autocorrelation")
	conditional_show()
	plt.clf()

	markers = []

	for index, value in enumerate(autocorr):
		value = value * 10
		# print("value: %d vs thresh: %d" % (value, correlation_threshold))
		if value >= correlation_threshold:
			markers.append(index)

	index = 0
	while (index + 1) < len(markers):
		if (abs(markers[index] - markers[index + 1]) < marker_distance_threshold):
			next_marker = markers[index + 1]
			current_marker = markers[index]
			if (autocorr[current_marker] < autocorr[next_marker]):
				markers.remove(current_marker)
				index = 0				
			else:
				markers.remove(markers[index + 1])
				index = 0
		else:
			index += 1
	
	plt.plot(autocorr, marker='p', markevery=markers)
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
