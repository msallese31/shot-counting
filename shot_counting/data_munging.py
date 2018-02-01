import pandas as pd
import math

def mag_from_file(filename):
	data = pd.read_csv(filename)
	x_data = get_data(data, 'x')
	y_data = get_data(data, 'y')
	z_data = get_data(data, 'z')
	return magnitude(x_data, y_data, z_data)

def magnitude(x_data, y_data, z_data):
	print(type(x_data))
	x2 = x_data * x_data
	y2 = y_data * y_data
	z2 = z_data * z_data
	m2 = x2 + y2 + z2
	m = m2.apply(lambda x: math.sqrt(x))
	return m

def get_data(data, axis):
	ret_data = data[axis]	
	return ret_data