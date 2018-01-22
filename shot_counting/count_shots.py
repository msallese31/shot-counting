
def count(raw_data, classifier, classifier_threshold):
	return count_shots_new(raw_data, classifier, classifier_threshold)

def count_shots_new(mag, classifer, thresh):
	shot_count = 0
	indexes = build_prediction_list(mag, classifer)
	streaks = countlist(indexes)
	shots = [i for i in streaks if i >= thresh]
	shot_count += len(shots)
	return shot_count


def build_prediction_list(mag, classifier):
	i = 0
	predictions = []
	indexes = []
	# print(mag.shape[1])
	while i < mag.shape[1]:
		if(i + 100 > mag.shape[1]):
			break
		# print("Mag shape:")
		# print(mag.shape)
		# print(mag)
		mag_reshape = mag[:, i:i+100]
		prediction = classifier.predict(mag_reshape)
		# print("Prediction for mag[i:i+100]")
		# print(prediction)
		predictions.append(prediction)
		if prediction == 1:
			indexes.append(i)
		# print("Index: %d;\n Window: mag[%d:%d];\nClass: %d" % (i, i, i + 100, prediction))
		i += 1
	return indexes

def countlist(random_list):
	count = 0 
	retlist = []
	# Avoid IndexError for  random_list[i+1]
	for i in range(len(random_list) - 1):
		# Check if the next number is consecutive
		if random_list[i] + 1 == random_list[i+1]:
			count += 1
		else:
			# If it is not append the count and restart counting
			retlist.append(count)
			count = 1
	# Since we stopped the loop one early append the last count
	retlist.append(count)
	return retlist

if __name__ == '__main__':
		run_tests()