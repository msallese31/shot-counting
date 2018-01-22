import glob
from build_classifiers import train_classifier
import data_munging as dm
import count_shots

def evaluate(X, Y):
	mlp, mlp_threshold = train_classifier("MLP", X, Y)
	log, log_threshold = train_classifier("LOG", X, Y)
	svm, svm_threshold = train_classifier("SVM", X, Y)
	gau, gau_threshold = train_classifier("GAU", X, Y)
	ran, ran_threshold = train_classifier("RAN", X, Y)

	print("########## MLP #########")
	mlp_count = load_all_files_and_count_shots(mlp, mlp_threshold)
	print("########################")
	print("########## LOG #########")
	log_count = load_all_files_and_count_shots(log, log_threshold)
	print("########################")
	print("########## SVM #########")
	svm_count = load_all_files_and_count_shots(svm, svm_threshold)
	print("########################")
	print("########## GAU #########")
	gau_count = load_all_files_and_count_shots(gau, gau_threshold)
	print("########################")
	print("########## RAN #########")
	ran_count = load_all_files_and_count_shots(ran, ran_threshold)
	print("########################")

	total_count = mlp_count + log_count + svm_count + gau_count + ran_count
	avg_count = total_count / 5
	print("Avg count of all classifiers = %d" % avg_count)

def load_all_files_and_count_shots(classifier, classifier_threshold):
	total_counted_shots = 0
	for filename in glob.iglob('shot_counting/*.txt'):
		mag = dm.mag_from_file(filename)
		mag = mag.reshape(1,-1)
		name = filename.split('.')[0]
		counted_shots = count_shots.count(mag, classifier, classifier_threshold)
		total_counted_shots += counted_shots
		print("Counted %d shots for %s" % (counted_shots, filename))
	print("total counted shots: %d" % total_counted_shots)
	return total_counted_shots