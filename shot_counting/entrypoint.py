from __future__ import print_function
import count_shots
from build_classifiers import train_classifier
from manual_data_labelling import label_data
import data_munging as dm
import evaluate_classifiers
from flask import Flask, request
import sys
import json
import pandas as pd
app = Flask(__name__)

X, Y = label_data()
classifier = "dummy"
classifier_threshold = "dummy"

@app.route('/')
def hello_world():
	return('Hello, World!\n')

@app.before_first_request
def main():
	global classifier_svm, classifier_threshold_svm, classifier_mlp, classifier_threshold_mlp, classifier_gau, classifier_threshold_gau, classifier_log, classifier_threshold_log, classifier_ran, classifier_threshold_ran
	print("Starting up flask")
	# evaluate_classifiers.evaluate(X,Y)
	classifier_svm, classifier_threshold_svm = train_classifier("SVM", X, Y)
	classifier_mlp, classifier_threshold_mlp = train_classifier("MLP", X, Y)
	classifier_gau, classifier_threshold_gau = train_classifier("GAU", X, Y)
	classifier_log, classifier_threshold_log = train_classifier("LOG", X, Y)
	classifier_ran, classifier_threshold_ran = train_classifier("RAN", X, Y)

	test_file()

	print("Flask started up succesfully! Listening for requests")


	# handle_request(classifier, classifier_threshold)


@app.route('/count', methods=['GET', 'POST'])
def handle_request():
	print(classifier_svm)
	print("hello?")
	# Placeholder until we know what kind of data will come in from the frontend
	# return("Request handling is not yet fully implemented!!\n")
	print(classifier)
	print(classifier_threshold)
	data = prepare_data(request.get_json())
	shots_svm = count_shots.count(data, classifier_svm, classifier_threshold_svm)
	shots_mlp = count_shots.count(data, classifier_mlp, classifier_threshold_mlp)
	shots_gau = count_shots.count(data, classifier_gau, classifier_threshold_gau)
	shots_log = count_shots.count(data, classifier_log, classifier_threshold_log)
	shots_ran = count_shots.count(data, classifier_ran, classifier_threshold_ran)
	print("%d shots SVM!" % shots_svm)
	print("%d shots MLP!" % shots_mlp)
	print("%d shots GAU!" % shots_gau)
	print("%d shots LOG!" % shots_log)
	print("%d shots RAN!" % shots_ran)
	return "good"

def prepare_data(data_from_request):
	x = pd.Series(data_from_request['X_Acc'])
	y = pd.Series(data_from_request['Y_Acc'])
	z = pd.Series(data_from_request['Z_Acc'])
	mag = dm.magnitude(x,y,z)
	mag = mag.reshape(1,-1)
	return mag

@app.route('/health')
def test_file():

	mag = dm.mag_from_file('data/3shots3.txt')
	mag = mag.reshape(1,-1)

	shots = count_shots.count(mag, classifier_svm, classifier_threshold_svm)
	return("Test file has %d shots\n" % shots)

if __name__ == '__main__':
		app.run()