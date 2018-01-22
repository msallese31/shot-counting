import count_shots
from build_classifiers import train_classifier
from manual_data_labelling import label_data
import data_munging as dm
import evaluate_classifiers
from flask import Flask
app = Flask(__name__)

X, Y = label_data()
classifier = "dummy"
classifier_threshold = "dummy"

@app.route('/')
def hello_world():
    return('Hello, World!\n')

@app.before_first_request
def main():
	global classifier, classifier_threshold
	print("Starting up flask")
	# evaluate_classifiers.evaluate(X,Y)
	classifier, classifier_threshold = train_classifier("SVM", X, Y)

	test_file()

	print("Flask started up succesfully! Listening for requests")


	# handle_request(classifier, classifier_threshold)


@app.route('/count')
def handle_request():
	# Placeholder until we know what kind of data will come in from the frontend
	return("Request handling is not yet fully implemented!!\n")
	data = prepare_data()
	shots = count_shots.count(data, classifier, classifier_threshold)

def prepare_data():
	# Placeholder until we know what kind of data will come in from the frontend
	# This most likely will move to data munging
	dummy = True

@app.route('/health')
def test_file():
	mag = dm.mag_from_file('data/3shots3.txt')
	mag = mag.reshape(1,-1)

	shots = count_shots.count(mag, classifier, classifier_threshold)
	return("Test file has %d shots\n" % shots)

if __name__ == '__main__':
		app.run()