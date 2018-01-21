from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression

def train_classifier(classifier_type, X, Y):

	MLP_threshold = 13
	SVM_threshold = 102
	GAU_threshold = 92
	LOG_threshold = 8
	RAN_threshold = 17

	if classifier_type == "MLP":
		MLP = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10, 10, 10), random_state=1)
		MLP.fit(X,Y)
		return MLP, MLP_threshold
	elif classifier_type == "SVM":
		supp_vec = svm.SVC(gamma=0.001, C=1.)
		supp_vec.fit(X, Y)
		return supp_vec, SVM_threshold
	elif classifier_type == "GAU":
		gnb = GaussianNB()
		gnb.fit(X, Y)
		return gnb, GAU_threshold
	elif classifier_type == "LOG":
		LogReg = LogisticRegression()
		LogReg.fit(X, Y)
		return LogReg, LOG_threshold
	elif classifier_type == "RAN":
		rand_for = RandomForestClassifier()
		rand_for.fit(X, Y)
		return rand_for, RAN_threshold
	else:
		raise RuntimeError("The classifier type %s is not supported!" % classifier_type)

