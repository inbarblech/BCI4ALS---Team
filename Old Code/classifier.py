import sklearn as skl
import sklearn.discriminant_analysis as skl_LDA
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib as plt


def classifier_knn(X, y, plot):
    knn_model = KNeighborsClassifier(n_neighbors=2)  # choose number of neighbors
    X_train, X_test, y_train, y_test = skl.model_selection.train_test_split(X,
                                                                            y,
                                                                            random_state=0,
                                                                            test_size=0.21)

    scaler = StandardScaler()
    scaler.fit(X_train)
    scaler.transform(X_train)
    scaler.transform(X_test)

    _ = knn_model.fit(X_train, np.ravel(y_train))
    y_predicted = knn_model.predict(X_test)

    confusion_matrix_ = skl.metrics.confusion_matrix(y_test, y_predicted)

    true_positive = confusion_matrix_[1, 1]
    false_positive = confusion_matrix_[0, 1]
    true_negative = confusion_matrix_[0, 0]
    false_negative = confusion_matrix_[1, 0]

    sensitivity = true_positive / (true_positive + false_negative)
    specificity = true_negative / (true_negative + false_positive)
    # precision = true_positive / (true_positive + false_positive)
    # f1_score = 2 / (1 / sensitivity + 1 / precision)

    a_score = skl.metrics.accuracy_score(y_test, y_predicted)

    if plot:
        confusion_matrix_plot(confusion_matrix_, 'knn classifier')

    return a_score, sensitivity, specificity   #, precision,f1_score


def classifier_svm(X, y, plot):
    X_train, X_test, y_train, y_test = skl.model_selection.train_test_split(X,
                                                                            y,
                                                                            random_state=0,
                                                                            test_size=0.25)

    scaler = StandardScaler()
    scaler.fit(X_train)
    scaler.transform(X_train)
    scaler.transform(X_test)

    svm_model = svm.SVC()
    _ = svm_model.fit(X_train, np.ravel(y_train))
    y_predicted = svm_model.predict(X_test)

    confusion_matrix_ = skl.metrics.confusion_matrix(y_test, y_predicted)

    true_positive = confusion_matrix_[1, 1]
    false_positive = confusion_matrix_[0, 1]
    true_negative = confusion_matrix_[0, 0]
    false_negative = confusion_matrix_[1, 0]

    sensitivity = true_positive / (true_positive + false_negative)
    specificity = true_negative / (true_negative + false_positive)
    # precision = true_positive / (true_positive + false_positive)
    # f1_score = 2 / (1 / sensitivity + 1 / precision)

    a_score = skl.metrics.accuracy_score(y_test, y_predicted)

    if plot:
        confusion_matrix_plot(confusion_matrix_, 'svm classifier')

    return a_score, sensitivity, specificity  #, precision, f1_score



def LDA_classifier(X, y, plot):
    X_train, X_test, y_train, y_test = skl.model_selection.train_test_split(X,
                                                                            y,
                                                                            random_state=0,
                                                                            test_size=0.2)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    LDA_model = skl_LDA.LinearDiscriminantAnalysis()
    _ = LDA_model.fit(X_train, np.ravel(y_train))
    y_predicted = LDA_model.predict(X_test)

    confusion_matrix_ = skl.metrics.confusion_matrix(y_test, y_predicted)

    true_positive = confusion_matrix_[1, 1]
    false_positive = confusion_matrix_[0, 1]
    true_negative = confusion_matrix_[0, 0]
    false_negative = confusion_matrix_[1, 0]

    sensitivity = true_positive / (true_positive + false_negative)
    specificity = true_negative / (true_negative + false_positive)
    # precision = true_positive / (true_positive + false_positive)
    # f1_score = 2 / (1 / sensitivity + 1 / precision)

    a_score = skl.metrics.accuracy_score(y_test, y_predicted)

    if plot:
        confusion_matrix_plot(confusion_matrix_, "LDA classifier")

    return a_score, sensitivity, specificity  #, precision, f1_score



def confusion_matrix_plot(confusion_matrix_, classifier_name):
    title = ['Confusion matrix for', classifier_name]

    disp = skl.metrics.ConfusionMatrixDisplay(confusion_matrix_)

    _ = disp.plot()

