import classifier
import pandas as pd
import os

path = os.path.join(os.getcwd(), "segmented_data")

X = pd.read_csv(os.path.join(path,'X.csv'))
Y = pd.read_csv(os.path.join(path,'Y.csv'), )
Y.drop(columns=['Unnamed: 0'], inplace=True)
X.dropna(1, inplace=True)

# precision_knn, f1_score_knn
a_score_knn, sensitivity_knn, specificity_knn = classifier.classifier_knn(X, Y, plot=1)
print("KNN Model")
print('the accuracy score is: ',a_score_knn, '\nthe sensitivity is: ', sensitivity_knn, '\nthe specificity is: ', specificity_knn)

a_score_svm, sensitivity_svm, specificity_svm = classifier.classifier_svm(X, Y, plot=1)
print("SVM Model")
print('the accuracy score is: ',a_score_svm, '\nthe sensitivity is: ', sensitivity_svm, '\nthe specificity is: ', specificity_svm)

a_score_lda, sensitivity_lda, specificity_lda = classifier.LDA_classifier(X, Y, plot=1)
print("LDA Model")
print('the accuracy score is: ',a_score_lda, '\nthe sensitivity is: ', sensitivity_lda, '\nthe specificity is: ', specificity_lda)