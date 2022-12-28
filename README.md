# spam_classification_using_Na-ve_Bayes

The dataset I used is SMS Spam Collection consisting of 5,574 instances with 2 classes of spam and ham (non-spam) available at UC Irvine Machine Learning Repository and could be obtained from “https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection”. 
The data folder of “smsspamcollection.zip” consist of 2 files “readme” and “SMSSpamCollection”. The “readme” consists of dataset description, format, and references and the “SMSSpamCollection” is the dataset you will use in this assignment.  

The following are the steps I performed:

Load the dataset (use the Pandas DataFrame to read the input file):
data_SMS = pd.read_csv('SMSSpamCollection', sep = '\t', header=None, names=["Class", "SMS_Text"])

Plot the frequencies of data classes (spam and ham).
data_SMS.Class.value_counts().plot(kind='barh')

Convert the class categories (spam and ham) into a binary format of 0 and 1. 
Print out the frequency of the top 10 words. 

Split the data into training and test sets. Use 80% for training and 20% for testing.

Extract the features using word frequencies.
WordVectorizer = CountVectorizer() 
TrainData_Freq = WordVectorizer.fit_transform(train_data.values)
TrainData_Freq.toarray()

Create the Multinomial Naïve Bayes classifier:
from sklearn.naive_bayes import MultinomialNB
NB_classifier = MultinomialNB()

Train the model:
NB_classifier.fit(TrainData_Freq, train_label)

Calculate the accuracy score (Note: perform feature extraction on test data):
TestData_Freq = WordVectorizer.transform(test_data)
NB_classifier.score(TestData_Freq, test_label)

Perform prediction on test data and print out the misclassified data 
For prediction: NB_classifier.predict()

Compute and print out the accuracy of a Naïve Bayes model in terms of true positive (TP), false positive (FP), true negative (TN), and false negative (FN). 
To validate calculations, plot the confusion matrix as follows: 

from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(NB_classifier, TestData_Freq, test_label)

Now, perform basic text data cleaning on input data such as removing special characters or punctuations, and then repeat the process from steps 5 to 10. 

Example of cleaning: data_SMS['SMS_Text'].str.replace('[^\w\s]','')

