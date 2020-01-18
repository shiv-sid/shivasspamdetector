#import statements

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
import pickle
from sklearn.pipeline import Pipeline




#----------------Input Data and Preprocessing-----------------

#1.Read the CSV file

df=pd.read_csv("spam.csv")

#2.Drop the columns Unnamed:2, Unnamed:3, Unnamed:4
df=df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"],axis=1)
#df
#3. Rename the columns v1 as label and v2 as message
df=df.rename(columns={"v1":"label","v2":"message"})
#df
#4. Map all ham labels to 0 and spam values to 1
#0-ham
#1-spam
df['label']=df.label.map({'ham':0, 'spam':1})
#print(df)
#5. Assign Message column to X
X=df['message']
#print(X)
#6. Assign label column to Y
Y=df['label']
#print(Y)
#print(df.shape)
#---------------------------Feature Extraction----------------

#7.Initialise the countvectorizer
cv=CountVectorizer(stop_words='english')
#print(cv)
#print(type(X))

"""
cv.fit(documents)
names=cv.get_feature_names()
print(names)
doc_array=cv.transform(documents).toarray()
print(doc_array)
frequency_matrix=pd.DataFrame(data=doc_array,columns=names)
print(frequency_matrix)
"""
#8.Fit tranform the data X in the vectorizer and store the result in X

#Xm=X.tolist()
#cv.fit(Xm)
#names=cv.get_feature_names()
#print(names)
#print(type(X))
X=cv.fit_transform(X)
#print(type(X))
#print(X)




#9.save your vectorizer in 'vector.pkl' file
pickle.dump(cv, open("vector.pkl", "wb"))


#------------------------Classification---------------------
from sklearn.model_selection import train_test_split 
##'''10. Split the dataset into training data and testing data with train_test_split function
##Note: parameters test_size=0.33, random_state=42'''

#X_train, X_test, y_train, y_test = train_test_split(df['message'],df['label'],random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.33, random_state=42)
#print(df.shape)
#print(X_train.shape)
#print(X_test.shape)

#11. Initialise multimimial_naive_bayes classifier
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()


#12.Fit the training data with labels in Naive Bayes classifier 'clf'
"""
cv=CountVectorizer(stop_words='english')
training_data=cv.fit_transform(X_train)
testing_data=cv.transform(X_test)
clf.fit(training_data,y_train)
"""
clf.fit(X_train,y_train)
predictions=clf.predict()
#
#from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
#
#print(accuracy_score(y_test,predictions))
#print(precision_score(y_test,predictions))
#print(recall_score(y_test,predictions))
#print(f1_score(y_test,predictions))
#count=0
#for i in names:
#    count+=1
#print(count)

#13. Store your classifier in 'NB_spam_model.pkl' file
joblib.dump(clf, 'NB_spam_model.pkl')






