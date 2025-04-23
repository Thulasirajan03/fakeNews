import pandas as pd
import re
import string
import pickle
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report

#Loading the Dataset
fake=pd.read_csv('Fake.csv')
true=pd.read_csv('True.csv')

#Labeling the  data
fake['class']=0
true['class']=1

#Merging and Shuffling the data
df=pd.concat([fake,true]).sample(frac=1, random_state=42)

#Deleting the column
df=df.drop(['title','subject','date'],axis=1)

#Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)     
    text = re.sub(r'\d+', '', text)                        
    text = re.sub(r'\b\w\b', '', text)                     
    text = text.translate(str.maketrans('', '', string.punctuation))  
    text = re.sub(r'\s+', ' ', text).strip()               
    return text

#Calling the function
df['text']=df['text'].apply(clean_text)

#Adding the label
X=df['text']
y=df['class']

#TF_IDF Vectorizing
vectorizer=TfidfVectorizer(stop_words='english',max_df=0.7,min_df=5,ngram_range=(1,2))
X_vect = vectorizer.fit_transform(X)

#Train-Test split
X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.25, stratify=y, random_state=42)

#Handling imbalance data
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

#Navie Bayes and RandomForest
nb_model = MultinomialNB()
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

#Creating voting classifier
ensemble_model = VotingClassifier(
    estimators=[('nb', nb_model), ('rf', rf_model)],
    voting='soft'
)

#Training the model
ensemble_model.fit(X_resampled, y_resampled)

#Prediction
y_pred = ensemble_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

#Saving the model
with open('ensemble_model.pkl', 'wb') as f:
    pickle.dump(ensemble_model, f)

with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Trained Successfully.")