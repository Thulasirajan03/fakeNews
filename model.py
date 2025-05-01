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

# Loading the datasets
fake = pd.read_csv('Fake.csv')
true = pd.read_csv('True.csv')
combined = pd.read_csv('dataset.csv')  

# Labeling fake and true data
fake['class'] = 0
true['class'] = 1

# Checking and handling combined dataset
if 'class' not in combined.columns:
    print("Warning: 'class' column not found in Combined dataset! Adding manually...")
    combined['class'] = -1  # Temporary label for unknown

# Merging and shuffling
df = pd.concat([fake, true, combined]).sample(frac=1, random_state=42)

# Removing unknown labels (-1)
df = df[df['class'] != -1]

# Dropping unnecessary columns
df = df.drop(['title', 'subject', 'date'], axis=1, errors='ignore')

# Handling missing text entries
df['text'] = df['text'].fillna('')

# Text cleaning function
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\b\w\b', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Applying text cleaning
df['text'] = df['text'].apply(clean_text)

# Features and Labels
X = df['text']
y = df['class']

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, min_df=5, ngram_range=(1,2))
X_vect = vectorizer.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.25, stratify=y, random_state=42)

# Handling Imbalanced Data
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Models
nb_model = MultinomialNB()
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Ensemble Model
ensemble_model = VotingClassifier(
    estimators=[('nb', nb_model), ('rf', rf_model)],
    voting='soft'
)

# Training
ensemble_model.fit(X_resampled, y_resampled)

# Prediction
y_pred = ensemble_model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

# Classification Report
report = classification_report(y_test, y_pred)

# Saving the model
with open('ensemble_model.pkl', 'wb') as f:
    pickle.dump(ensemble_model, f)

with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

# Saving the training report
with open('training_report.txt', 'w') as f:
    f.write(f"Model Accuracy: {accuracy:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report)

print("Training and Saving Completed Successfully.")
