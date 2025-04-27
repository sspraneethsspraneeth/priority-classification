import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv("email_dataset_balanced.csv")

df["Text"] = df["Subject"] + " " + df["Body"]


df["Email_Length"] = df["Text"].apply(len)


label_encoder = LabelEncoder()
df["LabelEncoded"] = label_encoder.fit_transform(df["Label"])


X_train, X_test, y_train, y_test = train_test_split(df[["Text", "Email_Length"]], df["LabelEncoded"], test_size=0.2, random_state=42)


vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train["Text"])
X_test_tfidf = vectorizer.transform(X_test["Text"])


with open("tfidf_vectorizer.pkl", "wb") as vec_file:
    pickle.dump(vectorizer, vec_file)

print("TF-IDF vectorizer saved as tfidf_vectorizer.pkl")


import scipy.sparse as sp
X_train_final = sp.hstack((X_train_tfidf, X_train[["Email_Length"]].values))
X_test_final = sp.hstack((X_test_tfidf, X_test[["Email_Length"]].values))


param_grid_rf = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5, 10]
}

grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=3, scoring="accuracy", n_jobs=-1)
grid_rf.fit(X_train_final, y_train)
best_rf = grid_rf.best_estimator_

with open("random_forest_model.pkl", "wb") as model_file:
    pickle.dump(best_rf, model_file)

print("Random Forest model saved as random_forest_model.pkl")