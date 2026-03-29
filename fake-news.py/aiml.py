import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Better dataset
data = {
    "text": [
        "Breaking news something shocking happened",
        "Government announces new policy today",
        "Scientists discover new planet in space",
        "New education reforms introduced",
        "Click here to win money instantly",
        "You won lottery claim now",
        "Fake cure for disease spreading online",
        "Celebrity rumor spreads on internet",
        "India launches new satellite successfully",
        "Stock market reaches new high today"
    ],
    "label": [0,1,1,1,0,0,0,0,1,1]
}

df = pd.DataFrame(data)

# Split
X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Vectorization
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model (important fix 👇)
model = LogisticRegression(class_weight='balanced')
model.fit(X_train_vec, y_train)

# Accuracy
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Prediction loop
while True:
    text = input("\nEnter news (type 'exit' to stop): ")

    if text.lower() == "exit":
        break

    text_vec = vectorizer.transform([text])
    result = model.predict(text_vec)

    if result[0] == 0:
        print("Fake News ❌")
    else:
        print("Real News ✅")