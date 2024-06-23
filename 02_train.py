import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import config
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import word_tokenize


if __name__ == '__main__':

    # Load preprocessed data
    df = pd.read_csv(config.MAIN_DATASET_PATH)
    df = df[['url', 'main_category', 'cleaned_website_text']]

    df['tokens'] = df['cleaned_website_text'].apply(word_tokenize)
    print(df['main_category'].unique())
    # Count the occurrences of each category
    category_counts = df['main_category'].value_counts()

    # Set the style of seaborn
    sns.set(style="whitegrid")

    # Create a bar plot
    plt.figure(figsize=(10, 8))  # Adjust the size of your plot
    sns.barplot(x=category_counts.values, y=category_counts.index, palette="viridis")

    # Set the labels and title
    plt.xlabel('Frequency')
    plt.ylabel('Main Category')
    plt.title('Distribution of Main Categories')

    # Display the plot
    plt.show()

    print(df)

    # Split data into features (tokens) and target (main_category)
    X = df['tokens']
    y = df['main_category']
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=79)

    # Vectorize the tokens using TF-IDF

    # Concatenate the strings in X_train and X_test
    X_train_concatenated = [' '.join(doc) for doc in X_train]
    X_test_concatenated = [' '.join(doc) for doc in X_test]

    # Apply TF-IDF vectorization
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_concatenated)
    X_test_tfidf = tfidf_vectorizer.transform(X_test_concatenated)

    print("Training model")
    # Train the logistic regression model
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train_tfidf, y_train)

    # Evaluate the model
    y_pred = logreg.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred,average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print("Accuracy Score:", accuracy)
    print("Precision Score:", precision)
    print("Recall Score:", recall)
    print("F1 Score:", f1)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=logreg.classes_, yticklabels=logreg.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    # Save the trained model and TfidfVectorizer to a file
    with open(config.TRAINED_MODEL, 'wb') as file:
        pickle.dump((logreg, tfidf_vectorizer), file)

    print("Model was saved successfully.")
