import pickle
import config
import argparse
from functions import scrape_url

parser = argparse.ArgumentParser(description='Website category prediction app')
parser.add_argument('-u', '--url', help='URL for website to predict category')

args = parser.parse_args()

if args.url:
    url = args.url
    print(url)
    
    # Scrape the website content to extract tokens
    tokens = scrape_url(url)
    if tokens:
        # Convert the list of tokens to a string
        tokens_str = ' '.join(tokens)  

        # Load the trained model and the TfidfVectorizer
        with open(config.TRAINED_MODEL, 'rb') as file:
            model, tfidf_vectorizer = pickle.load(file)

        # Vectorize the tokens using the same TF-IDF vectorizer
        tokens_tfidf = tfidf_vectorizer.transform([tokens_str])

        # Predict the category of the new website
        predicted_category = model.predict(tokens_tfidf)[0]

        if predicted_category:
            print("Predicted Category:", predicted_category)
else:
    parser.error("Please specify websites link. More info 'python 03_predict.py -h'")
