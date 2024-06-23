from flask import Flask, request, jsonify
import pickle
import config
import json
from functions import predict_from_meta
import os
import time

app = Flask(__name__)

pickle_in2 = open(config.META_MODEL_PATH, "rb")
words_frequency_meta = pickle.load(pickle_in2)


# Load or initialize the cache
cache_file = 'url_category_cache.json'
if os.path.exists(cache_file):
    with open(cache_file, 'r') as f:
        url_category_cache = json.load(f)
else:
    url_category_cache = {}

def save_cache():
    with open(cache_file, 'w') as f:
        json.dump(url_category_cache, f)

@app.route('/get_category/<path:url>', methods=['GET'])
def get_category(url):
    print("Received GET request with parameter:", url)

    # Check if the URL is in cache
    if url in url_category_cache:
        print("Cache hit for URL:", url)
        predicted_category = url_category_cache[url]
    else:
        print("Cache miss for URL:", url)
        # Scrape the website content to extract tokens
        full_domain = "https://" + url
        start_time = time.time()
        predicted_category = "unknown"

        start_time = time.time()     
        predicted_category_meta = predict_from_meta(full_domain, words_frequency_meta)
        end_time3=time.time() - start_time
        print("--- META %s seconds ---" % end_time3)

        if predicted_category_meta:
            print("Predicted Category META:", predicted_category_meta)
            # Save to cache
            url_category_cache[url] = predicted_category
            save_cache()

    response = {"category": predicted_category}
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, port=3000)
