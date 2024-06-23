
import config
import re
import requests
import numpy as np
from datetime import datetime
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer

wnl = WordNetLemmatizer()

import nltk 

nltk.download('words')

def word_break(s, word_dict):
    n = len(s)
    dp = [[] for _ in range(n+1)]
    dp[0] = [""]
    
    for end in range(1, n+1):
        for start in range(end):
            if dp[start] and s[start:end] in word_dict:
                for word in dp[start]:
                    dp[end].append((word + ' ' + s[start:end]).strip())
    
    return dp[n]


def extract_domain_name(url):
    url = url.replace('http://', '').replace('https://', '')
    url = url.replace('www.', '')
    parts = url.split('.')
    domain_name_parts = parts[:-1]
    domain_name = ''.join(domain_name_parts)
    return domain_name

def predict_from_url(s, words_frequency):

    cleaned_text = extract_domain_name(s)
    # print(cleaned_text)

    word_set = set(nltk.corpus.words.words())

    custom_words = ["sex", "porn"]

    for word in custom_words:
        word_set.add(word)

    results = word_break(cleaned_text, word_set)
    
    # Filter out words with length less than or equal to 3
    filtered_results = [
        [word for word in result.split() if len(word) > 3] 
        for result in results
    ]
    
    # Sort results based on the length of words in each combination
    sorted_results = sorted(filtered_results, key=lambda x: len(' '.join(x)), reverse=True)
    
    # Select the longest combination
    longest_combination = sorted_results[0] if sorted_results else []
    # print(longest_combination)
    if (longest_combination):
        return predict_category(words_frequency, longest_combination)
    else:
        return False


def scrape_url(url):
    try:
        res = requests.get(url, headers=config.REQUEST_HEADERS, timeout=5)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, "html.parser")
            [tag.decompose() for tag in soup("script")]
            [tag.decompose() for tag in soup("style")]
            text = soup.get_text()
            cleaned_text = re.sub('[^a-zA-Z]+', ' ', text).strip()
            tokens = word_tokenize(cleaned_text)
            tokens_lemmatize = remove_stopwords(tokens)
            return tokens_lemmatize
        else:
            print(
                f'Request failed ({res.status_code}). Please check if website do not blocking or it is still existing')
    except Exception as e:
        print(f'Request to {url} failed. Error code:\n {e}')
        return False
    
def predict_from_meta(url, words_frequency):
    try:
        res = requests.get(url, headers=config.REQUEST_HEADERS, timeout=15)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, "html.parser")
            [tag.decompose() for tag in soup("script")]
            [tag.decompose() for tag in soup("style")]
            [tag.decompose() for tag in soup("body")]
            text = soup.get_text()
            cleaned_text = re.sub('[^a-zA-Z]+', ' ', text).strip()
            tokens = word_tokenize(cleaned_text)
            tokens_lemmatize = remove_stopwords(tokens)
            # print(tokens_lemmatize)
            return predict_category(words_frequency, tokens_lemmatize)
        else:
            print(
                f'Request failed ({res.status_code}). Please check if website do not blocking or it is still existing')
    except Exception as e:
        print(f'Request to {url} failed. Error code:\n {e}')
        return False


def predict_category(words_frequency, tokens):
    category_weights = []
    for category in words_frequency:
        weight = 0
        intersect_words = set(words_frequency[category]).intersection(set(tokens))
        for word in intersect_words:
            if word in tokens:
                index = words_frequency[category].index(word)
                weight += config.FREQUENCY_TOP_WORDS - index
        category_weights.append(weight)

    category_index = category_weights.index(max(category_weights))
    main_category = list(words_frequency.keys())[category_index]
    category_weights[category_index] = 0
    category_index = category_weights.index(max(category_weights))
    return main_category

def remove_stopwords(tokens):
    tokens_list = []
    for word in tokens:
        word = wnl.lemmatize(word.lower())
        if word not in config.STOPWORDS:
            tokens_list.append(word)
    return list(filter(lambda x: len(x) > 1, tokens_list))
