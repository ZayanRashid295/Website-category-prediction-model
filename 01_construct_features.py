import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
import config
import pickle
from functions import timeit, scrape, parse_request

if __name__ == '__main__':
    # Load csv dataset and fix column names
    df = pd.read_csv(config.MAIN_DATASET_PATH)
    df = df.rename(columns={'main_category:confidence': 'main_category_confidence'})
    df = df[['url', 'main_category', 'main_category_confidence']]

    # Process websites with a confidence level of 100%
    df = df[(df['main_category'] != 'Not_working') & (df['main_category_confidence'] >= 1)]
    # Add http://
    df['url'] = df['url'].apply(lambda x: 'http://' + x)
    df['tld'] = df.url.apply(lambda x: x.split('.')[-1])
    df = df[df.tld.isin(config.TOP_LEVEL_DOMAIN_WHITELIST)].reset_index(drop=True)
    df['tokens'] = ''

    # Scrape websites
    print("Scraping begins. Start: ", datetime.now())
    with ThreadPoolExecutor(config.THREADING_WORKERS) as executor:
        start = datetime.now()
        results = executor.map(scrape, [(i, elem) for i, elem in enumerate(df['url'])])
    time1 = timeit(start)
    print('Scraping finished. Execution time: ', time1)

    # Check if website returns code 200 and parse tokens
    print("Analyzing responses. Start: ", datetime.now())
    with ProcessPoolExecutor(config.MULTIPROCESSING_WORKERS) as ex:
        start = datetime.now()
        res = ex.map(parse_request, [(i, elem) for i, elem in enumerate(results)])

    for props in res:
        i = props[0]
        tokens = props[1]
        df.at[i, 'tokens'] = tokens
    time2 = timeit(start)
    print('Analyzing responses. Execution time: ', time2)

    print('Saving data to pickle file: ', datetime.now())
    start = datetime.now()
    # Removing lines from dataset where no tokens 
    df = df[df['tokens'].apply(lambda x: len(x) > config.FREQUENCY_MIN_WORDS)]
    df.reset_index(drop=True, inplace=True)

    # Save new dataset to pickle file
    selected_columns = df[['main_category', 'tokens']]
    selected_columns.to_pickle(config.DATASET_WITH_TOKENS_PATH)

    time3 = timeit(start)

    print('Generating new dataset finished. Execution time: ', time3)

    print('Script finished.\n\nTimes log:\nScraping: ', time1, '\nExtract content: ', time2, '\nSave new dataset: ', time3, '\n')
