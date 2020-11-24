import string
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
import operator
import itertools
import webbrowser
from WordNetImprovement import WordNetImprovement
from collections import Counter


np.seterr(all='ignore')
# Importing the stored files
vocab = pd.read_pickle(r'./Storage/words.pkl')
idf = pd.read_pickle(r'./Storage/inv_doc_freq.pkl')
doc_vector = pd.read_pickle('./Storage/doc_vec.pkl', 'bz2')
zone = pd.read_pickle(r'./Storage/zone.pkl')
zone_vec = pd.read_pickle(r'./Storage/zone_vec.pkl')

# Creating the data-frame to store our query vector and zone vector
buffer = pd.read_pickle('./Storage/df.pkl', 'bz2')
buffer.drop(buffer.index, inplace=True)
buffer.loc[0] = 0
zone_buffer = pd.read_pickle('./Storage/zone_df.pkl', 'bz2')
zone_buffer.drop(zone_buffer.index, inplace=True)
zone_buffer.loc[0] = 0


def preprocess_query(query):
    """
    Remove all punctuation and special characters
    :param query: Preprocess the query to remove punctuations
    :return: tokenized query terms as list
    """

    query = query.lower()
    query = query.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
    query = query.translate(str.maketrans("‘’’–——−", '       '))

    # Tokenizing the query
    query_words = []
    query_words = word_tokenize(query)
    query_words = list(set(query_words))

    return query_words


def query_relaxation(tokenized_query, mode='hypernym'):
    """
    Query relaxation using WordNet hypernyms and synonyms
    :param tokenized_query: list of tokenized query terms
    :param mode: choose mode: hypernym for hypernym based relaxation, 'synonym' for synonym based relaxation
    :return: dictionary containing alternate tokenized queries
    """
    # Threshold idf as decided by heuristic
    threshold = 0.8

    parallel_query_dict = {}
    syn = {}
    hyp = {}
    temp = []
    idx = 0

    for token in tokenized_query:
        # Iterating through query term list to check if the term is present in the corpus and qualifies the threshold
        if idf.get(token) is not None and idf[token] > threshold:
            # Instantiating the WordNet Improvement object
            relaxer = WordNetImprovement(token)
            # hyp and syn lists corresponding to the hypernyms and synonyms of most common context
            hyp = relaxer.extract_hypernyms()[0]
            syn = relaxer.extract_synonyms()[0]

    if mode == 'hypernym':
        # Replacing the query term which crosses threshold with its hypernyms
        for w in hyp:
            # creating a copy of the original tokenized query
            temp = tokenized_query.copy()
            for i, term in enumerate(temp):
                # finding the index to make the replacement
                if term is not w and term in idf and idf[term] > threshold:
                    # Making sure the original term and its hypernym are not the same
                    temp[i] = w
                    # print(w)
                    # adding the relaxed query to parallel dictionary
                    parallel_query_dict[idx] = temp
            idx += 1

    elif mode == 'synonym':
        for w in syn:
            # n is hypernym considered for replacement in the hyp list
            temp = tokenized_query.copy()
            for i, term in enumerate(temp):
                # finding the index to make the replacement
                if term in idf and idf[term] > threshold and w != term:
                    # Making sure the original term and its hypernym are not the same
                    temp[i] = w
                    # adding the relaxed query to parallel dictionary
                    parallel_query_dict[i] = temp
            idx += 1

    '''parallel_query_dict is a dictionary containing list of tokenized query terms after relaxation'''
    return parallel_query_dict


def get_scores(query_words, open_web, use_zones):
    """
    Function to process and retrieve the docs
    :param query_words: tokenized query as a list of strings
    :param open_web: set to True if results are to be opened on browser window
    :param use_zones: set to True to enable Zonal indexing
    :return: dictionary of scores corresponding to their document id
    """
    # Resetting buffer and zone_buffer
    buffer.loc[0] = 0
    zone_buffer.loc[0] = 0

    # Populating the query term frequency data-frame
    threshold = 0.1
    # This is the idf below which which do not want to consider the words. Removes very frequent words from the zone.
    for token in query_words:
        if token in buffer.columns:
            buffer[token] += 1
            if token in zone_buffer.columns and idf[token] > threshold:
                zone_buffer[token] += idf[token]

    # Vectorising the query doc frequency and calculating weights
    query_vec = (1+np.log10(np.array(buffer.loc[0])))*list(idf.values())
    query_vec[query_vec == -np.inf] = 0
    query_vec = query_vec/(np.sqrt(sum(query_vec**2)))
    # Converting NaN values to zero
    query_vec = np.nan_to_num(query_vec)

    # Vectorising the query zone doc frequency and calculating weights
    zone_query_vec = np.array(zone_buffer.loc[0])
    zone_query_vec = zone_query_vec/(np.sqrt(sum(zone_query_vec**2)))
    zone_query_vec = np.nan_to_num(zone_query_vec)

    # Computing scores for the query vector corresponding to each document
    scores = {}
    for doc_id, sub_vector in doc_vector.items():
        scores[doc_id] = np.sum(np.multiply(query_vec, sub_vector))
    # max-val stores the highest score recorded for document content matching
    # We are adding extra score if the title also matches
    if use_zones:
        max_val = max(scores.values())
        for doc_id, sub_vector in zone_vec.items():
            scores[doc_id] += np.sum(np.multiply(zone_query_vec, sub_vector))*max_val*0.75

    return scores


def search(query, open_web, use_zones, enable_query_relaxation=1):
    """
    Searching a free text query to print top 10 doc ids, with their score and title
    :param query: query as a string
    :param open_web: set to True if results are to be opened on browser window
    :param use_zones: set to True to enable Zonal indexing
    :param enable_query_relaxation: set to 1 to enable hypernym query relaxation, 2 for synonym based relaxation
    :return: None
    """
    # Processing query to remove punctuations and special characters
    processed_query = preprocess_query(query)

    '''parallel_dict: dictionary containing relaxed queries as list of tokenized query terms
    parallel scores: dictionary containing combined score of all relaxed queries
    scored_doc_ids: final score of top 10 docs after query relaxation (if applicable)
    '''
    parallel_dict = {}
    parallel_scores = {}
    added_score = {}
    scored_doc_ids = []

    if enable_query_relaxation:

        if enable_query_relaxation == 1:
            parallel_dict = query_relaxation(processed_query, mode='hypernym')
            for i in range(len(parallel_dict.keys())):
                # Finding scores corresponding to each relaxed query
                try:
                    temp_score = get_scores(parallel_dict[i], open_web=False, use_zones=False)
                except KeyError:
                    temp_score = {}
                # Sorting the scores
                temp_score = dict(sorted(temp_score.items(), key=operator.itemgetter(1), reverse=True))
                # updating the temp_score in parallel dict by addition
                parallel_scores = Counter(parallel_scores) + Counter(temp_score)

        elif enable_query_relaxation == 2:
            parallel_dict = query_relaxation(processed_query, mode='synonym')
            for i in range(len(parallel_dict.keys())):
                try:
                    temp_score = get_scores(parallel_dict[i], open_web=False, use_zones=False)
                except KeyError:
                    temp_score = {}
                temp_score = dict(sorted(temp_score.items(), key=operator.itemgetter(1), reverse=True))
                parallel_scores = Counter(parallel_scores) + Counter(temp_score)

        # computing scores for the original query
        original_scores = get_scores(processed_query, open_web=False, use_zones=False)

        if parallel_scores is None:
            # if there is no relaxation possible (no hypernyms/synonyms), parallel_scores is empty
            added_score = original_scores
        else:
            # print(parallel_dict)
            num = len(parallel_dict.keys())
            # print(num)
            # weight to be given to scores of original query term
            weight = num

            for key, value in parallel_scores.items():
                # weighing the scores of relaxed queries by heuristic
                parallel_scores[key] = (1/value*original_scores[key])

            # added_score is the combined score after incorporating query relaxation
            added_score = Counter(original_scores) + Counter(parallel_scores)

        # sorting in decreasing order
        added_score = dict(sorted(added_score.items(), key=operator.itemgetter(1), reverse=True))
        # retrieving top 10 as list
        scored_doc_ids = list(itertools.islice(added_score.items(), 10))

    elif enable_query_relaxation is False:
        # Scoring without query relaxation
        temp_score = get_scores(processed_query, open_web=open_web, use_zones=use_zones)
        temp_score = dict(sorted(temp_score.items(), key=operator.itemgetter(1), reverse=True))
        scored_doc_ids = list(itertools.islice(temp_score.items(), 10))

    for k, v in scored_doc_ids:
        print(k, round(v, 3), zone[k])
        # Opening the web-pages in a browser for easy checking
        if open_web:
            webbrowser.open('https://en.wikipedia.org/wiki?curid=' + str(k))
