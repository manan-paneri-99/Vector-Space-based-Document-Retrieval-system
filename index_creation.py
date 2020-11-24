import re
import pickle
import os
import string
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import bz2


def corpus_parser(location):
    """
    Creates index as per lnc formalism
    :param location: address to the text corpus
    :return: None
    """
    # Creating a list of document ids
    doc_no = []
    # Creating a list of words in the documents
    words = []
    # Creating a list of words in the document zones i.e headings
    zone_words = []

    # Stores the document id and it's corresponding zone i.e heading
    zone = {}

    # Stores the document id and corresponding tokenised words of the document
    tokenised = {}

    # Stores the document id and corresponding tokenised words of the document zone
    zone_tokenised = {}

    # Opening the corpus and reading the file
    f = open(location, 'r', encoding='utf8')
    content = f.read()
    content = str(content)

    # Removing <a>...</a> tags
    pattern = re.compile("<(/)?a[^>]*>")
    content_new = re.sub(pattern, "", content)

    # Creating a folder to hold the seperated documents
    if not os.path.exists("./Documents"):
        os.mkdir("./Documents")

    # Creating the folder to store dictionaries as pickle files
    if not os.path.exists("./Storage"):
        os.mkdir("./Storage")

    # Creating a soup using a html parser and iterating through each 'doc'
    soup = BeautifulSoup(content_new, 'html.parser')
    for doc in soup.findAll('doc'):
        # Opening a file to write the contents of the doc
        o = open('./Documents/' + str(doc['id']) + ".txt", 'w', encoding='utf8')

        # Adding the document id to doc_no and extracting the text in that doc
        doc_no = doc_no + [(int(doc['id']))]
        text = doc.get_text()

        # Writing the text and closing the file
        o.write(doc.get_text())
        o.close()

        # Storing the heading of the document in the dictionary called 'zone'
        zone[int(doc['id'])] = str(text).partition('\n\n')[0][1:]

        # Extracting the heading of the document
        zone_text = zone[int(doc['id'])]

        # Making all the text lowercase
        text = text.lower()
        zone_text = zone_text.lower()

        # Replaces punctuations with spaces
        text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
        zone_text = zone_text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))

        # Removes weird punctuations. Add a sapce and symbol you want to replace respectively
        text = text.translate(str.maketrans("‘’’–——−", '       '))
        zone_text = zone_text.translate(str.maketrans("‘’’–——−", '       '))

        # Tokenizing word from the doc and adding it to 'words' dictionary
        words = words + word_tokenize(text)
        zone_words = zone_words + word_tokenize(zone_text)

        # Adding the token stream to a dictionary indexed by doc_id
        tokenised[int(doc['id'])] = word_tokenize(text)
        zone_tokenised[int(doc['id'])] = word_tokenize(zone_text)

        # Eliminating the duplicate words
        words = list(set(words))
        zone_words = list(set(zone_words))

        # Printing progress of processing documents
        print("\r" + "Parsing Progress: Document_id = " + doc['id'] + " : " + zone[int(doc['id'])], end='')
    f.close()

    zone_file = open('./Storage/zone.pkl', 'wb')
    pickle.dump(zone, zone_file)
    zone_file.close()

    doc_no_file = open('./Storage/doc_no.pkl', 'wb')
    pickle.dump(doc_no, doc_no_file)
    doc_no_file.close()

    words_file = open('./Storage/words.pkl', 'wb')
    pickle.dump(words, words_file)
    words_file.close()

    zone_words_file = open('./Storage/zone_words.pkl', 'wb')
    pickle.dump(zone_words, zone_words_file)
    zone_words_file.close()

    zone_file = open('./Storage/zone.pkl', 'wb')
    pickle.dump(zone, zone_file)
    zone_file.close()

    tokeinsed_file = open('./Storage/tokeinsed.pkl', 'wb')
    pickle.dump(tokenised, tokeinsed_file)
    tokeinsed_file.close()

    zone_tokeinsed_file = open('./Storage/zone_tokeinsed.pkl', 'wb')
    pickle.dump(zone_tokenised, zone_tokeinsed_file)
    zone_tokeinsed_file.close()
    print("\nDocuments separated and parsed")

    # Creating empty dataframe
    df = pd.DataFrame(0, index=doc_no, columns=words)
    zone_df = pd.DataFrame(0, index=doc_no, columns=zone_words)

    # Populating Document-Term Frequency Table
    for doc_id, tokenstream in tokenised.items():
        print("\r" + "Populating Document-Term Frequency Table with doc " + str(doc_id), end="")
        for token in tokenstream:
            df[token].loc[doc_id] += 1

    df.to_pickle('./Storage/df.pkl', 'bz2')

    # Populating Zone-Term Frequency Table
    for doc_id, tokenstream in zone_tokenised.items():
        print("\r" + "Populating Zone-Term Frequency Table with doc " + str(doc_id), end="")
        for token in tokenstream:
            zone_df[token].loc[doc_id] += 1

    zone_df.to_pickle('./Storage/zone_df.pkl', 'bz2')
    print("\nPopulating Term-Frequency Table done")

    # Constructing a dictionary containing the term and it's inverse document frequency. Formula: idf=log(N/tf)
    inv_doc_freq = {}
    no_of_docs = len(doc_no)
    for word in words:
        inv_doc_freq[word] = np.log10(no_of_docs / sum(df[word] > 0))

    inv_doc_freq_file = open('./Storage/inv_doc_freq.pkl', 'wb')
    pickle.dump(inv_doc_freq, inv_doc_freq_file)
    inv_doc_freq_file.close()

    # Creating and population a dictionary containg the vector of the documents
    doc_vec = {}
    for doc_id in doc_no:
        # Creating a vector for each document
        vec = (1 + np.log10(np.array(df.loc[doc_id])))  # *list(doc_freq.values())
        # Replacing all -inf values with zeros. -inf reached when we take log of 0
        vec[vec == -np.inf] = 0
        # Normalizing the vector
        vec = vec / (np.sqrt(sum(vec ** 2)))
        # Storing the vector
        doc_vec[doc_id] = vec
        print("\r" + "Document Vector created for doc_no:" + str(doc_id), end="")

    doc_vec_file = bz2.BZ2File('./Storage/doc_vec.pkl', 'w')
    pickle.dump(doc_vec, doc_vec_file)
    doc_vec_file.close()

    # Creating and population a dictionary containg the vector of the documents
    zone_vec = {}
    for doc_id in doc_no:
        # Creating a vector for each document
        vec = (1 + np.log10(np.array(zone_df.loc[doc_id])))  # *list(doc_freq.values())
        # Replacing all -inf values with zeros. -inf reached when we take log of 0
        vec[vec == -np.inf] = 0
        # Normalizing the vector
        vec = vec / (np.sqrt(sum(vec ** 2)))
        # Storing the vector
        zone_vec[doc_id] = vec
        print("\r" + "Zone Vector created for doc_no:" + str(doc_id), end="")

    zone_vec_file = open('./Storage/zone_vec.pkl', 'wb')
    pickle.dump(zone_vec, zone_vec_file)
    zone_vec_file.close()
    print("\nDocument vector creation done")

if __name__ == "__main__":
    location = './Text_corpus/wiki_00'
    corpus_parser(location)
