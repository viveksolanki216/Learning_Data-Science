import re

import nltk
import numpy as np
# NLP utilities from the person matching
import pandas as pd
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# Download stopwords list
# nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


def get_dirs(home_dir):
    ''':returns all the relevant directories'''
    return home_dir, f'{home_dir}/input/', f'{home_dir}/output/', f'{home_dir}/plots/', f'{home_dir}/analyses/'


def get_data(in_dir):
    ''':returns train, test and sample submission file'''
    train = pd.read_csv(f'{in_dir}train.csv')
    test = pd.read_csv(f'{in_dir}test.csv')
    sample_sub = pd.read_csv(f'{in_dir}sample_submission.csv')
    return train, test, sample_sub


#
def check_if_vectors_l2_normalized(X):
    print(np.power(X, 2).sum())


# POS Tagging
def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


# Cleaning Textual Data
def preprocess_text(doc_set,  # A List of Strings,
                    pos_tag=False,
                    tokenizer=RegexpTokenizer(r'\w{1,}'),
                    word_selection_patterns=re.compile(r'[a-z]+'),  # Select words that have at-least one alphabet
                    lemmatizer=WordNetLemmatizer(),
                    stop_words=set(stopwords.words('english')),
                    common_words={}):
    doc_set_tokens = []
    doc_set_POS_tags = []
    for i in doc_set:
        # print(i)
        raw = i.lower()
        tokens = tokenizer.tokenize(raw)
        tokens = [i for i in tokens if
                  not i in stop_words.union(common_words)]  # removing stop tokens and general tokens
        tokens = [i for i in tokens if word_selection_patterns.match(i) is not None]  # Keeping tokens with Patterns
        tokens = [lemmatizer.lemmatize(i) for i in tokens]  # Lemmatizing tokens
        doc_set_tokens.append(" ".join(tokens))
        if pos_tag == True:
            POS_tags = pd.DataFrame(nltk.pos_tag(tokens))
            doc_set_POS_tags.append(" ".join(POS_tags))
    return doc_set_tokens, doc_set_POS_tags


# Get Topics for Semantic Analysis
def get_topics(doc_embeddings,
               tokens,
               top_n_comps_to_display=10,
               top_m_tokens_to_display=5):
    '''
    This function returns topics for all the components in the document embeddings
    :param doc_embeddings:  Vectors of all documents
    :param n_comps_to_display: Top N Components to Display
    :return:
    '''
    lst = []
    for index, component in enumerate(doc_embeddings):
        if index >= top_n_comps_to_display:
            break
        zipped = zip(tokens, component)
        top_terms_key = sorted(zipped, key=lambda t: t[1], reverse=True)[:top_m_tokens_to_display]
        top_terms_list = list(dict(top_terms_key).keys())
        # print("Topic "+str(index)+": ",top_terms_list)
        lst.append(", ".join(top_terms_list))
    return pd.DataFrame({"Topics": np.arange(0, top_n_comps_to_display), "Top Terms": lst})


# QA
def get_cosine_similarity_sparse_BOWs(spare_tdm1, sparse_tdm2):
    similarity = np.zeros((spare_tdm1.shape[0], spare_tdm1.shape[0]))
    for i in range(spare_tdm1.shape[0]):
        if i % 1000 == 0:
            print(i)
        for j in range(spare_tdm1.shape[0]):
            similarity[i, j] = (spare_tdm1[i, :].todense() * spare_tdm1[i, :].todense().T)[0, 0]


def compare_tokens_weights(text1, mat1, row1, text2, mat2, row2):
    print(text1[row1])
    print(text2[row2])
    sim, words = get_cosine_similarity_BoWs(
        get_tokens_and_weights_for_a_document_BoWs(mat1, row1),
        get_tokens_and_weights_for_a_document_BoWs(mat2, row2))
    print(sim)
    return words


def get_similarity_score_BoW(doc_list1, doc_list2, doc_list_universe):
    # LSI + Cosine Similarity
    tfidf = TfidfVectorizer(binary=True)

    tdm_universe = tfidf.fit_transform(doc_list_universe)
    tdm1 = tfidf.transform(doc_list1)
    tdm2 = tfidf.transform(doc_list2)

    tokens = tfidf.get_feature_names()
    print(tokens.__len__())

    similarity_bow = np.dot(tdm1, tdm2.T)
    return similarity_bow, tfidf


def get_similarity_score_LSI(doc_list1, doc_list2, doc_list_universe):
    # LSI + Cosine Similarity
    tfidf = TfidfVectorizer(binary=True, max_df=0.7)

    tdm_universe = tfidf.fit_transform(doc_list_universe)
    tdm1 = tfidf.transform(doc_list1)
    tdm2 = tfidf.transform(doc_list2)

    tokens = tfidf.get_feature_names()
    # print(tokens.__len__())

    LSA = TruncatedSVD(n_components=3000, n_iter=2, random_state=42)
    LSA.fit(tdm_universe)
    print(LSA.explained_variance_ratio_.sum())
    tdm1_lsi = normalize(LSA.transform(tdm1))  # L2 Normalize
    tdm2_lsi = normalize(LSA.transform(tdm2))

    similarity_sementic = np.dot(tdm1_lsi, tdm2_lsi.T)

    return similarity_sementic


def get_tokens_and_weights_for_a_document_BoWs(sparse_tdm, row_number):
    '''
    This function takes to Scipy Spare Matrix and a row number (a document) and returns the tokens and their weights
    :param sparse_mat: Term Document Matrix (spare matrix) BoW
    :param row_number: Provide the Row number for the vector/document
    :return: Set of tokens in that document with the scores
    '''
    cx = coo_matrix(sparse_tdm)
    flags = cx.row == row_number
    temp = pd.DataFrame({'tokens': [], 'weights': []}, )
    for i, j, v in zip(cx.row[flags], cx.col[flags], cx.data[flags]):
        temp = pd.concat([temp, pd.DataFrame({'tokens': terms[j], 'weights': v}, index=[0])])
        # print("(%d, %s), %s" % (i,j,v))
    return temp.set_index('tokens')


def get_cosine_similarity_BoWs(tokens1, tokens2):
    '''
    :param tokens1: Output of "get_tokens_and_weights_for_a_document_BoWs"
    :param tokens2: Output of "get_tokens_and_weights_for_a_document_BoWs"
    :return: Cosine Similarity, And DataFrame with tokens and thier weights put together for both the documnets to analyse
    '''
    temp = pd.merge(tokens1, tokens2, on='tokens', how='outer')
    similarity = np.sum(temp['weights_x'] * temp['weights_y'])
    print("similarity", similarity)
    return similarity, temp
