import pandas as pd
import re
import torch
from transformers import AutoTokenizer, AutoModelWithLMHead
from rouge_score import rouge_scorer

pd.options.mode.chained_assignment = None

import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
import gensim
from gensim import downloader

w2v_path = 'word2vec-google-news-300'
w2v = downloader.load(w2v_path)


def clean_body(text):
    text = re.sub(r'[.]+[\n]+[,]', ".\n", text)
    return text


def clean_headline(summary):
    summary = summary.replace(".,", ".")
    return summary


def t5_summarizer(text):
    tokenizer = AutoTokenizer.from_pretrained("deep-learning-analytics/wikihow-t5-small")
    model = AutoModelWithLMHead.from_pretrained("deep-learning-analytics/wikihow-t5-small")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    tokenized_text = tokenizer.encode(text, return_tensors="pt")
    len = tokenized_text.shape[1]
    if len < 512:
        indices = torch.tensor(range(len))
        tokenized_text = torch.index_select(tokenized_text, 1, indices).to(device)
    else:
        indices = torch.tensor(range(512))
        tokenized_text = torch.index_select(tokenized_text, 1, indices).to(device)

    summary_ids = model.generate(
        tokenized_text,
        max_length=150,
        num_beams=2,
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def read_article(text):
    sentences = []
    for sentence in text.split('.'):
        sentences.append([sentence[i].replace("[^a-zA-Z]", " ") for i in range(len(sentence))])
    sentences.pop()
    return sentences


def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
    all_words = list(set(sent1 + sent2))
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
    return 1 - cosine_distance(vector1, vector2)


def build_similarity_matrix(sentences, stop_words):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))

    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2:  # ignore if both are same sentences
                continue
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix


def tfidf_pagerank_summarizer(text):
    top_n = min(5, len(text.split('.')))
    stop_words = stopwords.words('english')
    summarize_text = []

    # step1: read the text
    sentences = read_article(text)

    # step2: generate similarity matrix across sentences
    sentence_similarity_matrix = build_similarity_matrix(sentences, stop_words)

    # step3: rank sentences in similarity matrix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
    scores = nx.pagerank_numpy(sentence_similarity_graph)
    ranked_sentence = sorted([(scores[i], s, i) for i, s in enumerate(sentences)], key=lambda x: x[0], reverse=True)[
                      :top_n]
    top_n_ordered = sorted(ranked_sentence, key=lambda x: x[2])
    for i in range(len(top_n_ordered)):
        summarize_text.append("".join(top_n_ordered[i][1]))
    return ".".join(summarize_text)


def create_sents_representation(sentences):
    sentence_vectors = []
    for sen in sentences:
        if len(sen) != 0:
            v = np.zeros((300,))
            count = 0
            for w in sen:
                if w in w2v.key_to_index.keys() and w not in stopwords.words('english'):
                    v += w2v[w]
                    count += 1
            v /= count
            sentence_vectors.append(v)
        else:
            v = np.zeros((300,))
            sentence_vectors.append(v)
    return sentence_vectors


def build_similarity_matrix_embd(sentences):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))

    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2:  # ignore if both are same sentences
                continue
            similarity_matrix[idx1][idx2] = 1 - cosine_distance(sentences[idx1], sentences[idx2])

    return similarity_matrix


def embedding_pagerank_summarizer(text, top_n=5, Flag=False):
    top_n = min(5, len(text.split('.')))
    if Flag:
        top_n = int(0.8 * len(text.split('.')))
    stop_words = stopwords.words('english')
    summarize_text = []

    # step1: read the text
    sentences = read_article(text)

    # step2:create embedding representation for each sentence
    sentences_embd = create_sents_representation(sentences)

    # step3:generate similarity matrix across sentences
    sentence_similarity_matrix = build_similarity_matrix_embd(sentences_embd)

    # step4: rank sentences in similarity matrix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
    scores = nx.pagerank_numpy(sentence_similarity_graph)

    # step5: sort the rank and pick top sentences
    ranked_sentences = sorted([(scores[i], s, i) for i, s in enumerate(sentences)], key=lambda x: x[0], reverse=True)[
                       :top_n]

    top_n_ordered = sorted(ranked_sentences, key=lambda x: x[2])

    for i in range(len(top_n_ordered)):
        summarize_text.append("".join(top_n_ordered[i][1]))
    return ".".join(summarize_text)


def hybrid_summarizer(text):
    extractive = embedding_pagerank_summarizer(text, top_n=int(0.3 * len(text.split('.'))), Flag=True)
    abstractive = t5_summarizer(extractive)
    return abstractive


def calc_rouge(vec):
    real_summary = vec[0]
    pred_summary = vec[1]
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    return scorer.score(real_summary, pred_summary)


data = pd.read_csv('/home/student/Desktop/wikihowAll.csv')

data = data.astype(str)

data['clean_text'] = data['text'].apply(clean_body)
data['clean_summary'] = data['headline'].apply(clean_headline)

data['num_of_sents'] = data['clean_text'].apply(lambda x: len(x.split('.')))
data = data.loc[data['num_of_sents'] > 9]  # in order to avoid too-short articles
data = data.sample(frac=1)  # shuffle the dataframe

# Extractive summarization with tf-idf representation
data['pagerank_summary_tfidf'] = data['clean_text'].apply(tfidf_pagerank_summarizer)
data['rouge_score_tfidf'] = data[['clean_summary', 'pagerank_summary_tfidf']].apply(calc_rouge, axis=1)
data[['rouge_score_tfidf']].to_csv('/home/student/tfidf_results.csv')

# Extractive summarization with embedding representation
data['pagerank_summary_embedding'] = data['clean_text'].apply(embedding_pagerank_summarizer)
data['rouge_score_embd'] = data[['clean_summary', 'pagerank_summary_embedding']].apply(calc_rouge, axis=1)
data[['rouge_score_embd']].to_csv('/home/student/embd_results.csv')


# Abstractive summarization with t5 transformer
data['t5_summary'] = data['clean_text'].apply(t5_summarizer)
data['rouge_score_t5'] = data[['clean_summary', 't5_summary']].apply(calc_rouge, axis=1)
data[['rouge_score_t5']].to_csv('/home/student/T5_results.csv')


# Hybrid summarization
data['hybrid_summary'] = data['clean_text'].apply(hybrid_summarizer)
data['rouge_score_hybrid'] = data[['clean_summary', 'hybrid_summary']].apply(calc_rouge, axis=1)
data[['rouge_score_hybrid']].to_csv('/home/student/hybrid_results_03.csv')

# Save all summarization calculated by all models
data.to_csv('/home/student/results.csv')
