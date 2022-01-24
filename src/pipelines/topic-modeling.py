"""
Pipeline for topic classification
"""
from contextualized_topic_models.models import ctm
from contextualized_topic_models.utils.preprocessing import WhiteSpacePreprocessing, WhiteSpacePreprocessingStopwords

from config import TRAIN_FILE, CLEANED_PATH, EMBEDDING_DATA_PATH, EMBEDDING_MODEL_TYPE
from data import JSONLinesReader, EmbeddingStore
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pycountry

from contextualized_topic_models.models.ctm import ZeroShotTM
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
from nltk.corpus import stopwords


def document_creation(text, lang):
    if lang in documents:
        values = documents[lang] + text
        documents[lang] = values
    else:
        documents[lang] = text


def document_cleaning(docs, lang):
    if lang in stopwords.fileids():
        sp = WhiteSpacePreprocessing(docs, stopwords_language=lang)
    else:
        sp = WhiteSpacePreprocessingStopwords(docs)
    return sp

def data_preparation(text_for_bow):
    vectorizer = CountVectorizer()  # from sklearn

    train_bow_embeddings = vectorizer.fit_transform(text_for_bow)
    vocab = vectorizer.get_feature_names()
    id2token = {k: v for k, v in zip(range(0, len(vocab)), vocab)}

    return train_bow_embeddings, vocab, id2token


def topic_modeling(contextualized_embeddings, bow_embeddings, vocab, id2token):
    training_dataset = qt.load(contextualized_embeddings, bow_embeddings, id2token)
    ctm = ZeroShotTM(bow_size=len(vocab), contextual_size=len(vocab), n_components=50, num_epochs=20)
    ctm.fit(training_dataset)  # run the model
    ctm.get_topics(5)


if __name__ == "__main__":
    reader = JSONLinesReader(CLEANED_PATH + TRAIN_FILE)
    result = list(tqdm(reader.get_news_data()))
    train, test = train_test_split(result, test_size=0.3)
    documents = {}
    train_em = []
    for p_id, n1_data, n2_data, scores in train:
        #add to a dictionary
        if n2_data.meta_lang == n1_data.meta_lang:
            text = [n1_data.text.strip(), n2_data.text.strip()]
            document_creation(text, n1_data.meta_lang)
        else:
            document_creation([n1_data.text.strip()], n1_data.meta_lang)
            document_creation([n2_data.text.strip()], n2_data.meta_lang)
        #store contextualized embeddings
        p_id1, p_id2 = p_id.split("_")
        try:
            train_em.append(EmbeddingStore(EMBEDDING_DATA_PATH).read(p_id1))
            train_em.append(EmbeddingStore(EMBEDDING_DATA_PATH).read(p_id2))
        except:
            print("FileNotFoundError")

    #clean texts
    for key in documents:
        #convert language codes for nltk stopwords
        lang = pycountry.languages.get(alpha_2=key)
        if lang != None:
            lang = lang.name.lower()
        else:
            lang = key
        #clean documents from stopwords
        documents[key] = document_cleaning(documents[key], lang)

    #create needed elements
    preprocessed_documents, unpreprocessed_corpus, vocab = documents["en"].preprocess()
    for key in documents:
        if key != "en":
            pd, uc, v = documents[key].preprocess()
            preprocessed_documents += pd
            unpreprocessed_corpus += uc
            vocab += v

    bow_embeddings, vocab, id2token = data_preparation(preprocessed_documents)
    qt = TopicModelDataPreparation()
    topic_modeling(train_em, bow_embeddings, vocab, id2token)

    #try with testing dataset
    test_em = []
    for p_id, _, _, _ in tqdm(test):
        p_id1, p_id2 = p_id.split("_")
        try:
            test_em.append(EmbeddingStore(EMBEDDING_DATA_PATH + EMBEDDING_MODEL_TYPE).read(p_id1))
            test_em.append(EmbeddingStore(EMBEDDING_DATA_PATH + EMBEDDING_MODEL_TYPE).read(p_id2))
        except:
            print("FileNotFoundError")
    testing_dataset = qt.load(test_em)
    ctm.get_doc_topic_distribution(testing_dataset, n_samples=20)
