"""
Pipeline for topic classification
"""
import pycountry
from contextualized_topic_models.models.ctm import CombinedTM
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

from src.config import TRAIN_FILE, CLEANED_PATH, EMBEDDING_DATA_PATH, EMBEDDING_MODEL_TYPE
from src.data import JSONLinesReader, EmbeddingStore


def document_creation(text, lang):
    if lang in documents:
        values = documents[lang] + text
        documents[lang] = values
    else:
        documents[lang] = text


def document_cleaning(docs, lang):
    if lang in stopwords.fileids():
        for text in docs:
            tokens = word_tokenize(text)
            tokens = [word for word in tokens if not word in stopwords.words(lang)]
            text = ' '.join(tokens)

    return docs


def data_preparation(text_for_bow, text_for_contextual):
    vectorizer = CountVectorizer()  # from sklearn

    train_bow_embeddings = vectorizer.fit_transform(text_for_bow)
    vocab = vectorizer.get_feature_names()
    id2token = {k: v for k, v in zip(range(0, len(vocab)), vocab)}

    return train_bow_embeddings, vocab, id2token


def topic_modeling(contextualized_embeddings, bow_embeddings, vocab, id2token):
    qt = TopicModelDataPreparation()

    training_dataset = qt.load(contextualized_embeddings, bow_embeddings, id2token)
    ctm = CombinedTM(bow_size=len(vocab), contextual_size=len(contextualized_embeddings), n_components=5)
    ctm.fit(training_dataset)  # run the model
    ctm.get_topics()


if __name__ == "__main__":
    reader = JSONLinesReader(CLEANED_PATH + TRAIN_FILE)
    documents = {}
    for p_id, n1_data, n2_data, scores in tqdm(reader.get_news_data()):
        #add to a dictionary
        if n2_data.meta_lang == n1_data.meta_lang:
            text = [n1_data.text.strip(), n2_data.text.strip()]
            document_creation(text, n1_data.meta_lang)
        else:
            document_creation([n1_data.text.strip()], n1_data.meta_lang)
            document_creation([n2_data.text.strip()], n2_data.meta_lang)

    #clean texts
    for key in documents:
        #convert language codes for nltk stopwords
        lang = pycountry.languages.get(alpha2=key)
        lang = lang.name.lower()

        #clean documents from stopwords
        documents[key] = document_cleaning(documents[key], lang)

        #create needed elements
        preprocessed_documents, unpreprocessed_corpus, vocab = documents[key].preprocess()
        bow_embeddings, vocab, id2token = data_preparation(preprocessed_documents, unpreprocessed_corpus)
        contextualized_embeddings = EmbeddingStore(EMBEDDING_DATA_PATH+EMBEDDING_MODEL_TYPE).read()
        topic_modeling(contextualized_embeddings, bow_embeddings, vocab, id2token)
"""
en_test= ["On this year Nadal has won several competing prizes, most of them against the famous award-winning sportman Djokovic",]
de_test= ["In diesem Jahr hat Nadal mehrere Wettbewerbe gewonnen, die meisten davon gegen den berühmten, preisgekrönten Sportler Djokovic",]
#documents = [line.strip() for line in open(text_file, encoding="utf-8").readlines()]
documents = [line.strip() for line in en_test]
sp = WhiteSpacePreprocessing(documents, stopwords_language='english')
preprocessed_documents, unpreprocessed_corpus, vocab = sp.preprocess()
#DONE

# since we are doing multilingual topic modeling, we do not need the BoW in
# ZeroShotTM when doing cross-lingual experiments (it does not make sense, since we trained with an english Bow
# to use the spanish BoW)
tp = TopicModelDataPreparation("paraphrase-multilingual-mpnet-base-v2")
training_dataset = tp.fit(text_for_contextual=unpreprocessed_corpus, text_for_bow=preprocessed_documents)

ctm = ZeroShotTM(bow_size=len(tp.vocab), contextual_size=768, n_components=5, num_epochs=20)
ctm.fit(training_dataset)
ctm.get_topic_lists(5)

#testing_dataset = qt.transform(testing_text_for_contextual)

# n_sample how many times to sample the distribution (see the doc)
#ctm.get_doc_topic_distribution(testing_dataset, n_samples=20) # returns a (n_documents, n_topics) matrix with the topic distribution of each document
"""