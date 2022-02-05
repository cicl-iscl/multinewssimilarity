"""
Use all the generated embeddings of different types to calculate similarity score per entity.
"""
import argparse
import pickle
import pycountry

from gensim.models.fasttext import load_facebook_vectors
from nltk import word_tokenize
from nltk.corpus import stopwords
from numpy import mean, min, max, median
from pandas import DataFrame
from pathlib import Path
from spacy import load
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import util

from tqdm import tqdm

from src.config import (EmbeddingType, CLEANED_PATH, DATA_FILE, DataType, EmbeddingModels, STORE_PATH,
                        REQUIRED_SCORES, LingFeatureType, SimType, TFIDF_VECTOR, FASTTEXT_MODEL)
from src.data import JSONLinesReader, EmbeddingStore
from src.logger import log


print("Loading models...")
nlp = load('xx_ent_wiki_sm')
spacy_tokenizer = lambda data: nlp(data, disable=["tok2vec", "parser", "attribute_ruler", "lemmatizer"])
fasttext_model = load_facebook_vectors(FASTTEXT_MODEL)
print("Model loaded.")


def remove_stopwords(text, lang):
    #convert language code to nltk
    if len(lang) > 0:
        lang = pycountry.languages.get(alpha_2=lang)
        lang = lang.name.lower()

    if lang in stopwords.fileids():
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in stopwords.words(lang)]
        text = ' '.join(tokens)
    return text


def _calculate_mean_sent_similarity(source_embeddings, target_embeddings):
    results = util.semantic_search(source_embeddings, target_embeddings, top_k=1)
    return mean([res[0]['score'] for res in results])


def _calculate_median_sent_similarity(source_embeddings, target_embeddings):
    results = util.semantic_search(source_embeddings, target_embeddings, top_k=1)
    return median([res[0]['score'] for res in results])


def _calculate_min_sent_similarity(source_embeddings, target_embeddings):
    results = util.semantic_search(source_embeddings, target_embeddings, top_k=1)
    return min([res[0]['score'] for res in results])


def _calculate_max_sent_similarity(source_embeddings, target_embeddings):
    results = util.semantic_search(source_embeddings, target_embeddings, top_k=1)
    return max([res[0]['score'] for res in results])


def _calculate_cos_similarity(source_embeddings, target_embeddings):
    return float(util.cos_sim(source_embeddings, target_embeddings)[0][0])


def _calculate_ner_similarity(source_obj, target_obj):
    n1_data = spacy_tokenizer(source_obj.text)
    n2_data = spacy_tokenizer(target_obj.text)

    n1_ents = set(map(str.lower, map(str, set(n1_data.ents))))
    n2_ents = set(map(str.lower, map(str, set(n2_data.ents))))

    return len(n1_ents.intersection(n2_ents))/max([len(n1_ents), len(n2_ents)]) if len(n1_ents) and len(n2_ents) else 0.0


def _calculate_tfidf_similarity(source_obj, target_obj):
    n1_text = remove_stopwords(source_obj.text.strip(), source_obj.meta_lang)
    n2_text = remove_stopwords(target_obj.text.strip(), target_obj.meta_lang)
    q1_tfidf = vectorizer.transform([n1_text])
    q2_tfidf = vectorizer.transform([n2_text])
    return cosine_similarity(q1_tfidf, q2_tfidf).flatten()[0]


def _calculate_wmd_similarity(source_obj, target_obj):
    n1_text = remove_stopwords(source_obj.text.strip(), source_obj.meta_lang)
    n2_text = remove_stopwords(target_obj.text.strip(), target_obj.meta_lang)

    return fasttext_model.wmdistance(n1_text, n2_text)


def get_score_values(scores):
    return [scores.__dict__[s] for s in REQUIRED_SCORES]


def calculate_similarities(reader: JSONLinesReader,
                           data_type: DataType,
                           embedding_model: EmbeddingModels):
    """
    1. loop through data
    2. Decide which sim_score needs to be calculated
    4. Get appropriate embedding store for this sim score
    5. Get embeddings for both entities
    6. calculate sim score between entities
    """
    sim_scores = [["pair_id"] + [sim_score.name for sim_score in sim_type_func_map.keys()] +
                  [ling_feature.name for ling_feature in ling_feature_sim_func_map]]
    if data_type.name == 'train':
        sim_scores[0] += REQUIRED_SCORES
    for p_id, n1_data, n2_data, scores in tqdm(reader.get_news_data()):
        n1_id, n2_id = p_id.split('_')
        p_id_sim_score = [p_id]

        for sim_score in sim_type_func_map.keys():
            e1_t, e2_t = sim_type_to_embedding_store_map[sim_score]

            s1_path = Path(STORE_PATH.format(data_type=data_type.name, embedding_model=embedding_model.name,
                                             embedding_entity=e1_t.name))
            s2_path = Path(STORE_PATH.format(data_type=data_type.name, embedding_model=embedding_model.name,
                                             embedding_entity=e2_t.name))
            s1, s2 = EmbeddingStore(s1_path), EmbeddingStore(s2_path)

            try:
                if sim_score not in [SimType.n1_title_n1_text, SimType.n2_title_n2_text]:
                    n1_embeddings, n2_embeddings = s1.read(n1_id), s2.read(n2_id)
                else:
                    n1_embeddings, n2_embeddings = s1.read(n1_id), s2.read(n1_id)
            except FileNotFoundError:
                log.error(f"Couldn't retrieve {e1_t.name} embeddings for {n1_id} or {e2_t.name} for {n2_id}")
                break

            p_id_sim_score.append(sim_type_func_map[sim_score](n1_embeddings, n2_embeddings))
        else:
            # Add lexical similarity features
            p_id_sim_score.extend([sim_func(n1_data, n2_data) for _, sim_func in ling_feature_sim_func_map.items()])
            sim_scores.append(p_id_sim_score+get_score_values(scores) if scores else p_id_sim_score)
    return sim_scores


sim_type_func_map = {
    SimType.sentences_mean: _calculate_mean_sent_similarity,
    SimType.sentences_min: _calculate_min_sent_similarity,
    SimType.sentences_max: _calculate_max_sent_similarity,
    SimType.sentences_med: _calculate_median_sent_similarity,

    SimType.title: _calculate_cos_similarity,

    SimType.n1_title_n2_text: _calculate_cos_similarity,
    SimType.n2_title_n1_text: _calculate_cos_similarity,
    SimType.n1_title_n1_text: _calculate_cos_similarity,
    SimType.n2_title_n2_text: _calculate_cos_similarity,

    SimType.start_para: _calculate_cos_similarity,
    SimType.end_para: _calculate_cos_similarity,
}

sim_type_to_embedding_store_map = {
    SimType.sentences_mean: [EmbeddingType.sentences, EmbeddingType.sentences],
    SimType.sentences_min: [EmbeddingType.sentences, EmbeddingType.sentences],
    SimType.sentences_max: [EmbeddingType.sentences, EmbeddingType.sentences],
    SimType.sentences_med: [EmbeddingType.sentences, EmbeddingType.sentences],

    SimType.title: [EmbeddingType.title, EmbeddingType.title],

    SimType.n1_title_n2_text: [EmbeddingType.title, EmbeddingType.start_para],
    SimType.n2_title_n1_text: [EmbeddingType.title, EmbeddingType.start_para],
    SimType.n1_title_n1_text: [EmbeddingType.title, EmbeddingType.start_para],
    SimType.n2_title_n2_text: [EmbeddingType.title, EmbeddingType.start_para],

    SimType.start_para: [EmbeddingType.start_para, EmbeddingType.start_para],
    SimType.end_para: [EmbeddingType.end_para, EmbeddingType.end_para]
}

ling_feature_sim_func_map = {
    LingFeatureType.ner: _calculate_ner_similarity,
    LingFeatureType.tf_idf: _calculate_tfidf_similarity,
    LingFeatureType.wmd_dist: _calculate_wmd_similarity
}


def store_sim_scores(out_path, data):
    df = DataFrame.from_records(data[1:], columns=data[0])
    df.to_csv(out_path, sep=',', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate and Store Embeddings.')
    parser.add_argument('-d', '--data_type', type=DataType.from_string, choices=list(DataType), required=True,
                        help='Train or Test data type')
    parser.add_argument('-em', '--embedding_model', type=EmbeddingModels.from_string, choices=list(EmbeddingModels),
                        required=True, help='Embedding model enum')
    parser.add_argument('-out', '--output_file', type=str,
                        required=True, help='Output file path')
    args = parser.parse_args()

    reader_path = (CLEANED_PATH + DATA_FILE).format(data_type=args.data_type)
    vectorizer = pickle.load(open(TFIDF_VECTOR.format(data_type=args.data_type.name), "rb"))

    reader = JSONLinesReader(reader_path)
    sim_scores = calculate_similarities(reader, args.data_type, args.embedding_model)
    store_sim_scores(args.output_file, sim_scores)
