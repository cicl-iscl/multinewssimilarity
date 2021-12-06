"""
Pipeline to create & store embeddings to disk.
"""
import csv

from numpy import mean
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from spacy import load
from tqdm import tqdm

from src.config import EMBEDDING_MODEL, EMBEDDING_MODEL_TYPE, EMBEDDING_DATA_PATH, TRAIN_FILE, CLEANED_PATH
from src.data import JSONLinesReader, EmbeddingStore

nlp = load('xx_ent_wiki_sm')
nlp.add_pipe('sentencizer')


def calculate_overall_similarity(source_embeddings, target_embeddings):
    results = util.semantic_search(source_embeddings, target_embeddings, top_k=1)
    return mean([res[0]['score'] for res in results])*4


def store_embeddings(s, n1_em, n1_id, n2_em, n2_id):
    s.store(n1_em, n1_id)
    s.store(n2_em, n2_id)


if __name__ == "__main__":
    reader = JSONLinesReader(CLEANED_PATH+TRAIN_FILE)
    model = SentenceTransformer(EMBEDDING_MODEL)

    store_path = Path(EMBEDDING_DATA_PATH+EMBEDDING_MODEL_TYPE)
    if not store_path.exists():
        store_path.mkdir()
    e_store = EmbeddingStore(store_path)

    with open('early.csv', 'w') as csvfile:
        w = csv.writer(csvfile, delimiter=',')
        w.writerow(['pair_id', 'original_score', 'computed_score'])

        for p_id, n1_data, n2_data, scores in tqdm(reader.get_news_data()):
            n1_doc = nlp(n1_data.text, disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])
            n2_doc = nlp(n2_data.text, disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])
            n1_sents, n2_sents = [sent.text for sent in n1_doc.sents], [sent.text for sent in n2_doc.sents]
            n1_embeddings, n2_embeddings = model.encode(n1_sents), model.encode(n2_sents)
            store_embeddings(e_store, n1_embeddings, n1_data.id, n2_embeddings, n2_data.id)

            cal_sim = calculate_overall_similarity(n1_embeddings, n2_embeddings)

            w.writerow([p_id, scores.overall, cal_sim])
