"""
Use all the generated embeddings of different types to calculate similarity score per entity.
"""
import argparse
import os

from numpy import mean
from pandas import DataFrame
from pathlib import Path
from tqdm import tqdm

from sentence_transformers import util

from src.config import (EmbeddingType, CLEANED_PATH, DATA_FILE, DataType, EmbeddingModels, STORE_PATH,
                        REQUIRED_SCORES)
from src.data import JSONLinesReader, EmbeddingStore
from src.logger import log


def _calculate_sent_similarity(source_embeddings, target_embeddings):
    results = util.semantic_search(source_embeddings, target_embeddings, top_k=1)
    return mean([res[0]['score'] for res in results])


def _calculate_cos_similarity(source_embeddings, target_embeddings):
    return float(util.cos_sim(source_embeddings, target_embeddings)[0][0])


def _calculate_topics_similarity(source_embeddings, target_embeddings):
    return None


def get_score_values(scores):
    return [scores.__dict__[s] for s in REQUIRED_SCORES]


def calculate_similarities(reader: JSONLinesReader,
                           embedding_types: [EmbeddingType],
                           data_type: DataType,
                           embedding_model: EmbeddingModels):

    sim_scores = [["pair_id"]+[e_t.name for e_t in embedding_types]]
    if data_type.name == 'train':
        sim_scores[0] += REQUIRED_SCORES
    for p_id, _, _, scores in tqdm(reader.get_news_data()):
        n1_id, n2_id = p_id.split('_')
        p_id_sim_score = [p_id]

        for e_t in embedding_types:
            store_path = Path(STORE_PATH.format(data_type=data_type.name, embedding_model=embedding_model.name,
                                                embedding_entity=e_t.name))
            s = EmbeddingStore(store_path)
            try:
                n1_embeddings, n2_embeddings = s.read(n1_id), s.read(n2_id)
            except FileNotFoundError:
                log.error(f"Couldn't retrieve {e_t.name} embeddings for : {p_id}")
                continue

            p_id_sim_score.append(entitity_sim_func_map[e_t](n1_embeddings, n2_embeddings))
        if len(p_id_sim_score) == len(embedding_types) + 1:
            sim_scores.append(p_id_sim_score+get_score_values(scores) if scores else p_id_sim_score)

    return sim_scores


entitity_sim_func_map = {
    # TODO: Get rid of dict
    EmbeddingType.sentences: _calculate_sent_similarity,
    EmbeddingType.title: _calculate_cos_similarity,
    EmbeddingType.start_para: _calculate_cos_similarity,
    EmbeddingType.end_para: _calculate_cos_similarity,
    EmbeddingType.topics: _calculate_topics_similarity,
    EmbeddingType.summary: _calculate_cos_similarity
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

    embedding_entities = os.listdir(STORE_PATH.format(data_type=args.data_type,
                                                      embedding_model=args.embedding_model,
                                                      embedding_entity=''))
    embedding_entities = [EmbeddingType.from_string(e_t) for e_t in embedding_entities]

    reader = JSONLinesReader(reader_path)

    sim_scores = calculate_similarities(reader, embedding_entities, args.data_type, args.embedding_model)
    store_sim_scores(args.output_file, sim_scores)
