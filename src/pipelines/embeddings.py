"""
Pipeline to create & store embeddings to disk.
"""
import argparse

from pathlib import Path
from sentence_transformers import SentenceTransformer
from spacy import load
from transformers import AutoTokenizer
from tqdm import tqdm

from src.config import (EmbeddingModels, EmbeddingType, DataType, STORE_PATH,
                        DATA_FILE, CLEANED_PATH)
from src.data import JSONLinesReader, EmbeddingStore
from src.logger import log

nlp = load('xx_ent_wiki_sm')
nlp.add_pipe('sentencizer')


def compute_and_store_embeddings(reader: JSONLinesReader,
                                 model_name: EmbeddingModels,
                                 data_type: DataType,
                                 embedding_type: EmbeddingType = EmbeddingType.all):

    """
    Compute Embeddings given entity type.

    For all, loop through each type, compute embeddings and store in respective directories.
    """

    def store_embeddings(em_type, n1_em, n1_id, n2_em, n2_id):
        store_path = Path(STORE_PATH.format(data_type=data_type.name, embedding_model=model_name.name,
                                            embedding_entity=em_type.name))
        if not store_path.exists():
            store_path.mkdir(parents=True)
        s = EmbeddingStore(store_path)

        s.store(n1_em, n1_id)
        s.store(n2_em, n2_id)

    embedding_types = [type for type in EmbeddingType if type != EmbeddingType.all] if embedding_type == EmbeddingType.all else [embedding_type]
    spacy_tokenizer = lambda data: nlp(data, disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])
    hf_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/"+model_name.value)
    model = SentenceTransformer(model_name.value)

    for p_id, n1_obj, n2_obj, scores in tqdm(reader.get_news_data()):

        """
        For title, compute a single embedding vector of size <1, 512> and store it on disk
        
        For n-sentences, compute n-embedding vectors of size <n, 512> and store them on disk
        
        for start_para, compute a single embedding vector of size <1, 512> and store it on disk
        
        for end_para, compute a single embedding vector of size <1, 512> and store it on disk
        
        for full_para, compute a single embedding vector of size <1, 512> and store it on disk
        
        for topic, TBD
        for summary, TBD
        """

        for e_t in embedding_types:

            if e_t == EmbeddingType.title:
                n1_data = [" ".join(n1_obj.title.strip().split())]
                n2_data = [" ".join(n2_obj.title.strip().split())]
            elif e_t == EmbeddingType.sentences:
                n1_data = [sent.text for sent in spacy_tokenizer(n1_obj.text).sents]
                n2_data = [sent.text for sent in spacy_tokenizer(n2_obj.text).sents]
            elif e_t == EmbeddingType.start_para:
                n1_data = [" ".join(n1_obj.text.strip().split())]
                n2_data = [" ".join(n2_obj.text.strip().split())]
            elif e_t == EmbeddingType.end_para:
                n1_tokens = hf_tokenizer.tokenize(n1_obj.text.strip())
                n2_tokens = hf_tokenizer.tokenize(n2_obj.text.strip())
                n1_data = [hf_tokenizer.convert_tokens_to_string(n1_tokens[-512:])]
                n2_data = [hf_tokenizer.convert_tokens_to_string(n2_tokens[-512:])]
            else:
                continue

            if len(n2_data) > 0 and len(n1_data) > 0:
                n1_embeddings, n2_embeddings = model.encode(n1_data), model.encode(n2_data)
                store_embeddings(e_t, n1_embeddings, n1_obj.id, n2_embeddings, n2_obj.id)

            else:
                log.error(f"{p_id} does not have {e_t.value}.")


if __name__ == "__main__":
    """
    1. Take input of data_type, embedding_model, embedding_type
    2. Calculate embedding for given type 
    3. Store each embeddings
    """
    parser = argparse.ArgumentParser(description='Calculate and Store Embeddings.')
    parser.add_argument('-d', '--data_type', type=DataType.from_string, choices=list(DataType), required=True,
                        help='Train or Test data type')
    parser.add_argument('-em', '--embedding_model', type=EmbeddingModels.from_string, choices=list(EmbeddingModels),
                        required=True, help='')
    parser.add_argument('-et', '--embedding_type', type=EmbeddingType.from_string, choices=list(EmbeddingType),
                        required=True, help='')

    args = parser.parse_args()
    reader_path = (CLEANED_PATH+DATA_FILE).format(data_type=args.data_type)

    reader = JSONLinesReader(reader_path)

    compute_and_store_embeddings(reader, args.embedding_model, args.data_type, args.embedding_type)
