"""All kinds of config."""

from enum import Enum
from sklearn.linear_model import LinearRegression


class EmbeddingType(Enum):
    all = 'all'
    title = 'title'
    start_para = 'start_para'
    end_para = 'end_para'
    sentences = 'sentences'
    summary = 'summary'

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s):
        try:
            return EmbeddingType[s]
        except KeyError:
            raise ValueError()


class SimType(Enum):
    sentences_mean = 'sentences_mean'
    sentences_min = 'sentences_min'
    sentences_max = 'sentences_max'
    sentences_med = 'sentences_med'

    title = 'title'

    n1_title_n2_text = 'n1_title_n2_text'
    n2_title_n1_text = 'n2_title_n1_text'

    n1_title_n1_text = 'n1_title_n1_text'
    n2_title_n2_text = 'n2_title_n2_text'

    start_para = 'start_para'
    end_para = 'end_para'

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s):
        try:
            return EmbeddingType[s]
        except KeyError:
            raise ValueError()


class LingFeatureType(Enum):
    ner = 'ner'
    tf_idf = 'tf_idf'
    wmd_dist = 'wmd_dist'

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s):
        try:
            return EmbeddingType[s]
        except KeyError:
            raise ValueError()


class EmbeddingModels(Enum):
    labse = 'setu4993/LaBSE'
    pmpnet = 'paraphrase-multilingual-mpnet-base-v2'

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s):
        try:
            return EmbeddingModels[s]
        except KeyError:
            raise ValueError()


class DataType(Enum):
    train = 'train'
    test = 'test'

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s):
        try:
            return DataType[s]
        except KeyError:
            raise ValueError()


ALLOWED_FILE_TYPES = ['json', 'csv', 'jsonl']
REQUIRED_SCORES = ["geography", "entities", "time", "narrative", "overall"]
model_config = {}

DUMP_PATH = 'data_dump/{data_type}/'
CLEANED_PATH = 'data/{data_type}/'
UNCLEANED_PATH = 'uncleaned_data/{data_type}/'
STORE_PATH = 'data/EStore/{data_type}/{embedding_model}/{embedding_entity}'
RESULT_PATH = 'results/{model_name}/'
MODEL_PATH = 'models/'

RAW_FILE = '{data_type}_data.csv'
DATA_FILE = '{data_type}.jsonl'

INFERENCE_FILE = 'all_sim_scores.csv'


class WandbParams:
    WANDB_INIT_PARAMS = {
        "project": "SemEval-Task-8",
        "entity": "notsomonk",
    }
    WANDB_MODELS_AND_HPARAMS = {
        "lr": {"model": LinearRegression,
               "config_path": {}},
    }


TFIDF_VECTOR = 'models/tfidf_vec/{data_type}.pkl'
FASTTEXT_MODEL = 'models/fasttext/cc.en.300.bin'
# SCORE_CSV = '{}_{}.csv'.format(EMBEDDING_ENTITY, EMBEDDING_MODEL_TYPE)
