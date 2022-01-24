"""All kinds of config."""

from enum import Enum


class EmbeddingType(Enum):
    all = 'all'
    title = 'title'
    start_para = 'start_para'
    end_para = 'end_para'
    sentences = 'sentences'
    topics = 'topics'
    summary = 'summary'

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

RAW_FILE = '{data_type}_data.csv'
DATA_FILE = '{data_type}.jsonl'


class WandbParams:
    WANDB_INIT_PARAMS = {
        "project": "SemEval-Task-8",
        "entity": "notsomonk",
    }
    WANDB_MODEL_CONFIG = {
        "learning_rate": 5e-5,
        "epochs": 100,
        "batch_size": 128
    }


# SCORE_CSV = '{}_{}.csv'.format(EMBEDDING_ENTITY, EMBEDDING_MODEL_TYPE)
