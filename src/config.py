"""All kinds of config."""


model_config = {}

ALLOWED_FILE_TYPES = ['json', 'csv', 'jsonl']

DUMP_PATH = '/home/monk/Projects/UT/multinewssimilarity/data_dump/'
CLEANED_PATH = '/home/monk/Projects/UT/multinewssimilarity/data/'
UNCLEANED_PATH = '/home/monk/Projects/UT/multinewssimilarity/uncleaned_data/'

RAW_TRAIN_FILE = 'train_data.csv'
TRAIN_FILE = 'train.jsonl'

RAW_TEST_FILE = 'test_data.csv'
TEST_FILE = 'test.jsonl'


EMBEDDING_MODEL = 'paraphrase-multilingual-mpnet-base-v2'
EMBEDDING_MODEL_TYPE = 'pMPNet'

EMBEDDING_ENTITY = 'title'
# EMBEDDING_ENTITY = 'text'
# EMBEDDING_MODEL = 'setu4993/LaBSE'
# EMBEDDING_MODEL_TYPE = 'LaBSE'
EMBEDDING_DATA_PATH = f"/home/monk/Projects/UT/multinewssimilarity/data/EStore/{EMBEDDING_MODEL_TYPE}/{EMBEDDING_ENTITY}"


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


SCORE_CSV = '{}_{}.csv'.format(EMBEDDING_ENTITY, EMBEDDING_MODEL_TYPE)
