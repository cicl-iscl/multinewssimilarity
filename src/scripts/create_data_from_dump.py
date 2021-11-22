""" Merge data from train_csv and scrapped data into a single jsonl file."""
import jsonlines

from pathlib import Path
from src.logger import log
from src.data import CSVReader, JSONReader, News
from src.config import DUMP_PATH, RAW_TRAIN_FILE, TRAIN_PATH, FINAL_TRAIN_FILE
from tqdm import tqdm

FIELD_TO_INDEX_MAP = {'url1_lang': 0, 'url2_lang': 1, 'pair_id': 2, 'Geography': 7,
                      'Entities': 8, 'Time': 9, 'Narrative': 10, 'Overall': 11,
                      'Style': 12, 'Tone': 13}


def write_to_jsonl(w, data_row):
    w.write(data_row)


def id_to_json_path(pair_id):
    j1_name, j2_name = pair_id.split("_")
    j1_path = DUMP_PATH + 'articles/' + j1_name[-2:] + '/' + j1_name + '.json'
    j2_path = DUMP_PATH + 'articles/' + j2_name[-2:] + '/' + j2_name + '.json'
    return j1_path, j2_path


if __name__ == "__main__":
    if not (Path(TRAIN_PATH).exists() and Path(TRAIN_PATH+RAW_TRAIN_FILE).exists()):
        raise FileNotFoundError(f"Training data does not exist at: {TRAIN_PATH}")
    csv_reader = CSVReader(TRAIN_PATH+RAW_TRAIN_FILE)
    writer = jsonlines.Writer(open(TRAIN_PATH+FINAL_TRAIN_FILE, 'w'))
    empty_pairs = []
    t, processed = tqdm(total=csv_reader.df.shape[0]), 0

    for row in csv_reader.df.itertuples(index=False):
        pair_id = row[FIELD_TO_INDEX_MAP['pair_id']]
        n1_path, n2_path = id_to_json_path(pair_id)
        n1_reader, n2_reader = JSONReader(n1_path, News), JSONReader(n2_path, News)

        if n1_reader.data and n2_reader.data:
            d_row = {'pair_id': pair_id,
                     'n1_data': n1_reader.data.__dict__,
                     'n2_data': n2_reader.data.__dict__,
                     'scores': {'geography': row[FIELD_TO_INDEX_MAP['Geography']],
                                'entities': row[FIELD_TO_INDEX_MAP['Entities']],
                                'time': row[FIELD_TO_INDEX_MAP['Time']],
                                'narrative': row[FIELD_TO_INDEX_MAP['Narrative']],
                                'overall': row[FIELD_TO_INDEX_MAP['Overall']],
                                'style': row[FIELD_TO_INDEX_MAP['Style']],
                                'tone': row[FIELD_TO_INDEX_MAP['Tone']]}
                     }
            write_to_jsonl(writer, d_row)
        else:
            # TODO: Log such pairs to file and create proper log line
            log.error(f"Data doesn't exist for pair {pair_id}")
            empty_pairs.append(pair_id)
        processed += 1
        # t.update(processed)
    writer.close()
    print("Processed pairs: {}".format(processed))
    print("#Empty Pairs: {}".format(len(empty_pairs)))
