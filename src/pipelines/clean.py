"""
Data Cleaning and Filtering pipeline.

Functionalities:
    1. Filter empty text pairs
    2. Remove copyright text
"""


import jsonlines
import re

from src.config import TRAIN_FILE, UNCLEANED_PATH, CLEANED_PATH
from src.data import JSONLinesReader
from src.logger import log

COPYRIGHT_PATTERN = re.compile(r"(\.\s|\n)Copyright[^$]*")


def write_to_jsonl(w, pair_id, n1, n2, s):
    d_row = {'pair_id': pair_id,
             'n1_data': n1.__dict__,
             'n2_data': n2.__dict__,
             'scores': s.__dict__
             }
    w.write(d_row)


def remove_copyright(text):
    result = COPYRIGHT_PATTERN.search(text, 1)
    if result:
        text = re.sub(COPYRIGHT_PATTERN, "", text)
    return text


if __name__ == "__main__":

    reader = JSONLinesReader(UNCLEANED_PATH+TRAIN_FILE)
    writer = jsonlines.Writer(open(CLEANED_PATH+TRAIN_FILE, 'w'))

    for p_id, n1_data, n2_data, scores in reader.get_news_data():
        len_1, len_2 = len(re.split('.|\n', n1_data.text)), len(re.split('.|\n', n2_data.text))
        if len_1 <= 1 or len_2 <= 1:
            log.error(f"pair {p_id} doesn't have enough news content.")
            continue
        n1_data.text = remove_copyright(n1_data.text).strip().replace("\n", "")
        n2_data.text = remove_copyright(n2_data.text).strip().replace("\n", "")
        write_to_jsonl(writer, p_id, n1_data, n2_data, scores)
    writer.close()
