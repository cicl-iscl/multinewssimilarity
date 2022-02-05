"""
Data Cleaning and Filtering pipeline.

Functionalities:
    1. Filter empty text pairs
    2. Remove copyright text
    3. Remove urls
    4. Remove cookies warning and images copyright
    5. Remove stopwords
"""

import argparse
import jsonlines
import pycountry
import re

from nltk import word_tokenize
from nltk.corpus import stopwords
from pathlib import Path

from nltk import word_tokenize
from tqdm import tqdm
from nltk.corpus import stopwords
import pycountry

from src.config import DATA_FILE, UNCLEANED_PATH, CLEANED_PATH, DataType
from src.data import JSONLinesReader
from src.logger import log

COPYRIGHT_PATTERN = re.compile(r"(\.\s|\n)Copyright[^$]*")
URL_PATTERN = re.compile(r"\(?https?:\/\/[^\s\)\）。]*")
COOKIES_PATTERN = re.compile(r"\.[\s\w][^\.\“]*[^,]\s[Cc]ookies([^$](?![a-z][A-Z]))*")
IMAGES_PATTERN = re.compile(r"(\((Image:|Photo by)[^\)]*\))")


def write_to_jsonl(w, pair_id, n1, n2, s):
    if s:
        d_row = {'pair_id': pair_id,
                 'n1_data': n1.__dict__,
                 'n2_data': n2.__dict__,
                 'scores': s.__dict__
                 }
    else:
        d_row = {'pair_id': pair_id,
                 'n1_data': n1.__dict__,
                 'n2_data': n2.__dict__}
    w.write(d_row)


def remove_pattern(text, pattern):
    result = pattern.search(text, 1)
    if result:
        text = re.sub(pattern, "", text)
    return text


def remove_cookies(text):
    result = COOKIES_PATTERN.search(text, 1)
    if result and len(result.group(0)) < 1500:
        pattern = re.compile(r"Chrome")
        r = pattern.search(result.group(0), 1)
        if not r:
            text = re.sub(COOKIES_PATTERN, "", text)
    return text


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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Create single file from the scraped data dump.')
    parser.add_argument('-d', '--data_type', type=DataType.from_string, choices=list(DataType), required=True,
                        help='Train or Test data type')
    args = parser.parse_args()
    IP_FILE = DATA_FILE.format(data_type=args.data_type)
    OP_FILE = DATA_FILE.format(data_type=args.data_type)

    CLEANED_PATH = CLEANED_PATH.format(data_type=args.data_type)
    UNCLEANED_PATH = UNCLEANED_PATH.format(data_type=args.data_type)

    if not Path(CLEANED_PATH).exists():
        Path(CLEANED_PATH).mkdir()

    reader = JSONLinesReader(UNCLEANED_PATH+IP_FILE)
    writer = jsonlines.Writer(open(CLEANED_PATH+OP_FILE, 'w', encoding="utf8"))
    for p_id, n1_data, n2_data, scores in tqdm(reader.get_news_data()):
        len_1, len_2 = len(re.split('.|\n', n1_data.text)), len(re.split('.|\n', n2_data.text))
        if len_1 <= 1 or len_2 <= 1:
            log.error(f"pair {p_id} doesn't have enough news content.")
            continue
        n1_data.text = remove_pattern(n1_data.text, COPYRIGHT_PATTERN).strip().replace("\n", "")
        n2_data.text = remove_pattern(n2_data.text, COPYRIGHT_PATTERN).strip().replace("\n", "")
        n1_data.text = remove_pattern(n1_data.text, URL_PATTERN).strip().replace("\n", "")
        n2_data.text = remove_pattern(n2_data.text, URL_PATTERN).strip().replace("\n", "")
        n1_data.text = remove_cookies(n1_data.text).strip().replace("\n", "")
        n2_data.text = remove_cookies(n2_data.text).strip().replace("\n", "")
        n1_data.text = remove_pattern(n1_data.text, IMAGES_PATTERN).strip().replace("\n", "")
        n2_data.text = remove_pattern(n2_data.text, IMAGES_PATTERN).strip().replace("\n", "")
        write_to_jsonl(writer, p_id, n1_data, n2_data, scores)
    writer.close()
