import json
import jsonlines
import os
import pathlib

from dacite import from_dict
from dataclasses import dataclass
from numpy import save as n_save, load as n_load
from pandas import read_csv
from typing import Type, Union

from config import ALLOWED_FILE_TYPES
from logger import log


@dataclass
class News:
    id: int
    url: str
    title: str
    keywords: list
    meta_keywords: list
    meta_lang: str
    summary: str
    publish_date: Union[str, None]
    authors: list
    meta_description: str
    source_url: str
    text: str


@dataclass
class MinimalNews:
    id: int
    url: str
    title: str
    text: str


@dataclass
class Scores:
    geography: float
    entities: float
    time: float
    narrative: float
    overall: float
    style: float
    tone: float


class Reader:

    def __init__(self, path: str):
        self.path = pathlib.Path(path)
        self._check_args(path)

    def _check_args(self, *args):
        if args[0].split('.')[-1] not in ALLOWED_FILE_TYPES:
            raise NotImplementedError('{} filetype is not supported.'.format(args[0].split('.')[1]))
        if not pathlib.Path(args[0]).exists():
            raise FileNotFoundError(f'{self.path} does not exist.')


class CSVReader(Reader):

    def __init__(self, path: str):
        super().__init__(path)
        self.df = read_csv(self.path, sep=',')


class JSONReader(Reader):
    def __init__(self, path: str, d_class: Union[Type[News], Type[MinimalNews]]):
        try:
            super().__init__(path)
        except FileNotFoundError:
            log.error(f'Data does not exist for article: {path}')
            self.fobj, self.data, self.id = None, None, None
        else:
            self.fobj = open(self.path, 'r')
            self.id = self._get_id()
            self.data = self._get_data(d_class)

    def _get_data(self, d_class):
        content = self.fobj.read()
        if content.strip('\n') != '':
            # content += f"\n'id' : {self._get_id()}"
            content = content[:-1] + f', "id" : {self._get_id()}' + content[-1]
            return from_dict(d_class, json.loads(content))
        else:
            # TODO: Add logger and log Error
            print(f"Empty file: {self.path}")
            return None

    def _get_id(self):
        return self.path.stem

    def __del__(self):
        if self.fobj:
            self.fobj.close()


class JSONLinesReader(Reader):

    def __init__(self, path: str):
        try:
            super().__init__(path)
        except FileNotFoundError:
            log.error(f'Data does not exist for this pair: {path}')
            self._fobj = None
        else:
            self._fobj = open(self.path, 'r', encoding='utf8')
            self.reader = jsonlines.Reader(self._fobj)

    def get_news_data(self):
        for row in self.reader.iter():
            p_id = row['pair_id']
            n1_data = News(**row['n1_data'])
            n2_data = News(**row['n2_data'])
            scores = Scores(**row['scores'])
            yield p_id, n1_data, n2_data, scores

    def __del__(self):
        if self._fobj:
            self._fobj.close()


class EmbeddingStore(object):

    def __init__(self, path):
        self.path = pathlib.Path(path)
        if not self.path.exists():
            os.makedirs(self.path)

    def store(self, data, news_id):
        log.info("Storing News Embeddings for {} of len: {}".format(news_id, len(data)))
        news_id = str(news_id)
        group_name = news_id[-2:]
        print(group_name)
        group_path = pathlib.Path.joinpath(self.path, group_name)
        if not group_path.exists():
            log.info("News Group {} doesn't exist. Creating News group {}".format(group_name, group_name))
            group_path.mkdir()

        news_path = pathlib.Path.joinpath(group_path, news_id+".npy")
        n_save(news_path, data)
        log.info("News Embeddings stored successfully.")

    def read(self, news_id):
        group_name = news_id[-2:]
        group_path = pathlib.Path.joinpath(self.path, group_name)
        if not group_path.exists() or not group_path.is_dir():
            raise FileNotFoundError("News Group {} doesn't exist.".format(group_name))
        news_path = pathlib.Path.joinpath(group_path, news_id + ".npy")
        if not news_path.exists():
            raise FileNotFoundError("News Embeddings for {} doesn't exist".format(news_id))
        return n_load(news_path)


if __name__ == '__main__':
    # j = JSONReader('data_dump/articles/00/1483733400.json', News)
    # print(j.data.__dict__)

    import numpy as np
    x = np.random.random((5, 10))

    es = EmbeddingStore('data/ES/test')
    es.store(x, 'test_01')
    y = es.read('test_01')
    assert (x == y).all()
