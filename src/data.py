import json
import pathlib

from dataclasses import dataclass
from dacite import from_dict
from pandas import read_csv
from typing import Type, Union

from src.config import ALLOWED_FILE_TYPES
from src.logger import log


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


class Reader:

    def __init__(self, path: str):
        self.path = pathlib.Path(path)
        self._check_args(path)

    def _check_args(self, *args):
        if args[0].split('.')[1] not in ALLOWED_FILE_TYPES:
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


if __name__ == '__main__':
    j = JSONReader('data_dump/articles/00/1483733400.json', News)
    print(j.data.__dict__)