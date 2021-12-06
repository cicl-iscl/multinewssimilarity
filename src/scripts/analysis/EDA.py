import jsonlines
import re

from matplotlib.pyplot import hist, show
from src.config import CLEANED_PATH, TRAIN_FILE

if __name__ == '__main__':
    overall_score = []
    article_len = []

    with open(CLEANED_PATH+TRAIN_FILE, 'r') as fp:
        reader = jsonlines.Reader(fp)
        data = reader.iter()
        min_sent_len = 100
        for row in data:
            overall_score.append(row['scores']['overall'])
            len_1 = len(re.split('.|\n', row['n1_data']['text']))
            len_2 = len(re.split('.|\n', row['n2_data']['text']))
            if len_2 <= 1:
                print("sent 2: {}".format(row['n2_data']['text'].strip()))
                print(row['n2_data']['id'])
            if len_1 <= 1:
                print("sent 1: {}".format(row['n1_data']['text'].strip()))
                print(row['n1_data']['id'])
            min_sent_len = min(min(len_2, len_1), min_sent_len)
            article_len.append(len_1)
            article_len.append(len_2)
            # article_len.append(len(row['n1_data']['text'].strip().split(" ")))
            # article_len.append(len(row['n2_data']['text'].strip().split(" ")))
    print(min_sent_len)
    # hist(article_len, range=[0, 200])
    # hist(overall_score)
    # show()
