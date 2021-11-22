import jsonlines
from matplotlib.pyplot import hist, show
from src.config import TRAIN_PATH, FINAL_TRAIN_FILE

if __name__ == '__main__':
    overall_score = []
    article_len = []

    with open(TRAIN_PATH+FINAL_TRAIN_FILE, 'r') as fp:
        reader = jsonlines.Reader(fp)
        data = reader.iter()
        for row in data:
            overall_score.append(row['scores']['overall'])
            article_len.append(len(row['n1_data']['text'].strip().split(" ")))
            article_len.append(len(row['n2_data']['text'].strip().split(" ")))
    # hist(article_len, range=[0, 2000])
    hist(overall_score)
    show()
