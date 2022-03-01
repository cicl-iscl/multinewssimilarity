import ast
from pandas import read_csv


def build_pair_id_overall_score_dict():

    with open("data/labelled_evaluation.json") as f:
        data = ast.literal_eval(f.read())
    return {itms['pair_id']: itms['Overall'] for itms in data}


if __name__ == "__main__":

    id_overall = build_pair_id_overall_score_dict()
    s_df = read_csv("data/test/all_sim_scores_unlabelled.csv", sep=',')

    overall = []
    for row in s_df.itertuples(index=False):
        overall.append(id_overall[row[0]])

    s_df['overall'] = overall
    s_df.to_csv("data/test/all_sim_scores.csv", mode='w', index=False)
