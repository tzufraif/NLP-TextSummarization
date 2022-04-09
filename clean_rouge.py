import pandas as pd
import json


# All functions here used to parser the rouge metric from its complex origin structure.
def calc_rouge1(text):
    res = json.loads(text.replace("'", '"').replace(":", ':"').replace("),", ')",').replace(")}", ')"}'))
    splitted = res['rouge1'].split()
    results = []
    for i in range(3):
        if i == 0:
            results.append(float(splitted[i][16:-1]))
        elif i == 1:
            results.append(float(splitted[i][7:-1]))
        else:
            results.append(float(splitted[i][9:-1]))

    return results


def calc_rouge2(text):
    res = json.loads(text.replace("'", '"').replace(":", ':"').replace("),", ')",').replace(")}", ')"}'))
    splitted = res['rouge2'].split()
    results = []
    for i in range(3):
        if i == 0:
            results.append(splitted[i][16:-1])
        elif i == 1:
            results.append(splitted[i][7:-1])
        else:
            results.append(splitted[i][9:-1])

    return results


def calc_rougeL(text):
    res = json.loads(text.replace("'", '"').replace(":", ':"').replace("),", ')",').replace(")}", ')"}'))

    splitted = res['rougeL'].split()
    results = []
    for i in range(3):
        if i == 0:
            results.append(splitted[i][16:-1])
        elif i == 1:
            results.append(splitted[i][7:-1])
        else:
            results.append(splitted[i][9:-1])

    return results


df = pd.read_csv('/home/student/hybrid_results_09.csv')
df['rouge1'] = df['rouge_score_hybrid'].apply(calc_rouge1)
df['rouge2'] = df['rouge_score_hybrid'].apply(calc_rouge2)
df['rougeL'] = df['rouge_score_hybrid'].apply(calc_rougeL)
df.to_csv('/home/student/clean_hybrid_09.csv')
