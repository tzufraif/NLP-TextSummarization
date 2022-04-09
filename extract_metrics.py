import pandas as pd

# Ths code below used in order to reformat the structure of the metrics, before it gets into calculating and plotting.
methods = ['03', '05', '07', '09']
for method in methods:
    df = pd.read_csv(f'/home/student/clean_hybrid_{method}.csv')

    obj = 'tfidf'
    mettics = ['rouge1', 'rouge2', 'rougeL']
    for met in mettics:
        df[f'{met}_precision'] = df[f'{met}'].apply(lambda x: x.split(',')[0])
        df[f'{met}_recall'] = df[f'{met}'].apply(lambda x: x.split(',')[1])
        df[f'{met}_f'] = df[f'{met}'].apply(lambda x: x.split(',')[2])
    df.to_csv(f'/home/student/hybrid_{method}_seperate_rouge.csv')
