import plac
import pandas as pd
import numpy as np
import random
import os
from sklearn.preprocessing import LabelEncoder

from cfg import *


def npsave(path, arr: np.ndarray):
    print("%s: %s" % (path, arr.shape))

    if not os.path.exists('data'):
        os.mkdir('data')

    np.save(os.path.join('data', path), arr)


@plac.annotations(
    tvsplit=('Percentage of training data', 'option', 'v', float),
)
def main(tvsplit: float = 0.8):

    random.seed(0)
    np.random.seed(0)

    df_maccs = pd.read_csv('fingerprint.csv')
    df_train = pd.read_csv('train.csv')
    df_cell = pd.read_csv('gene_expression.csv')
    df_profiles = pd.read_csv('profiles.csv')

    le = LabelEncoder()
    le.fit(SMILES_SYMBOLS)

    cells = []

    labels_total = 0
    for cell in df_train['cell'].unique():
        X = []
        y = []

        for ridx, row in df_train[df_train['cell'] == cell].iterrows():
            # Get Cell Lines
            lines = df_cell[df_cell['cname'] == cell].iloc[0][LINES_LABELS].to_list()
            lines = np.pad(lines, [0, SMILES_LEN - len(lines)])
            lines = lines[..., np.newaxis]

            # Get MACCS
            maccs = [float(x) for x in
                     df_maccs[df_maccs['dname'] == row['drug']].iloc[0]['fps'][1:-1].replace(" ", "").split(",")]
            maccs = np.pad(maccs, [0, SMILES_LEN - len(maccs)])
            maccs = maccs[..., np.newaxis]

            # Get SMILES
            profile = df_profiles[df_cell['cname'] == row['cell']].iloc[0]
            smiles = profile['smiles']
            for key in SMILES_REPLACE:
                smiles = smiles.replace(key, SMILES_REPLACE[key])
            smiles = [c for c in smiles]
            smiles = le.transform(smiles)

            if len(smiles) < SMILES_LEN:
                smiles = np.pad(smiles, [0, SMILES_LEN - len(smiles)])
            elif len(smiles > SMILES_LEN):
                smiles = smiles[0:SMILES_LEN]
            smiles = smiles[..., np.newaxis]

            X.append(np.concatenate([smiles, lines, maccs], axis=1))

            y.append(row['ri'])

            labels_total += 1

        cells.append({
            'X': X,
            'y': y,
            'labels': len(y)
        })

    # Shuffle cells
    random.shuffle(cells)

    train = {'maccs': [], 'y': []}
    val = {'maccs': [], 'y': []}

    labels_fed = 0
    while labels_fed < tvsplit*labels_total:
        cell = cells.pop()
        labels_fed += cell['labels']
        del cell['labels']
        for key in cell:
            train[key].extend(cell[key])

    indices = np.arange(len(train['X']))
    np.random.shuffle(indices)

    for key in train:
        train[key] = np.array(key)
        train[key] = train[key][indices]
        npsave('%s_train.npy' % key, train[key])
        del train[key]

    while len(cells) > 0:

        cell = cells.pop()
        labels_fed += cell['labels']
        del cell['labels']
        for key in cell:
            val[key].extend(cell[key])

    indices = np.arange(len(val['X']))
    np.random.shuffle(indices)

    for key in val:
        val[key] = np.array(key)
        val[key] = val[key][indices]
        npsave('%s_val.npy' % key, val[key])
        del val[key]


if __name__ == '__main__':
    plac.call(main)
