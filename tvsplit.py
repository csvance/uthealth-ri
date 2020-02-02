import plac
import pandas as pd
import numpy as np
import random
import os
from rdkit import Chem
from feature import mol_to_feature
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from cfg import *
import pickle


def npsave(path, arr: np.ndarray):
    print("%s: %s" % (path, arr.shape))

    if not os.path.exists(TV_PATH):
        os.mkdir(TV_PATH)

    np.save(os.path.join(TV_PATH, path), arr)


@plac.annotations(
    tvsplit=('Percentage of training data', 'option', 'v', float),
    seed=('Random seed', 'option', 's', int),
    verbose=('Verbose output', 'option', 'V', int)
)
def main(tvsplit: float = 0.8,
         seed: int = 0,
         verbose: int = 0):
    random.seed(seed)
    np.random.seed(seed)

    target_nnf = pickle.loads(open(os.path.join(OBJ_PATH, 'target_gene_nnf.pickle'), 'rb').read())
    target_le = pickle.loads(open(os.path.join(OBJ_PATH, 'target_gene_labelencoder.pickle'), 'rb').read())
    lines_nnf = pickle.loads(open(os.path.join(OBJ_PATH, 'gene_expression_nnf.pickle'), 'rb').read())

    def run(stage: str = 'train'):

        last_lines = None

        print("Load gene expressions...")
        df_cell = pickle.loads(open(os.path.join(CSV_PATH, 'cell', 'gene_expression.pkl'), 'rb').read())
        print("Loading MACCS...")
        df_maccs = pickle.loads(open(os.path.join(CSV_PATH, 'drug', 'fingerprint.pkl'), 'rb').read())
        print("Loading profiles")
        df_profiles = pickle.loads(open(os.path.join(CSV_PATH, 'drug', 'profiles.pkl'), 'rb').read())
        print("Loading target genes...")
        df_target = pickle.loads(open(os.path.join(CSV_PATH, 'drug', 'target_gene.pkl'), 'rb').read())

        if stage == 'train':
            print("Loading training data...")
            df_train = pd.read_csv(os.path.join(CSV_PATH, 'train.csv'))

            # Attempt to balance classes
            df_train_pos = df_train[df_train['ri'] == 1]
            df_train_neg = df_train[df_train['ri'] == 0].sample(df_train_pos.shape[0])

            df_train = pd.concat([df_train_pos, df_train_neg], axis=0)
        elif stage == 'test':
            print("Loading test data...")
            df_train = pd.read_csv(os.path.join(CSV_PATH, 'test.csv'))

        cells = []

        labels_total = 0

        print("Transforming cell lines...")
        lines_all = df_cell.iloc[:, 1:].values.astype(np.float32)
        lines_all = lines_nnf.transform(lines_all)

        for cidx, cell in enumerate(df_train['cell'].unique()):

            # Get Cell Lines
            dfs_idx = df_cell.index[df_cell['cname'] == cell].to_list()
            if len(dfs_idx) != 1:
                print("Missing Cell: %s" % cell)
                if stage == 'test':
                    lines = last_lines
                else:
                    continue
            else:
                lines = lines_all[dfs_idx[0]]
                last_lines = lines

            X = []
            y = []
            dname = []
            cname = []

            print("Cell:\t%s" % cell)

            for ridx, row in df_train[df_train['cell'] == cell].iterrows():

                features = [lines]

                if MACCS_ENABLE:
                    # Get MACCS
                    dfs = df_maccs[df_maccs['dname'] == row['drug']]
                    if len(dfs) == 1:
                        maccs = df_maccs[df_maccs['dname'] == row['drug']].iloc[0]['fps']
                        maccs = np.array(maccs).astype(np.int32)
                        features.append(maccs)
                    else:
                        if verbose > 0:
                            print("Missing MACCS for %s" % row['drug'])
                        continue

                if TARGET_ENABLE:
                    # Get target genes
                    dfs = df_target[df_target['dname'] == row['drug']]
                    if len(dfs) == 1:
                        target = np.zeros((1, len(target_le.classes_, )))
                        for tidx, trow in dfs.iterrows():
                            x = target_le.transform(trow['gene'])
                            for xi in x:
                                target[0, xi] = 1
                        target = target_nnf.transform(target)[0]
                        features.append(target)
                    else:
                        if verbose > 0:
                            print("Missing target genes")

                if SMILES_ENABLE:
                    # Get SMILES
                    dfs = df_profiles[df_profiles['dname'] == row['drug']]
                    if len(dfs) > 0:
                        profile = dfs.iloc[0]
                        smiles = profile['smiles']
                        mol = Chem.MolFromSmiles(smiles)
                        smiles = np.array(mol_to_feature(mol, -1, 400))
                        features.append(smiles)
                    else:
                        print("Missing SMILES for %s" % row['drug'])
                        continue

                X.append(np.concatenate(features, axis=0))

                cname.append(str(cell))
                dname.append(str(row['drug']))

                if stage == 'train':
                    y.append(np.array(row['ri'])[..., np.newaxis])

                labels_total += 1

            cdict = {
                'X': X,
                'labels': len(X),
                'cname': cname,
                'dname': dname
            }

            if stage == 'train':
                cdict['y'] = y

            cells.append(cdict)

        if stage == 'train':
            # Shuffle cells
            random.shuffle(cells)

            train = {'X': [], 'y': [], 'cname': [], 'dname': []}
            val = {'X': [], 'y': [], 'cname': [], 'dname': []}

            labels_fed = 0
            while labels_fed < tvsplit * labels_total:

                print("%d < %f : %d left" % (labels_fed, tvsplit * labels_total, len(cells)))
                cell = cells.pop()
                labels_fed += cell['labels']
                del cell['labels']
                for key in cell:
                    train[key].extend(cell[key])

            indices = np.arange(len(train['X']))
            np.random.shuffle(indices)

            for key in train:
                train[key] = np.array(train[key])
                train[key] = train[key][indices]
                npsave('%s_train.npy' % key, train[key])
                train[key] = None

            while len(cells) > 0:
                cell = cells.pop()
                labels_fed += cell['labels']
                del cell['labels']
                for key in cell:
                    val[key].extend(cell[key])

            for key in val:
                val[key] = np.array(val[key])
                npsave('%s_val.npy' % key, val[key])
                val[key] = None

        elif stage == 'test':
            test = {'X': [], 'cname': [], 'dname': []}

            while len(cells) > 0:
                cell = cells.pop()
                del cell['labels']
                for key in cell:
                    test[key].extend(cell[key])

            for key in test:
                test[key] = np.array(test[key])
                npsave('%s_test.npy' % key, test[key])
                test[key] = None

    # run(stage='train')
    run(stage='test')


if __name__ == '__main__':
    plac.call(main)
