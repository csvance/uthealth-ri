import functools

from keras.layers import *
from keras.models import Model
from keras.wrappers.scikit_learn import KerasClassifier
from keras.regularizers import l2
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import pickle
import plac

from rectified_adam import RectifiedAdam

from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_auc_score

from cfg import *

OPTIMIZER = RectifiedAdam(lr=LR, decay=LR / EPOCHS)


def fix_tf():
    # Fix Tensorflow Memory Usage
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def build_keras():
    ip = Input((INPUT_DIMS,))

    def f_lambda_smiles(x):
        x = x[:, SMILES_IDX_I:SMILES_IDX_F]
        x = tf.reshape(x, (-1, SMILES_LEN, SMILES_DIMS))
        return x

    def f_lambda_lines(x):
        return x[:, LINES_IDX_I:LINES_IDX_F]

    def f_lambda_maccs(x):
        return x[:, MACCS_IDX_I:MACCS_IDX_F]

    def f_lambda_target(x):
        return x[:, TARGET_IDX_I:TARGET_IDX_F]

    lambda_smiles = functools.partial(f_lambda_smiles)
    lambda_smiles.__name__ = 'lambda_smiles'
    lambda_lines = functools.partial(f_lambda_lines)
    lambda_lines.__name__ = 'lambda_lines'
    lambda_maccs = functools.partial(f_lambda_maccs)
    lambda_maccs.__name__ = 'lambda_maccs'
    lambda_target = functools.partial(f_lambda_target)
    lambda_target.__name__ = 'lambda_target'

    features = []

    # Cell Lines
    l_dropout = Dropout(LINES_DROPOUT)

    # forward
    lx = Lambda(lambda x: lambda_lines(x))(ip)
    if LINES_DROPOUT > 0:
        lx = l_dropout(lx)
    features.append(lx)

    if SMILES_ENABLE:
        s_conv1 = Conv1D(SMILES_KERNEL_DIMS, 11, padding='same', kernel_regularizer=l2(L2))
        s_bn1 = BatchNormalization()
        s_relu1 = ReLU()
        s_ap1 = AveragePooling1D(pool_size=5)
        s_conv2 = Conv1D(SMILES_KERNEL_DIMS, 11, padding='same', kernel_regularizer=l2(L2))
        s_bn2 = BatchNormalization()
        s_relu2 = ReLU()
        s_ap2 = AveragePooling1D(pool_size=5)
        s_gmp = GlobalMaxPooling1D()

        # forward
        sx = Lambda(lambda x: lambda_smiles(x))(ip)
        sx = s_conv1(sx)
        sx = s_bn1(sx)
        sx = s_relu1(sx)
        sx = s_ap1(sx)
        sx = s_conv2(sx)
        sx = s_bn2(sx)
        sx = s_relu2(sx)
        sx = s_ap2(sx)
        sx = s_gmp(sx)

        features.append(sx)

    # MACCS
    if MACCS_ENABLE:
        m_dropout = Dropout(LINES_DROPOUT)

        # forward
        mx = Lambda(lambda x: lambda_maccs(x))(ip)
        if MACCS_DROPOUT > 0:
            mx = m_dropout(mx)
        features.append(mx)

    if TARGET_ENABLE:
        t_dropout = Dropout(TARGET_DROPOUT)

        # forward
        tx = Lambda(lambda x: lambda_target(x))(ip)
        if MACCS_DROPOUT > 0:
            tx = t_dropout(tx)
        features.append(tx)

    j_concat = Concatenate()
    j_hidden = Dense(HIDDEN_DIMS, activation='tanh')
    j_cls = Dense(1, activation='sigmoid')

    jx = j_concat(features)
    if HIDDEN_DIMS > 0:
        jx = j_hidden(jx)

    y = j_cls(jx)

    model = Model(inputs=ip, outputs=y)
    model.compile(optimizer=OPTIMIZER,
                  loss='binary_crossentropy',
                  metrics=['acc'])

    if SUMMARY:
        model.summary()

    return model


def build_sklearn():
    return KerasClassifier(build_fn=build_keras,
                           epochs=EPOCHS,
                           batch_size=BATCH_SIZE,
                           verbose=1)


def build_adaboost():
    sklearn_model = build_sklearn()
    model = AdaBoostClassifier(base_estimator=sklearn_model, n_estimators=N_ESTIMATORS)

    return model


def idgc20(y_true, y_pred, cname, dname, submission: bool = False):
    results = {
        'uid': [],
        'y_true': [],
        'y_pred': [],
        'cell': [],
        'drug': []
    }
    if submission:
        del results['y_true']

    for i in range(0, y_pred.shape[0]):
        results['uid'].append(i)
        if not submission:
            results['y_true'].append(y_true[i, 0])
        results['y_pred'].append(y_pred[i, 1])
        results['cell'].append(cname[i])
        results['drug'].append(dname[i])

    df = pd.DataFrame.from_dict(results)
    if submission:
        df['rank'] = 100

    cells_drugs = []

    all_idgc_20 = []
    for cell in df['cell'].unique():

        dfc = df[df['cell'] == cell]
        dfc.sort_values(by=['y_pred'], ascending=False, inplace=True)

        dgc_20 = 0
        for i in range(0, min(20, dfc.shape[0])):
            if submission:
                df.loc[df['uid'] == dfc.iloc[i]['uid'], 'rank'] = i
            if not submission:
                dgc_20 += dfc.iloc[i]['y_true'] / np.log2(i + 2)
        idgc_20 = 0
        for i in range(0, min(20, dfc.shape[0])):
            idgc_20 += 1 / np.log2(i + 2)

        idgc_20 = dgc_20 / idgc_20
        all_idgc_20.append(idgc_20)

    if submission:
        dfs = df[df['rank'] != 100]
        dfs.sort_values(by=['cell', 'rank'], ascending=True, inplace=True)
        dfs.drop(labels=['uid', 'y_pred'], axis=1, inplace=True)
        dfs.to_csv('submission.csv', index=False)

    return np.mean(all_idgc_20)


@plac.annotations(
    cmd=('train, val, rank', 'option', 'c', str)
)
def main(cmd: str = 'train'):
    fix_tf()

    X_train = np.load(os.path.join(TV_PATH, 'X_train.npy'), mmap_mode='r+')
    y_train = np.load(os.path.join(TV_PATH, 'y_train.npy'))
    cname_train = np.load(os.path.join(TV_PATH, 'cname_train.npy'))
    dname_train = np.load(os.path.join(TV_PATH, 'dname_train.npy'))

    X_val = np.load(os.path.join(TV_PATH, 'X_val.npy'), mmap_mode='r+')
    y_val = np.load(os.path.join(TV_PATH, 'y_val.npy'))

    cname_val = np.load(os.path.join(TV_PATH, 'cname_val.npy'))
    dname_val = np.load(os.path.join(TV_PATH, 'dname_val.npy'))

    if cmd == 'train':
        model = build_adaboost()

        model.fit(X_train, y_train)
        y_pred = np.array(model.predict_proba(X_val))

        idgc_20 = idgc20(y_val, y_pred, cname_val, dname_val)

        print("IDGC_20: %f" % idgc_20)

        print("Estimator Weights")
        print(model.estimator_weights_)

        for n, est in enumerate(model.estimators_):
            print("Saving estimator %d..." % n)
            est.model.save_weights('weights/%s_%d.h5' % (SESSION, n))

        model.estimators_ = []
        open('weights/%s.pkl' % SESSION, 'wb').write(pickle.dumps(model))

    elif cmd == 'val':
        model = pickle.loads(open('weights/%s.pkl' % SESSION, 'rb').read())

        for n in range(0, N_ESTIMATORS):
            m_n = build_sklearn()
            m_n.model = build_keras()
            m_n.model.load_weights('weights/%s_%d.h5' % (SESSION, n))
            model.estimators_.append(m_n)

        y_pred = np.array(model.predict_proba(X_val))

        idgc_20 = idgc20(y_val, y_pred, cname_val, dname_val)
        print("IDGC_20: %f" % idgc_20)


    elif cmd == 'rank':
        model = pickle.loads(open('weights/%s.pkl' % SESSION, 'rb').read())

        X_test = np.load(os.path.join(TV_PATH, 'X_test.npy'), mmap_mode='r+')
        cname_test = np.load(os.path.join(TV_PATH, 'cname_test.npy'))
        dname_test = np.load(os.path.join(TV_PATH, 'dname_test.npy'))

        for n in range(0, N_ESTIMATORS):
            m_n = build_sklearn()
            m_n.model = build_keras()
            m_n.model.load_weights('weights/%s_%d.h5' % (SESSION, n))
            model.estimators_.append(m_n)

        y_pred = np.array(model.predict_proba(X_test))

        idgc_20 = idgc20(None, y_pred, cname_test, dname_test, submission=True)


if __name__ == '__main__':
    plac.call(main)

