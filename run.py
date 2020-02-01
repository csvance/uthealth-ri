import functools

from keras.layers import *
from keras.models import Model
from keras.wrappers.scikit_learn import KerasClassifier
from keras.regularizers import l2
import tensorflow as tf

from rectified_adam import RectifiedAdam
from utils import fix_tf

from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_auc_score

from cfg import *

OPTIMIZER = RectifiedAdam(lr=LR)


def build_keras():

    ip = Input((SEQ_LEN, 3))

    def f_lambda_smiles(x):
        x = x[:, 0:SMILES_LEN, SMILES_IDX]
        x = tf.cast(x, tf.int32)
        x = tf.one_hot(x, axis=2, depth=SMILES_DIMS)
        return x

    def f_lambda_lines(x):
        return x[:, 0:LINES_LEN, LINES_IDX]

    def f_lambda_maccs(x):
        return x[:, 0:MACCS_LEN, MACCS_IDX]

    lambda_smiles = functools.partial(f_lambda_smiles)
    lambda_smiles.__name__ = 'lambda_smiles'
    lambda_lines = functools.partial(f_lambda_lines)
    lambda_lines.__name__ = 'lambda_lines'
    lambda_maccs = functools.partial(f_lambda_maccs)
    lambda_maccs.__name__ = 'lambda_maccs'

    features = []

    # Cell lines features
    l_dropout = Dropout(LINES_DROPOUT)
    l_dense1 = Dense(DIMS, activation='tanh', kernel_regularizer=l2(L2))
    lx = Lambda(lambda x: lambda_lines(x))(ip)
    if LINES_DROPOUT > 0:
        lx = l_dropout(lx)
    lx = l_dense1(lx)
    features.append(lx)

    # SMILES
    if SMILES_ENABLE:
        s_conv1 = Conv1D(DIMS, 3, padding='same', activation='relu')
        s_ap1 = AveragePooling1D()
        s_conv2 = Conv1D(DIMS, 3, padding='same', activation='relu')
        s_ap2 = AveragePooling1D()
        s_gmp = GlobalMaxPooling1D()

        sx = Lambda(lambda x: lambda_smiles(x))(ip)
        sx = s_conv1(sx)
        sx = s_ap1(sx)
        sx = s_conv2(sx)
        sx = s_ap2(sx)
        sx = s_gmp(sx)

        features.append(sx)

    # MACCS
    if MACCS_ENABLE:
        m_dropout = Dropout(LINES_DROPOUT)
        m_dense1 = Dense(DIMS, activation='tanh', kernel_regularizer=l2(L2))
        mx = Lambda(lambda x: lambda_maccs(x))(ip)
        if MACCS_DROPOUT > 0:
            mx = m_dropout(mx)
        mx = m_dense1(mx)
        features.append(mx)

    j_concat = Concatenate()
    cls = Dense(1, activation='sigmoid')

    jx = j_concat(features)

    y = cls(jx)

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


def main(cmd: str = 'train', dummy_data: bool = True):
    fix_tf()

    if not dummy_data:
        pass
    else:
        xpa = np.round(np.random.uniform(0, 42.5, size=(2048, 400, 1)))
        xpb = np.random.normal(size=(2048, 400, 2), loc=0.1, scale=4)
        xp = np.concatenate([xpa, xpb], axis=-1)
        yp = np.ones((2048, 1))

        xna = np.round(np.random.uniform(0.5, 43, size=(2048, 400, 1)))
        xnb = np.random.normal(size=(2048, 400, 2), loc=-0.1, scale=4)
        xn = np.concatenate([xna, xnb], axis=-1)
        yn = np.zeros((2048, 1))

        X = np.concatenate([xp, xn], axis=0)
        y = np.concatenate([yp, yn], axis=0)

        indices = np.arange(len(X))
        np.random.shuffle(indices)

        X = X[indices]
        y = y[indices]

    if cmd == 'train':
        model = build_adaboost()

        model.fit(X, y)
        y_pred = model.predict(X)

        print("AUC-ROC: %f" % roc_auc_score(y, y_pred))

        for n, est in enumerate(model.estimators_):
            est.model.save_weights('weights/%s_%d.h5' % (SESSION, n))

    elif cmd == 'val':

        models = []
        preds = np.zeros_like(y)

        for n in range(0, N_ESTIMATORS):
            model = build_keras()
            model.load_weights('weights/%s_%d.h5' % (SESSION, n))
            models.append(model)

            preds = preds + model.predict(X)

        y_pred = preds / N_ESTIMATORS

        print("AUC-ROC: %f" % roc_auc_score(y, y_pred))


if __name__ == '__main__':
    main('val')

