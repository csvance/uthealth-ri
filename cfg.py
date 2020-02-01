SESSION = 'n-estimators-5'

ITERS = 0

N_ESTIMATORS: int = 50

DIMS: int = 8
BATCH_SIZE: int = 256
EPOCHS: int = 10
LR: float = 0.01
L2: float = 0.

SMILES_ENABLE: bool = True
SMILES_DIMS: int = 42
SMILES_LEN: int = 400
SMILES_IDX: int = 0
SMILES_REPLACE = {
    'Cl': 'D',
    'Br': 'E'
}
SMILES_SYMBOLS = ".-=#$:/\\BCNOPSFIDE()#@"

MACCS_ENABLE: bool = True
MACCS_LEN: int = 128
MACCS_IDX: int = 2
MACCS_DROPOUT: float = 0.5

LINES_LEN: int = 128
LINES_IDX: int = 1
LINES_DROPOUT: float = 0.5
LINES_LABELS = ['A1BG', 'A1BG-AS1', 'A1CF', 'A2M', 'A2ML1-AS1', 'A2ML1-AS2', 'A2MP1', 'ZWINT', 'ZXDA', 'ZXDB', 'ZXDC',
                'ZYG11A']

SEQ_LEN: int = max(SMILES_LEN, LINES_LEN)

SUMMARY: bool = True
