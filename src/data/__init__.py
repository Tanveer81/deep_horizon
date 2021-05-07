"""
Constants for data usage.
"""
SMALL_DATA = 3000

VERSIONS = {0.1: 'new', 0.11: 'new2', 0.2: 'ext', 0.21: 'ext2', 0.22: 'ext3', 0.23: '023'}
SPLIT = ['train', 'test']
TRANS = ['quantile', 'power', 'robust', 'normal']

COLUMNS = {
    'CHANNEL': ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7'],
    'POS': ['x', 'y', 'z', 'rdist'],
    'OMNI': [
        'AE_index', 'SYM-H_index', 'F107', 'BimfxGSE', 'BimfyGSE',
        'BimfzGSE', 'VxSW_GSE', 'VySW_GSE', 'VzSW_GSE', 'NpSW', 'Pdyn', 'Temp'
    ],
    'FOOT_TYPE_RAW': ['FootType'],
    'FOOT_TYPE': ['FootType0', 'FootType1', 'FootType2'],
    'DATETIME': ['DateTime'],
    'COUNTS': ['counts']
}
LABELS = COLUMNS['CHANNEL']
FEATURES_RAW = COLUMNS['POS'] + COLUMNS['OMNI'] + COLUMNS['FOOT_TYPE_RAW']
FEATURES_RAW_WITH_TIME = FEATURES_RAW + COLUMNS['DATETIME']
FEATURES_NO_FOOT_TYPE = COLUMNS['POS'] + COLUMNS['OMNI']
FEATURES = FEATURES_NO_FOOT_TYPE + COLUMNS['FOOT_TYPE']
FEATURES_WITH_TIME = FEATURES + COLUMNS['DATETIME']
FEATURES_NO_FOOT_TYPE_WITH_TIME = FEATURES_NO_FOOT_TYPE + COLUMNS['DATETIME']

SEGMENTS = {
    'LABELS': LABELS,
    'FEATURES': FEATURES_WITH_TIME,
    'DATETIME': COLUMNS['DATETIME']
}

SEGMENTS_ALLOWED = {
    'LABELS': LABELS + COLUMNS['COUNTS'],
    'FEATURES': FEATURES_WITH_TIME,
    'DATETIME': COLUMNS['DATETIME']
}
