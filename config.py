import os

# Root of every cached artifact.  Defaults to the original 'temp_data', so an
# unset environment reproduces the historical paths byte for byte.  The
# reproduction driver overrides it per run, because the artifact names below do
# not encode --random_seed: without an isolated root, a second seed would silently
# reload the first seed's community file instead of re-detecting communities.
_DATA_ROOT = os.environ.get('CCGU_DATA_ROOT', 'temp_data').rstrip('/') + '/'

RAW_DATA_PATH = _DATA_ROOT + 'raw_data/'
PROCESSED_DATA_PATH = _DATA_ROOT + 'processed_data/'
MODEL_PATH = _DATA_ROOT + 'models/'
ANALYSIS_PATH = _DATA_ROOT + 'analysis_data/'
COM_PATH = _DATA_ROOT + 'com_data/'
NU_PATH = _DATA_ROOT + 'nucleus_data/'

# Downloaded DGL datasets are read-only and large, so they can be shared across
# runs even when the rest of the artifacts are isolated.
DGL_PATH = os.environ.get('CCGU_DGL_DATA', _DATA_ROOT + 'dgl_data/').rstrip('/') + '/'

# database name
DATABASE_NAME = "unlearning_gnn"
