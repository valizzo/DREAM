"""Versioned DREAM-owned transport models and their validated inference contract."""

import hashlib
import json
from pathlib import Path


MODEL_ROOT = Path(__file__).resolve().parents[3] / 'models' / 'transport'
FEATURE_NAMES = [
    'n_imp_neutral_norm', 'Ne_ion_norm', 'te_rel_to_initial', 'psi_norm', 'abs_q',
    'prox_q1', 'present_q2', 'grad_p', 'wth_norm', 'dwth_norm_dt_scaled',
    'ip_norm', 'dip_norm_dt_scaled', 'li', 'j_tor_norm',
]


def resolve_model(model=None, version=None):
    if model is not None and version is not None:
        raise ValueError('Specify model or version, not both')
    if model is None:
        selected = version or (MODEL_ROOT / 'DEFAULT_MODEL').read_text(encoding='utf-8').strip()
        if not selected or Path(selected).name != selected or selected in ('.', '..'):
            raise ValueError('Invalid bundled NN model version')
        directory = MODEL_ROOT / selected
    else:
        supplied = Path(model).expanduser()
        directory = supplied.parent if supplied.suffix == '.onnx' else supplied
    directory = directory.resolve()
    graph = supplied.resolve() if model is not None and supplied.suffix == '.onnx' else directory / 'transport.onnx'
    contract = json.loads((directory / 'deployment.json').read_text(encoding='utf-8'))
    if contract.get('deployment_schema_version') != 1 or contract.get('feature_names') != FEATURE_NAMES:
        raise ValueError('Unsupported NN transport feature contract')
    if contract['input_shape'] != [1, contract['window_size'], len(contract['rho_grid']), len(FEATURE_NAMES)]:
        raise ValueError('NN model input shape disagrees with its manifest')
    if contract.get('forecast_horizon_samples') != 1:
        raise ValueError('NN transport requires a one-sample-ahead model')
    if hashlib.sha256(graph.read_bytes()).hexdigest() != contract['sha256']['transport.onnx']:
        raise ValueError('NN transport model checksum mismatch')
    return graph, contract