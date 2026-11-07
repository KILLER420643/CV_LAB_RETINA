import numpy as np, pytest
from lowlight.pipeline import run
@pytest.mark.parametrize('shape', [(64, 80, 3), (300, 200, 3)])
@pytest.mark.parametrize('mode', ['classical', 'dl', 'hybrid'])
def test_modes(mode, shape):
    x = (np.random.rand(*shape)*.05).astype(np.float32); r = run(x, mode=mode)
    assert r['output'].shape == x.shape and np.isfinite(r['output']).all() and 0 <= r['output'].min() and r['output'].max() <= 1
def test_black_and_white_inputs():
    for v in (0., 1.): assert np.isfinite(run(np.full((64, 64, 3), v, np.float32))['output']).all()
