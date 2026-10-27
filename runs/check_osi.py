import numpy as np
rng = np.random.default_rng(0)
v = np.array([official_osi(rng.standard_normal((32, 32))) for _ in range(1000)])
print('official OSI on 32x32 Gaussian noise: mean %.3f, 95%% CI +-%.3f, median %.3f (paper: 1.97 +- 0.08)' % (v.mean(), 1.96 * v.std() / np.sqrt(len(v)), np.median(v)))
