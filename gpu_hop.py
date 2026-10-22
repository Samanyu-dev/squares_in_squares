S_HOP = 1.0 + 1e-5      # GPU inflation for warm starts (tight lattices cannot afford more)
S_HOPCHK = 1.0 + 2e-6   # float64 acceptance check
def hop_gpu(Pb, Lb, Lt, chains, iters, seed, sig0=0.15, T0=1e-3, kick_frac=0.05):
    """Warm start from packing Pb (side Lb): squeeze to Lt, kick a few squares per chain, relax with exact squares on GPU."""
    rng = np.random.default_rng(seed); n = len(Pb)
    P0 = np.empty((chains, n, 3), np.float32)
    for c in range(chains):
        Q = Pb.copy(); Q[:, :2] *= Lt / Lb
        m = int(rng.integers(1, max(2, int(n * kick_frac)) + 1))
        idx = rng.choice(n, m, replace=False)
        Q[idx] = np.c_[rng.uniform(.5, Lt - .5, m), rng.uniform(.5, Lt - .5, m), rng.uniform(0, 1.57, m)]
        Q[:, :2] += rng.normal(0, 0.01, (n, 2))
        P0[c] = Q
    dP = cuda.to_device(P0)
    states = cuda.to_device(((np.arange(chains, dtype=np.int64) + 1) * 2654435761 + int(seed) * 40503 + 12345) % 4294967295 + 1)
    kern[chains, BD](dP, n, np.float32(Lt), np.float32(0.0), np.float32(S_HOP), iters, np.float32(sig0), np.float32(T0), states)
    cuda.synchronize()
    return dP.copy_to_host().astype(np.float64)
