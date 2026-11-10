"""Bake-off for the lidar kernel (SPEC.md): pure Python (the truth), NumPy, numba, C++, Rust and Go.
Checks each against the reference bit for bit on real tracks, then measures poses/s on 1 and all threads and the
cost of one call with one pose (what an env step does).   ./build.sh && python make_data.py && python bench.py
"""
import argparse, ctypes, json, os, time
import numpy as np
import kernel_py

HERE = os.path.dirname(os.path.abspath(__file__))
D = ctypes.POINTER(ctypes.c_double)
STEP, MAXD = 1.0, 60.0


def c_impl(path):
    f = ctypes.CDLL(path).lidar
    f.restype = None
    f.argtypes = [D, D, ctypes.c_int, D, D, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double, D, ctypes.c_int]
    def run(d, P, threads):
        out = np.empty(P * d['R'])
        p = lambda a: a.ctypes.data_as(D)
        f(p(d['tiles']), p(d['centers']), d['T'], p(d['pos']), p(d['dirs']), P, d['R'], STEP, MAXD, p(out), threads)
        return out
    return run


def numba_impl():
    import numba
    serial, parallel = kernel_py.make_numba()
    def run(d, P, threads):
        out = np.empty(P * d['R'])
        if threads == 1:
            serial(d['tiles'], d['centers'], d['T'], d['pos'], d['dirs'], P, d['R'], STEP, MAXD, out)
        else:
            numba.set_num_threads(threads)
            parallel(d['tiles'], d['centers'], d['T'], d['pos'], d['dirs'], P, d['R'], STEP, MAXD, out)
        return out
    return run


def numpy_impl(d, P, threads):
    return kernel_py.lidar_numpy(d['tiles'], d['centers'], d['T'], d['pos'][:2 * P], d['dirs'][:2 * P * d['R']],
                                 P, d['R'], STEP, MAXD)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-ref', type=int, default=300, help='poses per track checked against pure Python')
    ap.add_argument('--threads', type=int, default=os.cpu_count())
    a = ap.parse_args()
    load = os.getloadavg()[0]
    if load > 1.0:
        print(f'WARNING: load average {load:.1f}: timings are noisy. Exactness results are valid.')
    data = []
    for s in [0, 1, 2]:
        z = np.load(os.path.join(HERE, 'out', f'data_{s}.npz'))
        d = {k: np.ascontiguousarray(z[k]) for k in ['tiles', 'centers', 'pos', 'dirs']}
        d.update(P=int(z['P']), R=int(z['R']), T=len(z['tiles']) // 8)
        data.append(d)
    refs, t_py, n_py = [], 0.0, 0
    for d in data:
        t = time.perf_counter()
        refs.append(kernel_py.lidar_ref(d['tiles'], d['centers'], d['T'], d['pos'][:2 * a.n_ref],
                                        d['dirs'][:2 * a.n_ref * d['R']], a.n_ref, d['R'], STEP, MAXD))
        t_py += time.perf_counter() - t; n_py += a.n_ref
    py_ps = n_py / t_py
    impls = {'NumPy (vectorised)': numpy_impl}
    try:
        impls['numba'] = numba_impl()
    except ImportError:
        print('skip numba')
    for name, lib in [('C++ -O2', 'libkernel_cpp.so'), ('C++ -O3 native', 'libkernel_cpp_native.so'),
                      ('Rust', 'libkernel_rust.so'), ('Go', 'libkernel_go.so')]:
        path = os.path.join(HERE, 'out', lib)
        if os.path.exists(path):
            impls[name] = c_impl(path)
    one_py = time.perf_counter()
    for _ in range(20):
        kernel_py.lidar_ref(data[0]['tiles'], data[0]['centers'], data[0]['T'], data[0]['pos'][:2],
                            data[0]['dirs'][:2 * data[0]['R']], 1, data[0]['R'], STEP, MAXD)
    one_py = (time.perf_counter() - one_py) / 20 * 1e6
    rows = [dict(impl='Python (reference)', bit_identical=1.0, max_abs_diff=0.0, poses_per_s_1=py_ps,
                 poses_per_s_all=None, one_pose_us=one_py, speedup_1=1.0)]
    for name, run in impls.items():
        run(data[0], 2, 1)
        same, n, mad = 0, 0, 0.0
        for d, ref in zip(data, refs):
            got = run(d, a.n_ref, 1)
            same += int((got == ref).sum()); n += ref.size; mad = max(mad, float(np.abs(got - ref).max()))
            if name != 'NumPy (vectorised)':
                assert (run(d, a.n_ref, a.threads) == got).all(), f'{name}: threads change the result'
        row = dict(impl=name, bit_identical=same / n, max_abs_diff=mad)
        for th, key in [(1, 'poses_per_s_1'), (a.threads, 'poses_per_s_all')]:
            if name == 'NumPy (vectorised)' and th > 1:
                row[key] = None; continue
            P = 300 if name == 'NumPy (vectorised)' else data[0]['P']
            best = 1e9
            for _ in range(3):
                t = time.perf_counter()
                for d in data:
                    run(d, P, th)
                best = min(best, time.perf_counter() - t)
            row[key] = 3 * P / best
        best = 1e9
        for _ in range(5):
            t = time.perf_counter()
            for _ in range(300):
                run(data[0], 1, 1)
            best = min(best, (time.perf_counter() - t) / 300)
        row['one_pose_us'] = best * 1e6
        row['speedup_1'] = row['poses_per_s_1'] / py_ps
        rows.append(row)
    meta = dict(load_avg=round(load, 1), n_ref_poses=n_py, rays=data[0]['R'], step=STEP, maxd=MAXD,
                threads=a.threads, cpu_count=os.cpu_count(), tracks=[d['T'] for d in data])
    json.dump({'meta': meta, 'rows': rows}, open(os.path.join(HERE, 'out', 'results.json'), 'w'), indent=1)
    print(meta)
    print(f'| impl | bit-identical | max diff | poses/s 1 thread | x Python | poses/s {a.threads} threads | one pose (us) |')
    print('|---|---|---|---|---|---|---|')
    f = lambda v, fmt: '-' if v is None else format(v, fmt)
    for r in rows:
        print(f"| {r['impl']} | {r['bit_identical']:.1%} | {r['max_abs_diff']:.0e} | {f(r['poses_per_s_1'], ',.0f')} | "
              f"{r['speedup_1']:.0f}x | {f(r['poses_per_s_all'], ',.0f')} | {r['one_pose_us']:.1f} |")


if __name__ == '__main__':
    main()
