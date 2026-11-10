// SPEC.md (lidar) in Rust (C ABI, std only). Rust never fuses a*b+c into an FMA unless asked.
use std::slice;

#[inline(always)]
fn inside(v: &[f64], px: f64, py: f64) -> bool {
    let mut inn = false;
    let (mut p1x, mut p1y) = (v[0], v[1]);
    for i in 0..5 {
        let (p2x, p2y) = (v[(i % 4) * 2], v[(i % 4) * 2 + 1]);
        if py > p1y.min(p2y) && py <= p1y.max(p2y) && px <= p1x.max(p2x) {
            let xin = if p1y != p2y { (py - p1y) * (p2x - p1x) / (p2y - p1y + 1e-8) + p1x } else { p1x };
            if p1x == p2x || px <= xin { inn = !inn; }
        }
        p1x = p2x; p1y = p2y;
    }
    inn
}

fn cast(tiles: &[f64], centers: &[f64], pos: &[f64], dirs: &[f64], p: usize, r_n: usize, step: f64, maxd: f64,
        out: &mut [f64], cand: &mut Vec<usize>) {
    let (x, y) = (pos[2 * p], pos[2 * p + 1]);
    let lim = (maxd + 10.0) * (maxd + 10.0);
    cand.clear();
    for t in 0..centers.len() / 2 {
        let (dx, dy) = (centers[2 * t] - x, centers[2 * t + 1] - y);
        if dx * dx + dy * dy <= lim { cand.push(t); }
    }
    let k_n = (maxd / step) as i32;
    let mut last = 0usize;
    for r in 0..r_n {
        let q = (p * r_n + r) * 2;
        let (dx, dy) = (dirs[q], dirs[q + 1]);
        let mut res = 1.0;
        for k in 1..=k_n {
            let px = x + dx * (step * k as f64);
            let py = y + dy * (step * k as f64);
            let mut hit = !cand.is_empty() && inside(&tiles[cand[last] * 8..cand[last] * 8 + 8], px, py);
            if !hit {
                for (c, &t) in cand.iter().enumerate() {
                    if inside(&tiles[t * 8..t * 8 + 8], px, py) { hit = true; last = c; break; }
                }
            }
            if !hit { res = step * (k - 1) as f64 / maxd; break; }
        }
        out[r] = res;
    }
}

#[no_mangle]
pub unsafe extern "C" fn lidar(tiles: *const f64, centers: *const f64, t_n: i32, pos: *const f64, dirs: *const f64,
                               p_n: i32, r_n: i32, step: f64, maxd: f64, out: *mut f64, threads: i32) {
    let (t_n, p_n, r_n) = (t_n as usize, p_n as usize, r_n as usize);
    let tiles = slice::from_raw_parts(tiles, t_n * 8);
    let centers = slice::from_raw_parts(centers, t_n * 2);
    let pos = slice::from_raw_parts(pos, p_n * 2);
    let dirs = slice::from_raw_parts(dirs, p_n * r_n * 2);
    let out = slice::from_raw_parts_mut(out, p_n * r_n);
    let threads = (threads.max(1) as usize).min(p_n.max(1));
    if threads == 1 {
        let mut cand = Vec::with_capacity(t_n);
        for p in 0..p_n { cast(tiles, centers, pos, dirs, p, r_n, step, maxd, &mut out[p * r_n..(p + 1) * r_n], &mut cand); }
        return;
    }
    let chunk = (p_n + threads - 1) / threads;
    std::thread::scope(|sc| {
        for (c, o) in out.chunks_mut(chunk * r_n).enumerate() {
            sc.spawn(move || {
                let mut cand = Vec::with_capacity(t_n);
                for k in 0..o.len() / r_n {
                    let p = c * chunk + k;
                    cast(tiles, centers, pos, dirs, p, r_n, step, maxd, &mut o[k * r_n..(k + 1) * r_n], &mut cand);
                }
            });
        }
    });
}
