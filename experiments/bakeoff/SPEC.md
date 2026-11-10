# Bake-off kernel: the lidar ray cast (the project's own hot loop)

Every training step casts R rays from the car against the road tiles. Inputs, all float64:

- `tiles[T*8]`: the T road quads (4 vertices x, y each, CarRacing's own fixture vertices), `centers[T*2]` their means
- `pos[P*2]`: the car positions, `dirs[P*R*2]`: unit ray directions (cos, sin), computed in Python so that no
  trigonometry happens in the kernel (libm vs Go's own `math.Sin` cannot differ)
- `step`, `maxd`: sample spacing and range; `K = floor(maxd / step)` samples per ray

Per pose p, ray r: candidate tiles are those with `(cx-x)^2 + (cy-y)^2 <= (maxd+10)^2` (index order; a tile is at
most ~7.5 units from its centre, so this cull never drops a tile a sample can be in). For k = 1..K:

    px = x + dx * (step * k);  py = y + dy * (step * k)
    if no candidate tile contains (px, py):  out = step * (k - 1) / maxd ; stop
    (all K inside: out = 1.0)

"Contains" is the crossing-number test of the v1 code (`_point_in_poly`), operation for operation:

    inside = false; p1 = v[0]
    for i in 0..4: p2 = v[i % 4]
        if py > min(p1y,p2y) and py <= max(p1y,p2y) and px <= max(p1x,p2x):
            xin = p1y != p2y ? (py-p1y)*(p2x-p1x)/(p2y-p1y+1e-8) + p1x : p1x
            if p1x == p2x or px <= xin: inside = !inside
        p1 = p2

Checking the tile that matched last time first is allowed (it changes the order, never the answer).

C ABI: `void lidar(const double* tiles, const double* centers, int T, const double* pos, const double* dirs,
int P, int R, double step, double maxd, double* out, int threads)`
