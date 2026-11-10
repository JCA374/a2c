// SPEC.md (lidar) in Go (C ABI via cgo, go build -buildmode=c-shared). float64(a*b) forbids FMA fusion.
package main

/*
#include <stdint.h>
*/
import "C"

import (
	"sync"
	"unsafe"
)

func inside(v []float64, px, py float64) bool {
	in := false
	p1x, p1y := v[0], v[1]
	for i := 0; i < 5; i++ {
		p2x, p2y := v[(i%4)*2], v[(i%4)*2+1]
		if py > min(p1y, p2y) && py <= max(p1y, p2y) && px <= max(p1x, p2x) {
			xin := p1x
			if p1y != p2y {
				xin = float64(float64((py-p1y)*(p2x-p1x))/(p2y-p1y+1e-8)) + p1x
			}
			if p1x == p2x || px <= xin {
				in = !in
			}
		}
		p1x, p1y = p2x, p2y
	}
	return in
}

func cast(tiles, centers, pos, dirs []float64, p, R int, step, maxd float64, out []float64, cand []int) {
	x, y := pos[2*p], pos[2*p+1]
	lim := float64((maxd + 10.0) * (maxd + 10.0))
	cand = cand[:0]
	for t := 0; t < len(centers)/2; t++ {
		dx, dy := centers[2*t]-x, centers[2*t+1]-y
		if float64(dx*dx)+float64(dy*dy) <= lim {
			cand = append(cand, t)
		}
	}
	K := int(maxd / step)
	last := 0
	for r := 0; r < R; r++ {
		q := (p*R + r) * 2
		dx, dy := dirs[q], dirs[q+1]
		res := 1.0
		for k := 1; k <= K; k++ {
			px := x + float64(dx*float64(step*float64(k)))
			py := y + float64(dy*float64(step*float64(k)))
			hit := len(cand) > 0 && inside(tiles[cand[last]*8:cand[last]*8+8], px, py)
			if !hit {
				for c, t := range cand {
					if inside(tiles[t*8:t*8+8], px, py) {
						hit, last = true, c
						break
					}
				}
			}
			if !hit {
				res = float64(step*float64(k-1)) / maxd
				break
			}
		}
		out[r] = res
	}
}

//export lidar
func lidar(tp, cp *C.double, cT C.int, pp, dp *C.double, cP, cR C.int, step, maxd C.double, op *C.double, cth C.int) {
	T, P, R, threads := int(cT), int(cP), int(cR), int(cth)
	tiles := unsafe.Slice((*float64)(unsafe.Pointer(tp)), T*8)
	centers := unsafe.Slice((*float64)(unsafe.Pointer(cp)), T*2)
	pos := unsafe.Slice((*float64)(unsafe.Pointer(pp)), P*2)
	dirs := unsafe.Slice((*float64)(unsafe.Pointer(dp)), P*R*2)
	out := unsafe.Slice((*float64)(unsafe.Pointer(op)), P*R)
	run := func(lo, hi int) {
		cand := make([]int, 0, T)
		for p := lo; p < hi; p++ {
			cast(tiles, centers, pos, dirs, p, R, float64(step), float64(maxd), out[p*R:(p+1)*R], cand)
		}
	}
	if threads <= 1 {
		run(0, P)
		return
	}
	var wg sync.WaitGroup
	for k := 0; k < threads; k++ {
		lo, hi := P*k/threads, P*(k+1)/threads
		wg.Add(1)
		go func() { defer wg.Done(); run(lo, hi) }()
	}
	wg.Wait()
}

func main() {}
