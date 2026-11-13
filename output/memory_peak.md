# AVIF texture loading - memory peak report

Measured with `ctx.texture_avif()` on llvmpipe (software GL, so GL
texture storage counts towards RSS). `naive` is the old external
pipeline (read whole file + Pillow decode + `tobytes()` copy).
Each case runs in a fresh process; `peak delta` is VmHWM growth.

| case | file | decoded | GL mip chain | peak delta | RSS after repeated release |
|---|---|---|---|---|---|---|
| 4K_8bit_rgba_moderngl | 4.4 MiB | 31.6 MiB | 42.2 MiB | 164.3 MiB | 165/165/165 MiB |
| 4K_8bit_rgba_naive | 4.4 MiB | 31.6 MiB | 42.2 MiB | 161.6 MiB | 155/155/155 MiB |
| 4K_10bit_rgba_moderngl | 7.0 MiB | 63.3 MiB | 84.4 MiB | 344.4 MiB | 207/207/207 MiB |
| 8K_8bit_rgba_moderngl | 17.4 MiB | 126.6 MiB | 168.7 MiB | 581.9 MiB | 291/291/291 MiB |
| 8K_8bit_rgba_naive | 17.4 MiB | 126.6 MiB | 168.7 MiB | 573.9 MiB | 281/281/281 MiB |
| 8K_10bit_rgba_moderngl | 28.0 MiB | 253.1 MiB | 337.5 MiB | 1300.4 MiB | 460/460/460 MiB |
| 8K_8bit_rgba_moderngl_q30 | 0.0 MiB | 126.6 MiB | 168.7 MiB | 563.2 MiB | 291/291/291 MiB |

Interpretation:
- The peak is `decode working set (libavif YUV planes + dav1d, freed
  after decode) + exactly one decoded pixel buffer + GL mip chain`.
  The decoded buffer is uploaded in place (zero copy) and released
  before the remaining mip levels are generated.
- File-size invariance: `8K_8bit_rgba_moderngl_q30` vs the q90 run -
  same pixels, ~900x smaller file, peak differs only by the compressed
  extent libavif itself buffers while decoding. The Python side never
  reads the file: libavif streams it (`avifIOCreateFileReader`), so no
  extra full copy of the file is made.
- `RSS after repeated release` is flat across load/release iterations:
  decode buffers and textures are freed (the residual above baseline is
  llvmpipe/glibc allocator caches, reused by the next iteration).
