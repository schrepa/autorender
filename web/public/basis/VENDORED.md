# Basis Universal transcoder

`basis_transcoder.js` and `basis_transcoder.wasm` are the unmodified WebGL
transcoder build of [Basis Universal](https://github.com/BinomialLLC/basis_universal)
(`webgl/transcoder/build/`) at commit
`38168767f42d5a77fb777594630825e0547b4489` (2026-07-18), under the Apache
License 2.0 in `LICENSE`.

| File | SHA-256 |
|---|---|
| `basis_transcoder.js` | `720dd9bd09c7cada6d87f1b7b70cec713df04da88cd641ac3212559353834dc8` |
| `basis_transcoder.wasm` | `a0f65d4a30ecb3269d01ead7d0a3477d2b0208146d083625a90623f473f6c139` |

Pages load it only when a scene references a KTX2 image, and transcode at
load, never per frame (`web/src/ktx2.ts`). To update, replace
both files from one upstream commit, record the commit and hashes here, and
run the texture smoke check.
