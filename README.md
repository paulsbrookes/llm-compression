# llm-compression

How few bits does it take to store an LLM weight with no loss at all?

The question: do the fp16/bf16 weights people ship actually use all 16 bits, or could a
plain entropy coder remove some slack for free? This repo works that out. It computes the
entropy floor for Gaussian data rounded to a float, implements a few coders that get close
to it, and runs the same machinery over a real model (Qwen2.5-0.5B) to see what its
weights cost. C++17, no dependencies, with a few HTML pages for the plots.

## What it comes out to

Qwen2.5-0.5B stores weights in bf16, so 16 bits each. The empirical entropy is around
10.5 bits/weight, so the model could be stored losslessly in about a third less space. The
catch: almost none of that slack is in the mantissa. It's in the exponent.

## Where 10.5 comes from

Take a Gaussian, round it to fp16, and the result is a discrete distribution that can be
written down exactly. Its entropy H is the lossless floor: over many samples, no code
beats H bits each on average.

Two things about it stand out:

- H barely depends on scale. Sweeping sigma over a factor of 160 moves H by about 0.001
  bits (it sits near 13.46 for fp16). Floating point has relative precision, so the
  spacing grows with the value and the distribution looks the same to the coder at any
  scale.
- Split by field, the entropy is sign 1.0, exponent 2.54, mantissa 9.97. The 2.54 bits
  of slack (16 minus 13.46) is almost entirely in the exponent; the mantissa is a
  near-flat 10 bits, i.e. close to incompressible.

bf16 keeps fp16's exponent but has 7 mantissa bits instead of 10. Dropping ~3 bits off the
mantissa term lands around 10.5, which is roughly what the real weights show, so the toy
Gaussian is a decent model for them.

## The coders

Each builds a static code from the exact pmf, and each is checked by round-tripping a
million samples:

| Coder          | bits/sample | overhead vs H |
|----------------|------------:|--------------:|
| naive storage  |      16.000 |           n/a |
| Huffman        |      13.503 |        +0.039 |
| range (64-bit) |      13.464 |          ~0   |
| rANS           |      13.462 |          ~0   |
| floor H        |      13.464 |           n/a |

(sigma=1, N=1e6.)

Huffman gives up only ~0.04 bits, not the up-to-1-bit that integer codeword lengths can
cost in the worst case. That worst case needs a symbol with large probability; here the
slack is spread across a ~30k-symbol alphabet with no dominant symbol, so the per-symbol
rounding is tiny. Range and rANS drop the integer-length constraint and close what's left.

One caveat: H is the entropy of the marginal distribution, treating weights as independent
draws. Whether nearby weights correlate is not tested here. If they do, a context model
could beat these numbers, so 10.5 is the iid floor rather than the final answer.

## The real weights

For the big matrices the weights are close to zero-mean Gaussian (kurtosis 3.5–4.7),
which is why the model above transfers. Two exceptions:

- Query and key projections are heavy-tailed (kurtosis 14–23), driven by a few outlier
  "attention-sink" dimensions.
- RMSNorm gains are bimodal: the magnitude clusters near 0.04 and the sign is split
  roughly 50/50. The sign can be folded into the following linear layer, so only the
  magnitude is really being learned.

The dashboard shows the per-matrix histograms.

## Layout

```
src/      C++17 tools (no deps)
  half.h        fp16/bf16 bit decode + round-to-nearest
  pmf.h/.cpp    exact pmf + entropy floor of N(mu,sigma^2)->fp16
  huffman.cpp   Huffman codec, lossless round-trip
  range.cpp     64-bit range/arithmetic coder
  rans.cpp      64-bit rANS coder
  model.h       shared pmf->integer-frequency model (range+rANS)
  weights.cpp   safetensors reader: stats + bf16 entropy
docs/     HTML figures (serve with scripts/serve.sh)
  index.html              overview + the numbers
  entropy_curve.html      Shannon entropy H(p)
  weights_dashboard.html  per-matrix weight distributions
scripts/  build / download / bake-off / analyze / serve
data/     generated artifacts + downloaded model (git-ignored)
```

## Quickstart

```bash
make                              # build all tools into bin/
scripts/bakeoff.sh 1.0 0.0 1000000   # entropy floor vs Huffman/range/rANS
scripts/download_model.sh         # fetch Qwen2.5-0.5B (~988 MB) into data/
scripts/analyze_weights.sh        # -> docs/weights.json
scripts/serve.sh                  # view docs/ at http://localhost:8000
```

## Why wouldn't this be used

The lossless floor is ~10.5 bits/weight, but deployments quantize to 4–8 bits, below the
floor. Two reasons.

Lossy schemes are allowed to lose information. Int4/int8/NF4 methods (GPTQ, AWQ,
bitsandbytes) discard detail the model tolerates, so they get under H, which a lossless
coder cannot do by definition. The lossless number is mainly useful as a reference: the
most that can be saved before any decision is made about what is safe to throw away.

The other reason is decode speed. A range or rANS decoder is sequential and branchy; a
quantized weight is a fixed-width field that unpacks with a shift and a mask and feeds
straight into the matmul. Fixed width also allows random access, which an entropy-coded
stream does not, and that matters for paging and sharding.

The part that would be compressed losslessly is mostly the mantissa, which is both the
hardest to compress and the least significant numerically, so low-bit quantization drops
it first anyway. The Q/K outliers above are why naive low-bit quant struggles, and why
SmoothQuant, AWQ and similar handle outliers as a special case.

```
naive   16 ─────────────┐
lossless ~10.5 (entropy floor) ──┐
lossy    4–8 (quantization)  ────┘  (below the floor: information is discarded)
```
