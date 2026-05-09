# MLA Completion Gates

Status: active checklist for deciding when `mla.tex` is complete.

This file restates the relevant gates from `plan_mla.md` for the current
`mla.tex` target.  The project is complete only when every required gate below
is either passed or explicitly marked out of scope.

## Required Gates

| Gate | Required Evidence | Current Status |
| --- | --- | --- |
| Source grounding | `source_ledger.md`, `external_sources.md`, and explicit evidence boundary in `mla.tex` | passed |
| Stable notation | `notation.md`; dimensions \(B,L,M,D,H,N,R,V,Q,C,T,X,Y\) used consistently in `mla.tex` | passed |
| Step-by-step derivation | `mla.tex` derives cache ladder, query path, KV path, materialized MLA, score decomposition, absorption, value projection, RoPE separation, external equivalence, validation, and manual example in order | passed after `mla_018` |
| No implementation listings in TeX | scan `mla.tex` for code listings, pseudocode, implementation identifiers, and verbatim blocks | passed |
| Shape evolution | `mla.tex` states mathematical shape assertions; `notation.md` keeps implementation-aligned suffix trace | passed |
| Executable reference parity | `verification/torch_mla_reference_check.py` and `.json` | passed |
| Manual example | `verification/two_head_manual_example.py`, `.json`, and the corresponding hand calculation in `mla.tex` | passed |
| External structural comparison | pinned vLLM/SGLang notes plus vLLM dummy-engine shape probe | passed |
| TeX build | two-pass `pdflatex` to `verification/mla.pdf` without errors or overfull hbox warnings | passed |
| Ledger consistency | valid `tasks.jsonl`, valid `events.jsonl`, updated `state.md`, updated `reports/final.md` | passed |

## Out Of Scope For This Version

- Full-model parity for all attention-mask forms.
- Training dropout behavior.
- YaRN-specific scaling proof.
- Paged-cache mutation semantics beyond the base cache object.
- Quantized, sparse/NSA, fused-kernel, and CUDA-graph-specific proofs.

These are valid future tasks but are not required for the dense base MLA
mathematical derivation.

## Final Verification Commands

```bash
.venv/bin/python math/mla/verification/torch_mla_reference_check.py --device auto --output math/mla/verification/torch_mla_reference_check.json
.venv/bin/python math/mla/verification/two_head_manual_example.py
.venv/bin/python math/mla/verification/vllm_dummy_mla_shape_trace.py --dims small --device auto --output math/mla/verification/vllm_dummy_mla_shape_trace.small.json
.venv/bin/python math/mla/verification/vllm_dummy_mla_shape_trace.py --dims profile --device auto --output math/mla/verification/vllm_dummy_mla_shape_trace.profile.json
.venv/bin/python math/mla/verification/vllm_dummy_engine_probe.py --output math/mla/verification/vllm_dummy_engine_probe.json --gpu-memory-utilization 0.20
pdflatex -interaction=nonstopmode -halt-on-error -output-directory math/mla/verification math/mla/mla.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory math/mla/verification math/mla/mla.tex
python - <<'PY'
import json, pathlib
for path in ['math/mla/tasks.jsonl', 'math/mla/events.jsonl']:
    for line in pathlib.Path(path).read_text().splitlines():
        if line.strip():
            json.loads(line)
print('jsonl ok')
PY
rg -n 'verbatim|def |class |import |einsum|MLAAttention|q_a_proj|kv_a_proj|kv_b_proj|load_format' math/mla/mla.tex
```

The final `rg` command should return no matches except terms intentionally kept
as validation labels, if any.
