# Quantized KV Cache for vLLM Ascend

Owner-led migration carrier for INT8, KIVI INT4, and dynamic quantized KV-cache work. This is distinct from offline model quantization and the Adaptive Quantized KV observer project.

**Status: quantized KV dtype, packed-layout, and fail-closed resolution contracts are installable and tested; device kernels and vLLM Ascend attachment remain blocked until `HOST_CONTRACT.md` is implemented.**

Technical ownership belongs to @hustcui, @SuccinctPaul. Source extraction must preserve exact authorship, license, tests, constraints, and evidence before activation is considered.

See [MAINTAINERS.md](MAINTAINERS.md) and [PROVENANCE.md](PROVENANCE.md).

## Extension framework

Extension ID: `org.vllm-hust.quantized-kv-cache`

This repository follows the vLLM-HUST Extension Template. The current package
is deliberately `import_only`: it can be built, installed, discovered, and
inspected, but Extension Manager must refuse enablement until the maintainers
land a real host contract, implementation, compatibility evidence, and tests.

```bash
python -m pip install "vllm-hust-ext @ git+https://github.com/vLLM-HUST/extension-manager.git@main"
python -m pip install -e ".[test]"
vllm-hust-ext extension inspect org.vllm-hust.quantized-kv-cache
vllm-hust-ext extension check org.vllm-hust.quantized-kv-cache
pytest -q
```

The static Manifest 0.2 descriptor lives inside the Python distribution under
`src/`. Installation alone changes no vLLM behavior.
