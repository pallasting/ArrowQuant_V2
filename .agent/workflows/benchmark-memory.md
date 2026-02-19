---
description: How to run the Localized Memory Fidelity and Association Benchmark
---

This workflow validates the performance of the localized ArrowEngine-Native memory system, including 4-bit reconstruction fidelity and knowledge graph associative recall.

// turbo-all
1. Ensure the ArrowEngine model is loaded in the `models/` directory.
2. Run the fidelity benchmark:
   ```bash
   python tests/compression_fidelity_benchmark.py
   ```
3. Verify that Fidelity is >= 0.85 and Recall is [EXCELLENT].
