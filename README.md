# PAD

Official repository for **“Prompt-Anchored Vision–Text Distillation for Lifelong Person Re-identification”** (CVPR 2026).

## Overview

We propose **PAD** (Prompt-Anchored Vision–Text Distillation), a framework for **exemplar-free lifelong person re-identification (LReID)**.

PAD introduces a frozen text encoder as a stable semantic anchor across domains, and combines:

* **TA-Prompt** for text-side semantic alignment
* **VA-Prompt** for visual-side incremental adaptation
* **Fixed textual distillation** to preserve semantic consistency
* **EMA-based visual distillation** to mitigate semantic drift and catastrophic forgetting

The proposed method achieves strong performance on both **seen** and **unseen** domains under standard lifelong ReID benchmarks.

## Status

Code will be released soon.

## Paper

**Prompt-Anchored Vision–Text Distillation for Lifelong Person Re-identification**
CVPR 2026

## Contact

Wen Wen ([zuzi666666@gmail.com](mailto:zuzi666666@gmail.com))

## Citation

```bibtex
@inproceedings{wen2026pad,
  title={Prompt-Anchored Vision–Text Distillation for Lifelong Person Re-identification},
  author={Wen, Wen and Chen, Hao and Zhang, Shiliang},
  booktitle={CVPR},
  year={2026}
}
```
