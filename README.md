# resiliencepy

**resiliencepy** is an open-source, recovery-centric Python library for modeling and measuring **supply chain resilience**.

Unlike traditional supply chain tools that focus on cost or optimization, resiliencepy focuses on **how systems absorb shocks and recover over time**, using standardized recovery curves and resilience metrics.

---

## Why resiliencepy?

Supply chain resilience is widely discussed, but rarely standardized in code.

Most existing tools:
- Model disruption *impact*, not *recovery*
- Use ad-hoc resilience metrics
- Are difficult to reuse across studies or applications

**resiliencepy fills this gap** by providing a simple, reusable computational core for recovery-based resilience analysis.

---

## Key Features

- **Recovery-first modeling** (not network-heavy)
- **Standardized resilience metrics**
  - Time-to-Recovery (TTR)
  - Area of Loss
  - Resilience Index
- **Interpretable recovery strategies**
  - Safety stock
  - Expediting
  - Dual sourcing
  - Rerouting
- **Vectorized simulations**
  - Single or batch scenarios
- **Minimal dependencies**
  - NumPy only

---

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17906957.svg)](https://doi.org/10.5281/zenodo.17906957)


## Installation

```bash
pip install resiliencepy
