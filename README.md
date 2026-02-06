# Modular Forms ML

Generate datasets of **weakly holomorphic modular forms** for machine learning experiments. Train neural networks to distinguish true modular forms from asymptotic mimics.

## Installation

```bash
# Clone the repository
git clone https://github.com/brandonrobinson/modular-forms-ml.git
cd modular-forms-ml/modular_forms

# Install with pip
pip install -e .

# Or install with ML dependencies
pip install -e ".[ml]"
```

## Quick Start

### Generate a Dataset

```bash
# Balanced dataset (modular + fake forms)
python -m modular_forms.dataset

# Only modular forms
python -m modular_forms.dataset --modular-only -o modular_only.npz

# Only fake forms
python -m modular_forms.dataset --fakes-only --n-fakes 5000 -o fakes.npz

# Custom configuration
python -m modular_forms.dataset --n-polynomials 500 --n-etas 500 --max-level 6 -o custom.npz
```

### Python API

```python
from modular_forms.dataset import generate_modular_forms, generate_fake_forms, generate_dataset

# Generate only modular forms
modular_data = generate_modular_forms(n_polynomials=100, n_eta_quotients=100)

# Generate only fakes
fake_data = generate_fake_forms(count=200)

# Generate complete dataset
generate_dataset(output_path="my_dataset.npz", n_polynomials=1000)
```

### Train a Classifier

```bash
pip install torch matplotlib  # if not already installed
python train_classifier.py
```

## Dataset Format

The `.npz` file contains:

| Key | Shape | Description |
|-----|-------|-------------|
| `coeffs` | `(N, 111)` | q-expansion coefficients for $q^{-10}$ to $q^{100}$ |
| `labels` | `(N,)` | `1` = modular, `0` = fake |
| `metadata` | `(N, 4)` | `(weight, q_start, level, type_id)` |

Type IDs: `0`=Fake, `1`=Polynomial, `2`=Eta Quotient, `3`=Shifted

## CLI Options

```
python -m modular_forms.dataset --help

Options:
  -o, --output          Output file path
  --modular-only        Generate only modular forms
  --fakes-only          Generate only fake forms
  --n-polynomials       Number of polynomial samples
  --n-etas              Number of eta quotient samples
  --n-shifted           Number of shifted form samples
  --n-fakes             Number of fake samples
  --order               q-expansion precision (default: 100)
  --max-level           Maximum Γ₀(N) level (default: 10)
  --k-min, --k-max      Weight range (default: -10 to 10)
  -q, --quiet           Suppress progress output
```

## Results

| Task | Accuracy |
|------|----------|
| Real vs Fake Classification | **100%** |
| Generator Identification | **97.6%** |
| Hecke Saliency Correlation | **+0.37** |

## License

MIT
