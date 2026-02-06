"""
Modular Forms Dataset Generator

Generates datasets of modular forms and/or fake (non-modular) forms for ML experiments.

CLI Usage:
    python -m modular_forms.dataset                          # Balanced dataset
    python -m modular_forms.dataset --modular-only           # Only modular forms
    python -m modular_forms.dataset --fakes-only             # Only fake forms
    python -m modular_forms.dataset --n-polynomials 500      # Custom counts
"""

import numpy as np
import argparse
from collections import defaultdict
from .generators import generate_polynomials, generate_eta_quotients, apply_weight_shifters, generate_fake_data


def generate_modular_forms(
    n_polynomials=1000,
    n_eta_quotients=1000,
    n_shifted=1000,
    order=100,
    max_level=10,
    max_poly_degree=5,
    k_range=(-10, 10),
    verbose=True
):
    """
    Generate modular form samples only.
    
    Returns:
        list of tuples: (coeffs, q_start, weight, level, type_id)
        where type_id: 1=Polynomial, 2=Eta Quotient, 3=Shifted
    """
    if verbose:
        print("Generating Polynomials...")
    polys = []
    per_N = max(1, n_polynomials // max_level)
    for N in range(1, max_level + 1):
        batch = generate_polynomials(N, max_degree=max_poly_degree, count=per_N, order=order)
        polys.extend([(c, q, w, n, 1) for c, q, w, n in batch])  # type_id=1
    
    if verbose:
        print(f"  Generated {len(polys)} polynomials.")
    
    if verbose:
        print("Generating Eta Quotients...")
    etas_raw = generate_eta_quotients(N_max=max_level, k_range=k_range, count=n_eta_quotients, order=order)
    etas = [(c, q, w, n, 2) for c, q, w, n in etas_raw]  # type_id=2
    if verbose:
        print(f"  Generated {len(etas)} eta quotients.")
    
    shifted = []
    if n_shifted > 0:
        if verbose:
            print("Generating Shifted Forms...")
        source = [item[:4] for item in polys[:len(polys)//2] + etas[:len(etas)//2]]
        if len(source) == 0:
            source = [item[:4] for item in polys + etas]
        
        count_per = max(1, n_shifted // len(source)) if source else 0
        shifted_raw = apply_weight_shifters(source, count_per_item=count_per, k_range=k_range, order=order)
        shifted = [(c, q, w, n, 3) for c, q, w, n in shifted_raw]  # type_id=3
        
        if len(shifted) > n_shifted:
            import random
            random.shuffle(shifted)
            shifted = shifted[:n_shifted]
        if verbose:
            print(f"  Generated {len(shifted)} shifted forms.")
    
    return polys + etas + shifted


def generate_fake_forms(
    count=1000,
    order=100,
    k_range=(-10, 10),
    pole_range=(1, 5),
    verbose=True
):
    """
    Generate fake (non-modular) form samples.
    
    Args:
        count: Number of fake samples to generate
        order: q-expansion precision
        k_range: Weight range (min, max)
        pole_range: Pole order range (min, max)
        verbose: Print progress
    
    Returns:
        list of tuples: (coeffs, q_start, weight, level=0, type_id=0)
    """
    if verbose:
        print(f"Generating {count} fake samples...")
    
    k_min, k_max = k_range
    m_min, m_max = pole_range
    
    # Group by (weight, pole_order) for efficiency
    reqs = defaultdict(int)
    for _ in range(count):
        w = np.random.randint(k_min, k_max + 1)
        m = np.random.randint(m_min, m_max + 1)
        reqs[(w, m)] += 1
    
    fake_data = []
    for (w, m), cnt in reqs.items():
        batch = generate_fake_data(weight=w, m=m, count=cnt, order=order)
        
        for c_n in batch:
            full = np.zeros(m + 1 + len(c_n))
            full[0] = 1  # q^-m term
            full[m+1:] = c_n
            fake_data.append((full, -m, w, 0, 0))  # level=0, type_id=0
    
    if verbose:
        print(f"  Generated {len(fake_data)} fakes.")
    
    return fake_data


def _align_to_grid(items, start_n=-10, end_n=100):
    """Align samples to a common coefficient grid [q^start_n, q^end_n]."""
    X = []
    Metadata = []
    
    for coeffs, q_start, weight, level, type_id in items:
        vec = np.zeros(end_n - start_n + 1)
        
        valid_start = max(start_n, q_start)
        valid_end = min(end_n, q_start + len(coeffs) - 1)
        
        if valid_end >= valid_start:
            src_start = valid_start - q_start
            src_end = valid_end - q_start + 1
            dst_start = valid_start - start_n
            dst_end = valid_end - start_n + 1
            vec[dst_start:dst_end] = coeffs[src_start:src_end]
        
        X.append(vec)
        Metadata.append((weight, q_start, level, type_id))
    
    return np.array(X), np.array(Metadata)


def generate_dataset(
    output_path="modular_forms_dataset.npz",
    n_polynomials=1000,
    n_eta_quotients=1000,
    n_shifted=1000,
    n_fakes=None,
    order=100,
    max_level=10,
    max_poly_degree=5,
    k_range=(-10, 10),
    modular_only=False,
    fakes_only=False,
    verbose=True
):
    """
    Generate a complete dataset and save to .npz file.
    
    Args:
        output_path: Output file path
        n_polynomials: Number of polynomial samples
        n_eta_quotients: Number of eta quotient samples
        n_shifted: Number of shifted form samples
        n_fakes: Number of fake samples (default: match modular count)
        order: q-expansion precision
        max_level: Maximum Γ₀(N) level
        max_poly_degree: Maximum polynomial degree
        k_range: Weight range
        modular_only: Generate only modular forms
        fakes_only: Generate only fake forms
        verbose: Print progress
    
    Returns:
        str: Path to saved dataset
    """
    all_items = []
    labels = []
    
    if not fakes_only:
        modular_data = generate_modular_forms(
            n_polynomials=n_polynomials,
            n_eta_quotients=n_eta_quotients,
            n_shifted=n_shifted,
            order=order,
            max_level=max_level,
            max_poly_degree=max_poly_degree,
            k_range=k_range,
            verbose=verbose
        )
        all_items.extend(modular_data)
        labels.extend([1] * len(modular_data))
    
    if not modular_only:
        n_modular = len(all_items)
        if n_fakes is None:
            n_fakes = n_modular if n_modular > 0 else 1000
        
        fake_data = generate_fake_forms(
            count=n_fakes,
            order=order,
            k_range=k_range,
            verbose=verbose
        )
        all_items.extend(fake_data)
        labels.extend([0] * len(fake_data))
    
    X, Metadata = _align_to_grid(all_items)
    Y = np.array(labels)
    
    # Shuffle
    perm = np.random.permutation(len(Y))
    X = X[perm]
    Y = Y[perm]
    Metadata = Metadata[perm]
    
    np.savez(output_path, coeffs=X, labels=Y, metadata=Metadata, info=np.array([-10, 100]))
    
    if verbose:
        print(f"Saved dataset to {output_path} with shape {X.shape}")
        print(f"  Modular: {np.sum(Y == 1)}, Fake: {np.sum(Y == 0)}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate modular forms dataset for ML experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m modular_forms.dataset                          # Balanced dataset
  python -m modular_forms.dataset --modular-only           # Only modular forms
  python -m modular_forms.dataset --fakes-only             # Only fake forms
  python -m modular_forms.dataset --n-polynomials 500 -o custom.npz
        """
    )
    
    parser.add_argument("-o", "--output", default="modular_forms_dataset.npz",
                        help="Output file path (default: modular_forms_dataset.npz)")
    
    parser.add_argument("--modular-only", action="store_true",
                        help="Generate only modular forms (no fakes)")
    parser.add_argument("--fakes-only", action="store_true",
                        help="Generate only fake forms (no modular)")
    
    parser.add_argument("--n-polynomials", type=int, default=1000,
                        help="Number of polynomial samples (default: 1000)")
    parser.add_argument("--n-etas", type=int, default=1000,
                        help="Number of eta quotient samples (default: 1000)")
    parser.add_argument("--n-shifted", type=int, default=1000,
                        help="Number of shifted form samples (default: 1000)")
    parser.add_argument("--n-fakes", type=int, default=None,
                        help="Number of fake samples (default: match modular count)")
    
    parser.add_argument("--order", type=int, default=100,
                        help="q-expansion precision (default: 100)")
    parser.add_argument("--max-level", type=int, default=10,
                        help="Maximum Γ₀(N) level (default: 10)")
    parser.add_argument("--k-min", type=int, default=-10,
                        help="Minimum weight (default: -10)")
    parser.add_argument("--k-max", type=int, default=10,
                        help="Maximum weight (default: 10)")
    
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Suppress progress output")
    
    args = parser.parse_args()
    
    if args.modular_only and args.fakes_only:
        parser.error("Cannot use --modular-only and --fakes-only together")
    
    generate_dataset(
        output_path=args.output,
        n_polynomials=args.n_polynomials,
        n_eta_quotients=args.n_etas,
        n_shifted=args.n_shifted,
        n_fakes=args.n_fakes,
        order=args.order,
        max_level=args.max_level,
        k_range=(args.k_min, args.k_max),
        modular_only=args.modular_only,
        fakes_only=args.fakes_only,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
