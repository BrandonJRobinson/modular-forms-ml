import numpy as np
import scipy.special
from itertools import product
from .core import QSeries, eisenstein_E4, eisenstein_E6, discriminant_Delta
from .hauptmodul import get_hauptmodul, check_ligozat_condition, get_factors

def generate_polynomials(N, max_degree, count, order=100):
    """
    Generates 'count' random polynomials of the Hauptmodul j_N.
    Returns list of (coeffs, q_pow, weight, N)
    """
    j, const_adj = get_hauptmodul(N, order=order)

    # j starts with q^-1.
    # To avoid huge negative powers, we might want to multiply by Delta^k?
    # But user wants Weight 0 forms.
    # So we just keep them as Laurent series in q.
    # Our QSeries implementation handles positive powers of q.
    # j has a pole.
    # We should track the pole order. P(j) will have pole of order degree.
    
    # Since QSeries supports negative powers via an offset or just implicitly?
    # My QSeries class doesn't support Laurent series directly, it assumes sum a_n q^n.
    # But j_N ~ q^-1 + ...
    # We returned j as (series_starting_q0, shift=-1).
    # So j = q^-1 * J_0(q).
    # j^k = q^-k * J_0(q)^k.
    
    # We need a data structure to represent general Laurent series for this dataset.
    # The network inputs will likely be the coefficients of q^n for n >= n0.
    # User constraint: "Pole order" - we assume we want coefficients $c(n)$ for $n$ from say -10 to 100?
    
    # Let's standardize the output format:
    # Returns list of (coefficients, min_power)
    
    # Precompute powers of J_0
    j_0 = j # This is the series part starting at q^0 (actually q^-1 in reality but 0 in object)
    # Wait, get_hauptmodul returns (series, shift).
    
    j_powers = []
    curr = QSeries([1], order) # j^0
    j_powers.append(curr)
    
    for _ in range(max_degree):
        curr = curr * j_0
        j_powers.append(curr)
        
    generated = []
    for _ in range(count):
        # Random polynomial coefficients
        coeffs = np.random.randint(-10, 11, size=max_degree + 1)
        # Avoid 0 polynomial
        if np.all(coeffs == 0): coeffs[0] = 1
        
        # Construct P(j) = sum c_k * j^k
        # j^k = q^(-k) * J_0^k
        # We need to align them.
        # Max pole is q^(-max_degree).
        # We align everything to start at q^(-max_degree).
        
        # Common min_power = -max_degree
        # For term k, shift is -k.
        # Relative shift: (-k) - (-max_degree) = max_degree - k.
        # We pad J_0^k with (max_degree - k) zeros at start.
        
        final_coeffs = np.zeros(order + 1) # This is heuristic size
        # We need to handle potential length mismatch
        
        # Let's use a large buffer
        buffer_size = order + max_degree + 1
        acc_coeffs = np.zeros(buffer_size)
        
        for k in range(max_degree + 1):
            if coeffs[k] == 0: continue
            
            # Term c_k * j^k
            # power shift = -k
            # index in buffer (where 0 is q^(-max_degree)):
            # q^(-max_degree) -> index 0
            # q^(-k) -> index (-k) - (-max_degree) = max_degree - k
            
            term_coeffs = j_powers[k].coeffs
            start_idx = max_degree - k
            end_idx = start_idx + len(term_coeffs)
            
            # Add to accumulator
            # handle truncation
            valid_len = min(len(term_coeffs), buffer_size - start_idx)
            acc_coeffs[start_idx : start_idx + valid_len] += coeffs[k] * term_coeffs[:valid_len]
            
        generated.append((acc_coeffs, -max_degree, 0, N)) # coeffs, q_pow, weight, Level
        
    return generated

def random_eta_quotient_params(N, max_r=24):
    """
    Finds a random valid r vector for Gamma0(N) eta quotient.
    Returns (r_dict, weight).
    """
    factors = get_factors(N)
    
    # We want random r_d such that conditions met.
    # Optimization: Randomly pick r_d for all but last 2 divisors, solve for last 2?
    # Or just standard lattice reduction?
    # For N <= 10, factors count is small (max 4).
    # Just rejection sampling might work if space is dense?
    # Constraints are mod 24 and sum=2k.
    
    # Try 1000 times
    for _ in range(1000):
        r_vals = np.random.randint(-max_r, max_r + 1, size=len(factors))
        r_dict = dict(zip(factors, r_vals))
        
        # Check conditions
        # We want integer weight k
        w_sum = sum(r_vals)
        if w_sum % 2 != 0: continue
        k = w_sum // 2
        
        # Check level conditions
        # sum d*r = 0 mod 24
        # sum (N/d)*r = 0 mod 24
        if sum(d * r_dict[d] for d in factors) % 24 != 0: continue
        if sum((N // d) * r_dict[d] for d in factors) % 24 != 0: continue
        
        # Check if non-trivial (not all zero)
        if w_sum == 0 and all(x == 0 for x in r_vals): continue
        
        # We don't enforce holomorphic at cusps here?
        # User goal: "Weakly Holomorphic Modular Forms".
        # So poles at cusps are allowed?
        # "Purpose: Efficiently populate the dataset with diverse integer weights and levels."
        # Usually we want holomorphic on H, but meromorphic at cusps is fine.
        # Eta quotients are always holomorphic on H (no zeros/poles in H).
        # So this is satisfied.
        
        return r_dict, k
        
    return None, None

import math

def generate_fake_data(weight, m, count, order=100):
    """
    Generates fake coefficients using the asymptotic formula (Hardy-Ramanujan-Rademacher mimic).
    c(n) ~ 2pi * (n/m)^((k-1)/2) * I_{k-1}(4pi sqrt(mn))
    
    We generate noise variations.
    """
    # Base coefficients
    coeffs_base = []
    
    # If m is pole order, we assume q^-m + ...
    # mimic start from n=1?
    # Formula is for n > 0.
    # "Strategy 1: Generate from scratch fake mmf with the same growth"
    
    k = weight
    # If weight is 0, k=0.
    if k == 0:
        # Use I_1 (since |0-1| = 1)
        order_bessel = 1
        pow_factor = -0.5
    else:
        order_bessel = abs(k - 1)
        pow_factor = (k - 1) / 2.0
        
    n_vals = np.arange(1, order + 1)
    
    # Compute base terms
    # arg = 4 * PI * sqrt(m * n)
    arg = 4 * np.pi * np.sqrt(m * n_vals)
    
    # prefactor = (2pi / sqrt(2n)) * (n/m)^...
    # wait, formula has sqrt(2n) in denominator?
    # Usually partition function p(n) ~ ... 1/n ...
    # Standard Rademacher for j (k=0, m=1):
    # c(n) ~ e^(4pi sqrt(n)) / sqrt(2) n^0.75 ...
    # Let's follow user formula exactly:
    # c(n) approx (2pi / sqrt(2n)) * (n/m)^pow * I_v(arg)
    
    prefactor = (2 * np.pi) / np.sqrt(2 * n_vals)
    term_pow = (n_vals / m) ** pow_factor
    bessel = scipy.special.iv(order_bessel, arg)
    
    c_n = prefactor * term_pow * bessel
    
    # Create noisy versions
    data = []
    for _ in range(count):
        # Add multiplicative noise? Or Structural noise?
        # "Strategy 2: add noise to the real dataset"
        # "Strategy 1: fake mmf with SAME GROWTH"
        
        # We can perturb the coefficients slightly
        # c_n_fake = c_n * (1 + epsilon)
        noise = np.random.normal(1, 0.05, size=len(c_n)) # 5% noise
        c_fake = c_n * noise
        
        # Prepend zeros up to q^1 ?
        # Or if m=1, we have term q^-1?
        # The formula gives c(n) for n >= 1.
        # We assume q^-m + sum c(n) q^n.
        # Let's just output the positive part c(n) for n=1..order.
        
        data.append(c_fake)
        
    return data

def generate_eta_quotients(N_max, k_range, count, order=100):
    """
    Generates a list of valid eta quotients with random levels N <= N_max and weights in k_range.
    """
    generated = []
    attempts = 0
    k_min, k_max = k_range
    
    while len(generated) < count and attempts < count * 10:
        attempts += 1
        N = np.random.randint(1, N_max + 1)
        r_dict, k = random_eta_quotient_params(N)
        
        if r_dict is None: continue
        if k < k_min or k > k_max: continue
        
        # Determine if we calculate it now or just store params?
        # Let's calculate the series.
        # But wait, we need 'compute_eta_quotient' from hauptmodul?
        # That was intended for Hauptmoduls. It works generally for any r_dict.
        
        # We need to import compute_eta_quotient from .hauptmodul
        # But currently it's not exposed in generators.py imports.
        # I'll add the import inside function or update top level properly.
        from .hauptmodul import compute_eta_quotient
        
        series, q_pow = compute_eta_quotient(r_dict, order)
        
        # q_pow is the starting power.
        # The series returned starts at q^0 relative to the shift.
        # q_pow is total_q_exponent // 24.
        # Usually for modular forms, q_pow >= 0 (holomorphic).
        # Weakly holomorphic can have q_pow < 0.
        
        # Store as (coeffs, q_pow, weight, Level)
        generated.append((series.coeffs, q_pow, k, N))
        
    return generated

def apply_weight_shifters(data, count_per_item, k_range, order=100, max_coeff=1e30):
    """
    Applies E4^a * E6^b * Delta^c to input forms to shift weights.
    # data: list of (coeffs, q_pow, current_weight, Level)
    # max_coeff: reject forms with coefficients exceeding this magnitude (prevents overflow)
    """
    new_data = []
    k_min, k_max = k_range
    
    e4 = eisenstein_E4(order)
    e6 = eisenstein_E6(order)
    delta = discriminant_Delta(order)
    # Delta usually starts q^1 (so order + 1 needed for full precision? 
    # Delta q_pow = 1.
    
    # Precompute some powers?
    # Or just random walk.
    
    for (coeffs, q_pow, k, N) in data:
        current_series = QSeries(coeffs, order)
        
        for _ in range(count_per_item):
            # Random a, b, c
            # We want to shift weight 4a + 6b + 12c (c can be negative).
            # Constraint: new_k inside range.
            
            # Try to pick shift that lands in range
            target_k = np.random.randint(k_min, k_max + 1)
            diff = target_k - k
            
            # Find integer solution to 4a + 6b + 12c = diff
            # This is 2(2a + 3b + 6c) = diff -> diff must be even?
            # Modular forms of integer weight are even/odd?
            # Level 1 forms are always even weight.
            # But we have level N forms.
            # Shifters are level 1, so they add even weight.
            # So parity of weight is invariant under shifters.
            
            if diff % 2 != 0:
                # Can't shift parity with level 1 forms
                continue
                
            # Randomly pick c (BOUNDED: -1 to 1 to avoid extreme coefficients)
            c = np.random.randint(-1, 2)  # -1, 0, or 1
            rem = diff - 12 * c
            
            # 4a + 6b = rem
            # 2a + 3b = rem / 2 = R
            # Diophantine equation.
            
            # Simple solver: pick b, solve for a.
            # 2a = R - 3b -> R - 3b must be even.
            # b has same parity as R.
            
            R = rem // 2
            # Try a few b's (BOUNDED: 0 to 3 to limit coefficient growth)
            found = False
            for b in range(0, 4):  # cap at b=3
                if (R - 3*b) % 2 == 0:
                    a = (R - 3*b) // 2
                    if 0 <= a <= 3:  # cap at a=3
                        found = True
                        break
            
            if not found: continue
            
            # Apply transformation
            # New form = f * E4^a * E6^b * Delta^c
            
            # Compute multiplier
            # We need to handle Delta^c for c < 0.
            # Delta^-1 has a pole q^-1.
            
            term = QSeries([1], order)
            shift = 0
            
            if a > 0: term = term * (e4 ** a)
            if b > 0: term = term * (e6 ** b)
            
            if c > 0:
                term = term * (delta ** c)
                shift += c # Delta starts q^1
            elif c < 0:
                # Delta^-1 starts q^-1
                # We need inverse series.
                # delta = q * (1 - 24q ...)
                # delta_inv = q^-1 * (1 - 24q ...)^-1
                
                # We can't elevate delta to negative power inside QSeries directly 
                # if delta starts with 0.
                
                # Manually handle:
                d_coeffs = delta.coeffs
                # verify d[0]=0, d[1]!=0
                val = next((i for i,x in enumerate(d_coeffs) if x!=0), 0)
                d_unit = QSeries(d_coeffs[val:], order) # shift down
                
                d_inv = d_unit.invert()
                # effectively this is q^-val * d_inv
                
                term = term * (d_inv ** (-c))
                shift += (-c) * (-val)
                
            # Now multiply original
            res = current_series * term
            new_shift = q_pow + shift
            
            # Check coefficient magnitude - reject if too large
            max_abs = np.max(np.abs(res.coeffs))
            if max_abs > max_coeff:
                continue  # Skip this sample to avoid overflow issues
            
            new_data.append((res.coeffs, new_shift, k + diff, N))

            
    return new_data

