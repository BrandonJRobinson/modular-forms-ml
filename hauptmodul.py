import numpy as np
from itertools import product
from .core import QSeries, dedekind_eta, sigma_k

# Standard Hauptmoduls for N where they are known simple eta quotients
# Format: {divisor: power}
# N=1: j (special case)
KNOWN_HAUPTMODULS = {
    2: {1: 24, 2: -24}, # (eta(tau)/eta(2tau))^24 + 24 -> Order 1 at inf? No, order -1 at inf.
    # q^(1/24*24) / q^(2/24*-24) = q^(1 - 2) = q^-1. Correct.
    # Constant term adjustment needed: +24
    
    3: {1: 12, 3: -12}, # (eta(tau)/eta(3tau))^12 + 12
    4: {1: 8, 4: -8},   # (eta(tau)/eta(4tau))^8 + 8
    5: {1: 6, 5: -6},   # (eta(tau)/eta(5tau))^6 + 6
    7: {1: 4, 7: -4},   # (eta(tau)/eta(7tau))^4 + 4
    # N=6, 8, 9, 10 are missing from the simple pattern
    # N=13: (eta/eta13)^2 + 2...
}

def get_factors(n):
    return [i for i in range(1, n + 1) if n % i == 0]

def check_ligozat_condition(N, r_dict):
    """
    Checks if the eta quotient prod eta(d*tau)^r_d is a modular function on Gamma0(N).
    conditions:
    1. sum r_d = 0
    2. sum d * r_d = 0 mod 24
    3. sum (N/d) * r_d = 0 mod 24
    4. Product extends to squares? 
       Actually, just need (prod d^r_d) is square of rational? 
       Usually this is for character triviality.
       Assuming trivial character if sum r_d is even?
       Let's stick to the 3 linear conditions for weight 0 and level N.
    """
    r_list = [r_dict.get(d, 0) for d in get_factors(N)]
    
    # 1. Weight 0
    if sum(r_list) != 0:
        return False
        
    # 2. Level condition 1
    sum2 = sum(d * r_dict.get(d, 0) for d in get_factors(N))
    if sum2 % 24 != 0:
        return False
        
    # 3. Level condition 2
    sum3 = sum((N // d) * r_dict.get(d, 0) for d in get_factors(N))
    if sum3 % 24 != 0:
        return False
        
    return True

def get_cusp_order(N, r_dict, c_prime):
    """
    Computes order of vanishing at cusp 1/c' (where c'|N).
    Formula: v = (N / (24 * gcd(c'^2, N))) * sum_{d|N} (gcd(d, c')^2 * r_d / d)
    Note: We want pole at infinity (c'=N corresponds to 1/N ~ 0, wait.)
    Cusps are 1/c' for c'|N.
    Infinity is represented by 1/N (or 1/0 in projective).
    Ligozat formula usually gives order at cusp a/c.
    
    Standard cusps for Gamma0(N) are represented by fractions a/d where d|N.
    Order at cusp 1/d:
    Let's use the formula:
    v_{1/d} = (N / 24) * sum_{delta|N} ( r_delta * gcd(d, delta)^2 / (gcd(d^2, N) * delta) )
    
    Correction:
    v_{1/d}(f) = \frac{N}{24 \gcd(d^2, N)} \sum_{\delta|N} \frac{\gcd(d, \delta)^2}{\delta} r_\delta
    
    Infinity corresponds to d=N (1/N is typically equivalent to 1/0 under action?).
    Wait, at infinity (q-expansion), the order is just 1/24 * sum(delta * r_delta).
    Let's check this.
    For N=2, r={1:24, 2:-24}.
    Sum = 1*24 + 2*(-24) = 24 - 48 = -24.
    Order = -24/24 = -1. Correct. Simple pole at infinity.
    
    So for infinity, we just check sum(delta * r_delta) = -24.
    
    For other cusps (d|N, d < N), we want order >= 0.
    """
    
    # Check infinity
    # This is equivalent to d=N in the formula?
    # d=N: gcd(N^2, N) = N.
    # v = (N / 24N) * sum ( gcd(N, delta)^2 / delta * r )
    #   = (1/24) * sum ( delta^2 / delta * r ) = 1/24 sum (delta * r).
    # Yes, consistent.
    
    val = 0
    den = 24 * np.gcd(c_prime**2, N)
    
    num_sum = 0
    for delta, r in r_dict.items():
        g = np.gcd(delta, c_prime)
        # We need integer arithmetic, but terms might be fractional before sum?
        # The formula result is integer.
        # sum term: (g^2 * r) / delta. 
        # So we multiply by N/delta to keep common denominator?
        
        # term = (g^2 * r) / delta
        # total = (N/den) * term
        
        # Let's compute in float to be safe, then check integer-ness
        term = (g**2 * r) / delta
        num_sum += term
        
    v = (N / den) * num_sum
    return v

def find_hauptmodul_exponents(N, search_range=10):
    factors = get_factors(N)
    # Search for r_delta
    # We fix r_N to some values to prune?
    # No, brute force for small N is fast.
    # Dimension is number of divisors.
    # N=6: 1, 2, 3, 6 (4 dims). Range 10 -> 20^4 = 160,000. Feasible.
    # N=10: 1, 2, 5, 10 (4 dims). Feasible.
    # N=8: 1, 2, 4, 8 (4 dims). Feasible.
    # N=9: 1, 3, 9 (3 dims). Very fast.
    
    # To ensure pole at infinity, we need sum(d*r) = -24.
    
    ranges = [range(-search_range, search_range + 1) for _ in factors]
    
    for r_values in product(*ranges):
        r_dict = dict(zip(factors, r_values))
        
        # Fast check: pole at infinity
        if sum(d * r_dict[d] for d in factors) != -24:
            continue
            
        if not check_ligozat_condition(N, r_dict):
            continue
            
        # Check other cusps
        valid_cusps = True
        for d in factors:
            if d == N: continue # Already checked infinity
            v = get_cusp_order(N, r_dict, d)
            if v < 0: # We want holomorphic at other cusps (v >= 0)
                valid_cusps = False
                break
        
        if valid_cusps:
            # We found a modular function with simple pole at infinity and holomorphic elsewhere.
            # This is a Hauptmodul candidate (up to constant).
            return r_dict
            
    return None

def compute_eta_quotient(r_dict, order=100):
    # r_dict: {delta: power}
    res = QSeries([1], order)
    
    # Optimization: precompute etas?
    # Or just compute on fly.
    
    base_eta = dedekind_eta(order) # This is prod(1-q^n)
    
    # eta(d*tau) -> q^(d/24) * prod(1-q^(dn))
    # We ignore the q factor for QSeries multiplication, handle it at end.
    
    total_q_exponent = 0
    
    for delta, power in r_dict.items():
        # q-factor: (delta/24) * power
        total_q_exponent += (delta * power)
        
        # Series part: eta(q^delta)
        # We need to stretch base_eta: q -> q^delta
        # This inserts zeros.
        
        coeffs = np.zeros(order + 1)
        # fill
        # original indices k -> new indices k*delta
        limit = order // delta
        old_coeffs = base_eta.coeffs[:limit+1]
        coeffs[0 : (limit+1)*delta : delta] = old_coeffs
        
        subset_series = QSeries(coeffs, order)
        
        term = subset_series ** power
        res = res * term
        
    # The total q exponent should be -1 (normalized to -24/24).
    # Verify
    if total_q_exponent != -24:
        # This shouldn't happen for our Hauptmoduls
        pass
        
    # Shift result by q^-1 (since total_q_exponent is -24/24 = -1)
    # We return QSeries starting at q^-1?
    # Our QSeries class starts at q^0.
    # Let's return (res, min_pow)
    # If min_pow is -1, it means valid Hauptmodul start.
    
    # Actually, user expects QSeries.
    # Let's adjust to be a polynomial in terms of q^{-1}?
    # Or just return the raw series starting at q^-1.
    # Currently QSeries holds [c0, c1, c2...] corresponding to q^0, q^1...
    # We can hack it or wrap it.
    
    return res, total_q_exponent // 24

def get_hauptmodul(N, order=100):
    # Returns QSeries for j_N (starting q^-1)
    
    if N == 1:
        # Special case: j-invariant
        from .core import j_invariant
        return j_invariant(order), 744 # j(q) = q^-1 + 744 + ...

    r_dict = None
    if N in KNOWN_HAUPTMODULS:
        r_dict = KNOWN_HAUPTMODULS[N]
    
    if r_dict is None:
        # Search
        # Try finding one.
        # Cache results?
        r_dict = find_hauptmodul_exponents(N, search_range=24) # Increased range for N=6,8,10
    
    if r_dict is None:
        raise ValueError(f"Could not find Hauptmodul for N={N}")
        
    series, shift = compute_eta_quotient(r_dict, order)
    
    # We want to normalize such that expansion is q^-1 + c0 + ...
    # series current is 1 + ... (because eta starts with 1)
    # shift is -1.
    # So q^-1 * (1 + ...).
    # Coefficient of q^-1 is 1.
    
    # We often want the constant term to be 0 or specific.
    # Standard Hauptmoduls often have specific constant terms (like 24, 744, etc.)
    # For generated dataset, does it matter?
    # "Best way (constants are arbitrary)"
    # User listed: N=2 -> +24.
    # Let's just make the constant term 0?
    # Or just return as is.
    # The formula N=2: (eta/eta2)^24 + 24.
    # eta/eta2 starts q^-1 - 24 + ...
    # So +24 makes it q^-1 + 0 + ...
    
    # Let's standardize to have constant term 0.
    # q^-1 * (1 + c1 q + c2 q^2 ...) = q^-1 + c1 + c2 q ...
    # Constant term is c1 in series.coeffs[1].
    # We subtract c1.
    
    c1 = series.coeffs[1] if len(series.coeffs) > 1 else 0
    
    # We can't subtract scalar from q^-1 series easily in this structure
    # unless we interpret the QSeries as the q-part.
    # Let's just return the raw series and let the consumer handle the "q^-1" knowledge.
    # But wait, QSeries represents sum a_n q^n.
    # Here we have q^-1 P(q).
    
    return series, c1 # Return series and the constant term adjustment needed
    
