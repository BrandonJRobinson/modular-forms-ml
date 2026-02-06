import numpy as np

class QSeries:
    """
    A class representing a truncated q-series expansion.
    """
    def __init__(self, coeffs, order=None):
        if isinstance(coeffs, list):
            coeffs = np.array(coeffs)
        self.coeffs = coeffs
        if order is None:
            self.order = len(coeffs) - 1
        else:
            self.order = order
            if len(self.coeffs) > order + 1:
                 self.coeffs = self.coeffs[:order+1]
            elif len(self.coeffs) < order + 1:
                self.coeffs = np.pad(self.coeffs, (0, order + 1 - len(self.coeffs)))

    def __repr__(self):
        return f"QSeries(order={self.order}, coeffs={self.coeffs[:5]}...)"

    def __add__(self, other):
        if isinstance(other, (int, float, complex)):
            new_coeffs = self.coeffs.copy()
            new_coeffs[0] += other
            return QSeries(new_coeffs, self.order)
        order = min(self.order, other.order)
        return QSeries(self.coeffs[:order+1] + other.coeffs[:order+1], order)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (int, float, complex)):
            new_coeffs = self.coeffs.copy()
            new_coeffs[0] -= other
            return QSeries(new_coeffs, self.order)
        order = min(self.order, other.order)
        return QSeries(self.coeffs[:order+1] - other.coeffs[:order+1], order)

    def __rsub__(self, other):
        if isinstance(other, (int, float, complex)):
            new_coeffs = -self.coeffs.copy()
            new_coeffs[0] += other
            return QSeries(new_coeffs, self.order)
        # Assuming other is QSeries, handled by __sub__
        return NotImplemented

    def __neg__(self):
        return QSeries(-self.coeffs, self.order)

    def __mul__(self, other):
        if isinstance(other, (int, float, complex)):
            return QSeries(self.coeffs * other, self.order)
        
        # Polynomial multiplication (convolution)
        # Using numpy.convolve is generally efficient for moderate sizes
        # For very large sizes, FFT might be better, but we need exact arithmetic if possible
        # However, for ML datasets, float precision is usually fine.
        # Let's stick to numpy.convolve.
        
        order = min(self.order, other.order)
        full_conv = np.convolve(self.coeffs, other.coeffs)
        return QSeries(full_conv[:order+1], order)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __pow__(self, power):
        # Handle numpy scalars
        if isinstance(power, (np.integer, int)):
            power = int(power)
        else:
            raise ValueError(f"Only integer powers supported for now. Got {type(power)}")
        
        if power == 0:
            coeffs = np.zeros(self.order + 1)
            coeffs[0] = 1
            return QSeries(coeffs, self.order)
        
        if power < 0:
             return self.invert() ** (-power)

        res = QSeries([1], self.order)
        base = self
        while power > 0:
            if power % 2 == 1:
                res = res * base
            base = base * base
            power //= 2
        return res

    def invert(self):
        # Power series inversion
        # If A(q) * B(q) = 1, then b_0 = 1/a_0
        # b_n = -1/a_0 * sum_{k=1}^n a_k b_{n-k}
        
        if self.coeffs[0] == 0:
            # Handle valuation shift? For now assume usually invertible or just shift
            # If leading term is q^k, we return q^-k * ...
            # But QSeries structure assumes q^0 start.
            # Let's raise error for now unless unit.
             raise ValueError("Series not invertible (constant term is 0).")

        n = self.order
        b = np.zeros(n + 1)
        a = self.coeffs
        b[0] = 1.0 / a[0]
        
        for k in range(1, n + 1):
             # term = sum_{i=1}^k a[i] * b[k-i]
             # We need a[1]...a[k] and b[k-1]...b[0]
             # slice a[1:k+1], reverse b[0:k]
             # Actually, just use convolve concept or loop
             term = np.dot(a[1:k+1], b[k-1::-1])
             b[k] = -term / a[0]
        
        return QSeries(b, self.order)

    def valuate(self):
        """Returns the order of the first non-zero coefficient."""
        for i, c in enumerate(self.coeffs):
            if not np.isclose(c, 0):
                return i
        return self.order + 1

    def hecke(self, p, k):
        """
        Applies the Hecke Operator T_p to the q-series, assuming it is a Modular Form of weight k.
        Formula: a_n(T_p f) = a_{np} + p^(k-1) * a_{n/p} (if p|n, else 0)
        Returns a new QSeries with reduced order (approx order/p).
        """
        new_order = self.order // p
        new_coeffs = np.zeros(new_order + 1)
        
        pk_1 = p**(k-1)
        
        for n in range(new_order + 1):
            # Term 1: a_{np}
            term1 = self.coeffs[n * p]
            
            # Term 2: p^(k-1) * a_{n/p}
            term2 = 0
            if n % p == 0:
                idx = n // p
                term2 = pk_1 * self.coeffs[idx]
                
            new_coeffs[n] = term1 + term2
            
        return QSeries(new_coeffs, new_order)


def sigma_k(n, k):
    """Sum of k-th powers of divisors of n."""
    if n <= 0: return 0
    # Create divisors
    divs = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divs.append(i)
            if i*i != n:
                divs.append(n // i)
    return sum(d**k for d in divs)

def get_sigma_series(k, order):
    """Returns q-series for sum_{n=1} sigma_k(n) q^n."""
    coeffs = np.zeros(order + 1)
    for n in range(1, order + 1):
        coeffs[n] = sigma_k(n, k)
    return QSeries(coeffs, order)

def eisenstein_E4(order=100):
    # E4 = 1 + 240 * sum sigma_3(n) q^n
    s3 = get_sigma_series(3, order)
    return 1 + 240 * s3

def eisenstein_E6(order=100):
    # E6 = 1 - 504 * sum sigma_5(n) q^n
    s5 = get_sigma_series(5, order)
    return 1 - 504 * s5

def dedekind_eta(order=100):
    # eta(q) = q^(1/24) * prod_{n=1}^inf (1 - q^n)
    # This function returns the series WITHOUT the q^(1/24) factor,
    # i.e., prod(1 - q^n). The fractional power is handled logically.
    
    # We can use Euler's pentagonal number theorem for sparse multiplication
    # prod(1 - q^n) = sum_{k} (-1)^k q^(k(3k-1)/2)
    
    coeffs = np.zeros(order + 1)
    k = 0
    
    # k=0 term (for completeness in loop logic, though theorem sums over k in Z)
    # The sum range for k is such that generalized pentagonal number <= order
    # Generalized pentagonal numbers: p_k = k(3k-1)/2 for k = 0, 1, -1, 2, -2, ...
    
    # Actually, simpler:
    coeffs[0] = 1 # k=0 -> p_0=0 term is not in product expansion directly usually, wait.
    # The formula is sum_{m=-inf}^{inf} (-1)^m q^(m(3m-1)/2)
    
    # m=0: q^0, coeff 1
    # m=1: q^1, coeff -1
    # m=-1: q^2, coeff -1
    # m=2: q^5, coeff 1
    # m=-2: q^7, coeff 1
    
    m = 1
    while True:
        p1 = m * (3 * m - 1) // 2
        p2 = m * (3 * m + 1) // 2 # Represents -m
        
        if p1 > order and p2 > order:
            break
            
        if p1 <= order:
            coeffs[p1] = (-1)**m
        
        if p2 <= order:
            coeffs[p2] = (-1)**m
            
        m += 1
        
    return QSeries(coeffs, order)

def discriminant_Delta(order=100):
    # Delta = eta^24
    # Note: eta is q^(1/24) * series.
    # eta^24 = q * (series)^24.
    # So we compute (series)^24, then shift by q^1 (pad 0 at start).
    
    eta_ser = dedekind_eta(order) # This is just prod(1-q^n)
    # To avoid huge coefficients in intermediate powers, we can compute eta^24 carefully?
    # Or just use E4^3 - E6^2 / 1728
    
    # Using E4 and E6 is often computationally more standard for Delta
    # Delta = (E4^3 - E6^2) / 1728
    
    e4 = eisenstein_E4(order)
    e6 = eisenstein_E6(order)
    
    num = (e4**3) - (e6**2)
    delta = num * (1.0/1728.0)
    
    return delta

def j_invariant(order=100):
    # j = E4^3 / Delta
    # Delta starts with q^1, so it has no constant term.
    # We need to handle division by series starting with 0.
    
    delta = discriminant_Delta(order + 1) # Need slightly more order to be safe
    # Delta = q * (1 - 24q + ...)
    # factor out q
    
    # Check valuation
    val = delta.valuate()
    if val != 1:
        # Should be 1
        # If order is small, it might be 0? No, Delta is O(q).
        pass
        
    # Remove the q factor from Delta for inversion
    delta_shifted_coeffs = delta.coeffs[1:] 
    # Pad to maintain order
    delta_shifted_coeffs = np.pad(delta_shifted_coeffs, (0, 1))
    
    delta_shifted = QSeries(delta_shifted_coeffs, order)
    
    delta_inv = delta_shifted.invert()
    
    e4 = eisenstein_E4(order)
    e4_3 = e4**3
    
    # j_shifted = E4^3 * (1/Delta_shifted)
    # This corresponds to j * q
    # So j_series = q^-1 * (res)
    # Since we can't represent q^-1 in QSeries easily without shifting logic,
    # we return a tuple or special object? 
    # Or just return the coefficients starting from q^-1.
    
    # Let's return coefficients where index 0 corresponds to q^-1. (Laurent Series-ish)
    
    res = e4_3 * delta_inv
    # coeff[0] is coeff of q^0 in (j*q), so it is c_{-1} for j.
    # coeff[1] is c_0
    
    # j(q) = q^-1 + 744 + 196884q + ...
    
    return res

