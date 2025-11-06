import random

def miller_rabin_test(d: int, n: int) -> bool:
    """
    Perform one iteration of the Miller-Rabin test.

    Args:
        d (int): The odd component of n - 1 (i.e., n - 1 = 2^r * d).
        n (int): The number to test for primality.

    Returns:
        bool: True if this round passes (n is probably prime), False otherwise.
    """
    # Pick a random base 'a' in the range [2, n - 2]
    a = random.randint(2, n - 2)

    # compute x = a^d % n
    x = pow(a, d, n)

    # If x == 1 or x == n - 1, n passes this round
    if x == 1 or x == n - 1:
        return True
    
    # Repeatedly square x while doubling d, stopping if x becomes 1 or n - 1
    # If neither condition occurs before d reaches n - 1, n is composite
    while d != n - 1:
        x = (x * x) % n
        d *= 2

        # If x becomes 1, n is definitely composite
        if x == 1:
            return False
        # If x becomes n - 1, n passes this round
        if x == n - 1:
            return True

    return False

def is_prime(n: int, k: int = 40) -> bool:
    """
    Determine if a number is prime using the Miller-Rabin primality test.

    This is a probabilistic test: each iteration reduces the chance of a
    false positive (composite identified as prime) to at most 1/4.
    After k iterations, the error probability is <= (1/4)^k.

    Args:
        n (int): The integer to test.
        k (int): Number of accuracy iterations. Default is 40.

    Returns:
        bool: True if n is probably prime, False otherwise.
    """
    # Base cases
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0:
        return False
    
    # Write n - 1 as 2^r * d with d odd
    d = n - 1
    while d % 2 == 0:
        d //= 2

    # Repeat the test k times to reduce error probability
    for _ in range(k):
        if not miller_rabin_test(d, n):
            return False
    
    # Probably prime if all tests passed
    return True
