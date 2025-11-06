import math

def is_prime(n: int) -> bool:
    """
    Check if a number is prime using the trial division method.

    Args:
        n (int): The integer to test.

    Returns:
        bool: True if n is prime, False otherwise.
    """
    # Base cases
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0:
        return False
    
    # Only check divisibility up to sqrt(n)
    root_n = math.isqrt(n)

    # Test only odd divisors since even numbers are already excluded
    for i in range(3, root_n + 1, 2):
        if n % i == 0:
            return False

    # If no divisors found, n is prime
    return True
