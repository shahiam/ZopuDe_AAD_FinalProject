import time
from typing import Callable
from miller_rabin import is_prime as mr_is_prime
from trial_division import is_prime as td_is_prime

def benchmark(func: Callable[[int], bool], numbers: list[int]) -> float:
    """
    Benchmark a primality test function.

    Args:
        func (Callable[[int], bool]): Primality test function.
        numbers (list[int]): List of integers to test.

    Returns:
        float: Total elapsed time in seconds.
    """
    start: float = time.perf_counter()
    for n in numbers:
        func(n)
    end: float = time.perf_counter()

    return end - start


def main() -> None:
    """
    Benchmark the Miller-Rabin and Trial Division algorithms
    across multiple input sizes and save the results.

    Results are written to 'benchmarks/miller_rabin_benchmarks.json'.
    """
    pass


if __name__ == "__main__":
    main()
