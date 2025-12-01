from statistics import mean, stdev
import time
import random

from trial_division import is_prime as trial_prime
from miller_rabin import is_prime as miller_prime

def summarize(times):
    """Return mean and std deviation of a list of runtime pairs."""
    trials = [t[0] for t in times]
    mrs    = [t[1] for t in times]
    return (
        mean(trials), stdev(trials), mean(mrs), stdev(mrs)
    )

def time_call(func, n, repeats=3):
    """Return average runtime of func(n) over multiple runs."""
    total = 0.0
    for _ in range(repeats):
        start = time.perf_counter()
        func(n)
        total += time.perf_counter() - start
    return total / repeats


def generate_prime(bit_size, tester, k):
    """Generate one random prime candidate of a given bit size."""
    while True:
        n = random.getrandbits(bit_size) | 1
        if tester(n, k):
            return n


def generate_prime_list(bit_size, count, tester, k):
    """Generate a list of prime numbers of the given bit size."""
    return [generate_prime(bit_size, tester, k) for _ in range(count)]


def main():
    k = 5
    repeats = 3

    print("\n=== Primality Test Benchmark ===\n")

    batch_sizes = {
        8: 10,
        16: 10,
        32: 10
    }

    for bits, count in batch_sizes.items():
        print(f"\n{bits}-bit primes:")
        print("-" * 60)

        nums = generate_prime_list(bits, count, miller_prime, k)

        results = []

        for n in nums:
            td = time_call(trial_prime, n, repeats)
            mr = time_call(lambda x: miller_prime(x, k), n, repeats)
            results.append((td, mr))

            print(f"n={n:<12} | Trial Division: {td:.6f}s | Miller Rabin: {mr:.6f}s")

        # Summary statistics
        mean_td, std_td, mean_mr, std_mr = summarize(results)
        
        print("-" * 60)
        print(f" Mean  | Trial Division: {mean_td:.6f}s | Miller Rabin: {mean_mr:.6f}s")
        print(f" STD   | Trial Division: {std_td:.6f} | Miller Rabin: {std_mr:.6f}")


if __name__ == "__main__":
    main()
