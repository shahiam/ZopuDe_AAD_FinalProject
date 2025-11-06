import time
import os
from typing import Callable
from miller_rabin import is_prime as mr_is_prime
from trial_division import is_prime as td_is_prime

def benchmark(func: Callable[[int], bool], numbers: list[int]) -> tuple[float, int]:
    pass


def main() -> None:
    pass


if __name__ == "__main__":
    main()
