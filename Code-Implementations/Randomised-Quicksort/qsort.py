import random
import time
import sys

sys.setrecursionlimit(10000)


def quicksort(arr):
    """
    simple quicksort implementation.
    selects first element as pivot,
    splits into left (< pivot) and right (>= pivot),
    recursively sorts sublists.
    """
    if len(arr) <= 1:
        return arr

    pivot = arr[0]

    left = [x for x in arr[1:] if x < pivot]
    right = [x for x in arr[1:] if x >= pivot]

    return quicksort(left) + [pivot] + quicksort(right)


def randomized_quicksort(arr):
    """
    randomized quicksort chooses random pivot,
    splits into left (< pivot),
    middle (== pivot), right (> pivot),
    recursively sorts left and right lists.
    """
    if len(arr) <= 1:
        return arr

    pivot = random.choice(arr)

    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    return randomized_quicksort(left) + middle + randomized_quicksort(right)


def benchmark(func, data):
    """
    runs sorting func on a copy of data,
    returns elapsed time rounded to 5 decimals.
    """
    start = time.time()
    func(data.copy())
    end = time.time()
    return round(end - start, 5)


def generate_datasets(size):
    """
    generates 4 datasets of given size:
    - random integers,
    - sorted ascending,
    - sorted descending,
    - lots of duplicates.
    """
    return {
        "Random": [random.randint(0, size) for _ in range(size)],
        "Sorted": list(range(size)),
        "Reverse Sorted": list(range(size, 0, -1)),
        "Duplicates": [random.choice([5, 10, 15, 20]) for _ in range(size)]
    }


def main():
    SIZE = 5000
    datasets = generate_datasets(SIZE)

    print(f"benchmarking on input size: {SIZE}\n")
    print(f"{'Dataset':<20} {'Regular QS':<15} {'Randomized QS'}")
    print("-" * 50)

    for name, data in datasets.items():
        t1 = benchmark(quicksort, data)
        t2 = benchmark(randomized_quicksort, data)
        print(f"{name:<20} {t1:<15} {t2}")


if __name__ == "__main__":
    main()
