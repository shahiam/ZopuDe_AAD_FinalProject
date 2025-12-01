import random
import sys

sys.setrecursionlimit(10000)

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
