import numpy as np
import time

def deterministic_verification(A, B, C):
    # C is verified by multiplying A and B in the standard way.
    calc_matrix = np.dot(A, B)
    difference = calc_matrix - C
    # checking if the difference matrix is all zeros with an absolute tolerance (atol)
    return np.allclose(difference, np.zeros_like(difference), rtol=0, atol=1e-7)
    # The above line uses np.allclose, which essentially checks if two matrices are close to each dependent on some parameters; it returns True if every single element is passed as close
    # This is checking if the difference matrix is close to the zero matrix of the same dimensions as the difference matrix (np.zeros_like(diff))
    # rtol, relative tolerance, is off - rtol usually defines closeness as a percentage of the second value which is not what we want
    # atol, absolute tolerance is set to recognise zero as any number smaller than 1e-7, allowing the algorithm to ignore tiny, unavoidable computer rounding errors but still detecting the main mismatch error, if any

def freivalds_algo(A, B, C, k=10):
    n = A.shape[0]
    # checking that all matrix dimensions are valid, if not - it gives False
    if A.shape[1] != B.shape[0] or B.shape[1] != C.shape[1] or A.shape[0] != C.shape[0]:
        return False

    # Now algorithm does k=10 internal iterations and gives True if all iterations return True
    for _ in range(k):
        r = np.random.randint(0, 2, size=(n, 1))
        A_Br = np.dot(A, np.dot(B, r))
        Cr = np.dot(C, r)

        diff = A_Br - Cr
        if not np.allclose(diff, np.zeros_like(diff), rtol=0, atol=1e-7):
            return False

    return True

if __name__ == "__main__":
    n = 1700 # large sample dataset for running & testing

    # Case 1: A * B = C (Matching Case)
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)
    C = np.dot(A, B)

    # time.perf_counter() is a performance counter that very accurately (most precise clock on Python) measures short time intervals, taking the time before and after the function is run when called
    start_time = time.perf_counter()
    result_det = deterministic_verification(A, B, C)
    end_time = time.perf_counter()
    print(f"Standard deterministic check: {result_det}\nTime: {end_time - start_time:.6f}s")

    start_time = time.perf_counter()
    result_freivalds = freivalds_algo(A, B, C, k=10)
    end_time = time.perf_counter()
    print(f"Frievald’s check: {result_freivalds}\nTime: {end_time - start_time:.6f}s")

# Case 2: A * B != C (Mismatching case)
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)
    C = np.dot(A, B)
    C[0][0] += 1 # We are deliberately introducing a change in C so that it does not match with product of A and B

    start_time = time.perf_counter()
    result_det = deterministic_verification(A, B, C)
    end_time = time.perf_counter()
    print(f"Standard deterministic check: {result_det}\nTime: {end_time - start_time:.6f}s")

# Due to the presence of rare false positive, the following section runs Freivald’s algorithm and reports the percentage of error/false positives when the algorithm gives True after being run 100 times - the number will be relatively very small, showing that the error probability is actually very low

    number_runs = 100
    totaltime = 0.0 
    true_count = 0
    for i in range(number_runs):
        start_time = time.perf_counter()
        result_freivalds = freivalds_algo(A, B, C, k=5)
        end_time = time.perf_counter()
        # The original code had a print statement here, but it would print 100 times,
        # making the output very long. I've commented it out for brevity, but it can be uncommented if needed.
        # print(f"Frievald's check: {result_freivalds}\nTime: {end_time - start_time:.6f}s")
        if result_freivalds:
            true_count += 1
        totaltime += (end_time - start_time)

    print(f"Percentage of false positives: {true_count / number_runs}\n Average Time: {totaltime/100:.6f}s")