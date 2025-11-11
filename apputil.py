import numpy as np
import seaborn as sns
from time import time
from typing import Any, Iterable, List, Optional, Tuple

from sklearn.cluster import KMeans

# load diamonds dataset and keep only numeric columns as a global dataframe
_diamonds = sns.load_dataset("diamonds")
DIAMONDS_NUMERIC = _diamonds.select_dtypes(include=[np.number]).copy()

# global step counter for the bonus
step_count: int = 0


# -------------------------------------------------------------------
# Exercise 1: kmeans
# -------------------------------------------------------------------
def kmeans(X: Any, k: int, random_state: Optional[int] = 0
           ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform k-means clustering on a numeric 2D array X.

    Returns:
        (centroids, labels)
        - centroids: ndarray of shape (k, n_features)
        - labels: ndarray of shape (n_samples,)
    """
    X_arr = np.asarray(X)
    if X_arr.ndim != 2:
        raise ValueError("X must be a 2D numeric array")
    if X_arr.size == 0:
        raise ValueError("X must not be empty")
    if X_arr.dtype.kind not in ("i", "u", "f"):
        raise ValueError("X must be numeric")

    km = KMeans(n_clusters=int(k), random_state=random_state, n_init=10)
    km.fit(X_arr)
    return km.cluster_centers_, km.labels_


# -------------------------------------------------------------------
# Exercise 2: kmeans_diamonds
# -------------------------------------------------------------------
def kmeans_diamonds(n: int, k: int,
                    random_state: Optional[int] = 0
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run kmeans on the first n rows of the numeric diamonds dataset.

    Returns:
        (centroids, labels)
    """
    if int(n) <= 0:
        raise ValueError("n must be a positive integer")
    n = min(int(n), DIAMONDS_NUMERIC.shape[0])
    X = DIAMONDS_NUMERIC.iloc[:n].values
    return kmeans(X, k, random_state=random_state)


# -------------------------------------------------------------------
# Exercise 3: kmeans_timer
# -------------------------------------------------------------------
def kmeans_timer(n: int, k: int, n_iter: int = 5,
                 random_state: Optional[int] = 0) -> float:
    """
    Run kmeans_diamonds(n, k) exactly n_iter times and return the
    average runtime in seconds.
    """
    timings: List[float] = []
    for _ in range(int(n_iter)):
        start = time()
        _ = kmeans_diamonds(n, k, random_state=random_state)
        timings.append(time() - start)
    return float(np.mean(timings))


# -------------------------------------------------------------------
# Bonus: binary search with step counting
# -------------------------------------------------------------------
def bin_search_count(n: int) -> Tuple[int, int]:
    """
    Binary search for value (n-1) in an array of size n while
    counting elementary steps in the global `step_count`.

    Returns:
        (index_found, step_count)
    """
    global step_count
    step_count = 0

    arr = np.arange(int(n))
    step_count += 1  # array creation

    left = 0
    right = int(n) - 1
    step_count += 2  # assignments

    x = int(n) - 1
    step_count += 1  # target assignment

    while left <= right:
        step_count += 1  # while-condition evaluated (entered)
        middle = left + (right - left) // 2
        step_count += 2  # compute middle and assign

        step_count += 1  # comparison arr[middle] == x
        if arr[middle] == x:
            step_count += 1  # equality true
            return middle, step_count

        step_count += 1  # comparison arr[middle] < x
        if arr[middle] < x:
            step_count += 1  # branch taken
            left = middle + 1
            step_count += 1  # assignment
        else:
            step_count += 1  # else branch
            right = middle - 1
            step_count += 1  # assignment

    step_count += 1  # final while-condition false
    return -1, step_count


def bin_search_stepcounts(n_values: Iterable[int]
                          ) -> Tuple[List[int], List[int]]:
    """
    Given an iterable of sizes n_values, return (ns, counts) where
    counts[i] is the step count for bin_search_count(ns[i]).
    """
    ns: List[int] = [int(n) for n in n_values]
    counts: List[int] = []
    for n in ns:
        _, cnt = bin_search_count(n)
        counts.append(cnt)
    return ns, counts