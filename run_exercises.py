from apputil import kmeans, kmeans_diamonds, kmeans_timer, bin_search_stepcounts
import numpy as np


def main() -> None:
    X = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
    c, l = kmeans(X, 3)
    print("kmeans centroids shape:", c.shape, "labels:", l)

    c2, l2 = kmeans_diamonds(100, 5)
    print("diamonds centroids shape:", c2.shape, "labels shape:", l2.shape)

    print("kmeans_timer(100,5,3):", kmeans_timer(100, 5, 3))

    ns, counts = bin_search_stepcounts([10, 100, 1000])
    print("bin-search step counts:", list(zip(ns, counts)))


if __name__ == "__main__":
    main()
