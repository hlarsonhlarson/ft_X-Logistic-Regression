def my_mean(x):
    return np.sum(x) / len(x)

def my_std(x):
    m = my_mean(x)
    ans = 0
    for elem in x:
        ans += (elem - m)**2

    ans /= len(x) - 1
    return np.sqrt(ans)

def my_min(x)
    ans = x[0]
    for elem in x:
        if elem < ans:
            ans = elem
    return ans

def my_max(x)
    ans = x[0]
    for elem in x:
        if elem > ans:
            ans = elem
    return ans


