# Problem statement

**This is the hard version of the problem. The difference between the versions is that in this version, you need to count how many sets satisfy the constraints. You can hack only if you solved all versions of this problem.**

Let $$ f(S,x)=\text{mex}\left(\left\{\left\lfloor \frac{y}{x}\right\rfloor : y\in S\right\}\right), $$ where $S$ is a set of non-negative integers and $x$ is a positive integer. $^{\text{∗}}$

Farmer John chooses a (possibly empty) subset $A\subseteq \{0,1,\ldots,n-1\}$. He then constructs an array $a$ of length $n$, where $a_k = f(A,k)$ for every $1\le k\le n$.

Bessie, being a mischievous cow, hides the set $A$. Farmer John is left only with the array $a_1,a_2,\ldots,a_n$.

Your task is to count how many sets (including the empty set) $B \subseteq \{0,1,\ldots,n-1\}$ such that $f(B,k)=a_k$ for every $1 \le k \le n$. Since the number can be very large, output it modulo $10^9+7$. **Farmer John will only give you arrays where such a $B$ exists**.

$^{\text{∗}}$$\operatorname{mex}(c)$ denotes the minimum excluded (MEX)$^{\text{∗}}$ of the collection $c$. (in this case, the minimum non-negative integer not present in the set)

# Notebook interface

## Input

Your function receives:
- `n` (int): the size of the universe `{0, 1, ..., n-1}` and the length of `a` (`1 ≤ n ≤ 10^4`)
- `a` (list of int): the array `a_1, ..., a_n` with `0 ≤ a_k ≤ n`, stored 0-indexed (`a[k-1]` is a_k); at least one valid set exists

## Output

Your function must return an int `count`: the number of sets B with f(B, k) = a_k for every k, modulo 10^9 + 7.

## Sample Test Cases

| n | a | Expected Output | Explanation |
|---|---|-----------------|-------------|
| 6 | `[0, 3, 2, 2, 2, 1]` | `6` | k = 1 forces 0 ∉ B; k = 5 forces 5 ∈ B; k = 2 then needs 1 ∈ B and one of {2, 3}, while 4 is free → 3 × 2 = 6 sets. |
| 5 | `[2, 1, 1, 1, 1]` | `1` | k = 1 forces 0, 1 ∈ B and 2 ∉ B; k = 2 and k = 3 forbid 3 and 4 → only {0, 1}. |
| 6 | `[1, 2, 1, 1, 1, 1]` | `1` | k = 1 forces 0 ∈ B, 1 ∉ B; k = 2 needs an element in [2, 3], k = 3 forbids 3 and k = 2 forbids 4, 5 → only {0, 2}. |
