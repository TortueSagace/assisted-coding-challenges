# Problem statement

Among the people from the city *GR*, there is a legend about a hill that leads to the backstage. An array $b$ consisting of $m$ integers is called a *hill* if there exists an index $k$ ($1\le k\le m$) such that the following conditions hold:

- The first $k$ elements are strictly increasing. Formally, for all $i$ ($1\le i\lt k$), $b_i\lt b_{i+1}$.
- The last $m-k+1$ elements are strictly decreasing. Formally, for all $i$ ($k\le i\lt m$), $b_i\gt b_{i+1}$.

You are given an array $a$ consisting of $n$ **distinct** integers. You may perform the following operation any number of times:

- Choose any index $i$ ($1\le i\le n-2$).
- Swap $a_i$ and $a_{i+2}$ in the array.

Your task is to determine whether it is possible to turn array $a$ into a *hill* by performing the operation any number of times.

# Notebook interface

## Input

Your function receives:
- `n` (int): the length of the array (`1 ≤ n ≤ 10^5`)
- `a` (list of int): the array, a permutation of `1..n` (distinct values with `1 ≤ a_i ≤ n`)

## Output

Your function must return a str `answer`: "YES" if the array can be turned into a hill, "NO" otherwise.

## Sample Test Cases

| n | a | Expected Output | Explanation |
|---|---|-----------------|-------------|
| 3 | `[2, 1, 3]` | `'NO'` | The smallest value 1 must end at position 1 or 3, but it sits on an even position and can only move between even positions → NO. |
| 4 | `[1, 2, 3, 4]` | `'YES'` | The array is already a hill → YES. |
| 6 | `[5, 3, 4, 6, 2, 1]` | `'YES'` | Swapping a₁ ↔ a₃ then a₂ ↔ a₄ gives the hill [4, 6, 5, 3, 2, 1] (see the Note) → YES. |
| 6 | `[1, 5, 2, 3, 6, 4]` | `'NO'` | 1 must go to an end: position 1 (odd, like 1). Then 2 must go to position 2 or 6, both even, but 2 is on an odd position → NO. |
| 7 | `[1, 2, 3, 4, 7, 5, 6]` | `'NO'` | In a hill, 6 must sit right next to the peak 7, but both are on odd positions and can never become neighbours → NO. |
| 5 | `[5, 4, 1, 3, 2]` | `'YES'` | Swapping a₁ ↔ a₃ gives the hill [1, 4, 5, 3, 2] → YES. |
| 6 | `[4, 1, 3, 2, 6, 5]` | `'NO'` | 1 is on an even position, so it must go to position 6. Then 2 (also even) must go to position 1 or 5, both odd → NO. |
