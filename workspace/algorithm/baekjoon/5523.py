# https://www.acmicpc.net/problem/5523

import sys
input = sys.stdin.readline

N = int(input())

A_win = 0
B_win = 0

for _ in range(N):
    A, B = map(int, input().split())
    if A > B:
        A_win += 1
    elif A < B:
        B_win += 1
    else:
        pass

print(f"{A_win} {B_win}")
