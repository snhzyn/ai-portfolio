# https://www.acmicpc.net/problem/12840

import sys
input = sys.stdin.readline

h, m, s = map(int, input().split())
q = int(input())

time = h * 3600 + m * 60 + s

for _ in range(q):
    value = input().split()
    T = int(value[0])

    if T == 3:
        h = time // 3600
        m = (time % 3600) // 60
        s = time % 60
        print(h, m, s)
    else:
        c = int(value[1])
        if T == 1:
            time = (time + c) % 86400
        else:
            time = (time - c) % 86400



