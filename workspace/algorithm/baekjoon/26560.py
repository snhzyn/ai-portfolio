# https://www.acmicpc.net/problem/26560

n = int(input())

for _ in range(n):
    i = input()
    if i[-1] != ".":
        i += "."
    print(i)