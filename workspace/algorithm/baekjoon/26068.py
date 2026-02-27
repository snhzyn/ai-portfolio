# https://www.acmicpc.net/problem/26068

n = int(input())

count = 0

for _ in range(n):
    x = int(input().split("-")[1])

    if x <= 90:
        count += 1
    else:
        pass

print(count)