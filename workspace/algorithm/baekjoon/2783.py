# https://www.acmicpc.net/problem/2783

X, Y = map(int, input().split())
prices = [X / Y * 1000]

N = int(input())

for _ in range(N):
    X, Y = map(int, input().split())
    prices.append(X / Y * 1000)

print(min(prices))
