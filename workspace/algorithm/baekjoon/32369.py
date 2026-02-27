# https://www.acmicpc.net/problem/32369

N, A, B = map(int, input().split())

X = 1   # 칭찬 양파
Y = 1   # 비난 양파

for _ in range(N):
    X += A
    Y += B

    if Y > X:
        X, Y = Y, X
    
    elif X == Y:
        Y -= 1

print(X, Y)