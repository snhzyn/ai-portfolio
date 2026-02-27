# https://www.acmicpc.net/problem/10988

a = input()

if a == a[::-1]:
    print(1)
else:
    print(0)