# https://www.acmicpc.net/problem/14535

case = 1

months = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]

while True:
    N = int(input())
    if N == 0:
        break

    cnt = [0] * 13
    
    for _ in range(N):
        d, m, y = map(int, input().split())
        cnt[m] += 1

    print(f"Case #{case}:")
    for i in range(1, 13):
        print(f"{months[i-1]}:" + "*" * cnt[i])

    case += 1
    