# https://www.acmicpc.net/problem/26350

n = int(input())

for _ in range(n):
    parts = list(map(int, input().split()))
    d = parts[0]
    coins = parts[1:]

    v = " ".join(map(str, coins))

    print(f"Denominations: {v}")

    good = True
    for i in range(1, d):
        if coins[i] < 2 * coins[i - 1]:
            good = False
            break
         
    if good:
        print("Good coin denominations!\n")
    else:
        print("Bad coin denominations!\n")
    

         