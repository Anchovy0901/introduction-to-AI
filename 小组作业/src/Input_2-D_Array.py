n = int(input())
line = [[0]*n]*n
for i in range(n):
    line[i] = input().split(" ")
    line[i] = [int(j) for j in line[i]]
print(line)  