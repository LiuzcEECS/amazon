import os
f = open("summit.csv", "r")
l = []
for line in f:
    if line[0] != 'i':
        l.append(line)
f.close()
f = open("summit.csv", "w")
f.writelines(l)
f.close()
print l
