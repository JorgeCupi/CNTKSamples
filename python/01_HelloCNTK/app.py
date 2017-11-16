import cntk
print("Tensor A = [1,2,3]")
print("Tensor B = [4,5,6]\n")

print("A+B:")
sum = cntk.plus([1, 2, 3], [4, 5, 6]).eval()
print("{}\n".format(sum))

print("A-B:")
minus = cntk.minus([1, 2, 3], [4, 5, 6]).eval() 
print("{}\n".format(minus))

print("A*B:")
times = cntk.times([1,3,4],[4,5,6]).eval()
print("{}\n".format(times))

print("A/B:")
divide = cntk.element_divide([4,32, 15], [2, 4, 5]).eval()
print("{}\n".format(divide))

print("A^B:")
pow = cntk.pow([1,3,4],[4,2,3]).eval()
print("{}\n".format(pow))

print("Min(A,B):")
min = cntk.element_min([1,2, 3], [4, 5, 6], [2,1,0]).eval()
print("{}\n".format(min))

print("Max(A,B):")
max = cntk.element_max([1,2,3],[4,5,6],[2,9,0]).eval()
print("{}\n".format(max))