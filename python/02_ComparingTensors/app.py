import cntk
A = [1, 3, 4]
B = [4, 3, 2]

print("less(A,B):")
less = cntk.less(A,B).eval()
print("{}\n".format(less))

print("equal(A,B):")
equal = cntk.equal(A,B).eval()
print("{}\n".format(equal))

print("greater(A,B)")
greater = cntk.greater(A,B).eval()
print("{}\n".format(greater))

print("greater_equal(A,B):")
greater_equal = cntk.greater_equal(A,B).eval()
print("{}\n".format(greater_equal))

print("not_equal(A,B):")
not_equal = cntk.not_equal(A,B).eval()
print("{}\n".format(not_equal))

print("less_equal(A,B):")
less_equal = cntk.less_equal(A,B).eval()
print("{}\n".format(less_equal))