def two_in_thtree(A, B, C):
    X = (A & B) - C
    X |= (B & C) - A
    X |= (C & A) - B
    return X

print(two_in_thtree({1, 2, 'a'}, {2, 'b', 0}, {3, 'a', 2}))

w = '100'
print(len(w*2))