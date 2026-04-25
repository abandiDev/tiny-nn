from core.value import Value

a = Value(1.0)
b = Value(2.0)

print(a)
print(b)

c = a + b
print(c)
print(a.grad)
print(b.grad)

d = Value(3.0)

e = d * c
e.backward()

