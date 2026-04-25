from core.value import Value

a = Value(1.0)
b = Value(2.0)
c = a + b
d = Value(3.0)
e = d * c
f = e ** 2
g = f.relu()
g.backward()

print(g)
print(a.grad)

