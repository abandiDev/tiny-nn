from core.value import Value
from nn.module import MLP

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

# TRAINIG EXAMLE #

xs = [
    [2.0, 3.0],
    [1.0, -1.0],
    [-2.0, 1.0],
    [-1.0, -3.0],
]

ys = [1.0, 1.0, -1.0, -1.0]

model = MLP(2, [4, 4, 6, 100, 1])

for step in range(100):
    ypred = [model(x) for x in xs]

    loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred))

    model.zero_grad()
    loss.backward()

    for p in model.parameters():
        p.data += -0.001 * p.grad
    
    print(step, loss.data)


print(model([2.0, 3.0]))
print(model([1.0, -1.0]))
print(model([-2.0, 1.0]))
print(model([-1.0, -3.0]))