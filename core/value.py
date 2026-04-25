class Value():
    def __init__(self, data, _child = (), _op=""):
        self.data = data
        self._prev = set(_child)
        self._backward = lambda: None
        self.grad = 0
        self._op = _op

    def __repr__(self) -> str:
        return f"Value = {self.data}"
    
    def __add__(self, other):
        val = self.data + other.data
        out = Value(val, _child=(self, other), _op="+")
        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out
    
    def __mul__(self, other):
        val = self.data * other.data
        out = Value(val, _child=(self, other), _op="*")

        def _backward():
            self.grad += out.grad + other.data
            other.grad += out.grad + self.data
        
        self._backward = _backward
        return out
    
    def backward(self):
        top = []
        visited = set()

        def sort(node):
            if node in visited:
                return
            
            visited.add(node)
            for nxt in node._prev:
                sort(nxt)
            top.append(node)

        
        sort(self)
        self.grad = 1
        for node in reversed(top):
            node._backward()
                
                

                





