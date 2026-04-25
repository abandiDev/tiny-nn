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
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Only supports int/ float for now"  
        val = self.data ** other
        out = Value(val, _child=(self, ), _op="pow")
        def _backward():
            self.grad += (other * (self.data ** (other -1))) * out.grad
        
        out._backward = _backward
        return out
            
    
    def __mul__(self, other):
        val = self.data * other.data
        out = Value(val, _child=(self, other), _op="*")

        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        
        out._backward = _backward
        return out
    
    def relu(self):
        # if x > 0 : x else 0
        val = max(0, self.data)
        out = Value(val, _child=(self,), _op="ReLU")

        def _backward():
            if self.data > 0:
                self.grad += out.grad
            else:
                self.grad += 0

        out._backward = _backward
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
                
                

                





