from reader import Reader

grad_file = './GRAD/scr.geom/grad.xyz'
data = Reader(grad_file)
print(data.grad)