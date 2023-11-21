import torch

def mandel(dev = torch.device('cpu'), a=(-2,-1.5), b=(1,1.5), pixels=1e6, iterations=5000):
    '''
    device:         torch.device
    a:              tuple, (x,y), bottom left corner of viewing window
    b:              tuple, (x,y), upper left corner of viewing window
    pixels:         int, number of pixels in output
    '''

    xs = torch.linspace(a[0],b[0], int(math.sqrt(pixels)), device=dev)
    ys = torch.linspace(a[1],b[1], int(math.sqrt(pixels)), device=dev)

    z = torch.complex( torch.meshgrid((xs,ys), indexing='xy') )
    out = torch.zeros_like(z, dtype=int)
    c = z.clone()
    
    for i in range(1,iterations):
        z = z**2 + c
        mask = torch.abs(z) > 4
        out[mask], z[mask], c[mask] = i, 0, 0

    return out