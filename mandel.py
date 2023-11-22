import torch
import math

def mandel(device = torch.device('cpu'), a=(-2,-1.5), b=(1,1.5), pixels=1e6, iterations=5000):
	'''
	device:         torch.device, 											default: cpu
	a:              tuple, (x,y), bottom left corner of viewing window		default: (-2,-1.5)
	b:              tuple, (x,y), upper left corner of viewing window		default: (1,1.5)
	pixels:         int, number of pixels in output, 						default: 1,000,000
	iterations:		int,													default: 5000
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


if __name__ == '__main__':
	import argparse
	from matplotlib import colormaps
	import matplotlib.pyplot as plt

	parser = argparse.ArgumentParser()
	parser.add_argument('--a', default=(-2,-1.5), help='bottom left corner of viewing window')
	parser.add_argument('--b', default=(1,1.5), help='upper right corner of viewing window')
	parser.add_argument('--pixels', default=1e6, help='number of pixels in output')
	parser.add_argument('--iterations', default=5000, help='number of iterations')
	args = parser.parse_args()

	out = mandel(a=args.a, b=args.b, pixels=args.pixels, iterations=args.iterations)

	cols = colormaps.get_cmap('twilight_shifted')
	cols.set_under(color='black')
	plt.imshow(out, cmap=cols, vmin=0.5)	