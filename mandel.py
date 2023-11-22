import torch
import math

def mandel(device = torch.device('cpu'), limits=(-2,0.75,-1.4,1.4), a=(-2,-1.4), b=(0.75,1.4), pixels=1e7, iterations=500):
	'''
	device:         torch.device, 											default: cpu
	limits:			tuple, position of output (xmin,xmax,ymin,ymax)			default: (-2,0.75,-1.4,1.4)
	pixels:         int, number of pixels in output, 						default: 10,000,000
	iterations:		int,													default: 5000
	'''

	xs = torch.linspace( limits[0],limits[1], int(math.sqrt(pixels)), device=device)
	ys = torch.linspace( limits[3],limits[2], int(math.sqrt(pixels)), device=device)

	z = torch.complex( *torch.meshgrid((xs,ys), indexing='xy') )
	out = torch.zeros_like(z, dtype=int)
	c = z.clone()
	
	for i in range(1,iterations):
		z = z**2 + c
		mask = torch.abs(z) > 4
		out[mask], z[mask], c[mask] = i, 0, 0

	return out.to('cpu')


if __name__ == '__main__':
	import argparse
	from matplotlib import colormaps
	import matplotlib.pyplot as plt

	parser = argparse.ArgumentParser()
	parser.add_argument('--iterations', default=5000, help='number of iterations')
	args = parser.parse_args()

	out = mandel(iterations=int(args.iterations))
	cols = colormaps.get_cmap('twilight_shifted')
	cols.set_under(color='black')
	plt.imshow(out, cmap=cols, vmin=1, extent=(-2,0.75,-1.4,1.4))
	plt.show(block=True)