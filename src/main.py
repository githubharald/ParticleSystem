import os
import random
import argparse
import cv2
import numpy as np
import pyopencl as cl


def render(numParticles, fireColors, dumpFrames=False, clDebug=False):
	"render particle system with specified number of particles and particle color"

	# show output of OpenCL compiler
	if clDebug:
		os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
		
	# if frames should be dumped, create directory if necessary
	dumpDir = 'dump'
	if dumpFrames and not os.path.exists(dumpDir):
		os.mkdir(dumpDir)
		
	# setup OpenCL
	platforms = cl.get_platforms() # a platform corresponds to a driver (e.g. AMD)
	platform = platforms[0] # take first platform
	devices = platform.get_devices(cl.device_type.GPU) # get GPU devices of selected platform
	device = devices[0] # take first GPU
	context = cl.Context([device]) # put selected GPU into context object
	queue = cl.CommandQueue(context, device) # create command queue for selected GPU and context
	
	# setup buffer for particles
	sizeParticleStruct = 32 # sizeof(struct Particle)
	bufParticles = cl.Buffer(context, cl.mem_flags.READ_WRITE, size=sizeParticleStruct*numParticles, hostbuf=None)
	
	# setup random values (for random speed and color)
	random.seed()
	randVals = np.array([random.random() - 0.5 for _ in range(2 * numParticles)], dtype=np.float32)
	bufRandVals = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=randVals)
	
	# setup output image
	windowSize = 480
	colorChannels = 4 # RGBA
	sizeofColorChannel = 4 # we need int32 to perform atomic operations in the kernel (multiple particles at same position)
	img = np.zeros([windowSize, windowSize, colorChannels], dtype=np.int32) # must be square image to ignore distortion
	imgShape = (windowSize, windowSize) # 2d shape of image
	imgBuf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, size=windowSize*windowSize*colorChannels*sizeofColorChannel)
	
	# setup kernels
	compilerSettings = ('-DWINDOW_SIZE=%d'%(windowSize)) + ' ' + ('-DFIRE_COLORS' if fireColors else '')
	program = cl.Program(context, open('kernel.cl').read()).build(compilerSettings)
	initParticles = cl.Kernel(program, 'initParticles')
	updateParticles = cl.Kernel(program, 'updateParticles')
	clearCanvas = cl.Kernel(program, 'clearCanvas')
	drawParticles = cl.Kernel(program, 'drawParticles')
	saturate = cl.Kernel(program, 'saturate')
	drawEmitter = cl.Kernel(program, 'drawEmitter')
	
	# init particles
	initParticles.set_arg(0, bufParticles)
	initParticles.set_arg(1, bufRandVals)
	cl.enqueue_nd_range_kernel(queue, initParticles, (numParticles,), None)
	
	# do some (invisible) iterations for smooth particle distribution
	for t in range(1000):
		updateParticles.set_arg(0, bufParticles)
		cl.enqueue_nd_range_kernel(queue, updateParticles, (numParticles,), None)
	
	# rendering loop
	ctr = 0
	while True:
		# clear canvas
		clearCanvas.set_arg(0, imgBuf)	
		cl.enqueue_nd_range_kernel(queue, clearCanvas, imgShape, None)
		
		# draw all particles
		drawParticles.set_arg(0, bufParticles)
		drawParticles.set_arg(1, imgBuf)
		cl.enqueue_nd_range_kernel(queue, drawParticles, (numParticles,), None)
		
		# saturate
		saturate.set_arg(0, imgBuf)
		cl.enqueue_nd_range_kernel(queue, saturate, imgShape, None)
		
		# draw emitter
		drawEmitter.set_arg(0, imgBuf)
		cl.enqueue_nd_range_kernel(queue, drawEmitter, (1,), None)
		
		# update particles
		updateParticles.set_arg(0, bufParticles)
		cl.enqueue_nd_range_kernel(queue, updateParticles, (numParticles,), None)
		
		# copy result from GPU
		cl.enqueue_copy(queue, img, imgBuf, is_blocking=True)
		
		# show image (and dump if specified)
		imgU8 = img[:,:,0:3].astype(np.uint8)
		cv2.imshow("Particle system [press ESC to exit]", imgU8)
		if dumpFrames:
			ctr += 1
			cv2.imwrite('%s/%d.png'%(dumpDir, ctr), imgU8)
		
		# exit with ESC
		keyPressed = cv2.waitKey(10)
		if keyPressed == 27:
			break
			
		
def main():
	# read options from command line
	parser = argparse.ArgumentParser(description='Particle system.')
	parser.add_argument('--number', type=int, default=200, help='number of particles (default value is 200)')
	parser.add_argument('--fire', help='use fire-like color', action='store_true')
	parser.add_argument('--dump', help='dump frames', action='store_true')
	args = parser.parse_args()
	
	# render particle system
	render(args.number, args.fire, args.dump)

if __name__ == '__main__':
	main()