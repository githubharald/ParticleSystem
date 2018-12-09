# Particle System

A simple particle system implemented with Python and OpenCL.

## Run demo

Go to the `src/` directory and run ```python main.py```.
This should open a window showing some firework-like particle behavior as seen in the animation below.

![animation](./doc/animation.gif)

Requirements:

* Python 3
* NumPy
* OpenCV
* OpenCL

## More options

Use the help option to see all program options.

```
> python main.py --help

Particle system.

optional arguments:
  -h, --help       show this help message and exit
  --number NUMBER  number of particles
  --fire           particles have fire-like colors
  --dump           dump frames
```

The following illustration shows three possible settings.

![animation](./doc/options.png)