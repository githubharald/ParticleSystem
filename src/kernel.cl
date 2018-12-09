// physical constants
#define STEP_POS 0.01f
#define STEP_AGE 0.05f
#define GRAVITY 1.25f
#define HORIZONTAL_SPEED 0.25f
#define VERTICAL_SPEED 1.0f

// drawing constants
#define COLOR_CHANNELS 4
#define FADE_OUT 0.05f
#define ILLUMINATE 1.5f
#define EMITTER_SHADOW 0.5f

// further constants set via compiler: FIRE_COLORS, WINDOW_SIZE

// state of a particle, sizeof(Particle)==32
struct Particle
{
	float2 speed;
	float2 pos;
	float2 randVal;
	float age;
};


// (re-)init a particle
void initOneParticle(__global struct Particle* particle, int n)
{
	const float speedX = particle->randVal.s0 * HORIZONTAL_SPEED;
	const float speedY = VERTICAL_SPEED + particle->randVal.s1;
	
	particle->speed = (float2)(speedX, speedY);
	particle->pos = (float2)(0.5f, 0.0f);
	particle->age = 0.0f;
}


// init particles for the first time
__kernel void initParticles(__global struct Particle* particles, __global float* randVals)
{
	const int n = get_global_id(0);
	particles[n].randVal = (float2)(randVals[2 * n], randVals[2 * n + 1]);
	initOneParticle(particles + n, n);
}


// update state of particles
__kernel void updateParticles(__global struct Particle* particles)
{
	const int n = get_global_id(0);
	
	// particle disappeared, therefore re-init particle
	if(particles[n].pos.y < 0.0f)
	{
		initOneParticle(particles + n, n);
	}
	
	// update position, speed and age
	particles[n].pos = particles[n].pos + particles[n].speed * STEP_POS;
	particles[n].speed.y = particles[n].speed.y - GRAVITY * STEP_POS;
	particles[n].age += STEP_AGE;
}


// set or add pixel at given image position
void drawPixel(__global int* img, int2 pos, int4 color, bool add)
{
	// don't draw outside of image
	if(pos.x < 0 || pos.x >= WINDOW_SIZE || pos.y < 0 || pos.y >= WINDOW_SIZE)
	{
		return;
	}

	// either set or add color value (ignore 4th color channel)
	const int base = COLOR_CHANNELS * (pos.y * WINDOW_SIZE + pos.x);
	switch(add)
	{
	case true:
		atomic_add(img + base, color.x);
		atomic_add(img + base + 1, color.y);
		atomic_add(img + base + 2, color.z);
		break;
		
	case false:
		img[base] = color.x;
		img[base + 1] = color.y;
		img[base + 2] = color.z;
		break;
	}
}


// clear canvas
__kernel void clearCanvas(__global int* img)
{
	drawPixel(img, (int2)(get_global_id(0), get_global_id(1)), (int4)(0, 0, 0, 0), false);
}


// limit color channel values to 255
__kernel void saturate(__global int4* img)
{
	const int base = get_global_id(1) * WINDOW_SIZE + get_global_id(0);
	img[base] = min(img[base], (int4)(255, 255, 255 ,0));
}


// compute color of a particle
float4 computeColor(__global struct Particle* particle)
{
	float4 color;
	
#ifdef FIRE_COLORS
	// fade out fire-like color
	color = mix((float4)(0.0f, 0.4f, 1.0f, 0.0f), (float4)(1.0f, 1.0f, 1.0f, 0.0f), exp(-particle->age)) ;
#else
	// create pseudo-random color
	float randVal = fabs(exp(particle->randVal.x / particle->randVal.y))*100.0f;
	randVal = fmod(randVal * 100.0f, 1.0f);
	color.x = randVal;
	randVal = fmod(randVal * 100.0f, 1.0f);
	color.y = randVal;
	randVal = fmod(randVal * 100.0f, 1.0f);
	color.z = randVal;
	color.w = 0.0f;
#endif

	return color * exp(-particle->age * FADE_OUT);
}


// draw particles
__kernel void drawParticles(__global struct Particle* particles, __global int* img)
{
	// compute position in image
	const int n = get_global_id(0);
	const int2 imgPos = (int2)(particles[n].pos.x * WINDOW_SIZE, WINDOW_SIZE - 1 - WINDOW_SIZE * particles[n].pos.y);

	// draw particle
	int radius = 3;
	for(int dx=-radius; dx<=radius; dx++)
	{
		for(int dy=-radius; dy<=radius; dy++)
		{
			const float dist = sqrt((float)(dx * dx + dy * dy));
			const float4 color = computeColor(particles + n) * exp(ILLUMINATE - dist);
			drawPixel(img, imgPos + (int2)(dx, dy), convert_int4(color * 255.0f), true);
		}
	}
}


// draw particel-emitter
__kernel void drawEmitter(__global int* img)
{	
	// position in image
	const int radius = 5;
	const int2 imgPos = (int2)(WINDOW_SIZE/2, WINDOW_SIZE-1-radius);
	
	// draw emitter
	for(int dx=-radius; dx<=radius; dx++)
	{
		for(int dy=-radius; dy<=radius; dy++)
		{
			const float dist = abs(dx);
			const float4 color = (float4)(0.0f, 0.5f, 1.0f, 0.0f) * exp(-dist * EMITTER_SHADOW);
			drawPixel(img, imgPos + (int2)(dx, dy), convert_int4(color * 255.0f), false);
		}
	}
	
}
