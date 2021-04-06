#include "utils.cuh"
#include "MM.cuh"


__host__
Image* mm(Image* image, Probe* probe, MMop mmOp, Version vrs){
	switch (mmOp) {
	case (EROSION):
		return erosion(image, probe, vrs);
		break;

	case (DILATATION):
		return dilatation(image, probe, vrs);
		break;

	case (OPENING):
		return opening(image, probe, vrs);
		break;

	case (CLOSING):
		return closing(image, probe, vrs);
		break;
	}
}

/**
 * @brief Operations implemented using __process common function.
 *
 * Basic: Dilatation D, Erosion E
 * Composed: Opening = D(E(image)), Closing = E(D(Image))
 */
__host__
Image* erosion(Image* image, Probe* probe, Version vrs) {

	float *imgData, *imgDataD, *outData, *outDataD, *probeData;
	int imgH = image->getHeight();
	int imgW = image->getWidth();
	int prbH = probe->getHeight();
	int prbW = probe->getWidth();

	imgData = image->getData();
	probeData = probe->getData();

	/// 1. Alloc GPU memories
	CUDA_CHECK_RETURN(
		cudaMalloc((void**)&imgDataD, sizeof(float) * imgH * imgW));

	CUDA_CHECK_RETURN(
		cudaMalloc((void**)&outDataD, sizeof(float) * imgH * imgW));

	/// 2. Host to Device
	CUDA_CHECK_RETURN(cudaMemcpy(
				  imgDataD, imgData, sizeof(float) * imgH * imgW, cudaMemcpyHostToDevice));

	CUDA_CHECK_RETURN(cudaMemcpyToSymbol(
				  probeDataD, probeData, sizeof(float) * prbH * prbW));

	/// 3. Kernel function compute...
	dim3 dimGrid(ceil((float) imgW/TILE_WIDTH), ceil((float) imgH/TILE_WIDTH));
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

	if(vrs == NAIVE) {
		naive::__process<<<dimGrid, dimBlock>>>(
			imgDataD, probeDataD, outDataD, imgH, imgW, prbH, prbW, EROSION);
		cudaDeviceSynchronize();
	}

	if(vrs == SHAREDOPT) {
		sharedOpt::__process<<<dimGrid, dimBlock>>>(
			imgDataD, probeDataD, outDataD, imgH, imgW, prbH, prbW, DILATATION);
		cudaDeviceSynchronize();
	}

	/// 4. Device To Host
	outData = (float*)malloc(sizeof(float) * imgH * imgW);
	CUDA_CHECK_RETURN(cudaMemcpy(
				  outData, outDataD, sizeof(float) * imgH * imgW, cudaMemcpyDeviceToHost));

	/// 5. Destroy GPU memory
	cudaFree(imgDataD);
	cudaFree(outDataD);
	cudaFree(probeDataD);

	Image *out = new Image(image->getHeight(), image->getWidth(), outData, bw);
	return out;
}

__host__
Image* dilatation(Image* image, Probe* probe, Version vrs) {

	float *imgData, *imgDataD, *outData = nullptr, *outDataD, *probeData;
	int imgH = image->getHeight();
	int imgW = image->getWidth();
	int prbH = probe->getHeight();
	int prbW = probe->getWidth();

	imgData = image->getData();
	probeData = probe->getData();

	/// 1. Alloc GPU memories
	CUDA_CHECK_RETURN(
		cudaMalloc((void**)&imgDataD, sizeof(float) * imgH * imgW));

	CUDA_CHECK_RETURN(
		cudaMalloc((void**)&outDataD, sizeof(float) * imgH * imgW));

	/// 2. Host to Device
	CUDA_CHECK_RETURN(cudaMemcpy(
				  imgDataD, imgData, sizeof(float) * imgH * imgW, cudaMemcpyHostToDevice));

	CUDA_CHECK_RETURN(cudaMemcpyToSymbol(
				  probeDataD, probeData, sizeof(float) * prbH * prbW));

	/// 3. Kernel function compute...
	dim3 dimGrid(ceil((float) imgW/TILE_WIDTH), ceil((float) imgH/TILE_WIDTH), 1);
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

	if(vrs == NAIVE) {
		naive::__process<<<dimGrid, dimBlock>>>(
			imgDataD, probeDataD, outDataD, imgH, imgW, prbH, prbW, DILATATION);
		cudaDeviceSynchronize();
	}

	if(vrs == SHAREDOPT) {
		sharedOpt::__process<<<dimGrid, dimBlock>>>(
			imgDataD, probeDataD, outDataD, imgH, imgW, prbH, prbW, DILATATION);
		cudaDeviceSynchronize();
	}

	/// 4. Device To Host
	outData = (float*)malloc(sizeof(float) * imgH * imgW);
	CUDA_CHECK_RETURN(cudaMemcpy(
				  outData, outDataD, imgH * imgW * sizeof(float), cudaMemcpyDeviceToHost));

	/// 5. Destroy GPU memory
	cudaFree(imgDataD);
	cudaFree(outDataD);
	cudaFree(probeDataD);

	Image *out = new Image(image->getHeight(), image->getWidth(), outData, bw);
	return out;
}

__host__
Image* opening(Image* image, Probe* probe, Version vrs) {
	Image* imageEroded = erosion(image, probe, vrs);
	Image* out = dilatation(imageEroded, probe, vrs);

	return out;
}

__host__
Image* closing(Image* image, Probe* probe, Version vrs) {
	Image* imageDilatated = dilatation(image, probe, vrs);
	Image* out = erosion(imageDilatated, probe, vrs);

	return out;
}


/**
 * @brief Naive version to process the images.
 *
 * A simple implementation that compute basic operations in parallel version.
 *  - DILATATION: select max value of neighborhood
 *  - EROSION: select min value of neighborhood
 *
 * In according CUDA documentation, this function is called by host and
 *      executed by the device.
 */
__global__
void naive::__process(float* imgData, const float*__restrict__ prbData,
                      float* outData, int imgH, int imgW, int prbH, int prbW, MMop mmOp){

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;

	// global index of thread to map image in global memory
	int colImg = bx * blockDim.x + tx;
	int rowImg = by * blockDim.y + ty;

	// compute max and min of neighborhood for image[rowImg * imgW + colImg]
	if(rowImg >= 0 && rowImg < imgH && colImg >= 0 && colImg < imgW) {
		float max = imgData[rowImg * imgW + colImg];
		float min = max;

		for(int rowPrb = -prbH/2; rowPrb < prbH/2; rowPrb++) {
			for(int colPrb = -prbW/2; colPrb < prbW/2; colPrb++) {
				if((rowImg + rowPrb) * imgW + colImg + colPrb >= 0 &&
				   (rowImg + rowPrb) * imgW + colImg + colPrb <= imgH * imgW) {

					if(max < imgData[(rowImg + rowPrb) * imgW + colImg + colPrb])
						max = imgData[(rowImg + rowPrb) * imgW + colImg + colPrb];

					if(min > imgData[(rowImg + rowPrb) * imgW + colImg + colPrb])
						min = imgData[(rowImg + rowPrb) * imgW + colImg + colPrb];

				}
			}
		}

		if(mmOp == EROSION)
			outData[rowImg * imgW + colImg] = min;

		if(mmOp == DILATATION)
			outData[rowImg * imgW + colImg] = max;

	}
}

/**
 * @brief Optimized parallel version to process the images.
 *
 * The optimizations consists of
 *  - Shared Memory usage
 *  - Simple Padding policy (value as -1)
 *
 * In according CUDA documentation, this function is called by host and
 *      executed by the device.
 */
__global__
void sharedOpt::__process(float* imgData, const float* __restrict__ prbData,
                          float* outData, int imgH, int imgW, int prbH, int prbW, MMop mmOp){

	int rowImg, colImg;
	__shared__ float tileDS[W][W];

	/// first batch loading
	int dest = threadIdx.y * TILE_WIDTH + threadIdx.x;
	int destY = dest / W;
	int destX = dest % W;
	int srcY = blockIdx.y * TILE_WIDTH + destY - MASK_RADIUS;
	int srcX = blockIdx.x * TILE_WIDTH + destX - MASK_RADIUS;
	int src = srcY * imgW + srcX;

	if(srcY >= 0 && srcY < imgH && srcX >= 0 && srcX < imgW) {
		tileDS[destY][destX] = imgData[src];
	}else{
		tileDS[destY][destX] = -1;
	}

	/// second batch loading
	dest = threadIdx.y * TILE_WIDTH + threadIdx.x + TILE_WIDTH * TILE_WIDTH;
	destY = dest / W;
	destX = dest % W,
	srcY = blockIdx.y * TILE_WIDTH + destY - MASK_RADIUS;
	srcX = blockIdx.x * TILE_WIDTH + destX - MASK_RADIUS;
	src = srcY * imgW + srcX;

	if(destY < W) {
		if(srcY >= 0 && srcY < imgH && srcX >= 0 && srcX < imgW) {
			tileDS[destY][destX] = imgData[src];
		}else{
			tileDS[destY][destX] = -1;
		}
	}

	__syncthreads();

	/// compute
	float max = tileDS[threadIdx.y][threadIdx.x];
	float min = max;

	for(int y = 0; y < MASK_WIDTH; y++) {
		for(int x = 0; x < MASK_WIDTH; x++) {
			if(tileDS[threadIdx.y + y][threadIdx.x + x] > -1) {
				if(max < tileDS[threadIdx.y + y][threadIdx.x + x])
					max = tileDS[threadIdx.y + y][threadIdx.x + x];

				if(min > tileDS[threadIdx.y + y][threadIdx.x + x])
					min = tileDS[threadIdx.y + y][threadIdx.x + x];
			}
		}
	}

	rowImg = blockIdx.y * TILE_WIDTH + threadIdx.y;
	colImg = blockIdx.x * TILE_WIDTH + threadIdx.x;
	if(rowImg < imgH && colImg < imgW) {
		if (mmOp == EROSION)
			outData[rowImg * imgW + colImg] = min;

		if(mmOp == DILATATION)
			outData[rowImg * imgW + colImg] = max;
	}

	__syncthreads();
}
