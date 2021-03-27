#include "utils.cuh"
#include "MM.cuh"

/******* Interface *******/
__host__
Image* mm(Image* image, Probe* probe, MMop mmOp, Version vrs){
    switch (mmOp){
        case(EROSION):
            return erosion(image, probe, vrs);
            break;

        case(DILATATION):
            return dilatation(image, probe, vrs);
            break;

        case(OPENING):
            return opening(image, probe, vrs);
            break;

        case(CLOSING):
            return closing(image, probe, vrs);
            break;
    }
}

/******* Operations for MM *******/
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

    if(vrs == NAIVE){
        naive::__process<<<dimGrid, dimBlock>>>(
            imgDataD, probeDataD, outDataD, imgH, imgW, prbH, prbW, EROSION);
        cudaDeviceSynchronize();
    }

    if(vrs == SHAREDOPT){
        int sharedMem = (TILE_WIDTH + prbW - 1) * (TILE_WIDTH + prbH - 1) * sizeof(float);
        sharedOpt::__process<<<dimGrid, dimBlock, sharedMem>>>(
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

    if(vrs == NAIVE){
        naive::__process<<<dimGrid, dimBlock>>>(
            imgDataD, probeDataD, outDataD, imgH, imgW, prbH, prbW, DILATATION);
        cudaDeviceSynchronize();
    }

    if(vrs == SHAREDOPT){
        int sharedMem = (TILE_WIDTH + prbW - 1) * (TILE_WIDTH + prbH - 1) * sizeof(float);
        sharedOpt::__process<<<dimGrid, dimBlock, sharedMem>>>(
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


/******* naive version *******/
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
    if(rowImg >= 0 && rowImg < imgH && colImg >= 0 && colImg < imgW){
        float max = imgData[rowImg * imgW + colImg];
        float min = max;

        for(int rowPrb = - prbH/2; rowPrb < prbH/2; rowPrb++) {
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

/******* sharedOpt version *******/
__global__
void sharedOpt::__process(float* imgData, const float* __restrict__ prbData,
    float* outData, int imgH, int imgW, int prbH, int prbW, MMop mmOp){

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int colImg = bx * blockDim.x + tx;
    int rowImg = by * blockDim.y + ty;

    // to handle shared memory
    extern __shared__ float tileDS[];
    int wideTile = TILE_WIDTH + prbW - 1;
    int dest = threadIdx.y * wideTile + threadIdx.x;
    int destY = dest / wideTile;
	int destX = dest % wideTile;

    /// load Tile
    if(rowImg >= 0 && rowImg < imgH && colImg >= 0 && colImg < imgW)
        tileDS[(destY + prbH / 2) * wideTile + destX + prbW / 2] = imgData[rowImg * imgW + colImg];
    else
        tileDS[(destY + prbH / 2) * wideTile + destX + prbW / 2] = -1;
    __syncthreads();

    /// padding
    // top row
    if(ty == 0){
        for(int row = 0; row < prbH/2; row++){
            if(rowImg == 0)
                tileDS[row * wideTile + destX + prbW/2] = -1;
            else
                tileDS[row * wideTile + destX + prbW/2] = imgData[(rowImg - prbH/2 + row) * imgW + colImg];
        }
    }

    // left column
    if(tx == 0){
        for(int col = 0; col < prbW/2; col++){
            if(colImg == 0)
                tileDS[(destY + prbH/2) * wideTile + col] = -1;
            else
                tileDS[(destY + prbH/2) * wideTile + col] = imgData[rowImg * imgW + (colImg - prbW/2 + col)];
        }
    }

    // bottom row
    if(ty == TILE_WIDTH - 1){
        for(int row = wideTile - prbH/2; row < wideTile; row++){
            int gy = rowImg + (row - wideTile + prbH/2);
            if(gy >= imgH)
                tileDS[row * wideTile + destX + prbW/2] = -1;
            else
                tileDS[row * wideTile + destX + prbW/2] = imgData[gy * imgW + colImg];
        }
    }

    // right column
    if(tx == TILE_WIDTH - 1){
        for(int col = prbW/2; col < wideTile; col++){
            int gx = colImg - wideTile + prbH/2 + col;
            if(gx >= imgW)
                tileDS[(destY + prbH/2) * wideTile + col] = -1;
            else{
                if(rowImg * imgW + gx > 0)
                    tileDS[(destY + prbH/2) * wideTile + col] = imgData[rowImg * imgW + gx];  //FIXME
            }
        }
    }

    // NW corner
    if(tx == 0 && ty == 0){
        for(int row = 0; row < prbH/2; row++){
            for(int col = 0; col < prbW/2; col++){

                int gx = colImg - prbW/2 + col;
                int gy = rowImg - prbH/2 + row;

                if(gx < 0 || gy < 0)
                    tileDS[row * wideTile + col] = -1;
                else
                    tileDS[row * wideTile + col] = imgData[gy * imgW + gx];
            }
        }
    }

    // NE corner
    if(tx == TILE_WIDTH - 1 && ty == 0){
        for(int row = 0; row < prbH/2; row++){
            for(int col = TILE_WIDTH + prbH/2; col < wideTile; col ++){
                int gx = colImg + 1 + col;
                int gy = rowImg - prbH/2 + row;

                if(gx >= imgW || gy < 0)
                    tileDS[row * wideTile + col] = -1;
                else
                    tileDS[row * wideTile + col] = imgData[gy * imgW + gx];
            }
        }
    }

    // SW corner
    if(tx == 0 && ty == TILE_WIDTH - 1){
        for(int row = TILE_WIDTH + prbH/2; row < wideTile; row++){
            for(int col = 0; col < prbW/2; col++){
                int gx = colImg - prbW/2 + col;
                int gy = rowImg + 1 + row;

                if(gx < 0 || gy >= imgH)
                    tileDS[row * wideTile + col] = -1;
                else
                    tileDS[row * wideTile + col] = imgData[gy * imgW + gx];
            }
        }
    }

    // SE corner
    if(tx == TILE_WIDTH - 1 && ty == TILE_WIDTH -1){
        for(int row = TILE_WIDTH + prbH/2; row < wideTile; row++){
            for(int col = TILE_WIDTH + prbW/2; col < wideTile; col++){
                int gx = colImg + 1 + col;
                int gy = rowImg + 1 + row;

                if(gx >= imgW || gy >= imgH)
                    tileDS[row * wideTile + col] = -1;
                else
                    tileDS[row * wideTile + col] = imgData[gy * imgW + gx];
            }
        }
    }

    __syncthreads();

    /// compute
    if(rowImg >= 0 && rowImg < imgH && colImg >= 0 && colImg < imgW){
        float max = tileDS[(destY + prbH/2) * wideTile + destX + prbW/2];
        float min = max;

        for(int rowPrb = - prbH/2; rowPrb < prbH/2; rowPrb++) {
            for(int colPrb = - prbW/2; colPrb < prbW/2; colPrb++) {

                // apply mask
                if(tileDS[(destY + prbH/2 + rowPrb) * wideTile + destX + prbW/2 + colPrb] > -1){

                    if(max < tileDS[(destY + prbH/2 + rowPrb) * wideTile + destX + prbW/2 + colPrb])
                        max = tileDS[(destY + prbH/2 + rowPrb) * wideTile + destX + prbW/2 + colPrb];

                    if(min > tileDS[(destY + prbH/2 + rowPrb) * wideTile + destX + prbW/2 + colPrb])
                        min = tileDS[(destY + prbH/2 + rowPrb) * wideTile + destX + prbW/2 + colPrb];
                }
            }
        }

        if(mmOp == EROSION)
            outData[rowImg * imgW + colImg] = min;

        if(mmOp == DILATATION)
            outData[rowImg * imgW + colImg] = max;
    }
    __syncthreads();
}
