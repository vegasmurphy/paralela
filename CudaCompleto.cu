#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#define BLOCK_SIZE 16
#define KERNEL_W 5
#define IMAGE_W 256
#define divider 16
#define KERNEL_RADIUS 2

bool convolve2D(unsigned char*, unsigned char*,int,int,float*, int, int);
bool convolve2D(float* , float* , int , int ,float* , int , int );
bool convolve2DSeparable(float* ,float* ,int ,int ,float* ,int ,float* ,int );


__device__ __constant__ float kernelC[25];
__device__ __constant__ float kernelVec[5];
/*
__global__ void gpuMM(float *A, float *B, float *C, int N)
{
	// Matrix multiplication for NxN matrices C=A*B
	// Each thread computes a single element of C
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int col = blockIdx.x*blockDim.x + threadIdx.x;

	float sum = 0.f;
	for (int m = 0; m < N; ++m)
		for (int n = 0; n < N; ++n)
		    sum += A[row*N+n]*kernelC[];

	C[row*N+col] = sum;
}
*/

__global__ void convolutionGPU(
                               float *d_Result,
                               float *d_Data,
                               int dataW,
                               int dataH )
{
 
    // global mem address for this thread
    const int gLoc = threadIdx.x + 
                     blockIdx.x * blockDim.x +
                     threadIdx.y * dataW +
                     blockIdx.y * blockDim.y * dataW; 

    float sum = 0;
    float value = 0;

    for (int i = -KERNEL_RADIUS; i <= KERNEL_RADIUS; i++)	// row wise
        for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)	// col wise
        {
            // check row first
            if (blockIdx.x == 0 && (threadIdx.x + i) < 0)	// left apron
                value = 0;
            else if ( blockIdx.x == (gridDim.x - 1) && 
                        (threadIdx.x + i) > blockDim.x-1 )	// right apron
                value = 0;
            else 
            { 
                // check col next
                if (blockIdx.y == 0 && (threadIdx.y + j) < 0)	// top apron
                    value = 0;
                else if ( blockIdx.y == (gridDim.y - 1) && 
                            (threadIdx.y + j) > blockDim.y-1 )	// bottom apron
                    value = 0;
                else	// safe case
                    value = d_Data[gLoc + i + j * dataW];
            } 
            sum += value * kernelVec[KERNEL_RADIUS + i] * kernelVec[KERNEL_RADIUS + j];
        }
        d_Result[gLoc] = sum; 
}

__global__ void convolutionColGPU(
                               float *d_Result,
                               float *d_Data,
                               int dataW,
                               int dataH )
{
 
    // global mem address for this thread
    const int gLoc = threadIdx.x + 
                     blockIdx.x * blockDim.x +
                     threadIdx.y * dataW +
                     blockIdx.y * blockDim.y * dataW; 

    float sum = 0;
    float value = 0;
    int col = threadIdx.y + (blockIdx.y * blockDim.y);
        for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)	// col wise
        {
            if (col+j<0 || col+j>=IMAGE_W)
            	value = 0;
            else	// safe case
                value = d_Data[gLoc + j * dataW];
        sum += value * kernelVec[KERNEL_RADIUS + j];
        } 
        
        d_Result[gLoc] = sum; 
}

__global__ void convolutionRowGPU(
                               float *d_Result,
                               float *d_Data,
                               int dataW,
                               int dataH )
{

    // global mem address for this thread
    const int gLoc = threadIdx.x + 
                     blockIdx.x * blockDim.x +
                     threadIdx.y * dataW +
                     blockIdx.y * blockDim.y * dataW; 

    float sum = 0;
    float value = 0;
    int row = threadIdx.x + (blockIdx.x * blockDim.x);
    for (int i = -KERNEL_RADIUS; i <= KERNEL_RADIUS; i++)	{// row wise
            // check row first
            if (row+i<0 || row+i>=IMAGE_W)
                value = 0;
            else 
                value = d_Data[gLoc + i ]; 
            sum += value * kernelVec[KERNEL_RADIUS + i];
        }
        d_Result[gLoc] = sum; 
}


int saveRawImage(char *fileName, int x, int y, unsigned char *data)
{
    // check params
    if(!fileName || !data)
        return 0;

    FILE *fp;
    if((fp = fopen(fileName, "w")) == NULL)
    {
        printf("Cannot open %s.\n", fileName);
        return 0;
    }

    // read pixel data
    fwrite(data, 1, x*y, fp);
    fclose(fp);
    return 1;
}

void loadRawImage(char *fileName, int x, int y, unsigned char *data)
{
    // check params
    if(!fileName || !data)
        return;

    FILE *fp;
    if((fp = fopen(fileName, "r")) == NULL)
    {
        printf("Cannot open %s.\n", fileName);
        return;
    }

    // read pixel data
    fread(data, 1, x*y, fp);
    fclose(fp);
}

int main(int argc, char *argv[])
{
	// Perform matrix multiplication C = A*B
	// where A, B and C are NxN matrices
	// Restricted to matrices where N = K*BLOCK_SIZE;

	time_t start,end;
  	double dif;

	int N,K;
	K = 16;			
	N = K*BLOCK_SIZE; //IMAGE_W
	
	int kernelSize = 5;
	char* fileName = "lena.raw";

	unsigned char Ha[256*256];
    float kernel[25] = { 1/256.0f,  4/256.0f,  6/256.0f,  4/256.0f, 1/256.0f,
                     4/256.0f, 16/256.0f, 24/256.0f, 16/256.0f, 4/256.0f,
                     6/256.0f, 24/256.0f, 36/256.0f, 24/256.0f, 6/256.0f,
                     4/256.0f, 16/256.0f, 24/256.0f, 16/256.0f, 4/256.0f,
                     1/256.0f,  4/256.0f,  6/256.0f,  4/256.0f, 1/256.0f };
    float sepKernel[5] = {0.0625,  0.25,  0.375,  0.25, 0.0625};
    //float sepKernel[5] = {0,  -.25,  1.5,  -.25, 0};
    
    loadRawImage(fileName, IMAGE_W, IMAGE_W, Ha);
    
    int j;
    int i;
	// Initialize matrices on the host
	printf("%i",N);
	
	float data[256*256];

	for (j=0; j<N; j++){
		//printf("\n");
	    for (i=0; i<N; i++){
	    	data[j*N+i]=(float)Ha[j*N+i];
	    	//printf("%f ",data[j*N+i]);
	   }
	}

		// Allocate memory on the device
	int size = N*N*sizeof(char);	// Size of the memory in bytes
	int kernelSize1 = 25*sizeof(float);	// Size of the memory in bytes
	
	char *dA;
	char *dC;
	float *fA;
	float *fB;
	float *fC;
	cudaMalloc(&dA,size);
	//cudaMalloc(&dB,size);
	//cudaMalloc(&dC,N*N*sizeof(char));
	cudaMalloc(&fA,N*N*sizeof(float));//Data matrix
	cudaMalloc(&fB,N*N*sizeof(float));//Result 1
	cudaMalloc(&fC,N*N*sizeof(float));//Result 2

	dim3 threadBlock(BLOCK_SIZE,BLOCK_SIZE);
	dim3 grid(K,K);
	
	double ini,fin;
	// Copy matrices from the host to device
	//cudaMemcpy(dA,Ha,size,cudaMemcpyHostToDevice);

	cudaMemcpy(fA,data,N*N*sizeof(float),cudaMemcpyHostToDevice);
	
	cudaMemcpyToSymbol(kernelC,kernel,sizeof(float)*25);
	cudaMemcpyToSymbol(kernelVec,sepKernel,sizeof(float)*5);
	
	//cudaMemcpy(dB,kernel,kernelSize,cudaMemcpyHostToDevice);

	//convHor<<<grid,threadBlock>>>(dA,dC);
	//convVer<<<grid,threadBlock>>>(dA,dC);

	float C[N*N];
  // Do some calculation.

  	ini=omp_get_wtime();
	convolve2D(data, C, 256, 256, kernel, 5, 5);//Serial
	fin=omp_get_wtime();
	printf("demoro en serial %f\n",fin-ini);

  ini=omp_get_wtime();
	convolutionGPU<<<grid,threadBlock>>>(fC,fA,256,256);//gpu completo
	
	//convolutionRowGPU<<<grid,threadBlock>>>(fB,fA,256,256);//gpu separable
	//convolutionColGPU<<<grid,threadBlock>>>(fC,fB,256,256);

	cudaDeviceSynchronize();
    


    unsigned char Serial[N*N];
    float f[256*256];
    
    // Now copy the GPU result back to CPU
    cudaMemcpy(f,fC,N*N*sizeof(float),cudaMemcpyDeviceToHost);
	fin=omp_get_wtime();
  printf("demoro en cuda %f\n",fin-ini);
	unsigned char res[256*256];
	unsigned char resSer[256*256];
	/*printf("\n\n\n Result:");
	for (j=0; j<N; j++){
		printf("\n");
	    for (i=0; i<N; i++){
	    	printf("%d ",(int)f[j*N+i]);
	   }
	}*/
	for (j=0; j<N; j++){
	    for (i=0; i<N; i++){
	    	//int temp = (int)f[j*N+i];
	    	//res[j*N+i]=temp & 0xFF;
	    	res[j*N+i]=(unsigned char) f[j*N+i];
	   }
	}

	for (j=0; j<N; j++){
	    for (i=0; i<N; i++){
	    	//int temp = (int)C[j*N+i];
	    	//resSer[j*N+i]=temp & 0xFF;
	    	resSer[j*N+i]= (unsigned char) C[j*N+i];
	   }
	}

	char* file="./resultCudaCompleto.raw";
	saveRawImage(file, 256, 256, res);
	char* file1="./resultSec.raw";
	saveRawImage(file1, 256, 256, resSer);

}

bool convolve2D(unsigned char* in, unsigned char* out, int dataSizeX, int dataSizeY, 
                float* kernel, int kernelSizeX, int kernelSizeY)
{
    int i, j, m, n;
    unsigned char *inPtr, *inPtr2, *outPtr;
    float *kPtr;
    int kCenterX, kCenterY;
    int rowMin, rowMax;                             // to check boundary of input array
    int colMin, colMax;                             //
    float sum;                                      // temp accumulation buffer

    // check validity of params
    if(!in || !out || !kernel) return false;
    if(dataSizeX <= 0 || kernelSizeX <= 0) return false;

    // find center position of kernel (half of kernel size)
    kCenterX = kernelSizeX >> 1;
    kCenterY = kernelSizeY >> 1;

    // init working  pointers
    inPtr = inPtr2 = &in[dataSizeX * kCenterY + kCenterX];  // note that  it is shifted (kCenterX, kCenterY),
    outPtr = out;
    kPtr = kernel;

    // start convolution
    for(i= 0; i < dataSizeY; ++i)                   // number of rows
    {
        // compute the range of convolution, the current row of kernel should be between these
        rowMax = i + kCenterY;
        rowMin = i - dataSizeY + kCenterY;

        for(j = 0; j < dataSizeX; ++j)              // number of columns
        {
            // compute the range of convolution, the current column of kernel should be between these
            colMax = j + kCenterX;
            colMin = j - dataSizeX + kCenterX;

            sum = 0;                                // set to 0 before accumulate

            // flip the kernel and traverse all the kernel values
            // multiply each kernel value with underlying input data
            for(m = 0; m < kernelSizeY; ++m)        // kernel rows
            {
                // check if the index is out of bound of input array
                if(m <= rowMax && m > rowMin)
                {
                    for(n = 0; n < kernelSizeX; ++n)
                    {
                        // check the boundary of array
                        if(n <= colMax && n > colMin)
                            sum += *(inPtr - n) * *kPtr;

                        ++kPtr;                     // next kernel
                    }
                }
                else
                    kPtr += kernelSizeX;            // out of bound, move to next row of kernel

                inPtr -= dataSizeX;                 // move input data 1 raw up
            }

            // convert negative number to positive
            *outPtr = (unsigned char)((float)fabs(sum) + 0.5f);

            kPtr = kernel;                          // reset kernel to (0,0)
            inPtr = ++inPtr2;                       // next input
            ++outPtr;                               // next output
        }
    }

    return true;
}

bool convolve2D(float* in, float* out, int dataSizeX, int dataSizeY, 
                float* kernel, int kernelSizeX, int kernelSizeY)
{
    int i, j, m, n;
    float *inPtr, *inPtr2, *outPtr, *kPtr;
    int kCenterX, kCenterY;
    int rowMin, rowMax;                             // to check boundary of input array
    int colMin, colMax;                             //

    // check validity of params
    if(!in || !out || !kernel) return false;
    if(dataSizeX <= 0 || kernelSizeX <= 0) return false;

    // find center position of kernel (half of kernel size)
    kCenterX = kernelSizeX >> 1;
    kCenterY = kernelSizeY >> 1;

    // init working  pointers
    inPtr = inPtr2 = &in[dataSizeX * kCenterY + kCenterX];  // note that  it is shifted (kCenterX, kCenterY),
    outPtr = out;
    kPtr = kernel;

    // start convolution
    for(i= 0; i < dataSizeY; ++i)                   // number of rows
    {
        // compute the range of convolution, the current row of kernel should be between these
        rowMax = i + kCenterY;
        rowMin = i - dataSizeY + kCenterY;

        for(j = 0; j < dataSizeX; ++j)              // number of columns
        {
            // compute the range of convolution, the current column of kernel should be between these
            colMax = j + kCenterX;
            colMin = j - dataSizeX + kCenterX;

            *outPtr = 0;                            // set to 0 before accumulate

            // flip the kernel and traverse all the kernel values
            // multiply each kernel value with underlying input data
            for(m = 0; m < kernelSizeY; ++m)        // kernel rows
            {
                // check if the index is out of bound of input array
                if(m <= rowMax && m > rowMin)
                {
                    for(n = 0; n < kernelSizeX; ++n)
                    {
                        // check the boundary of array
                        if(n <= colMax && n > colMin)
                            *outPtr += *(inPtr - n) * *kPtr;
                        ++kPtr;                     // next kernel
                    }
                }
                else
                    kPtr += kernelSizeX;            // out of bound, move to next row of kernel

                inPtr -= dataSizeX;                 // move input data 1 raw up
            }

            kPtr = kernel;                          // reset kernel to (0,0)
            inPtr = ++inPtr2;                       // next input
            ++outPtr;                               // next output
        }
    }

    return true;
}

bool convolve2DSeparable(float* in, float* out, int dataSizeX, int dataSizeY, 
                         float* kernelX, int kSizeX, float* kernelY, int kSizeY)
{
    int i, j, k, m, n;
    float *tmp, *sum;                               // intermediate data buffer
    float *inPtr, *outPtr;                          // working pointers
    float *tmpPtr, *tmpPtr2;                        // working pointers
    int kCenter, kOffset, endIndex;                 // kernel indice

    // check validity of params
    if(!in || !out || !kernelX || !kernelY) return false;
    if(dataSizeX <= 0 || kSizeX <= 0) return false;

    // allocate temp storage to keep intermediate result
    tmp = new float[dataSizeX * dataSizeY];
    if(!tmp) return false;  // memory allocation error

    // store accumulated sum
    sum = new float[dataSizeX];
    if(!sum) return false;  // memory allocation error

    // covolve horizontal direction ///////////////////////

    // find center position of kernel (half of kernel size)
    kCenter = kSizeX >> 1;                          // center index of kernel array
    endIndex = dataSizeX - kCenter;                 // index for full kernel convolution

    // init working pointers
    inPtr = in;
    tmpPtr = tmp;                                   // store intermediate results from 1D horizontal convolution

    // start horizontal convolution (x-direction)
    for(i=0; i < dataSizeY; ++i)                    // number of rows
    {

        kOffset = 0;                                // starting index of partial kernel varies for each sample

        // COLUMN FROM index=0 TO index=kCenter-1
        for(j=0; j < kCenter; ++j)
        {
            *tmpPtr = 0;                            // init to 0 before accumulation

            for(k = kCenter + kOffset, m = 0; k >= 0; --k, ++m) // convolve with partial of kernel
            {
                *tmpPtr += *(inPtr + m) * kernelX[k];
            }
            ++tmpPtr;                               // next output
            ++kOffset;                              // increase starting index of kernel
        }

        // COLUMN FROM index=kCenter TO index=(dataSizeX-kCenter-1)
        for(j = kCenter; j < endIndex; ++j)
        {
            *tmpPtr = 0;                            // init to 0 before accumulate

            for(k = kSizeX-1, m = 0; k >= 0; --k, ++m)  // full kernel
            {
                *tmpPtr += *(inPtr + m) * kernelX[k];
            }
            ++inPtr;                                // next input
            ++tmpPtr;                               // next output
        }

        kOffset = 1;                                // ending index of partial kernel varies for each sample

        // COLUMN FROM index=(dataSizeX-kCenter) TO index=(dataSizeX-1)
        for(j = endIndex; j < dataSizeX; ++j)
        {
            *tmpPtr = 0;                            // init to 0 before accumulation

            for(k = kSizeX-1, m=0; k >= kOffset; --k, ++m)   // convolve with partial of kernel
            {
                *tmpPtr += *(inPtr + m) * kernelX[k];
            }
            ++inPtr;                                // next input
            ++tmpPtr;                               // next output
            ++kOffset;                              // increase ending index of partial kernel
        }

        inPtr += kCenter;                           // next row
    }
    // END OF HORIZONTAL CONVOLUTION //////////////////////

    // start vertical direction ///////////////////////////

    // find center position of kernel (half of kernel size)
    kCenter = kSizeY >> 1;                          // center index of vertical kernel
    endIndex = dataSizeY - kCenter;                 // index where full kernel convolution should stop

    // set working pointers
    tmpPtr = tmpPtr2 = tmp;
    outPtr = out;

    // clear out array before accumulation
    for(i = 0; i < dataSizeX; ++i)
        sum[i] = 0;

    // start to convolve vertical direction (y-direction)

    // ROW FROM index=0 TO index=(kCenter-1)
    kOffset = 0;                                    // starting index of partial kernel varies for each sample
    for(i=0; i < kCenter; ++i)
    {
        for(k = kCenter + kOffset; k >= 0; --k)     // convolve with partial kernel
        {
            for(j=0; j < dataSizeX; ++j)
            {
                sum[j] += *tmpPtr * kernelY[k];
                ++tmpPtr;
            }
        }

        for(n = 0; n < dataSizeX; ++n)              // convert and copy from sum to out
        {
            *outPtr = sum[n];                       // store final result to output array
            sum[n] = 0;                             // reset to zero for next summing
            ++outPtr;                               // next element of output
        }

        tmpPtr = tmpPtr2;                           // reset input pointer
        ++kOffset;                                  // increase starting index of kernel
    }

    // ROW FROM index=kCenter TO index=(dataSizeY-kCenter-1)
    for(i = kCenter; i < endIndex; ++i)
    {
        for(k = kSizeY -1; k >= 0; --k)             // convolve with full kernel
        {
            for(j = 0; j < dataSizeX; ++j)
            {
                sum[j] += *tmpPtr * kernelY[k];
                ++tmpPtr;
            }
        }

        for(n = 0; n < dataSizeX; ++n)              // convert and copy from sum to out
        {
            *outPtr = sum[n];                       // store final result to output buffer
            sum[n] = 0;                             // reset before next summing
            ++outPtr;                               // next output
        }

        // move to next row
        tmpPtr2 += dataSizeX;
        tmpPtr = tmpPtr2;
    }

    // ROW FROM index=(dataSizeY-kCenter) TO index=(dataSizeY-1)
    kOffset = 1;                                    // ending index of partial kernel varies for each sample
    for(i=endIndex; i < dataSizeY; ++i)
    {
        for(k = kSizeY-1; k >= kOffset; --k)        // convolve with partial kernel
        {
            for(j=0; j < dataSizeX; ++j)
            {
                sum[j] += *tmpPtr * kernelY[k];
                ++tmpPtr;
            }
        }

        for(n = 0; n < dataSizeX; ++n)              // convert and copy from sum to out
        {
            *outPtr = sum[n];                       // store final result to output array
            sum[n] = 0;                             // reset to 0 for next sum
            ++outPtr;                               // next output
        }

        // move to next row
        tmpPtr2 += dataSizeX;
        tmpPtr = tmpPtr2;                           // next input
        ++kOffset;                                  // increase ending index of kernel
    }
    // END OF VERTICAL CONVOLUTION ////////////////////////

    // deallocate temp buffers
    delete [] tmp;
    delete [] sum;
    return true;
}