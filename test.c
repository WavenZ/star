#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <omp.h>
#include <pthread.h>

typedef unsigned char uint8;

#define NTHREAD         8
#define PI              3.1415926
#define PERCENTAGE      0.005
#define MIN_CONNECT     30
#define MAX_SIZE        2048 * 2048
#define KERNEL_SIZE     13
#define EPS             1e-7

int queue[MAX_SIZE];
int front, back;

void queue_clear(){
    front = back = 0;
}
void queue_push(int x, int y){
    queue[back] = x;
    queue[back + 1] = y;
    back += 2;
}
void queue_pop(int* x, int *y){
    *x = queue[front];
    *y = queue[front + 1];
    front += 2;
}
int queue_empty(){
    return front == back;
}
int queue_size(){
    return (back - front) / 2;
}

void conv(uint8 *src, int h, int w, int x0, int y0, uint8 *res, 
                                    double kernels[181][KERNEL_SIZE][KERNEL_SIZE]){
    /*
        Convolution.
    */
    double cx = x0 + EPS;
    double cy = y0 + EPS;

    #pragma omp parallel for num_threads(NTHREAD)
    for(int i = KERNEL_SIZE / 2; i < h - KERNEL_SIZE / 2; ++i){
        int row = h - 1 - i;
        for(int j = KERNEL_SIZE / 2; j < w - KERNEL_SIZE / 2; ++j){
            int col = j;
            double temp = 0;
            int n = (int)((atan((col - cx) * 1.0 / (row - cy)) * 180 / PI) + 90);
            for(int x = 0; x < KERNEL_SIZE; ++x){
                int offset = (row - KERNEL_SIZE / 2 + x) * w + col - KERNEL_SIZE / 2;
                for(int y = 0; y < KERNEL_SIZE; ++y){
                    temp += kernels[n][x][y] * src[offset + y];
                }
            }
            if(temp >= 255) 
                res[row * w + col] = 255;
            else if(temp <= 0) 
                res[row * w + col] = 0;
            else 
                res[row * w + col] = (int)temp;
        }
    }
}

void conv_and_bin(uint8 *src, int h, int w, int x0, int y0, uint8 *res, 
                                    double kernels[181][KERNEL_SIZE][KERNEL_SIZE]){
    /*
        Convolution and binarization.
    */
    printf("\033[1m\033[;33m[call conv() in conv.dll ...]\033[0m\n");
    printf("\033[1m\033[;33m[(h, w) = (%d, %d) (x0, y0) = (%d, %d)]\033[0m\n", h, w, x0, y0);
    
    conv(src, h, w, x0, y0, res, kernels);

    // return;      
    /*
     * Calculate the cumulative distribution. 
     */
    int* hist = (int*)malloc((256 + 1) * sizeof(int));
    memset(hist, 0, sizeof(int) * (256 + 1));
    for(int i = 0; i < h; ++i){
        for(int j = 0; j < w; ++j){
            hist[res[i * w + j]]++;
        }
    }
    for(int i = 256; i >= 0; --i){
        if(i != 256) hist[i] += hist[i + 1];
    }

    /*
     * Calculate the threshold of binarization. 
     */
    int thresh = 256;
    while(hist[thresh] < PERCENTAGE * h * w) thresh--;

    // printf("%d\n", thresh);

    /*
     * Binarization.
     */
    for(int i = 0; i < h; ++i){
        for(int j = 0; j < w; ++j){
            res[i * w + j] = (res[i * w + j] > thresh) ? 254 : 0;
        }
    }

    /*
     * Connected component analysis.
     */
    for(int i = KERNEL_SIZE / 2; i < h - KERNEL_SIZE / 2; ++i){
        for(int j = KERNEL_SIZE / 2; j < w - KERNEL_SIZE / 2; ++j){
            if(res[i * w + j] == 254){
                int cnt = 1;
                queue_clear();
                res[i * w + j] = 255;
                queue_push(i, j);
                int x, y;
                while(!queue_empty()){
                    for(int k = queue_size(); k > 0; --k){
                        queue_pop(&x, &y);
                        for(int dx = -1; dx <= 1; ++dx){
                            for(int dy = -1; dy <= 1; ++dy){
                                if(res[(x + dx) * w + y + dy] == 254){
                                    res[(x + dx) * w + y + dy] = 255;
                                    queue_push(x + dx, y + dy);
                                    cnt++;
                                }
                            }
                        }
                    }
                }
                if(cnt < MIN_CONNECT){  // Delete small connected blocks.
                    queue_clear();
                    res[i * w + j] = 0;
                    queue_push(i, j);
                    while(!queue_empty()){
                        for(int k = queue_size(); k > 0; --k){
                            queue_pop(&x, &y);
                            for(int dx = -1; dx <= 1; ++dx){
                                for(int dy = -1; dy <= 1; ++dy){
                                    if(res[(x + dx) * w + y + dy] == 255){
                                        res[(x + dx) * w + y + dy] = 0;
                                        queue_push(x + dx, y + dy);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    free(hist);
}