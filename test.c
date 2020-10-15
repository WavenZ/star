#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <omp.h>
#include <pthread.h>

typedef unsigned char uint8;

#define NTHREAD 8
#define PI 3.1415926
#define PERCENTAGE 0.003
#define MIN_CONNECT 50
#define MAX_SIZE 2048 * 2048

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

void conv(uint8 *src, int h, int w, int x0, int y0, uint8 *res, double kernels[181][11][11]){
    /*
        Convolution.
    */
    printf("\033[1m\033[;33m[call conv() in conv.dll ...]\033[0m\n");
    printf("\033[1m\033[;33m[(h, w) = (%d, %d) (x0, y0) = (%d, %d)]\033[0m\n", h, w, x0, y0);
    
    double cx = x0 + 0.0001;
    double cy = y0 + 0.0001;

    #pragma omp parallel for num_threads(NTHREAD)
    for(int i = 5; i < h - 5; ++i){
        int row = h - 1 - i;
        for(int j = 5; j < w - 5; ++j){
            int col = j;
            double temp = 0;
            double a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10;
            int n = (int)((atan((col - cx) * 1.0 / (row - cy)) * 180 / PI) + 90);
            for(int x = 0; x < 11; ++x){
                int offset = (row - 5 + x) * w + col - 5;
                a0 = kernels[n][x][0] * src[offset + 0];
                a1 = kernels[n][x][1] * src[offset + 1];
                a2 = kernels[n][x][2] * src[offset + 2];
                a3 = kernels[n][x][3] * src[offset + 3];
                a4 = kernels[n][x][4] * src[offset + 4];
                a5 = kernels[n][x][5] * src[offset + 5];
                a6 = kernels[n][x][6] * src[offset + 6];
                a7 = kernels[n][x][7] * src[offset + 7];
                a8 = kernels[n][x][8] * src[offset + 8];
                a9 = kernels[n][x][9] * src[offset + 9];
                a10 = kernels[n][x][10] * src[offset + 10];
                temp += (a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10);
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

void conv_and_bin(uint8 *src, int h, int w, int x0, int y0, uint8 *res, double kernels[181][11][11]){
    /*
        Convolution and binarization.
    */
    printf("\033[1m\033[;33m[call conv() in conv.dll ...]\033[0m\n");
    printf("\033[1m\033[;33m[(h, w) = (%d, %d) (x0, y0) = (%d, %d)]\033[0m\n", h, w, x0, y0);
    
    double cx = x0 + 0.0001;
    double cy = y0 + 0.0001;

    #pragma omp parallel for num_threads(NTHREAD)
    for(int i = 5; i < h - 5; ++i){
        int row = h - 1 - i;
        for(int j = 5; j < w - 5; ++j){
            int col = j;
            double temp = 0;
            double a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10;
            int n = (int)((atan((col - cx) * 1.0 / (row - cy)) * 180 / PI) + 90);
            for(int x = 0; x < 11; ++x){
                int offset = (row - 5 + x) * w + col - 5;
                a0 = kernels[n][x][0] * src[offset + 0];
                a1 = kernels[n][x][1] * src[offset + 1];
                a2 = kernels[n][x][2] * src[offset + 2];
                a3 = kernels[n][x][3] * src[offset + 3];
                a4 = kernels[n][x][4] * src[offset + 4];
                a5 = kernels[n][x][5] * src[offset + 5];
                a6 = kernels[n][x][6] * src[offset + 6];
                a7 = kernels[n][x][7] * src[offset + 7];
                a8 = kernels[n][x][8] * src[offset + 8];
                a9 = kernels[n][x][9] * src[offset + 9];
                a10 = kernels[n][x][10] * src[offset + 10];
                temp += (a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10);
            }
            if(temp >= 255) 
                res[row * w + col] = 255;
            else if(temp <= 0) 
                res[row * w + col] = 0;
            else 
                res[row * w + col] = (int)temp;
        }
    }

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
    for(int i = 5; i < h - 5; ++i){
        for(int j = 5; j < w - 5; ++j){
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