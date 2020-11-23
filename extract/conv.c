#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <omp.h>
#include <pthread.h>

typedef unsigned char uint8;

#define max(a, b)       ((a)>(b)?(a):(b))
#define min(a, b)       ((a)<(b)?(a):(b))

#define NTHREAD         8
#define PI              3.1415926
#define MIN_CONNECT     30
#define MAX_SIZE        2048 * 2048
#define KERNEL_SIZE     13
#define EPS             1e-7
#define MAX_COMNUM      200000
#define MAX_PIXEL       1000

/* Queue for bfs. */
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

int maxlen;
int size[MAX_COMNUM];                   // Size of each domain.
int mean[MAX_COMNUM];                   // Mean gray value of each domain.
int point[MAX_COMNUM][MAX_PIXEL][2];    // Coordinates of each domain.
int endpoint[MAX_COMNUM][2][2];         // Endpoints of each domain.

void push_back(int index, int x, int y){
    /*
     * Add coodinates to doamin.
     */
    point[index][size[index]][0] = x;
    point[index][size[index]][1] = y;
    size[index]++;
}
int get_size(int index){
    return size[index];
}
void clear(){
    /* 
     * Clear some global variables.
     */
    maxlen = 0;
    memset(mean, 0, sizeof(int) * MAX_COMNUM);
    memset(size, 0, sizeof(int) * MAX_COMNUM);
    memset(endpoint, 0, sizeof(int) * MAX_COMNUM * 2 * 2);
}

void update_endpoint(int index){
    /*
     * Update the endpoints of the giving domain.
     */
    int minx = 0x7fffffff, miny = 0x7fffffff, maxx = -1, maxy = -1;
    for(int i = 0; i < size[index]; ++i){
        minx = min(minx, point[index][i][0]);
        maxx = max(maxx, point[index][i][0]);
        miny = min(miny, point[index][i][1]);
        maxy = max(maxy, point[index][i][1]);
    }
    int dx = maxx - minx, dy = maxy - miny;
    if(dx > dy){
        for(int i = 0; i < size[index]; ++i){
            if(point[index][i][0] == minx){
                endpoint[index][0][0] = point[index][i][0];
                endpoint[index][0][1] = point[index][i][1];
            }
            if(point[index][i][0] == maxx){
                endpoint[index][1][0] = point[index][i][0];
                endpoint[index][1][1] = point[index][i][1];
            }
        }
    }else{
        for(int i = 0; i < size[index]; ++i){
            if(point[index][i][1] == miny){
                endpoint[index][0][0] = point[index][i][0];
                endpoint[index][0][1] = point[index][i][1];
            }
            if(point[index][i][1] == maxy){
                endpoint[index][1][0] = point[index][i][0];
                endpoint[index][1][1] = point[index][i][1];
            }
        }
    }
    maxlen = max(maxlen, sqrt(dx * dx + dy * dy));
    // printf("maxlen = %d\n", maxlen);
}

int get_distance(int x0, int y0, int x1, int y1){
    /*
     * Get distance of (x0, y0) to (x1, y1).
     */
    return sqrt((x0 - x1) * (x0 - x1) + (y0 - y1) * (y0 - y1));
}

double point_line(int x, int y, double k, double b){
    /*
     * Get distance of (x, y) to y = kx + b.
     */
    return fabs(k * x - y + b) * 1.0 / sqrt(k * k + 1);
}

int merge(int p, int q, uint8 *res, int w, int cx, int cy){
    /*
     * Try two merge two domains.
     */

    /* Calc minimun distance */
    int mindis = 0x7fffffff, maxdis = 0;
    for(int i = 0; i <= 1; ++i){
        for(int j = 0; j <= 1; ++j){
            int dis = get_distance(endpoint[p][i][0], endpoint[p][i][1], endpoint[q][j][0], endpoint[q][j][1]);
            mindis = min(mindis, dis);
            maxdis = max(maxdis, dis);
        }
    }
    /* Return if distance dissatisfy. */
    if(mindis > 50 || maxdis > maxlen * 1.5){
        return 0;
    }

    /* Get Le. */
    int x0, y0, x1, y1, dis = mindis;
    for(int i = 0; i <= 1; ++i){
        for(int j = 0; j <= 1; ++j){
            int temp = get_distance(endpoint[p][i][0], endpoint[p][i][1],
                                endpoint[q][j][0], endpoint[q][j][1]);
            if(temp >= dis){
                x0 = endpoint[p][i][0];
                y0 = endpoint[p][i][1];
                x1 = endpoint[q][j][0];
                y1 = endpoint[q][j][1];
                dis = temp;
            }
        }
    }
    double k = (x1 == x0) ? 9999.9999 : ((y1 - y0) * 1.0 / (x1 - x0));
    double b = y1 - k * x1;

    /* Calc the gap. */
    double gap = point_line(endpoint[p][0][0], endpoint[p][0][1], k, b) + 
                point_line(endpoint[p][1][0], endpoint[p][1][1], k, b) + 
                point_line(endpoint[q][0][0], endpoint[q][0][1], k, b) +
                point_line(endpoint[q][1][0], endpoint[q][1][1], k, b);
    if(gap > 3.14){
        return 0;
    }

    /* Get the center of each part. */
    double cx0 = 0, cy0 = 0, cx1 = 0, cy1 = 0;
    for(int i = 0; i < size[p]; ++i){
        cx0 += point[p][i][0];
        cy0 += point[p][i][1];
    }
    cx0 = cx0 / size[p];
    cy0 = cy0 / size[p];
    for(int i = 0; i < size[q]; ++i){
        cx1 += point[q][i][0];
        cy1 += point[q][i][1];
    }
    cx1 = cx1 / size[q];
    cy1 = cy1 / size[q];

    /* The line that goes through two centers. */
    k = (cx1 == cx0) ? 9999.9999 : ((cy1 - cy0) * 1.0 / (cx1 - cx0));
    b = cy1 - k * cx1;

    // printf("** %lf, %lf\n", atan(k) * 180 / 3.1415926, atan(((cy0 + cy1) / 2 - cy) / ((cx0 + cx1) / 2 - cx)) * 180 / 3.1415926);
    if(fabs(atan(k) * 180 / 3.1415926 + atan(((cy0 + cy1) / 2 - cy) / ((cx0 + cx1) / 2 - cx)) * 180 / 3.1415926) > 3){
        // printf("   fail.\n");
        return 0;
    }     

    /* Get the max distance to Lc. */
    int dx = max(max(endpoint[p][0][0], endpoint[p][1][0]),
                    max(endpoint[q][0][0], endpoint[q][1][0])) - 
             min(min(endpoint[p][0][0], endpoint[p][1][0]),
                    min(endpoint[q][0][0], endpoint[q][1][0]));
    int dy = max(max(endpoint[p][0][1], endpoint[p][1][1]),
                    max(endpoint[q][0][1], endpoint[q][1][1])) - 
             min(min(endpoint[p][0][1], endpoint[p][1][1]),
                    min(endpoint[q][0][1], endpoint[q][1][1]));
    // printf("%d %d\n", dx, dy);

    double maxwidth = 0.0;
    for(int i = 0; i < size[p]; ++i){
        maxwidth = max(maxwidth, point_line(point[p][i][0], point[p][i][1], k, b));
    }
    for(int i = 0; i < size[q]; ++i){
        maxwidth = max(maxwidth, point_line(point[q][i][0], point[q][i][1], k, b));
    }

    /* Merge two domains and fix the star. */
    if(dx > dy){
        int up = min(cx0, cx1);
        int down = max(cx0, cx1);
        // printf("%d %d\n", up, down);
        for(int i = up; i <= down; ++i){
            int mid = k * i + b + 0.5;
            for(int j = mid - 5; j <= mid + 5; ++j){
                if(res[i * w + j] != 255 && point_line(i, j, k, b) < maxwidth){
                    res[i * w + j] = 150;
                    push_back(q, i, j);
                }
            }
        }
    }else{
        int left = min(cy0, cy1);
        int right = max(cy0, cy1);
        // printf("%d %d\n", left, right);
        for(int j = left; j <= right; ++j){
            int mid = (j - b) / k + 0.5;
            for(int i = mid - 5; i <= mid + 5; ++i){
                if(res[i * w + j] != 255 && point_line(i, j, k, b) < maxwidth){
                    res[i * w + j] = 150;
                    push_back(q, i, j);
                }
            }
        }        
    }
    /* Merge domain p to dimain q. */
    for(int i = 0; i < size[p]; ++i){
        push_back(q, point[p][i][0], point[p][i][1]);
    }
    update_endpoint(q);
    size[p] = 0;

    // printf("   success.\n");
    return 1;
}


void conv(uint8 *src, int h, int w, double x0, double y0, uint8 *res, 
                                    double kernels[181][KERNEL_SIZE][KERNEL_SIZE]){
    /*
        Convolution.
    */
    // printf("%lf, %lf\n", x0, y0);
    if(fabs(x0 - 99999) + fabs(y0 - 99999) < EPS){
        for(int i = 0; i < h; ++i){
            for(int j = 0; j < w; ++j){
                res[i * w + j] = src[i * w + j];
            }
        }
        return;
    }

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

void conv_and_bin(uint8 *src, int h, int w, double x0, double y0, uint8 *res, 
                                    double kernels[181][KERNEL_SIZE][KERNEL_SIZE],
                                    double centers[1000][3], int* ccnt){

    // printf("\033[1m\033[;33m[call conv() in conv.dll ...]\033[0m\n");
    // printf("\033[1m\033[;33m[(h, w) = (%d, %d) (x0, y0) = (%.2lf, .%2lf)]\033[0m\n", h, w, x0, y0);
    
    /* Clear some global variable. */
    clear();

    /* Convolution and binarization. */
    conv(src, h, w, x0, y0, res, kernels);

    /* Calculate the threshold of binarization. */
    int thresh = 256;
    int pos[4][2] = {{16, 0}, {16, 256}, {16, 512}, {16, 1024}};
    #pragma omp parallel for num_threads(NTHREAD)
    for(int k = 0; k < 4; ++k){
        int sum = 0;
        for(int i = pos[k][0]; i < pos[k][0] + 128; ++i){
            for(int j = pos[k][1]; j < pos[k][1] + 128; ++j){
                sum += res[i * w + j];
            }
        }
        double mean = sum * 1.0 / (128 * 128);
        double var = 0.0;
        for(int i = pos[k][0]; i < pos[k][0] + 128; ++i){
            for(int j = pos[k][1]; j < pos[k][1] + 128; ++j){
                var += ((res[i * w + j] - mean) * (res[i * w + j] - mean));
            }
        }
        var /= (128 * 128);
        thresh = min(thresh, (int)(mean + sqrt(var) * 5));
    }

    /* Binarization. */
    #pragma omp parallel for num_threads(NTHREAD)
    for(int i = 0; i < h; ++i){
        for(int j = 0; j < w; ++j){
            res[i * w + j] = (res[i * w + j] > thresh) ? 254 : 0;
        }
    }

    /* Get connected domain by bfs method. */ 
    int cnt = 0;
    for(int i = KERNEL_SIZE / 2; i < h - KERNEL_SIZE / 2; ++i){
        for(int j = KERNEL_SIZE / 2; j < w - KERNEL_SIZE / 2; ++j){
            if(res[i * w + j] == 254){
                queue_clear();
                res[i * w + j] = 255;
                queue_push(i, j);
                int x, y;
                while(!queue_empty()){
                    for(int k = queue_size(); k > 0; --k){
                        queue_pop(&x, &y);
                        push_back(cnt, x, y);
                        for(int dx = -1; dx <= 1; ++dx){
                            for(int dy = -1; dy <= 1; ++dy){
                                if(res[(x + dx) * w + y + dy] == 254){
                                    res[(x + dx) * w + y + dy] = 255;
                                    queue_push(x + dx, y + dy);
                                }
                            }
                        }
                    }
                }
                /* Clear small domains. */
                if(size[cnt] < 20){
                    for(int i = 0; i < size[cnt]; ++i){
                        res[point[cnt][i][0] * w + point[cnt][i][1]] = 0;
                    }
                    size[cnt] = 0;
                }else 
                    cnt++;
            }
        }
    } 

    /* Connected domain analysis and star fix. */
    if(fabs(x0 - 99999) + fabs(y0 - 99999) > EPS){
        
        /* Update endpoints of each domain. */
        for(int i = 0; i < cnt; ++i){
            if(size[i] <= 10) continue;
            int temp = 0;
            for(int j = 0;j < size[i]; ++j){
                temp += src[point[i][j][0] * w + point[i][j][1]];
            }
            mean[i] = temp / size[i];
            update_endpoint(i);
        }

        /* Try to merge domain */
        int merge_cnt = 0;   // Successfully merged counter.
        for(int i = 0; i < cnt; ++i){
            // if(merge_cnt == 5) break;
            if(size[i] <= 10 || mean[i] > thresh) continue;
            for(int j = i + 1; j < cnt; ++j){
                // if(merge_cnt == 5) break;
                if(size[j] <= 10 || mean[j] > thresh) continue;
                if(merge(i, j, res, w, x0, y0)){
                    merge_cnt++;
                    break;
                }
            }
        }
        // printf("%d\n", merge_cnt);

    }

    /* Clear small domains after the merge. */
    for(int i = 0; i < cnt; ++i){
        if(size[i] <= 50){
            for(int j = 0; j < size[i]; ++j){
                res[point[i][j][0] * w + point[i][j][1]] = 0;
            }
            size[i] = 0;
        }else{
            for(int j = 0; j < size[i]; ++j){
                res[point[i][j][0] * w + point[i][j][1]] = 255;
            }
        }
    }
    

    /* Update the center of each star. */
    *ccnt = 0;
    for(int i = 0; i < cnt; ++i){
        if(size[i] > 0){
            double cx = 0, cy = 0, gray = 0.0;
            for(int j = 0; j < size[i]; ++j){
                cx += point[i][j][0];
                cy += point[i][j][1];
                gray += src[point[i][j][0] * w + point[i][j][1]];
            }
            
            cx /= size[i];
            cy /= size[i];
            gray /= size[i];
            centers[*ccnt][0] = cy;
            centers[*ccnt][1] = cx;
            centers[*ccnt][2] = gray;
            (*ccnt)++;
            res[(int)(cx + 0.5) * w + (int)(cy + 0.5)] = 0;
            // printf("(%lf, %lf) ", cx, cy);
        }
    }
    // printf("\n");
    // printf("\033[1m\033[;33m[Total stars: %d]\033[0m\n", *ccnt);
}