#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <functional>
#include <pthread.h>
#include "timer.h"
using namespace std;

#define PI 3.1415926
const int nthread = 8;



vector<vector<vector<double>>> kernels;
void kread(const string& filename){
    ifstream in(filename);
    kernels = vector<vector<vector<double>>>(181, vector<vector<double>>(11, vector<double>(11)));
    for(int i = 0; i < 181; ++i){
        for(int j = 0; j < 11; ++j){
            for(int k = 0; k < 11; ++k){
                in >> kernels[i][j][k];
            }
        }
    }
}

char *ans, *src;
int cx, cy;
int h, w;
void* conv_thread(void* arg){
    int tid = *(int*)arg;
    for(int i = 5 + tid; i < h - 5; i += nthread){
        int row = h - 1 - i;
        for(int j = 5; j < w - 5; ++j){
            int col = j;
            double temp = 0;
            auto& kernel = kernels[(atan((col - cx) * 1.0 / (row - cy)) * 180 / PI) + 90];
            for(int x = 0; x < 11; ++x){
                for(int y = 0; y < 11; ++y){
                    temp += kernel[x][y] * src[(row - 5 + x) * w + col - 5 + y];
                }
            }
            if(temp >= 255) ans[row * w + col] = 255;
            else if(temp <= 0) ans[row * w + col] = 0;
            else ans[row * w + col] = (int)temp;
        }
    }
    return nullptr;
}
char* conv(char* img){
    src = img;
    ans = new char[h * w];
    memcpy(ans, src, h * w);
    pthread_t threads[nthread];
    int tid[nthread];
    for(int i = 0; i < nthread; ++i){
        tid[i] = i;
        pthread_create(&threads[i], NULL, conv_thread, (void*)&tid[i]);
    }
    for(int i = 0; i < nthread; ++i){
        pthread_join(threads[i], NULL);
    }
    return ans;
}
char* imread(const string& filename){
    FILE* fp = fopen("img.txt", "rb");
    char* buf = new char[h * w];
    size_t size = fread(buf, 1, h * w, fp);
    if(size != h * w){
        cout << "Read error! size=" << size << " not equals to h*w=" << h * w << endl;
        return {};
    }
    return buf;
}
int main(int argc, char* argv[]){
    Timer t;
    if(argc > 1){
        h = atoi(argv[1]);
        w = atoi(argv[2]);
        cx = atoi(argv[3]);
        cy = atoi(argv[4]);
    }else{
        h = 2048, w = 2048;
        cx = 204800, cy = 204800;
    }
    cout << "[Call conv.exe...]" << endl;
    cout << "[h = " << h << "  w = " << w << "]" << endl;
    cout << "[x = " << cx << "  y = " << cy << "]" << endl;
    auto src = imread("img.txt");
    kread("kernel.txt");
    char* ret = conv(src);
    FILE* fp = fopen("img1.txt", "wb");
    fwrite(ret, 1, h * w, fp);
    fclose(fp);
    return 0;
}