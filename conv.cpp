#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cstdio>
#include "timer.h"
using namespace std;

vector<string> imread(const string& filename){
     ifstream in(filename);
     vector<string> res;
     string temp;
     while(in >> temp){
         res.push_back(temp);
     }
     return res;
}

vector<vector<vector<double>>> kread(const string& filename){
    ifstream in(filename);
    vector<vector<vector<double>>> kernels(180, vector<vector<double>>(11, vector<double>(11)));
    for(int i = 0; i < 180; ++i){
        for(int j = 0; j < 11; ++j){
            for(int k = 0; k < 11; ++k){
                in >> kernels[i][j][k];
            }
        }
    }
    return kernels;
}


int main(int argc, char* argv[]){
    Timer t;
    auto ret = imread("img.txt");
    auto kernels = kread("kernel.txt");
    for(int i = 0; i < 11; ++i){
        for(int j = 0; j < 11; ++j){
            cout << kernels[0][i][j] << " ";
        }
        cout << endl;
    }
    return 0;
}