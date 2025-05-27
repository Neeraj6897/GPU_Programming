#include<iostream>
using namespace std;

void sequential_scan(int in[], int out[], int n){
    out[0] = 0;
    for(int i=1; i<n; i++){
        out[i] = out[i-1] + in[i-1];
    }
}

int main(){
    int n = 8;
    int in[n] = {1,2,3,4,5,6,7,8};
    int out[n];
    sequential_scan(in, out, n);
    
    for(int i=0; i<n; i++){
        cout<<out[i]<<" ";
    }
    return 0;
}