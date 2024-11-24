#include "Neuron.h"
#include <cmath>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

// CUDA 커널 함수: 활성화 함수 (Sigmoid)
__device__ double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// CUDA 커널 함수: 활성화 함수 미분
__device__ double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

// 뉴런 생성자: GPU 메모리 할당
Neuron::Neuron(int num_inputs) : value(0.0), bias(0.0), gradient(0.0), num_inputs(num_inputs) {
    cudaMalloc((void**)&weights, num_inputs * sizeof(double));
    cudaMalloc((void**)&inputs, num_inputs * sizeof(Neuron*));

    // 랜덤 가중치 초기화
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    curandGenerateUniformDouble(gen, weights, num_inputs);
    curandDestroyGenerator(gen);

    // 편향 초기화
    bias = ((double)rand() / RAND_MAX) * 2 - 1; // -1 ~ 1 사이의 값
}

// 뉴런 소멸자: GPU 메모리 해제
Neuron::~Neuron() {
    cudaFree(weights);
    cudaFree(inputs);
}

// 순방향 계산 CUDA 커널
__global__ void forward_kernel(Neuron* neuron) {
    double sum = neuron->bias;
    for (int i = 0; i < neuron->num_inputs; ++i) {
        sum += neuron->inputs[i]->get_output() * neuron->weights[i];
    }
    neuron->value = sigmoid(sum);
}

// 순방향 계산 함수 호출
double Neuron::forward() {
    forward_kernel<<<1, 1>>>(this);
    cudaDeviceSynchronize();
    return value;
}

// 역전파 CUDA 커널
__global__ void backward_kernel(Neuron* neuron, double target, bool is_output) {
    if (is_output) {
        neuron->gradient = (target - neuron->value) * sigmoid_derivative(neuron->value);
    } else {
        double error = 0.0;
        for (int i = 0; i < neuron->num_inputs; ++i) {
            error += neuron->inputs[i]->gradient * neuron->weights[i];
        }
        neuron->gradient = error * sigmoid_derivative(neuron->value);
    }
}

// 역전파 계산 함수 호출
void Neuron::backward(double target, bool is_output) {
    backward_kernel<<<1, 1>>>(this, target, is_output);
    cudaDeviceSynchronize();
}

// 가중치 업데이트 CUDA 커널
__global__ void update_weights_kernel(Neuron* neuron, double learning_rate) {
    for (int i = 0; i < neuron->num_inputs; ++i) {
        neuron->weights[i] += learning_rate * neuron->gradient * neuron->inputs[i]->get_output();
    }
    neuron->bias += learning_rate * neuron->gradient;
}

// 가중치 업데이트 함수 호출
void Neuron::update_weights(double learning_rate) {
    update_weights_kernel<<<1, 1>>>(this, learning_rate);
    cudaDeviceSynchronize();
}

// 입력 값 설정
void Neuron::set_input(double input_value) {
    value = input_value;
}

// 출력 값 반환
double Neuron::get_output() const {
    return value;
}

// 입력 뉴런 연결
void Neuron::connect_input(Neuron* neuron) {
    // GPU 메모리에 직접 입력 뉴런 연결
    cudaMemcpy(&inputs[num_inputs++], &neuron, sizeof(Neuron*), cudaMemcpyHostToDevice);
}
