#ifndef NEURON_H
#define NEURON_H

#include <vector>
#include <cuda_runtime.h>

class Neuron {
private:
    double value;                        // 뉴런의 출력 값
    double bias;                         // 뉴런의 편향
    double* weights;                     // 연결된 가중치 (GPU 메모리 할당)
    Neuron** inputs;                     // 입력 뉴런 연결 (GPU 메모리 할당)
    double gradient;                     // 그래디언트 (역전파 계산용)

    int num_inputs;                      // 입력 뉴런 수

public:
    Neuron(int num_inputs);
    ~Neuron();

    double forward();                    // 순방향 계산
    void backward(double target, bool is_output); // 역전파 계산
    void update_weights(double learning_rate);    // 가중치 업데이트

    void set_input(double input_value);  // 입력 값 설정
    double get_output() const;           // 현재 출력 값
    void connect_input(Neuron* neuron);  // 입력 뉴런 연결
};

#endif
