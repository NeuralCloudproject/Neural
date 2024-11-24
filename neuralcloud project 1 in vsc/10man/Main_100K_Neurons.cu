#include <iostream>
#include "AICore.h"
#include <random>
#include <cuda_runtime.h>

int main() {
    const int TOTAL_NEURONS = 100000; // 총 뉴런 수
    const int INPUT_LAYER_SIZE = 1000;
    const int OUTPUT_LAYER_SIZE = 1000;
    const int HIDDEN_LAYER_SIZE = (TOTAL_NEURONS - INPUT_LAYER_SIZE - OUTPUT_LAYER_SIZE) / 3;

    AICore ai(INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE);

    // 랜덤 연결 설정 (최대 100,000개의 연결)
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(0, INPUT_LAYER_SIZE - 1);

    for (auto& neuron : ai.hidden_layer_1) {
        for (int i = 0; i < 100000; ++i) {
            neuron.connect_input(&ai.input_layer[distribution(generator)]);
        }
    }

    for (auto& neuron : ai.hidden_layer_2) {
        for (int i = 0; i < 100000; ++i) {
            neuron.connect_input(&ai.hidden_layer_1[distribution(generator)]);
        }
    }

    for (auto& neuron : ai.hidden_layer_3) {
        for (int i = 0; i < 100000; ++i) {
            neuron.connect_input(&ai.hidden_layer_2[distribution(generator)]);
        }
    }

    for (auto& neuron : ai.output_layer) {
        for (int i = 0; i < 100000; ++i) {
            neuron.connect_input(&ai.hidden_layer_3[distribution(generator)]);
        }
    }

    std::cout << "10만 개 뉴런 모델이 GPU에서 생성 및 연결되었습니다!" << std::endl;

    return 0;
}
