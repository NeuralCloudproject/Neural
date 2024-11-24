#include "AICore.h"
#include <iostream>
#include <random>

// 생성자
AICore::AICore(int input_size, int hidden_size, int output_size, double exploration_rate)
    : exploration_rate(exploration_rate) {
    // 입력층 생성
    for (int i = 0; i < input_size; ++i) {
        input_layer.emplace_back(0);
    }

    // 은닉층 생성
    for (int i = 0; i < hidden_size; ++i) {
        hidden_layer_1.emplace_back(input_size);
        hidden_layer_2.emplace_back(hidden_size);
        hidden_layer_3.emplace_back(hidden_size);
    }

    // 출력층 생성
    for (int i = 0; i < output_size; ++i) {
        output_layer.emplace_back(hidden_size);
    }
}

// 자율 행동
void AICore::think_and_act() {
    std::cout << "AI가 자율적으로 행동합니다." << std::endl;
}

// 입력에 응답
std::string AICore::respond(const std::string& input) {
    if (memory.find(input) != memory.end()) {
        return "기억에서 찾았습니다: " + memory[input];
    }
    return "제가 확실히 모르겠어요. 학습을 시도할까요?";
}

// 학습
void AICore::train(const std::string& input, const std::string& target) {
    remember(input, target);
}

// 기억 저장
void AICore::remember(const std::string& key, const std::string& value) {
    memory[key] = value;
}

// 기억 검색
std::string AICore::recall(const std::string& key) {
    if (memory.find(key) != memory.end()) {
        return memory[key];
    }
    return "기억이 없습니다.";
}
