#ifndef AICORE_H
#define AICORE_H

#include <string>
#include <unordered_map>
#include <vector>
#include "Neuron.h"

class AICore {
public:
    std::vector<Neuron> input_layer;     // 입력층
    std::vector<Neuron> hidden_layer_1;  // 은닉층 1
    std::vector<Neuron> hidden_layer_2;  // 은닉층 2
    std::vector<Neuron> hidden_layer_3;  // 은닉층 3
    std::vector<Neuron> output_layer;    // 출력층

private:
    std::unordered_map<std::string, std::string> memory; // 장기 기억
    double exploration_rate;                            // 자율 행동 확률

public:
    AICore(int input_size, int hidden_size, int output_size, double exploration_rate = 0.1);

    void think_and_act();                               // 자율 행동
    std::string respond(const std::string& input);      // 입력에 응답
    void train(const std::string& input, const std::string& target); // 학습
    void remember(const std::string& key, const std::string& value); // 기억 저장
    std::string recall(const std::string& key);         // 기억 검색
};

#endif
