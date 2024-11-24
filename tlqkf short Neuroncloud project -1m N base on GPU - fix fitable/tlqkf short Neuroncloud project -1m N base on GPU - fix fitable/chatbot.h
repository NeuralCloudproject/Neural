#ifndef CHATBOT_H
#define CHATBOT_H

#include <torch/script.h>
#include <string>
#include <vector>

class Chatbot {
private:
    torch::jit::script::Module model;

public:
    Chatbot(const std::string& model_path) {
        model = torch::jit::load(model_path);
        model.to(torch::kCUDA); // GPU 사용
    }

    std::string generate_response(const std::string& user_input, const std::vector<std::string>& contexts) {
        std::string context_combined;
        for (const auto& ctx : contexts) {
            context_combined += ctx + " ";
        }

        std::string input = "Context: " + context_combined + " User: " + user_input + " Bot:";
        torch::Tensor input_tensor = torch::ones({ 1, 1 }).to(torch::kCUDA);
        auto output = model.forward({ input_tensor }).toStringRef();
        return output;
    }
};

#endif // CHATBOT_H
