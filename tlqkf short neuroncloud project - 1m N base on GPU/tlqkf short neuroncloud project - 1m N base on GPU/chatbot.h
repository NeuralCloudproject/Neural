#ifndef CHATBOT_H
#define CHATBOT_H

#include <string>
#include <vector>
#include <torch/script.h>

class Chatbot {
private:
    torch::jit::script::Module model;

public:
    Chatbot(const std::string& model_path) {
        model = torch::jit::load(model_path);
    }

    std::string generate_response(const std::string& user_input, const std::vector<std::string>& contexts) {
        std::string context_combined;
        for (const auto& ctx : contexts) {
            context_combined += ctx + " ";
        }

        std::string input = "Context: " + context_combined + " User: " + user_input + " Bot:";
        torch::Tensor input_tensor = torch::ones({ 1, 1 }); // Dummy tensor
        auto output = model.forward({ input_tensor }).toStringRef();
        return output;
    }
};

#endif
