#include "neuron.h"
#include "search_cache.h"
#include "chatbot.h"
#include "gpu_utils.h"
#include <iostream>

int main() {
    // Neuron ???
    Neuron n1(1, 1.0, 0.0);
    Neuron n2(2, 1.0, 0.0);
    n1.connections.push_back(1);
    std::vector<Neuron> neurons = { n1, n2 };
    neurons[0].current_potential = 1.5;
    neurons[0].fire(neurons);

    // SearchCache ???
    SearchCache cache("search_cache.json");
    cache.set("test_query", { "Result 1", "Result 2" });

    // Chatbot ???
    Chatbot chatbot("model.pt");
    std::string response = chatbot.generate_response("Hello!", { "How can I assist you?" });
    std::cout << "Chatbot response: " << response << "\n";

    // GPU ???
    test_gpu_computation();

    return 0;
}
