#include "neurons.h"
#include "search.h"
#include "embedding.h"
#include "chatbot.h"
#include <iostream>

int main() {
    // Initialize cache and search engine
    SearchCache cache("search_cache.json");
    InternetSearch search("YOUR_API_KEY", "YOUR_CSE_ID", cache);

    // Perform search
    std::vector<std::string> results = search.search("What is quantum computing?");
    std::cout << "Search Results:\n";
    for (const auto& res : results) {
        std::cout << res << "\n";
    }

    // Generate response with chatbot
    Chatbot chatbot("model.pt");
    std::string response = chatbot.generate_response("What is quantum computing?", results);
    std::cout << "Chatbot Response:\n" << response << "\n";

    return 0;
}
