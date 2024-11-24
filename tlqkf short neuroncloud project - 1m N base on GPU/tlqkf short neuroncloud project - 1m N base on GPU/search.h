#ifndef SEARCH_H
#define SEARCH_H

#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <nlohmann/json.hpp>
#include <cpr/cpr.h>

class SearchCache {
private:
    std::unordered_map<std::string, std::vector<std::string>> cache;
    std::string cache_file;

    void load_cache() {
        std::ifstream file(cache_file);
        if (file) {
            nlohmann::json j;
            file >> j;
            for (auto& [key, value] : j.items()) {
                cache[key] = value.get<std::vector<std::string>>();
            }
        }
    }

    void save_cache() {
        nlohmann::json j(cache);
        std::ofstream file(cache_file);
        file << j.dump(4);
    }

public:
    SearchCache(const std::string& cache_file) : cache_file(cache_file) {
        load_cache();
    }

    bool is_cached(const std::string& query) {
        return cache.find(query) != cache.end();
    }

    std::vector<std::string> get(const std::string& query) {
        return cache[query];
    }

    void set(const std::string& query, const std::vector<std::string>& results) {
        cache[query] = results;
        save_cache();
    }
};

class InternetSearch {
private:
    std::string api_key;
    std::string cse_id;
    SearchCache& cache;

public:
    InternetSearch(const std::string& api_key, const std::string& cse_id, SearchCache& cache)
        : api_key(api_key), cse_id(cse_id), cache(cache) {
    }

    std::vector<std::string> search(const std::string& query, int num_results = 5) {
        if (cache.is_cached(query)) {
            return cache.get(query);
        }

        std::string url = "https://www.googleapis.com/customsearch/v1?q=" + query +
            "&key=" + api_key + "&cx=" + cse_id + "&num=" + std::to_string(num_results);

        auto response = cpr::Get(cpr::Url{ url });
        nlohmann::json j = nlohmann::json::parse(response.text);

        std::vector<std::string> results;
        for (auto& item : j["items"]) {
            results.push_back(item["title"].get<std::string>() + " - " + item["link"].get<std::string>());
        }

        cache.set(query, results);
        return results;
    }
};

#endif
