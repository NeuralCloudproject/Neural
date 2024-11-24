#ifndef SEARCH_CACHE_H
#define SEARCH_CACHE_H

#include <unordered_map>
#include <string>
#include <vector>
#include <fstream>
#include <nlohmann/json.hpp>
#include <mutex>

using json = nlohmann::json;

class SearchCache {
private:
    std::unordered_map<std::string, std::vector<std::string>> cache;
    std::string cache_file;
    std::mutex mtx;

    void load_cache() {
        std::ifstream file(cache_file);
        if (file) {
            json j;
            file >> j;
            std::lock_guard<std::mutex> lock(mtx);
            for (auto& [key, value] : j.items()) {
                cache[key] = value.get<std::vector<std::string>>();
            }
        }
    }

    void save_cache() {
        std::lock_guard<std::mutex> lock(mtx);
        json j(cache);
        std::ofstream file(cache_file);
        file << j.dump(4);
    }

public:
    SearchCache(const std::string& cache_file) : cache_file(cache_file) {
        load_cache();
    }

    bool is_cached(const std::string& query) {
        std::lock_guard<std::mutex> lock(mtx);
        return cache.find(query) != cache.end();
    }

    std::vector<std::string> get(const std::string& query) {
        std::lock_guard<std::mutex> lock(mtx);
        return cache[query];
    }

    void set(const std::string& query, const std::vector<std::string>& results) {
        {
            std::lock_guard<std::mutex> lock(mtx);
            cache[query] = results;
        }
        save_cache();
    }
};

#endif // SEARCH_CACHE_H
