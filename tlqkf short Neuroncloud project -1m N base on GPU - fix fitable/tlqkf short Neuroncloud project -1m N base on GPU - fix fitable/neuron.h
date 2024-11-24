#ifndef NEURON_H
#define NEURON_H

#include <vector>
#include <string>
#include <fstream>
#include <nlohmann/json.hpp>
#include <mutex>
#include <thread>

using json = nlohmann::json;

class Neuron {
public:
    int id;
    double threshold, resting_potential, current_potential;
    std::vector<int> connections;
    std::mutex mtx;

    Neuron(int id, double threshold, double resting_potential)
        : id(id), threshold(threshold), resting_potential(resting_potential), current_potential(resting_potential) {
    }

    void fire(std::vector<Neuron>& all_neurons) {
        if (current_potential > threshold) {
            std::vector<std::thread> threads;
            for (int conn : connections) {
                threads.emplace_back([&, conn]() {
                    std::lock_guard<std::mutex> lock(all_neurons[conn].mtx);
                    all_neurons[conn].receive_signal(1.0);
                    });
            }
            for (auto& t : threads) t.join();
            current_potential = resting_potential;
        }
    }

    void receive_signal(double strength) {
        std::lock_guard<std::mutex> lock(mtx);
        current_potential += strength;
    }

    void save_state(const std::string& folder) {
        json state = {
            {"id", id},
            {"threshold", threshold},
            {"resting_potential", resting_potential},
            {"connections", connections}
        };
        std::ofstream file(folder + "/neuron_" + std::to_string(id) + ".json");
        file << state.dump(4);
    }

    static Neuron load_state(const std::string& file_path) {
        std::ifstream file(file_path);
        json state;
        file >> state;
        Neuron neuron(state["id"], state["threshold"], state["resting_potential"]);
        neuron.connections = state["connections"].get<std::vector<int>>();
        return neuron;
    }
};

#endif // NEURON_H
