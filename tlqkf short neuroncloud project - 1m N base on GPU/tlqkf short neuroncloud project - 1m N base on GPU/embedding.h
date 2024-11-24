#ifndef EMBEDDING_H
#define EMBEDDING_H

#include <string>
#include <vector>
#include <cmath>

class Embedding {
public:
    static double cosine_similarity(const std::vector<double>& a, const std::vector<double>& b) {
        double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }
        return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
    }
};

#endif
