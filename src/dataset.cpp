// dataset.cpp
#include "dataset.h"
#include <fstream>
#include <sstream>
#include <random>
#include <cmath>
#include <iostream>
#include <algorithm>

// --------------------------------------------------
// Constructor
// --------------------------------------------------
Dataset::Dataset(
    const std::string& text_path,
    int window_size,
    int num_neg,
    int batch_size
)
    : window_size_(window_size),
      num_neg_(num_neg),
      batch_size_(batch_size),
      cursor_(0)
{
    load_text(text_path);
    build_neg_table();
}

// --------------------------------------------------
int Dataset::vocab_size() const {
    return static_cast<int>(id2word_.size());
}

// --------------------------------------------------
int Dataset::num_batches() const {
    // Each batch consumes batch_size center words
    return static_cast<int>(
        (corpus_.size() - 2 * window_size_) / batch_size_
    );
}

// --------------------------------------------------
void Dataset::reset() {
    cursor_ = window_size_;
}

// --------------------------------------------------
// Load and tokenize text
// --------------------------------------------------
void Dataset::load_text(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Failed to open text file");
    }

    std::vector<std::string> tokens;
    tokens.reserve(1'000'000);

    std::string word;
    while (file >> word) {
        tokens.push_back(word);
    }

    build_vocab(tokens);
    build_corpus(tokens);

    std::cout << "Loaded corpus with "
              << corpus_.size()
              << " tokens\n";
}

// --------------------------------------------------
// Build vocabulary (no pruning yet)
// --------------------------------------------------
void Dataset::build_vocab(const std::vector<std::string>& tokens) {
    for (const auto& w : tokens) {
        if (word2id_.count(w) == 0) {
            int id = static_cast<int>(id2word_.size());
            word2id_[w] = id;
            id2word_.push_back(w);
        }
    }

    std::cout << "Vocabulary size: "
              << id2word_.size()
              << "\n";
}

// --------------------------------------------------
// Convert tokens to IDs
// --------------------------------------------------
void Dataset::build_corpus(const std::vector<std::string>& tokens) {
    corpus_.reserve(tokens.size());
    for (const auto& w : tokens) {
        corpus_.push_back(word2id_[w]);
    }
}

// --------------------------------------------------
// Build negative sampling table (unigram^0.75)
// --------------------------------------------------
void Dataset::build_neg_table() {
    const double power = 0.75;
    const size_t table_size = 1'000'000;

    std::vector<int> freq(vocab_size(), 0);
    for (int w : corpus_) {
        freq[w]++;
    }

    double norm = 0.0;
    for (int f : freq) {
        norm += std::pow(f, power);
    }

    neg_table_.reserve(table_size);
    for (size_t i = 0; i < freq.size(); ++i) {
        double prob = std::pow(freq[i], power) / norm;
        int count = static_cast<int>(prob * table_size);
        for (int j = 0; j < count; ++j) {
            neg_table_.push_back(static_cast<int>(i));
        }
    }

    std::mt19937 rng(123);
    std::shuffle(neg_table_.begin(), neg_table_.end(), rng);

    std::cout << "Negative sampling table built\n";
}

// --------------------------------------------------
// Generate next batch
// --------------------------------------------------
Batch Dataset::next_batch() {
    Batch batch;
    batch.center.resize(batch_size_);
    batch.pos.resize(batch_size_);
    batch.neg.resize(batch_size_ * num_neg_);

    static thread_local std::mt19937 rng(42);
    std::uniform_int_distribution<int> neg_dist(
        0, static_cast<int>(neg_table_.size() - 1)
    );
    std::uniform_int_distribution<int> win_dist(
        -window_size_, window_size_
    );

    for (int i = 0; i < batch_size_; ++i) {
        int c = corpus_[cursor_];

        int offset = 0;
        while (offset == 0) {
            offset = win_dist(rng);
        }

        int p = corpus_[cursor_ + offset];

        batch.center[i] = c;
        batch.pos[i] = p;

        for (int k = 0; k < num_neg_; ++k) {
            batch.neg[i * num_neg_ + k] =
                neg_table_[neg_dist(rng)];
        }

        cursor_++;
    }

    return batch;
}