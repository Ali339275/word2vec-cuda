// dataset.h
#ifndef DATASET_H
#define DATASET_H

#include <vector>
#include <string>
#include <unordered_map>

// --------------------------------------------------
// A batch of Skip-gram training data
// --------------------------------------------------
struct Batch {
    std::vector<int> center;  // size = batch_size
    std::vector<int> pos;     // size = batch_size
    std::vector<int> neg;     // size = batch_size * num_neg
};

// --------------------------------------------------
// Dataset class
// --------------------------------------------------
class Dataset {
public:
    Dataset(
        const std::string& text_path,
        int window_size,
        int num_neg,
        int batch_size
    );

    int vocab_size() const;
    int num_batches() const;

    void reset();
    Batch next_batch();

private:
    // Text as word IDs
    std::vector<int> corpus_;

    // Vocabulary
    std::unordered_map<std::string, int> word2id_;
    std::vector<std::string> id2word_;

    // Negative sampling table
    std::vector<int> neg_table_;

    // Parameters
    int window_size_;
    int num_neg_;
    int batch_size_;

    // Cursor over corpus
    size_t cursor_;

    // Internal helpers
    void load_text(const std::string& path);
    void build_vocab(const std::vector<std::string>& tokens);
    void build_corpus(const std::vector<std::string>& tokens);
    void build_neg_table();
};

#endif // DATASET_H