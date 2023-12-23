#pragma once
#include <chrono>
#include <string>
#include <atomic>

struct Block
{
    static constexpr int num_fields = 7;
    std::size_t id;
    std::chrono::time_point<std::chrono::system_clock> timestamp;
    std::string data;
    std::string previous_hash;
    std::string hash;
    std::size_t difficulty;
    std::size_t nonce;
};

namespace block
{
    Block genesis();
    Block new_from_previous(const Block& previous_block);
    Block new_from_previous_pow(const Block& previous_block, std::atomic<bool>& stop, int difficulty);
    bool validation(const Block& previous_block, const Block& block);

    std::string to_string(const Block& block);
    std::string to_readable_string(const Block& block);
    Block from_string(const std::string& string);
} // namespace block
