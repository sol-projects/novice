#pragma once
#include "options.hpp"
#include <chrono>
#include <string>
#include <atomic>
#include "options.hpp"

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

    static Block genesis();
    static Block new_from_previous(const Block& previous_block);
    static Block new_from_previous_pow(const Block& previous_block, std::atomic<bool>& stop, int difficulty, const OptionFlags& options, int mpi_rank, int mpi_world_size);
    static Block deserialize(const std::vector<uint8_t>& buffer);

    std::vector<uint8_t> serialize() const;
    bool validation(const Block& previous_block) const;

    std::string to_string() const;
    std::string to_readable_string() const;
    static Block from_string(const std::string& string);
};
