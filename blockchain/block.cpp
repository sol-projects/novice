#include "block.hpp"
#include "hash.hpp"
#include <algorithm>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

Block block::genesis()
{
    Block block { .id = 0,
        .timestamp = std::chrono::system_clock::now(),
        .data = "",
        .previous_hash = "",
        .hash = "",
        .difficulty = 5,
        .nonce = 0 };

    block.hash = hash::block(block);

    return block;
}

Block block::new_from_previous(const Block& previous_block)
{
    Block block { .id = previous_block.id + 1,
        .timestamp = std::chrono::system_clock::now(),
        .data = "",
        .previous_hash = previous_block.hash,
        .hash = "",
        .difficulty = previous_block.difficulty,
        .nonce = 0 };

    block.hash = hash::block(block);

    return block;
}

bool block::validation(const Block& previous_block, const Block& block)
{
    return block.previous_hash == previous_block.hash && hash::block(block) == block.hash && block.id == (previous_block.id + 1);
}

std::string block::to_string(const Block& block)
{
    return std::to_string(block.id) + "\n" + std::to_string(std::chrono::system_clock::to_time_t(block.timestamp)) + "\n" + block.data + "\n" + block.previous_hash + "\n" + block.hash + "\n" + std::to_string(block.difficulty) + "\n" + std::to_string(block.nonce) + "\n";
}

std::string block::to_readable_string(const Block& block)
{
    auto t = std::chrono::system_clock::to_time_t(block.timestamp);
    auto time = *std::localtime(&t);

    std::ostringstream oss;
    oss << std::put_time(&time, "%d-%m-%Y %H:%M:%S");
    auto time_string = oss.str();

    return "id: " + std::to_string(block.id) + "\n" + "time: " + time_string + "\n" + "data: " + block.data + "\n" + "previous hash: " + block.previous_hash + "\n" + "hash: " + block.hash + "\n" + "difficulty: " + std::to_string(block.difficulty) + "\n" + "nonce: " + std::to_string(block.nonce) + "\n";
}

Block block::from_string(const std::string& string)
{
    Block block {};

    std::string id;
    std::string time;
    std::string nonce;
    std::string difficulty;
    auto it = string.begin();
    for (; *it != '\n'; ++it)
    {
        id += *it;
    }

    for (it = ++it; *it != '\n'; ++it)
    {
        time += *it;
    }

    for (it = ++it; *it != '\n'; ++it)
    {
        block.data += *it;
    }

    for (it = ++it; *it != '\n'; ++it)
    {
        block.previous_hash += *it;
    }

    for (it = ++it; *it != '\n'; ++it)
    {
        block.hash += *it;
    }

    for (it = ++it; *it != '\n'; ++it)
    {
        difficulty += *it;
    }

    for (it = ++it; it != string.end(); ++it)
    {
        nonce += *it;
    }

    block.id = std::stoi(id);
    block.timestamp = std::chrono::system_clock::from_time_t(std::stoi(time));
    block.difficulty = std::stoi(difficulty);
    block.nonce = std::stoi(nonce);
    return block;
}

Block block::new_from_previous_pow(const Block& previous_block, std::atomic<bool>& stop, int difficulty)
{
    auto block = block::new_from_previous(previous_block);

    static std::size_t nonce = 0;
    while (true)
    {
        if(nonce == std::numeric_limits<decltype(nonce)>::max())
        {
            nonce = 0;
        }

        if(stop)
        {
            return block;
        }

        block.nonce = nonce;
        block.difficulty = difficulty;
        block.hash = hash::block(block);
        if (std::all_of(std::begin(block.hash),
                std::begin(block.hash) + block.difficulty,
                [](auto c) { return c == '0'; }))
        {
            return block;
        }

        nonce++;
    }

    return block;
}
