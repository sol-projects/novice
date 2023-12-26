#include "block.hpp"
#include "hash.hpp"
#include "options.hpp"
#include <algorithm>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <thread>

Block Block::genesis()
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

Block Block::new_from_previous(const Block& previous_block)
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

bool Block::validation(const Block& previous_block) const
{
    return previous_hash == previous_block.hash && hash::block(*this) == hash && id == (previous_block.id + 1);
}

std::string Block::to_string() const
{
    return std::to_string(id) + "\n" + std::to_string(std::chrono::system_clock::to_time_t(timestamp)) + "\n" + data + "\n" + previous_hash + "\n" + hash + "\n" + std::to_string(difficulty) + "\n" + std::to_string(nonce) + "\n";
}

std::string Block::to_readable_string() const
{
    auto t = std::chrono::system_clock::to_time_t(timestamp);
    auto time = *std::localtime(&t);

    std::ostringstream oss;
    oss << std::put_time(&time, "%d-%m-%Y %H:%M:%S");
    auto time_string = oss.str();

    return "id: " + std::to_string(id) + "\n" + "time: " + time_string + "\n" + "data: " + data + "\n" + "previous hash: " + previous_hash + "\n" + "hash: " + hash + "\n" + "difficulty: " + std::to_string(difficulty) + "\n" + "nonce: " + std::to_string(nonce) + "\n";
}

Block Block::from_string(const std::string& string)
{
    Block block {};
    std::istringstream iss(string);

    std::string id;
    std::getline(iss, id);
    block.id = std::stoi(id);

    std::string time;
    std::getline(iss, time);
    block.timestamp = std::chrono::system_clock::from_time_t(std::stoi(time));

    std::getline(iss, block.data);

    std::getline(iss, block.previous_hash);

    std::getline(iss, block.hash);

    std::string difficulty;
    std::getline(iss, difficulty);
    block.difficulty = std::stoi(difficulty);

    std::string nonce;
    std::getline(iss, nonce);
    block.nonce = std::stoi(nonce);

    return block;
}

Block Block::new_from_previous_pow(const Block& previous_block, std::atomic<bool>& stop, int difficulty, const OptionFlags& options)
{
    auto new_block = new_from_previous(previous_block);
    std::atomic<bool> nonce_found(false);
    std::size_t nonce_increment = std::numeric_limits<std::size_t>::max() / options.threads;
    std::vector<std::thread> threads;

    for (int i = 0; i < options.threads; ++i)
    {
        threads.emplace_back([&, i, nonce_increment]()
        {
            auto block = new_block;
            std::size_t nonce_start = i * nonce_increment;
            std::size_t nonce_end = (i + 1) * nonce_increment;
            for (std::size_t nonce = nonce_start; nonce < nonce_end; ++nonce)
            {
                if (stop || nonce_found)
                {
                    new_block = block;
                    return;
                }

                block.nonce = nonce;
                block.difficulty = difficulty;
                block.hash = hash::block(block);
                if (std::all_of(std::begin(block.hash),
                        std::begin(block.hash) + block.difficulty,
                        [](auto c) { return c == '0'; }))
                {
                    nonce_found = true;
                    new_block = block;
                    return;
                }
            }
        });
    }

    for (auto& thread : threads)
    {
        thread.join();
    }

    return new_block;
}

