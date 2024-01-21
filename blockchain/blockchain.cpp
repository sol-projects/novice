#include "blockchain.hpp"
#include "options.hpp"
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <string>

Blockchain blockchain::init()
{
    return {Block::genesis()};
}

Blockchain blockchain::empty() {
    return {};
}

bool blockchain::validate(const Blockchain& blockchain)
{
    using namespace std::chrono_literals;
    return std::adjacent_find(std::begin(blockchain), std::end(blockchain), [](const auto& current, const auto& previous) {
        return current.validation(previous) && std::chrono::duration_cast<std::chrono::seconds>(current.timestamp - previous.timestamp) <= 60s;
    }) == std::end(blockchain);
}

std::string blockchain::to_string(const Blockchain& blockchain)
{
    return std::accumulate(
        std::next(std::cbegin(blockchain)), std::cend(blockchain),
        blockchain.at(0).to_string(), [](auto blockchain, auto block) {
            return std::move(blockchain) + block.to_string();
        });
}

std::string blockchain::to_readable_string(const Blockchain& blockchain)
{
    return std::accumulate(
        std::next(std::cbegin(blockchain)), std::cend(blockchain),
        blockchain.at(0).to_readable_string(),
        [](auto blockchain, auto block) {
            return std::move(blockchain) + block.to_readable_string();
        });
}

std::size_t blockchain::difficulty(const Blockchain& blockchain)
{
    return std::accumulate(std::cbegin(blockchain), std::cend(blockchain), 0,
        [](auto difficulty, auto block) {
            return difficulty + std::pow(2, block.difficulty);
        });
}

Blockchain blockchain::new_block_pow(const Blockchain& blockchain, const std::string& data, std::atomic<bool>& stop, const OptionFlags& options, int mpi_rank, int mpi_world_size)
{
    using namespace std::chrono_literals;
    const auto difficulty_change_interval = 10;
    static int difficulty = blockchain.back().difficulty;

    int blocks_below_expected_time = 0;
    int blocks_above_expected_time = 0;

    if(blockchain.size() >= 10)
    {
        for(auto i = blockchain.size() - 9; i < blockchain.size(); i++)
        {
            const auto time_expected = 10s;
            auto time_taken = std::chrono::duration_cast<std::chrono::microseconds>(
            blockchain.at(i).timestamp - blockchain.at(i - 1).timestamp);

            if (time_taken < time_expected / 2)
            {
                blocks_below_expected_time++;
            }
            else if (time_taken > time_expected * 2)
            {
                blocks_above_expected_time++;
            }
        }

        if (std::size(blockchain) % difficulty_change_interval == 0)
        {
            if (blocks_above_expected_time - 2 > blocks_below_expected_time)
            {
                difficulty--;
            }
            else if (blocks_above_expected_time + 2 < blocks_below_expected_time)
            {
                difficulty++;
            }
        }
    }

    Block block = Block::new_from_previous_pow(blockchain.back(), data, stop, difficulty, options, mpi_rank, mpi_world_size);
    auto out_blockchain = blockchain;
    out_blockchain.push_back(block);
    return out_blockchain;
}

Blockchain blockchain::from_string(const std::string& blockchain_)
{
    Blockchain blockchain;

    std::string block;
    int n = Block::num_fields;
    for (char c : blockchain_)
    {
        if (c == '\n')
        {
            block += c;
            n--;
            if (n == 0)
            {
                blockchain.push_back(Block::from_string(block));
                block.clear();
                n = Block::num_fields;
            }
        }
        else
        {
            block += c;
        }
    }

    return blockchain;
}
