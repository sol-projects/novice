#include "blockchain.hpp"
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <string>

Blockchain blockchain::init()
{
    Blockchain blockchain;
    blockchain.push_back(block::genesis());

    return blockchain;
}

bool blockchain::validate(const Blockchain& blockchain)
{
    using namespace std::chrono_literals;
    for (auto it = std::next(std::begin(blockchain)); it != std::end(blockchain);
         ++it)
    {
        if (!block::validation(*std::prev(it), *it))
        {
            return false;
        }

        if (std::chrono::duration_cast<std::chrono::seconds>(
                (*it).timestamp - (*std::prev(it)).timestamp)
            > 60s)
        {
            return false;
        }
    }

    return true;
}

std::string blockchain::to_string(const Blockchain& blockchain)
{
    return std::accumulate(
        std::next(std::cbegin(blockchain)), std::cend(blockchain),
        block::to_string(blockchain.at(0)), [](auto blockchain, auto block) {
            return std::move(blockchain) + block::to_string(block);
        });
}

std::string blockchain::to_readable_string(const Blockchain& blockchain)
{
    return std::accumulate(
        std::next(std::cbegin(blockchain)), std::cend(blockchain),
        block::to_readable_string(blockchain.at(0)),
        [](auto blockchain, auto block) {
            return std::move(blockchain) + block::to_readable_string(block);
        });
}

std::size_t blockchain::difficulty(const Blockchain& blockchain)
{
    return std::accumulate(std::cbegin(blockchain), std::cend(blockchain), 0,
        [](auto difficulty, auto block) {
            return difficulty + std::pow(2, block.difficulty);
        });
}

Blockchain blockchain::new_block_pow(Blockchain& blockchain, std::atomic<bool>& stop)
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

    Block block = block::new_from_previous_pow(blockchain.back(), stop, difficulty);
    blockchain.push_back(block);
    return std::move(blockchain);
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
                blockchain.push_back(block::from_string(block));
                block.clear();
                n = Block::num_fields;
            }
        }
        else
        {
            block += c;
        }

        std::cout << c << std::flush;
    }

    return blockchain;
}
