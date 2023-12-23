#pragma once
#include "block.hpp"
#include <string>
#include <vector>
#include <atomic>

using Blockchain = std::vector<Block>;

namespace blockchain
{
    Blockchain init();
    bool validate(const Blockchain& blockchain);
    Blockchain new_block_pow(Blockchain& blockchain, std::atomic<bool>& reset);

    std::string to_string(const Blockchain& blockchain);
    std::string to_readable_string(const Blockchain& blockchain);
    Blockchain from_string(const std::string& blockchain_);

    std::size_t difficulty(const Blockchain& blockchain);
} // namespace blockchain
