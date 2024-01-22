#pragma once
#include "block.hpp"
#include "options.hpp"
#include <atomic>
#include <string>
#include <vector>

using Blockchain = std::vector<Block>;

namespace blockchain
{
    Blockchain init();
    Blockchain empty();
    bool validate(const Blockchain& blockchain);
    Blockchain new_block_pow(const Blockchain& blockchain, const std::string& data, std::atomic<bool>& stop, const OptionFlags& options, int mpi_rank, int mpi_world_size);

    std::string to_string(const Blockchain& blockchain);
    std::string to_readable_string(const Blockchain& blockchain);
    Blockchain from_string(const std::string& blockchain_);

    std::size_t difficulty(const Blockchain& blockchain);
} // namespace blockchain
