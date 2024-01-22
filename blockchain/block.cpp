#include "block.hpp"
#include "hash.hpp"
#include "options.hpp"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mpi.h>
#include <string>
#include <thread>

namespace
{
    void check(int error_code, const std::string& msg)
    {
        if (error_code != MPI_SUCCESS)
        {
            char error_string[MPI_MAX_ERROR_STRING];
            int length_of_error_string;
            MPI_Error_string(error_code, error_string, &length_of_error_string);
            std::cerr << "MPI ERROR (" << msg << "): " << error_string << std::endl;
            MPI_Abort(MPI_COMM_WORLD, error_code);
        }
    }
}

Block Block::genesis()
{
    Block block { .id = 0,
        .timestamp = std::chrono::system_clock::now(),
        .data = "",
        .previous_hash = "",
        .hash = "",
        .difficulty = 6,
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
    return previous_hash == previous_block.hash && hash == hash::block(*this) && id == (previous_block.id + 1);
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

    std::string line;
    int field_counter = 0;

    while (std::getline(iss, line))
    {
        if (line.find('{') != std::string::npos)
        {
            std::string json_content = line + '\n';
            while (line.find('}') == std::string::npos && std::getline(iss, line))
            {
                json_content += line + '\n';
            }
            json_content.push_back('}');
            block.data = json_content;
        }
        else
        {
            switch (field_counter)
            {
                case 0:
                    block.id = std::stoi(line);
                    break;
                case 1:
                    block.timestamp = std::chrono::system_clock::from_time_t(std::stoull(line));
                    break;
                case 3:
                    block.previous_hash = line;
                    break;
                case 4:
                    block.hash = line;
                    break;
                case 5:
                    block.difficulty = std::stoi(line);
                    break;
                case 6:
                    block.nonce = std::stoull(line);
                    break;
            }
            field_counter++;
        }
    }
    return block;
}

Block Block::new_from_previous_pow(const Block& previous_block, const std::string& data, std::atomic<bool>& stop, int difficulty, const OptionFlags& options, int mpi_rank, int mpi_world_size)
{
    auto start_time = std::chrono::steady_clock::now();
    constexpr int mpi_block_found = 1;

    bool main_mpi_process = (mpi_rank == 0);

    auto new_block = new_from_previous(previous_block);
    new_block.data = data;

    std::vector<uint8_t> serialized;
    if (main_mpi_process)
    {
        serialized = new_block.serialize();
    }

    int serialized_size;
    if (main_mpi_process)
    {
        serialized_size = serialized.size();
    }
    int err = MPI_Bcast(&serialized_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    check(err, "serialized size");

    if (!main_mpi_process)
    {
        serialized.resize(serialized_size);
    }
    err = MPI_Bcast(serialized.data(), serialized_size, MPI_CHAR, 0, MPI_COMM_WORLD);
    check(err, "serialized data");

    err = MPI_Bcast(&difficulty, 1, MPI_INT, 0, MPI_COMM_WORLD);
    check(err, "difficulty");

    serialized.clear();
    serialized_size = 0;

    std::mutex totalNumHashesMutex;
    new_block = Block::deserialize(serialized);

    std::cout << "MPI process " << mpi_rank << " searching block with prev hash (last 5): " << std::string(std::end(new_block.previous_hash) - 5, std::end(new_block.previous_hash)) << std::endl;

    std::atomic<bool> nonce_found(false);
    std::size_t total_nonce_space = std::numeric_limits<std::size_t>::max();
    std::size_t nonce_range = total_nonce_space / mpi_world_size;
    std::size_t mpi_nonce_start = mpi_rank * nonce_range;
    std::size_t mpi_nonce_end = (mpi_rank + 1) * nonce_range;
    std::size_t nonce_increment = nonce_range / options.threads;

    std::vector<std::thread> threads;
    MPI_Request recv_request = MPI_REQUEST_NULL;
    MPI_Status recv_status;
    bool block_received = false;

    if (main_mpi_process)
    {
        serialized.resize(serialized_size + 3000);
        err = MPI_Irecv(serialized.data(), serialized_size + 3000, MPI_CHAR, MPI_ANY_SOURCE, mpi_block_found, MPI_COMM_WORLD, &recv_request);
        check(err, "irecv");
    }

    std::mutex nonceFoundMutex;
    std::atomic<std::size_t> total_num_hashes = 0;
    int main_thread = 0;
    for (int i = 0; i < options.threads; ++i)
    {
        threads.emplace_back([&, i, nonce_increment]() {
            std::size_t num_hashes = 0;
            std::lock_guard<std::mutex> lock(nonceFoundMutex);
            auto block = new_block;
            std::size_t nonce_start = mpi_nonce_start + i * nonce_increment;
            std::size_t nonce_end = mpi_nonce_start + (i + 1) * nonce_increment;
            for (std::size_t nonce = nonce_start; nonce < nonce_end; ++nonce)
            {
                bool n = nonce_found;
                if (main_thread)
                {
                    err = MPI_Bcast(&n, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
                    check(err, "stuck at nonce_found check");
                }
                nonce_found.store(n);

                if (stop)
                {
                    if (!main_mpi_process)
                    {
                        auto serialized_block = block.serialize();
                        err = MPI_Send(serialized_block.data(), serialized_block.size(), MPI_CHAR, 0, mpi_block_found, MPI_COMM_WORLD);
                        check(err, "sending discarded data to main process");

                        std::cout << "MPI process " << mpi_rank << " is discarding work due to the block being found by another program." << std::endl;
                        new_block = block;
                    }
                }

                if (nonce_found.load())
                {
                    std::lock_guard<std::mutex> lock(totalNumHashesMutex);
                    total_num_hashes += num_hashes;
                    return;
                }
                // static const auto timePoint = std::chrono::system_clock::from_time_t(0);
                // static const auto timestamp = std::chrono::system_clock::to_time_t(timePoint);
                block.timestamp = std::chrono::system_clock::now();
                block.nonce = nonce;
                block.difficulty = difficulty;
                block.hash = hash::block(block);
                num_hashes++;
                if (std::all_of(std::begin(block.hash),
                        std::begin(block.hash) + block.difficulty,
                        [](auto c) { return c == '0'; }))
                {
                    nonce_found.store(true);
                    if (!main_mpi_process)
                    {
                        auto serialized_block = block.serialize();
                        err = MPI_Send(serialized_block.data(), serialized_block.size(), MPI_CHAR, 0, mpi_block_found, MPI_COMM_WORLD);
                        check(err, "sending data to main process");

                        std::cout << "MPI process " << mpi_rank << " found block with hash: " << block.hash << "for prev block hash (last 5): " << std::string(std::end(block.previous_hash) - 5, std::end(block.previous_hash)) << std::endl;
                        new_block = block;
                    }
                    else
                    {
                        new_block = block;
                    }
                }

                if (main_mpi_process)
                {
                    int flag;
                    err = MPI_Test(&recv_request, &flag, &recv_status);
                    check(err, "testing if block found");
                    if (flag)
                    {
                        int count;
                        err = MPI_Get_count(&recv_status, MPI_CHAR, &count);
                        check(err, "receiving new block count");

                        if (count > 0)
                        {
                            std::cout << "Main MPI process receiving block" << std::endl;
                            block_received = true;
                            new_block = Block::deserialize(serialized);
                            std::cout << "Main MPI process received block with hash: " << new_block.hash << "for prev block hash (last 5): " << std::string(std::end(new_block.previous_hash) - 5, std::end(new_block.previous_hash)) << std::endl;
                            nonce_found.store(true);
                        }
                        else
                        {
                            std::cerr << "Received unexpected data size: " << count << std::endl;
                        }
                    }
                }
            }
        });
    }

    for (auto& thread : threads)
    {
        thread.join();
    }

    if (main_mpi_process)
    {
        std::cout << "Main MPI process is returning block with hash: " << new_block.hash << "for prev block hash (last 5): " << std::string(std::end(new_block.previous_hash) - 5, std::end(new_block.previous_hash)) << std::endl;
    }

    auto end_time = std::chrono::steady_clock::now();
    std::cout << "Process finished: " << mpi_rank << "\n";
    std::cout << "Total hashes: " << total_num_hashes << "\n";
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Time took: " << duration.count() << " milliseconds" << std::endl;
    std::cout << "Hashes/ms: " << (total_num_hashes + 1) / (duration.count() + 1) << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);

    return new_block;
}

std::vector<uint8_t> Block::serialize() const
{
    std::vector<uint8_t> buffer;

    auto id_bytes = reinterpret_cast<const uint8_t*>(&id);
    buffer.insert(buffer.end(), id_bytes, id_bytes + sizeof(id));

    uint64_t timestamp_count = timestamp.time_since_epoch().count();
    auto timestamp_bytes = reinterpret_cast<const uint8_t*>(&timestamp_count);
    buffer.insert(buffer.end(), timestamp_bytes, timestamp_bytes + sizeof(timestamp_count));

    std::size_t data_length = data.size();
    auto data_length_bytes = reinterpret_cast<const uint8_t*>(&data_length);
    buffer.insert(buffer.end(), data_length_bytes, data_length_bytes + sizeof(data_length));
    buffer.insert(buffer.end(), data.begin(), data.end());

    std::size_t hash_length = hash.size();
    auto hash_length_bytes = reinterpret_cast<const uint8_t*>(&hash_length);
    buffer.insert(buffer.end(), hash_length_bytes, hash_length_bytes + sizeof(hash_length));
    buffer.insert(buffer.end(), hash.begin(), hash.end());

    std::size_t prev_hash_length = previous_hash.size();
    auto prev_hash_length_bytes = reinterpret_cast<const uint8_t*>(&prev_hash_length);
    buffer.insert(buffer.end(), prev_hash_length_bytes, prev_hash_length_bytes + sizeof(prev_hash_length));
    buffer.insert(buffer.end(), previous_hash.begin(), previous_hash.end());

    auto difficulty_bytes = reinterpret_cast<const uint8_t*>(&difficulty);
    buffer.insert(buffer.end(), difficulty_bytes, difficulty_bytes + sizeof(difficulty));

    auto nonce_bytes = reinterpret_cast<const uint8_t*>(&nonce);
    buffer.insert(buffer.end(), nonce_bytes, nonce_bytes + sizeof(nonce));

    return buffer;
}

Block Block::deserialize(const std::vector<uint8_t>& buffer)
{
    Block block;
    int offset = 0;

    std::memcpy(&block.id, buffer.data() + offset, sizeof(block.id));
    offset += sizeof(block.id);

    uint64_t timestamp_count;
    std::memcpy(&timestamp_count, buffer.data() + sizeof(id), sizeof(timestamp_count));
    block.timestamp = std::chrono::time_point<std::chrono::system_clock>(
        std::chrono::duration_cast<std::chrono::system_clock::duration>(
            std::chrono::nanoseconds(timestamp_count)));

    offset += sizeof(timestamp_count);

    std::size_t data_length;
    std::memcpy(&data_length, buffer.data() + offset, sizeof(data_length));
    offset += sizeof(data_length);
    block.data = std::string(reinterpret_cast<const char*>(buffer.data() + offset), data_length);
    offset += data_length;

    std::size_t hash_length;
    std::memcpy(&hash_length, buffer.data() + offset, sizeof(hash_length));
    offset += sizeof(hash_length);
    block.hash = std::string(reinterpret_cast<const char*>(buffer.data() + offset), hash_length);
    offset += hash_length;

    std::size_t prev_hash_length;
    std::memcpy(&prev_hash_length, buffer.data() + offset, sizeof(prev_hash_length));
    offset += sizeof(prev_hash_length);
    block.previous_hash = std::string(reinterpret_cast<const char*>(buffer.data() + offset), prev_hash_length);
    offset += prev_hash_length;

    std::memcpy(&block.difficulty, buffer.data() + offset, sizeof(block.difficulty));
    offset += sizeof(block.difficulty);

    std::memcpy(&block.nonce, buffer.data() + offset, sizeof(block.nonce));
    offset += sizeof(block.nonce);

    return block;
}
