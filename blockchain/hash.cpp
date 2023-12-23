#include "hash.hpp"
#include <iomanip>
#include <iostream>
#include <openssl/sha.h>
#include <sstream>

std::string hash::string(const std::string& string)
{
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256_CTX sha256;
    SHA256_Init(&sha256);
    SHA256_Update(&sha256, string.c_str(), string.size());
    SHA256_Final(hash, &sha256);
    std::stringstream ss;
    for (int i = 0; i < SHA256_DIGEST_LENGTH; i++)
    {
        ss << std::hex << std::setw(2) << std::setfill('0')
           << static_cast<int>(hash[i]);
    }
#pragma GCC diagnostic pop
    return ss.str();
}

std::string hash::block(const Block& block)
{
    return hash::string(
        std::to_string(block.id) + std::to_string(std::chrono::system_clock::to_time_t(block.timestamp)) + block.data + block.previous_hash + std::to_string(block.difficulty) + std::to_string(block.nonce));
}
