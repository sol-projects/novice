#pragma once
#include <cargs.h>

struct OptionFlags
{
    int threads = 1;
};

// clang-format off
constexpr cag_option options_info[]
{
    {
        .identifier = 't',
        .access_letters = "t",
        .access_name = "threads",
        .value_name = "threads",
        .description = "Number of threads for mining on a single system."
    },

    {
        .identifier = 'h',
        .access_letters = "h",
        .access_name = "help",
        .value_name = nullptr,
        .description = "Information."
    }
};
