#include "Peer.hpp"
#include "block.hpp"
#include "blockchain.hpp"
#include <QApplication>
#include <asio.hpp>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <openssl/aes.h>
#include <string>
#include <thread>
#include <atomic>
#include "options.hpp"
#include <mpi.h>
#include "block.hpp"

OptionFlags parse(int argc, char* argv[])
{
    OptionFlags options {};

    cag_option_context context;
    cag_option_prepare(&context, options_info, CAG_ARRAY_SIZE(options_info), argc, argv);
    while (cag_option_fetch(&context))
    {
        auto identifier = cag_option_get(&context);
        switch (identifier)
        {
            case 't':
                options.threads = std::stoi(cag_option_get_value(&context));
                break;
            case 'b': {
                std::cout << "Running tests..." << std::endl;
                {
                    auto a = Block::genesis();
                    auto b = Block::deserialize(a.serialize());

                    if(a.to_string() != b.to_string()) {
                        std::cerr << "Tests failed" << std::endl;
                        std::cerr << a.to_readable_string() << '\n';
                        std::cerr << b.to_readable_string() << '\n';
                    } else {
                        std::cout << "Completed tests." << std::endl;
                    }
                }

                std::cout << "Running benchmarks..." << std::endl;
                auto b = blockchain::init();
                std::atomic<bool> stop = false;
                Block previous_block = b.at(0);
                auto start_time = std::chrono::high_resolution_clock::now();
                for(int i = 0; i < 100; i++) {
                    auto block = Block::new_from_previous_pow(previous_block, stop, 5, options, 0, 1);
                    b.push_back(block);
                    previous_block = block;
                }
                auto end_time = std::chrono::high_resolution_clock::now();
                if(!blockchain::validate(b)) {
                    std::cerr << "Benchmark failed: blockchain is invalid.";
                }
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
                std::cout << "Benchmark completed: elapsed time: " << duration.count() << " ms" << std::endl;
                std::exit(0);
                break;
                      }
            case 'h':
                cag_option_print(options_info, CAG_ARRAY_SIZE(options_info), stdout);
                std::exit(1);
        }
    }

    return options;
}

int main(int argc, char* argv[])
{
    auto options = parse(argc, argv);
    constexpr int port = 2310;
    const std::string ip = asio::ip::address_v4::any().to_string();

    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided < MPI_THREAD_MULTIPLE)
    {
        std::cerr << "ERROR: The MPI library does not have full thread support\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int world_rank;
    int world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if(world_rank == 0) {
        std::cout << "Starting main MPI process with GUI." << std::endl;
        auto app = new QApplication(argc, argv);
        Gui gui(app);

        std::thread t2([&]() {
            asio::io_context ioContext;
            std::unique_ptr<Server> server;
            if (!available(ip, port))
            {
                std::cout << "new node" << std::endl;
                server = std::make_unique<Server>(ioContext, port, world_rank, world_size);
                ioContext.run();
            }
        });

        using namespace std::chrono_literals;
        std::thread t([&]() {
            Client c(ip, port, options, world_rank, world_size);
            c.process([&]([[maybe_unused]] auto client) {

            });
        });

        app->exec();
        t.join();
        t2.join();
    } else {
        std::atomic<bool> stop = false;
        for(;;)
        {
            Block::new_from_previous_pow(Block::genesis(), stop, 0, options, world_rank, world_size);
            std::cout << "here" << world_rank << std::endl;
        }
    }

    MPI_Finalize();
}
