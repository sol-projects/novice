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
#include <sys/utsname.h>
#include <thread>
#include <atomic>
#include "options.hpp"

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
                std::cout << "Running benchmarks..." << std::endl;
                auto b = blockchain::init();
                std::atomic<bool> stop = false;
                Block previous_block = b.at(0);
                auto start_time = std::chrono::high_resolution_clock::now();
                for(int i = 0; i < 100; i++) {
                    auto block = Block::new_from_previous_pow(previous_block, stop, 5, options);
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

    auto app = new QApplication(argc, argv);
    Gui gui(app);

    std::thread t2([&]() {
        asio::io_context ioContext;
        std::unique_ptr<Server> server;
        if (!available(ip, port))
        {
            std::cout << "new node" << std::endl;
            server = std::make_unique<Server>(ioContext, port);
            ioContext.run();
        }
    });

    using namespace std::chrono_literals;
    std::thread t([&]() {
        Client c(ip, port, options);
        c.process([&]([[maybe_unused]] auto client) {

        });
    });

    app->exec();
    t.join();
    t2.join();
}