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

int main(int argc, char* argv[])
{

    if (argc > 1)
    {
        std::atomic<bool> never_stop = false;
        if (!std::strcmp(argv[1], "run_tests"))
        {
            Blockchain blockchain = blockchain::init();
            blockchain = blockchain::new_block_pow(blockchain, never_stop);

            for (int i = 0; i < 3; i++)
            {
                blockchain = blockchain::new_block_pow(blockchain, never_stop);
            }

            std::cout << (blockchain::validate(blockchain) ? "true" : "false")
                      << std::endl;

            blockchain.at(2).id = 10;
            std::cout << (blockchain::validate(blockchain) ? "true" : "false")
                      << std::endl;

            auto to = blockchain::to_string(blockchain);
            auto from = blockchain::from_string(to);

            if (blockchain::to_string(from) == to)
            {
                std::cout << "ok" << std::endl;
            }
            else
            {
                std::cout << "no" << std::endl;
            }
        }
    }
    else
    {
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
            Client c(ip, port);
            c.process([&]([[maybe_unused]] auto client) {

            });
        });

        app->exec();
        t.join();
        t2.join();
    }
}
