#include "service.hpp"
#include <httplib.h>
#ifndef CPPHTTPLIB_OPENSSL_SUPPORT
    #define CPPHTTPLIB_OPENSSL_SUPPORT
#endif

namespace
{
    std::thread server_thread;
    httplib::Server server;
}

void service::listen()
{
    server_thread = std::thread([&]() {
        try {
            server.listen("0.0.0.0", service::port);
        } catch(const std::exception& e) {
            std::cerr << "Error starting server: " << e.what() << std::endl;
        }
    });
}

void stop() {
    server.stop();
    server_thread.join();
}

httplib::Server* service::get_server() {
    return &server;
}
