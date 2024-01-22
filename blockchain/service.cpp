#include "service.hpp"
#include <httplib.h>
#include <thread>
#ifndef CPPHTTPLIB_OPENSSL_SUPPORT
    #define CPPHTTPLIB_OPENSSL_SUPPORT
#endif
#include "Peer.hpp"

namespace
{
    std::thread server_thread;
    httplib::Server server;
    bool publisher = true;
    std::vector<std::string> client_addresses;
    std::unique_ptr<httplib::Client> client = std::make_unique<httplib::Client>("127.0.0.1", service::port);

    std::unique_ptr<httplib::Client>& get_client() {
        return client;
    }
}

void service::register_client(std::string_view address) {
    client_addresses.push_back(std::string(address));
    std::cout << "New client registered with address " << address << std::endl;
}

void service::remove_client(std::string_view address) {
    client_addresses.erase(std::remove_if(std::begin(client_addresses), std::end(client_addresses), [&address](auto x) { return x == address; }));
}

void service::start()
{
    if(!available("0.0.0.0", service::port))
    {
        std::cout << "Message publisher already exists. Creating new listener." << std::endl;
        publisher = false;
        using std::chrono::operator""s;
        std::this_thread::sleep_for(1s);
        get_client();
        auto result = get_client()->Post("/register", "127.0.0.1" ,"text/plain");

        if (result && result->status == 200) {
            std::cout << "Registered with publisher successfully." << std::endl;
        } else {
            std::cerr << "Failed to register with publisher with error code " << result << std::endl;
        }
    } else {
        std::cout << "Creating message publisher." << std::endl;
        server_thread = std::thread([&]() {
            server.listen("0.0.0.0", service::port);
        });

        server.Post("/register", [](const httplib::Request& req, httplib::Response& res) {
            std::cout << "Attempting to register new client with address " << req.remote_addr + ":" + std::to_string(req.remote_port) << std::endl;
            register_client(req.remote_addr + ":" + std::to_string(req.remote_port));
            res.set_content("Registration successful", "text/plain");
        });
    }
}

void stop() {
    if(publisher)
    {
        server.stop();
        server_thread.join();
    }
}

httplib::Server* service::get_server() {
    if(!publisher) {
        std::cerr << "Fatal error, attempting to call service::get_server on a non-publisher." << std::endl;
        std::exit(1);
    }

    return &server;
}

std::unique_ptr<httplib::Client>& service::get_client() {
    if(publisher) {
        std::cerr << "Fatal error, attempting to call service::get_client on a publisher." << std::endl;
        std::exit(1);
    }

    return client;
}

bool service::is_publisher()
{
    return publisher;
}

std::vector<std::string>* service::get_clients()
{
    return &client_addresses;
}
