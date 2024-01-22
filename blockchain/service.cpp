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

    std::unique_ptr<httplib::Client>& get_client()
    {
        return client;
    }
}

void service::register_client(std::string_view address)
{
    client_addresses.push_back(std::string(address));
    std::cout << "New client registered with address " << address << std::endl;
}

void service::remove_client(std::string_view address)
{
    client_addresses.erase(std::remove_if(std::begin(client_addresses), std::end(client_addresses), [&address](auto x) { return x == address; }));
}

void service::send_through_sockets(const std::string& data, const std::string& client_ip, int client_port)
{
    std::cout << "Starting write through sockets." << std::endl;
    int client_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (client_socket == -1)
    {
        perror("Error creating socket");
        return;
    }
    std::cout << "Attempting to connect to IP: " << client_ip << " Port: " << client_port << std::endl;

    sockaddr_in client_address;
    memset(&client_address, 0, sizeof(client_address));
    client_address.sin_family = AF_INET;
    client_address.sin_port = htons(8080);
    inet_pton(AF_INET, client_ip.c_str(), &(client_address.sin_addr));

    if (connect(client_socket, reinterpret_cast<struct sockaddr*>(&client_address), sizeof(client_address)) == -1)
    {
        perror("Error connecting to client");
        close(client_socket);
        return;
    }

    send(client_socket, data.c_str(), data.size(), 0);

    close(client_socket);
}

std::string receive_through_sockets()
{
    std::cout << "Starting read through sockets for messages." << std::endl;
    int server_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (server_socket == -1)
    {
        perror("Error creating socket");
        return "";
    }

    sockaddr_in server_address;
    memset(&server_address, 0, sizeof(server_address));
    server_address.sin_family = AF_INET;
    server_address.sin_addr.s_addr = INADDR_ANY;
    server_address.sin_port = htons(8080);

    if (bind(server_socket, reinterpret_cast<struct sockaddr*>(&server_address), sizeof(server_address)) == -1)
    {
        perror("Error binding socket");
        close(server_socket);
        return "";
    }
    char server_ip_str[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &(server_address.sin_addr), server_ip_str, INET_ADDRSTRLEN);
    std::cout << "Server bound to IP: " << server_ip_str << " Port: " << ntohs(server_address.sin_port) << std::endl;
    if (listen(server_socket, 10) == -1)
    {
        perror("Error listening on socket");
        close(server_socket);
        return "";
    }

    int client_socket = accept(server_socket, nullptr, nullptr);
    if (client_socket == -1)
    {
        perror("Error accepting connection");
        close(server_socket);
        return "";
    }

    char buffer[8096];
    ssize_t received_bytes = recv(client_socket, buffer, sizeof(buffer), 0);

    if (received_bytes == -1)
    {
        perror("Error receiving data");
    }
    else
    {
        buffer[received_bytes] = '\0';
        std::cout << "Received data: " << buffer << std::endl;
    }

    close(client_socket);
    close(server_socket);

    return std::string(buffer);
}

void service::start()
{
    if (!available("0.0.0.0", service::port))
    {
        std::cout << "Message publisher already exists. Creating new listener." << std::endl;
        publisher = false;
        using std::chrono::operator""s;
        std::this_thread::sleep_for(1s);
        get_client();
        auto result = get_client()->Post("/register", "127.0.0.1", "text/plain");

        if (result && result->status == 200)
        {
            std::cout << "Registered with publisher successfully." << std::endl;
        }
        else
        {
            std::cerr << "Failed to register with publisher with error code " << result << std::endl;
        }
    }
    else
    {
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

void stop()
{
    if (publisher)
    {
        server.stop();
        server_thread.join();
    }
}

httplib::Server* service::get_server()
{
    if (!publisher)
    {
        std::cerr << "Fatal error, attempting to call service::get_server on a non-publisher." << std::endl;
        std::exit(1);
    }

    return &server;
}

std::unique_ptr<httplib::Client>& service::get_client()
{
    if (publisher)
    {
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
