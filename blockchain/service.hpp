#pragma once
#include <httplib.h>
#include <asio.hpp>
#include "Peer.hpp"
#include <unistd.h>
#include <arpa/inet.h>

namespace service {
    constexpr int port = 49321;
    void start();
    httplib::Server* get_server();
    bool is_publisher();
    void stop();
    std::vector<std::string>* get_clients();
    void register_client(std::string_view address);
    void remove_client(std::string_view address);
    std::unique_ptr<httplib::Client>& get_client();

inline void send_through_sockets(const std::string& data, const std::string& client_ip, int client_port) {
    std::cout << "Starting write through sockets." << std::endl;
    int client_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (client_socket == -1) {
        perror("Error creating socket");
        return;
    }
    std::cout << "Attempting to connect to IP: " << client_ip << " Port: " << client_port << std::endl;

    sockaddr_in client_address;
    memset(&client_address, 0, sizeof(client_address));
    client_address.sin_family = AF_INET;
    client_address.sin_port = htons(8080);
    inet_pton(AF_INET, client_ip.c_str(), &(client_address.sin_addr));

    if (connect(client_socket, reinterpret_cast<struct sockaddr*>(&client_address), sizeof(client_address)) == -1) {
        perror("Error connecting to client");
        close(client_socket);
        return;
    }

    send(client_socket, data.c_str(), data.size(), 0);

    close(client_socket);
}

inline std::string receive_through_sockets() {
    std::cout << "Starting read through sockets." << std::endl;
    int server_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (server_socket == -1) {
        perror("Error creating socket");
        return "";
    }

    sockaddr_in server_address;
    memset(&server_address, 0, sizeof(server_address));
    server_address.sin_family = AF_INET;
    server_address.sin_addr.s_addr = INADDR_ANY;
    server_address.sin_port = htons(8080);

    if (bind(server_socket, reinterpret_cast<struct sockaddr*>(&server_address), sizeof(server_address)) == -1) {
        perror("Error binding socket");
        close(server_socket);
        return "";
    }
    char server_ip_str[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &(server_address.sin_addr), server_ip_str, INET_ADDRSTRLEN);
    std::cout << "Server bound to IP: " << server_ip_str << " Port: " << ntohs(server_address.sin_port) << std::endl;
    if (listen(server_socket, 10) == -1) {
        perror("Error listening on socket");
        close(server_socket);
        return "";
    }

    int client_socket = accept(server_socket, nullptr, nullptr);
    if (client_socket == -1) {
        perror("Error accepting connection");
        close(server_socket);
        return "";
    }

    char buffer[8096];
    ssize_t received_bytes = recv(client_socket, buffer, sizeof(buffer), 0);

    if (received_bytes == -1) {
        perror("Error receiving data");
    } else {
        buffer[received_bytes] = '\0';
        std::cout << "Received data: " << buffer << std::endl;
    }

    close(client_socket);
    close(server_socket);

    return std::string(buffer);
}


    bool get_message_and_status(auto&& F) {
        if(service::is_publisher())
        {
            service::get_server()->Post("/message", [F](const httplib::Request& req, httplib::Response& res) {
                std::cout << req.body << std::endl;
                res.set_content("Success", "text/plain");

                for (auto client_address : *service::get_clients()) {
                    auto delimiter_pos = client_address.find(':');
                    if (delimiter_pos == std::string::npos) {
                        std::cout << "Client " << client_address << " has an invalid address." << std::endl;
                        continue;
                    }

                    std::string client_ip = client_address.substr(0, delimiter_pos);
                    int client_port = std::stoi(client_address.substr(delimiter_pos + 1).data());
                    send_through_sockets(req.body, client_ip, client_port);
                }

                F(req.body);
            });
        }
        else
        {
            F(receive_through_sockets());
        }

        return service::is_publisher();
    }
}
