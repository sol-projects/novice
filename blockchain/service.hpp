#pragma once
#include "Peer.hpp"
#include <arpa/inet.h>
#include <asio.hpp>
#include <httplib.h>
#include <unistd.h>

namespace service
{
    constexpr int port = 49321;
    void start();
    httplib::Server* get_server();
    bool is_publisher();
    void stop();
    std::vector<std::string>* get_clients();
    void register_client(std::string_view address);
    void remove_client(std::string_view address);
    std::unique_ptr<httplib::Client>& get_client();
    void send_through_sockets(const std::string& data, const std::string& client_ip, int client_port);
    std::string receive_through_sockets();

    bool get_message_and_status(auto&& F)
    {
        if (service::is_publisher())
        {
            service::get_server()->Post("/message", [F](const httplib::Request& req, httplib::Response& res) {
                std::cout << req.body << std::endl;
                res.set_content("Success", "text/plain");

                for (auto client_address : *service::get_clients())
                {
                    auto delimiter_pos = client_address.find(':');
                    if (delimiter_pos == std::string::npos)
                    {
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
