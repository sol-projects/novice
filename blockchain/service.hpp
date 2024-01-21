#pragma once
#include <httplib.h>

namespace service {
    constexpr int port = 49321;
    void listen();
    httplib::Server* get_server();

    void get_message(auto&& F) {
        service::get_server()->Post("/message", [F](const httplib::Request& req, httplib::Response& res) {
                std::cout << req.body << std::endl;
                F(req.body);
                res.set_content("Success", "text/plain");
            });
    }

    void stop();
}
