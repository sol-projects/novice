#pragma once
#include <QGraphicsView>
#include <QMainWindow>
#include <QtWidgets>
#include <asio.hpp>
#include <asio/ip/tcp.hpp>
#include <iostream>
#include <memory>
#include <string>
#include "options.hpp"

constexpr int buffer = 12000;
class TcpConnection : public std::enable_shared_from_this<TcpConnection>
{
public:
    TcpConnection(asio::ip::tcp::socket socket)
        : m_socket(std::move(socket))
        , m_read(buffer)
    {
    }
    void write(const std::string& msg);
    void read();

private:
    asio::ip::tcp::socket m_socket;
    std::string m_write;
    std::vector<char> m_read;
};

class Server
{
public:
    Server(asio::io_context& ioContext, int port, int world_rank, int mpi_world_size);
    static void broadcastFromAllServersToAllClients(const std::string& msg);
    void startAccepting();

private:
    int m_world_rank;
    int m_world_size;
    asio::ip::tcp::endpoint endpoint;
    asio::ip::tcp::acceptor acceptor;
    asio::ip::tcp::socket socket;
    std::string message;

};

class Gui : QWidget
{
public:
    Gui(QApplication* app);
    bool eventFilter(QObject* obj, QEvent* event) override;
};

bool available(const std::string& ip, int port);

class Client
{
public:
    Client();
    Client(const std::string& ip, int port, const OptionFlags& options, int world_rank, int mpi_world_size);
    ~Client();

    void write(const std::string& data);
    static void writeSignal(const std::string& data);
    void read();

    void guiSync();
    void mine();

    bool connect(const std::string& ip, int port);
    void handle_read(const asio::error_code code, std::size_t bytesTransferred);

    void process(auto fn)
    {
        std::thread t([this]() { ioContext.run(); });

        write("Client connected.");
        fn(this);
        mine();

        t.join();
    }

private:
    int m_world_rank;
    int m_world_size;
    std::string m_ip;
    int m_port;
    asio::io_context ioContext;
    std::unique_ptr<asio::ip::tcp::socket> socket;
    std::string m_write;
    std::string m_username;
    static inline std::string m_signalWrite = "";
    std::vector<char> m_read;
    OptionFlags m_options;
    std::condition_variable m_sensor_data_cv;
    std::mutex m_sensor_data_mutex;
    std::string m_sensor_data;
};
