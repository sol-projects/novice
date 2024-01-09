#include "Peer.hpp"
#include "blockchain.hpp"
#include "openssl/des.h"
#include <QKeyEvent>
#include <QLabel>
#include <QLineEdit>
#include <QVBoxLayout>
#include <QtWidgets>
#include <asio.hpp>
#include <asio/error.hpp>
#include <asio/ip/tcp.hpp>
#include <asio/placeholders.hpp>
#include <asio/streambuf.hpp>
#include <asio/system_error.hpp>
#include <atomic>
#include <functional>
#include <iostream>
#include <memory>
#include <netdb.h>
#include <openssl/rc4.h>
#include <random>
#include <sys/socket.h>
#include <sys/types.h>

namespace
{
    std::vector<std::shared_ptr<TcpConnection>> connections;
    Blockchain m_blockchain = blockchain::init();

    QWidget* m_window;
    QLineEdit* m_lineEdit;
    QVBoxLayout* m_layout;
    std::vector<QLabel*> m_labels;
    std::atomic<bool> sendSignal = false;
    std::atomic<bool> resetWrite = false;

    void outputToGui(const std::string& string)
    {
        for (auto label : m_labels)
        {
            if (label->text().toStdString() == "")
            {
                label->setText(QString::fromStdString(string));
                return;
            }
        }

        for (std::size_t i = 1; i < std::size(m_labels); i++)
        {
            m_labels.at(i)->setText(m_labels.at(i - 1)->text());
        }

        m_labels.back()->setText(QString::fromStdString(string));
    }
} // namespace

void TcpConnection::write(const std::string& msg)
{
    m_write = msg;
    auto self(shared_from_this());
    m_socket.async_send(asio::buffer(m_write),
        [this, self](std::error_code code, std::size_t) {
            if (!code)
            {
                m_write.clear();
                m_read.clear();
                m_read.resize(buffer);

                read();
            }
            else
            {
                m_write.clear();
                m_read.clear();
                m_read.resize(buffer);
                std::cerr << code.message();
            }
        });
}

void TcpConnection::read()
{
    auto self(shared_from_this());
    m_socket.async_receive(
        asio::buffer(m_read.data(), std::size(m_read)),
        [this, self](asio::error_code code, std::size_t length) {
            if (!code)
            {
                auto received = std::string(m_read.data());
                std::cout << "SERVER RECEIVED: " << received << std::endl;
                Server::broadcastFromAllServersToAllClients(received);
            }
            else
            {
                std::erase_if(connections, [&](auto connection){ return connection.get() == this; });
                std::cerr << code.message() << " Bytes transferred: " << length
                          << std::endl;
                std::cerr << "client disconnected" << std::endl;
            }
        });
}

Server::Server(asio::io_context& ioContext, int port)
    : endpoint(asio::ip::address_v4::any(), port)
    , acceptor(ioContext, endpoint)
    , socket(ioContext)
{
    startAccepting();
}

void Server::startAccepting()
{
    acceptor.async_accept(socket, [this](asio::error_code code) {
        if (!code)
        {
            auto connection = std::make_shared<TcpConnection>(std::move(socket));
            std::cout << "new connection" << std::endl;
            connection->read();
            connections.push_back(std::move(connection));
        }
        else
        {
            std::cerr << "Error with client connecting " << code.message();
        }

        startAccepting();
    });
}

void Server::broadcastFromAllServersToAllClients(const std::string& msg)
{
    std::cout << "SERVER broadcasting to " << connections.size() << " clients."
              << std::endl;
    for (auto connection : connections)
    {
        connection->write(msg);
    }
}

bool available(const std::string& ip, int port)
{
    auto true_ = 1;
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0)
    {
        std::cerr << "Error creating socket" << std::endl;
        close(sockfd);
        return false;
    }

    struct sockaddr_in server_addr;
    std::memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);
    int result = inet_pton(AF_INET, ip.c_str(), &server_addr.sin_addr);
    if (result < 0)
    {
        close(sockfd);
        std::cerr << "Error converting IP address" << std::endl;
        return false;
    }

    result = connect(sockfd, (struct sockaddr*)&server_addr, sizeof(server_addr));
    if (result < 0)
    {
        close(sockfd);
        return false;
    }
    else
    {
        close(sockfd);
        return true;
    }

    close(sockfd);
    setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &true_, sizeof(int));
    return true;
}

bool Gui::eventFilter(QObject* obj, QEvent* event)
{
    if (obj == m_lineEdit && event->type() == QEvent::KeyPress)
    {
        QKeyEvent* key = static_cast<QKeyEvent*>(event);
        if (key->key() == Qt::Key::Key_Control)
        {
            if (!m_lineEdit->text().toStdString().empty())
            {
                sendSignal = true;
            }
        }
    }

    return QObject::eventFilter(obj, event);
}

Gui::Gui(QApplication* app)
{
    app->installEventFilter(this);
    m_window = new QWidget();
    m_window->resize(700, 300);

    m_lineEdit = new QLineEdit();
    m_layout = new QVBoxLayout(m_window);

    for (int i = 0; i < 60; i++)
    {
        auto message = new QLabel();
        m_labels.push_back(std::move(message));
        m_layout->addWidget(m_labels.at(i));
    }

    m_layout->addWidget(m_lineEdit);

    m_window->show();
    m_window->setWindowTitle("Blockchain");
}

void Client::writeSignal(const std::string& data) { m_signalWrite = data; }

Client::Client()
{
    socket = std::make_unique<asio::ip::tcp::socket>(ioContext);
}

Client::Client(const std::string& ip, int port, const OptionFlags& options)
    : m_ip(ip)
    , m_port(port)
    , m_read(buffer)
    , m_options(options)
{
    socket = std::make_unique<asio::ip::tcp::socket>(ioContext);
    connect(m_ip, m_port);
}

bool Client::connect(const std::string& ip, int port)
{
    asio::ip::tcp::endpoint endpoint(asio::ip::address::from_string(ip.c_str()),
        port);
    socket->async_connect(endpoint, [](const asio::error_code& code) {
        if (code)
        {
            std::cerr << code.message();
            return false;
        }
        else
        {
            std::cout << "connected" << std::endl;
            return true;
        }
    });

    std::cerr << "not connected" << std::endl;
    return false;
}

void Client::write(const std::string& data)
{
    m_write = data;
    socket->async_send(
        asio::buffer(m_write),
        [this](const asio::error_code& code, std::size_t bytesTransferred) {
            if (!code)
            {
                std::cout << "CLIENT SENT: " << m_write << std::endl;
                m_read.clear();
                m_read.resize(buffer);
                m_write.clear();
                read();
            }
            else
            {
                std::cerr << code.message()
                          << " Bytes transferred: " << bytesTransferred;
                m_read.clear();
                m_read.resize(buffer);
                m_write.clear();
            }
        });
}

void Client::mine()
{
    std::thread t([this]() {
        for (;;)
        {
            resetWrite = false;
            m_blockchain = blockchain::new_block_pow(m_blockchain, resetWrite, m_options);
            if(resetWrite)
            {
                m_blockchain.erase(std::end(m_blockchain) - 1);
            }
            else if (blockchain::validate(m_blockchain))
            {
                write(blockchain::to_string(m_blockchain));
                outputToGui("Mined block. Diff: " + std::to_string(m_blockchain.back().difficulty) + " Hash of last 5: " + std::string(m_blockchain.back().hash.end() - 5, m_blockchain.back().hash.end()));
            }
        }
    });

    t.join();
}

void Client::read()
{
    socket->async_receive(
        asio::buffer(m_read),
        [this](const asio::error_code& code, std::size_t bytesTransferred) {
            if (!code)
            {
                auto received = std::string(m_read.data());
                std::cout << "CLIENT RECEIVED: " << received << std::endl;
                if (received.starts_with("Client connected.") || bytesTransferred == 0)
                {
                    read();
                    return;
                }

                auto newBlockchain = blockchain::from_string(received);
                if (blockchain::validate(newBlockchain))
                {
                    if (blockchain::difficulty(newBlockchain) >= blockchain::difficulty(m_blockchain))
                    {
                        m_blockchain = newBlockchain;
                        resetWrite = true;
                        outputToGui(
                            "Successfully validated new blocks. Local chain modified. Last 5 of hash: " + std::string(m_blockchain.back().hash.end() - 5, m_blockchain.back().hash.end()));
                    }
                    else
                    {
                        outputToGui("Ignored new blocks due to lower difficulty. Last 5 of hash: " + std::string(m_blockchain.back().hash.end() - 5, m_blockchain.back().hash.end()));
                    }
                }
                else
                {
                    outputToGui("Unsuccessfully validated new blocks, ignoring." + std::string(m_blockchain.back().hash.end() - 5, m_blockchain.back().hash.end()));
                }

                read();
            }
            else
            {
                std::cerr << code.message()
                          << " Bytes transferred: " << bytesTransferred;
                outputToGui("Central node disconnected.");
                return;

                read();
            }
        });
}
