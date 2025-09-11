#pragma once

#include "server_context.h"
#include "singleton.h"

struct Request {
    int id;
    std::string body;
    std::function<bool()> is_connection_closed = []() { return false; };
};

struct Response {
    int id;
    bool success;
    std::function<void(int)> complete;
    std::function<bool(int,const std::string&)> write;
    std::function<bool(int)> is_writable;
};

class Scheduler : public patterns::Singleton<Scheduler> {
    friend class patterns::Singleton<Scheduler>;

private:
    server_context ctx_server;
    bool running= false;
    std::thread tasks_thread;
    Scheduler();
    ~Scheduler();

public:
    bool start(const std::vector<std::string>& args);
    bool stop();
    void cleanup();

    bool is_running();

    void handle_completions(const Request & req, Response & res);
    void handle_completions_impl(server_task_type type,json & data,const std::vector<raw_buffer> & files,const std::function<bool()> & is_connection_closed,Response & res,oaicompat_type oaicompat);
    void handle_completions_oai(const Request & req, Response & res);
    void handle_chat_completions(const Request & req, Response & res);

    void handle_embeddings_impl(const Request & req, Response & res, oaicompat_type oaicompat);
    void handle_embeddings(const Request & req, Response & res);
    void handle_embeddings_oai(const Request & req, Response & res);
    void res_error(Response & res, const json & error_data);
    void res_ok(Response & res, const json & data);
};