#include "process.h"
#include "runner.h"
#include "log.h"
#include "whisper_service.h"
#include "scheduler.h"

extern "C" {
    void PushToChan(int id, const char* val);
    void CloseChan(int id);
}

bool llama_start(const char * args) {
    if (Scheduler::instance().is_running()) {
        return false;
    }

    std::istringstream iss(args);
    std::vector<std::string> v_args;
    std::string v_a;
    while (iss >> v_a) {
        v_args.push_back(v_a);
    }

    if (!Scheduler::instance().start(v_args)) {
        return false;
    }
    return true;
}

bool llama_stop() {
    if (!Scheduler::instance().is_running()) {
        return false;
    }
    if (!Scheduler::instance().stop()) {
        return false;
    }
    return true;
}

Result llama_gen(int id,const char * js_str) {
    if (!Scheduler::instance().is_running()) {
        return {false};
    }
    Request rq{id,std::string(js_str)};
    Response rp{id};

    rp.write = [](int id, const std::string& content) {
        PushToChan(id, content.c_str());
        return true;
    };
    rp.is_writable = [](int id) {
        return true;
    };
    rp.complete = [](int id) {
        CloseChan(id);
    };

    Scheduler::instance().handle_completions_oai(rq,rp);
    if (!rp.success) {
        return {false};
    }
    return {true};
}

Result llama_chat(int id,const char * js_str) {
    if (!Scheduler::instance().is_running()) {
        return {false};
    }

    Request rq{id,std::string(js_str)};
    Response rp{id};

    rp.write = [](int id, const std::string& content) {
        PushToChan(id, content.c_str());
        return true;
    };
    rp.is_writable = [](int id) {
        return true;
    };
    rp.complete = [](int id) {
        CloseChan(id);
    };

    Scheduler::instance().handle_chat_completions(rq,rp);
    if (!rp.success) {
        return {false};
    }

    return {true};
}

Result whisper_gen(const char * model,const char * input) {
    WhisperService ws;

    std::string result = ws.generate(std::string(model),std::string(input));
    if (result.empty()) {
        return {false};
    }
    char* arr = new char[result.size() + 1];
    std::copy(result.begin(), result.end(), arr);
    arr[result.size()] = '\0';

    return {true,arr};
}