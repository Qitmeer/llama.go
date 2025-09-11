#include <iostream>
#include <thread>
#include <chrono>
#include <cstdlib>
#include <future>
#include <unistd.h>

#include "process.h"

int main() {
    const char* env_model = "LLAMA_TEST_MODEL";
    const char* model = std::getenv(env_model);

    if (model == nullptr) {
        std::cerr << "error：can't find " << env_model << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "env: " << env_model << "=" << model << std::endl;

    std::stringstream ss;
    ss << "test_runner_gen -m " << model << " --seed 0";

    int ret = llama_start(ss.str().c_str());
    if (ret == EXIT_FAILURE) {
        return ret;
    }

    std::this_thread::sleep_for(std::chrono::seconds(2));

    std::future<void> ll_gen = std::async(std::launch::async, [](){
        std::string js_str="{\"prompt\":\"why the sky is blue\"}";
        Result result = llama_chat(js_str.c_str());
        if (!result.ret) {
            std::cout<<"fail"<<std::endl;
            return;
        }

        std::string content(result.content);
        if (content.empty()) {
            std::cout<<"fail"<<std::endl;
            return;
        }
        std::cout<<"Response:"<<content<<std::endl;

        bool ret =llama_stop();
        std::cout<<"Result1:"<<ret<<std::endl;
    });


    ll_gen.wait();


    std::cout<<"success"<<std::endl;

    return EXIT_SUCCESS;
}