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
    ss << "test_runner_gen -m " << model << " -i --seed 0";

    std::future<void> ll_main = std::async(std::launch::async, [&ss](){
        bool ret = llama_start(ss.str().c_str(), true,"");
        std::cout<<"Result0:"<<ret<<std::endl;
    });

    std::this_thread::sleep_for(std::chrono::seconds(2));

    std::future<void> ll_gen = std::async(std::launch::async, [](){
        const char* roles[] = { "system","user"};
        const char* contents[] = {,"why sky is blue"};
        int size = 2;

        std::string content = llama_chat(roles,contents,size);
        if (content.empty()) {
            return;
        }
        std::cout<<"Response:"<<content<<std::endl;
        });


    ll_gen.wait();
    ll_main.wait();


    std::cout<<"success"<<std::endl;

    return EXIT_SUCCESS;
}