#include <iostream>
#include <thread>
#include <chrono>
#include <cstdlib>

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
    ss << "test_runner -m " << model << " --seed 0";

    bool ret=llama_start(ss.str().c_str());
    if (!ret) {
        return 1;
    }
    int seconds = 2;
    std::cout << "sleep...:"<<seconds<<" seconds" << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(seconds));

    ret=llama_stop();
    if (!ret) {
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}