#include "process.h"
#include "runner.h"

bool llama_interactive(const char * args,const char * prompt) {
    std::istringstream iss(args);
    std::vector<std::string> v_args;
    std::string v_a;
    while (iss >> v_a) {
        v_args.push_back(v_a);
    }

    Runner runner(1,v_args, false,std::string(prompt));
    if (!runner.start()) {
        return false;
    }
    return runner.stop();
}
