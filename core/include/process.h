#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>

typedef struct Result {
    bool ret;
    const char *content;
} Result;

bool llama_start(const char * args);
bool llama_stop();
Result llama_gen(int id,const char * js_str);
Result llama_chat(int id,const char * js_str);

bool llama_interactive(const char * args,const char * prompt);

Result whisper_gen(const char * model,const char * input);

#ifdef __cplusplus
}
#endif