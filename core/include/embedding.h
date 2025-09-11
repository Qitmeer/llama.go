#pragma once
#include "process.h"
#ifdef __cplusplus
extern "C" {
#endif

Result llama_embedding(const char * args,const char * prompt);

#ifdef __cplusplus
}
#endif