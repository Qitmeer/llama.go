#include "scheduler.h"


Scheduler::Scheduler() {
    std::cout << "Scheduler Constructor"<< std::endl;
}

Scheduler::~Scheduler() {
    std::cout << "Scheduler Destructor"<< std::endl;
}

bool Scheduler::start(const std::vector<std::string>& args) {
    std::ostringstream oss;
    for (const auto& s : args) {
        oss << s << " ";
    }
    std::cout << "Scheduler Start:"<<oss.str()<< std::endl;

    std::vector<char*> v_argv;
    for (auto& t : args) {
        v_argv.push_back(const_cast<char*>(t.c_str()));
    }
    int argc = v_argv.size();

    common_params params;
    if (!common_params_parse(argc, v_argv.data(), params, LLAMA_EXAMPLE_SERVER)) {
        return false;
    }
    if (params.model_alias.empty() && !params.model.path.empty()) {
        std::filesystem::path fp(params.model.path);
        params.model_alias=fp.stem();
    }

    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    LOG_INF("system info: n_threads = %d, n_threads_batch = %d, total_threads = %d\n", params.cpuparams.n_threads, params.cpuparams_batch.n_threads, std::thread::hardware_concurrency());
    LOG_INF("\n");
    LOG_INF("%s\n", common_params_get_system_info(params).c_str());
    LOG_INF("\n");

    // Necessary similarity of prompt for slot selection
    ctx_server.slot_prompt_similarity = params.slot_prompt_similarity;

    //
    // Start the server
    //
    if (params.n_threads_http < 1) {
        // +2 threads for monitoring endpoints
        params.n_threads_http = std::max(params.n_parallel + 2, (int32_t) std::thread::hardware_concurrency() - 1);
    }

    // load the model
    LOG_INF("%s: loading model\n", __func__);

    if (!ctx_server.load_model(params)) {
        cleanup();
        LOG_ERR("%s: exiting due to model loading error\n", __func__);
        return false;
    }

    ctx_server.init();

    LOG_INF("%s: model loaded\n", __func__);

    // print sample chat example to make it clear which template is used
    LOG_INF("%s: chat template, chat_template: %s, example_format: '%s'\n", __func__,
            common_chat_templates_source(ctx_server.chat_templates.get()),
            common_chat_format_example(ctx_server.chat_templates.get(), ctx_server.params_base.use_jinja).c_str());

    ctx_server.queue_tasks.on_new_task([this](server_task && task) {
        ctx_server.process_single_task(std::move(task));
    });

    ctx_server.queue_tasks.on_update_slots([this]() {
        ctx_server.update_slots();
    });
    running= true;
    // this call blocks the main thread until queue_tasks.terminate() is called
    tasks_thread = std::thread([&](){
        ctx_server.queue_tasks.start_loop();
    });

    return true;
}

bool Scheduler::stop() {
    std::cout << "Scheduler stop: is_running="<<is_running()<< std::endl;
    if (!is_running()) {
        return false;
    }
    running= false;
    cleanup();
    if (tasks_thread.joinable()) {
        tasks_thread.join();
    }
    return true;
}

void Scheduler::cleanup() {
    // this will unblock start_loop()
    ctx_server.queue_tasks.terminate();
    llama_backend_free();
}

bool Scheduler::is_running() {
    return running;
}

void Scheduler::handle_completions(const Request & req, Response & res) {
    json data = json::parse(req.body);
    std::vector<raw_buffer> files; // dummy
    handle_completions_impl(
            SERVER_TASK_TYPE_COMPLETION,
            data,
            files,
            req.is_connection_closed,
            res,
            OAICOMPAT_TYPE_NONE);
}

// handle completion-like requests (completion, chat, infill)
// we can optionally provide a custom format for partial results and final results
void Scheduler::handle_completions_impl(
        server_task_type type,
        json & data,
        const std::vector<raw_buffer> & files,
        const std::function<bool()> & is_connection_closed,
        Response & res,
        oaicompat_type oaicompat) {
    GGML_ASSERT(type == SERVER_TASK_TYPE_COMPLETION || type == SERVER_TASK_TYPE_INFILL);

    auto completion_id = gen_chatcmplid();
    std::unordered_set<int> task_ids;
    try {
        std::vector<server_task> tasks;

        const auto & prompt = data.at("prompt");
        // TODO: this log can become very long, put it behind a flag or think about a more compact format
        //SRV_DBG("Prompt: %s\n", prompt.is_string() ? prompt.get<std::string>().c_str() : prompt.dump(2).c_str());

        // process files
        mtmd::bitmaps bitmaps;
        const bool has_mtmd = ctx_server.mctx != nullptr;
        {
            if (!has_mtmd && !files.empty()) {
                throw std::runtime_error("This server does not support multimodal");
            }
            for (auto & file : files) {
                mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_buf(ctx_server.mctx, file.data(), file.size()));
                if (!bmp.ptr) {
                    throw std::runtime_error("Failed to load image or audio file");
                }
                // calculate bitmap hash (for KV caching)
                std::string hash = fnv_hash(bmp.data(), bmp.n_bytes());
                bmp.set_id(hash.c_str());
                bitmaps.entries.push_back(std::move(bmp));
            }
        }

        // process prompt
        std::vector<server_tokens> inputs;

        if (oaicompat && has_mtmd) {
            // multimodal
            std::string prompt_str = prompt.get<std::string>();
            mtmd_input_text inp_txt = {
                    prompt_str.c_str(),
                    /* add_special */   true,
                    /* parse_special */ true,
            };
            mtmd::input_chunks chunks(mtmd_input_chunks_init());
            auto bitmaps_c_ptr = bitmaps.c_ptr();
            int32_t tokenized = mtmd_tokenize(ctx_server.mctx,
                                              chunks.ptr.get(),
                                              &inp_txt,
                                              bitmaps_c_ptr.data(),
                                              bitmaps_c_ptr.size());
            if (tokenized != 0) {
                throw std::runtime_error("Failed to tokenize prompt");
            }

            server_tokens tmp(chunks, true);
            inputs.push_back(std::move(tmp));
        } else {
            // non-multimodal version
            auto tokenized_prompts = tokenize_input_prompts(ctx_server.vocab, prompt, true, true);
            for (auto & p : tokenized_prompts) {
                auto tmp = server_tokens(p, ctx_server.mctx != nullptr);
                inputs.push_back(std::move(tmp));
            }
        }

        tasks.reserve(inputs.size());
        for (size_t i = 0; i < inputs.size(); i++) {
            server_task task = server_task(type);

            task.id    = ctx_server.queue_tasks.get_new_id();
            task.index = i;

            task.prompt_tokens    = std::move(inputs[i]);
            task.params           = server_task::params_from_json_cmpl(
                    ctx_server.ctx,
                    ctx_server.params_base,
                    data);
            task.id_selected_slot = json_value(data, "id_slot", -1);

            // OAI-compat
            task.params.oaicompat                 = oaicompat;
            task.params.oaicompat_cmpl_id         = completion_id;
            // oaicompat_model is already populated by params_from_json_cmpl

            tasks.push_back(std::move(task));
        }

        task_ids = server_task::get_list_id(tasks);
        ctx_server.queue_results.add_waiting_tasks(tasks);
        ctx_server.queue_tasks.post(std::move(tasks));
    } catch (const std::exception & e) {
        res_error(res, format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
        return;
    }

    bool stream = json_value(data, "stream", false);

    if (!stream) {
        ctx_server.receive_multi_results(task_ids, [&](std::vector<server_task_result_ptr> & results) {
            if (results.size() == 1) {
                // single result
                res_ok(res, results[0]->to_json());
            } else {
                // multiple results (multitask)
                json arr = json::array();
                for (auto & res : results) {
                    arr.push_back(res->to_json());
                }
                res_ok(res, arr);
            }
        }, [&](const json & error_data) {
            res_error(res, error_data);
        }, is_connection_closed);
        res.complete(res.id);
        ctx_server.queue_results.remove_waiting_task_ids(task_ids);
    } else {
        auto server_sent_event=[&](const char * event, const json & data) {
            const std::string str =
                    std::string(event) + ": " +
                    data.dump(-1, ' ', false, json::error_handler_t::replace) +
                    "\n\n"; // required by RFC 8895 - A message is terminated by a blank line (two line terminators in a row).

            LOG_DBG("data stream, to_send: %s", str.c_str());

            return res.write(res.id,str);
        };
        ctx_server.receive_cmpl_results_stream(task_ids, [&](server_task_result_ptr & result) -> bool {
            json res_json = result->to_json();
            if (res_json.is_array()) {
                for (const auto & res : res_json) {
                    if (!server_sent_event("data", res)) {
                        // sending failed (HTTP connection closed), cancel the generation
                        return false;
                    }
                }
                return true;
            } else {
                return server_sent_event("data", res_json);
            }
        }, [&](const json & error_data) {
            server_sent_event("error", error_data);
        }, [&res]() {
            // note: do not use req.is_connection_closed here because req is already destroyed
            return !res.is_writable(res.id);
        });
        if (oaicompat != OAICOMPAT_TYPE_NONE) {
            static const std::string ev_done = "data: [DONE]\n\n";
            res.write(res.id,ev_done);
        }
        res.success= true;
        res.complete(res.id);
        ctx_server.queue_results.remove_waiting_task_ids(task_ids);
    }
}


void Scheduler::handle_completions_oai(const Request & req, Response & res) {
    json data = oaicompat_completion_params_parse(json::parse(req.body));
    std::vector<raw_buffer> files; // dummy
    handle_completions_impl(
            SERVER_TASK_TYPE_COMPLETION,
            data,
            files,
            req.is_connection_closed,
            res,
            OAICOMPAT_TYPE_COMPLETION);
}

void Scheduler::handle_chat_completions(const Request & req, Response & res) {
    LOG_DBG("request: %s\n", req.body.c_str());

    auto body = json::parse(req.body);
    std::vector<raw_buffer> files;
    json data = oaicompat_chat_params_parse(
            body,
            ctx_server.oai_parser_opt,
            files);

    handle_completions_impl(
            SERVER_TASK_TYPE_COMPLETION,
            data,
            files,
            req.is_connection_closed,
            res,
            OAICOMPAT_TYPE_CHAT);
}

void Scheduler::handle_embeddings_impl(const Request & req, Response & res, oaicompat_type oaicompat) {
    if (!ctx_server.params_base.embedding) {
        res_error(res, format_error_response("This server does not support embeddings. Start it with `--embeddings`", ERROR_TYPE_NOT_SUPPORTED));
        return;
    }

    if (oaicompat != OAICOMPAT_TYPE_NONE && llama_pooling_type(ctx_server.ctx) == LLAMA_POOLING_TYPE_NONE) {
        res_error(res, format_error_response("Pooling type 'none' is not OAI compatible. Please use a different pooling type", ERROR_TYPE_INVALID_REQUEST));
        return;
    }

    const json body = json::parse(req.body);

    // for the shape of input/content, see tokenize_input_prompts()
    json prompt;
    if (body.count("input") != 0) {
        prompt = body.at("input");
    } else if (body.contains("content")) {
        oaicompat = OAICOMPAT_TYPE_NONE; // "content" field is not OAI compatible
        prompt = body.at("content");
    } else {
        res_error(res, format_error_response("\"input\" or \"content\" must be provided", ERROR_TYPE_INVALID_REQUEST));
        return;
    }

    bool use_base64 = false;
    if (body.count("encoding_format") != 0) {
        const std::string& format = body.at("encoding_format");
        if (format == "base64") {
            use_base64 = true;
        } else if (format != "float") {
            res_error(res, format_error_response("The format to return the embeddings in. Can be either float or base64", ERROR_TYPE_INVALID_REQUEST));
            return;
        }
    }

    auto tokenized_prompts = tokenize_input_prompts(ctx_server.vocab, prompt, true, true);
    for (const auto & tokens : tokenized_prompts) {
        // this check is necessary for models that do not add BOS token to the input
        if (tokens.empty()) {
            res_error(res, format_error_response("Input content cannot be empty", ERROR_TYPE_INVALID_REQUEST));
            return;
        }
    }

    int embd_normalize = 2; // default to Euclidean/L2 norm
    if (body.count("embd_normalize") != 0) {
        embd_normalize = body.at("embd_normalize");
        if (llama_pooling_type(ctx_server.ctx) == LLAMA_POOLING_TYPE_NONE) {
            SRV_DBG("embd_normalize is not supported by pooling type %d, ignoring it\n", llama_pooling_type(ctx_server.ctx));
        }
    }

    // create and queue the task
    json responses = json::array();
    bool error = false;
    std::unordered_set<int> task_ids;
    {
        std::vector<server_task> tasks;
        for (size_t i = 0; i < tokenized_prompts.size(); i++) {
            server_task task = server_task(SERVER_TASK_TYPE_EMBEDDING);

            task.id            = ctx_server.queue_tasks.get_new_id();
            task.index         = i;
            task.prompt_tokens = server_tokens(tokenized_prompts[i], ctx_server.mctx != nullptr);

            // OAI-compat
            task.params.oaicompat = oaicompat;
            task.params.embd_normalize = embd_normalize;

            tasks.push_back(std::move(task));
        }

        task_ids = server_task::get_list_id(tasks);
        ctx_server.queue_results.add_waiting_tasks(tasks);
        ctx_server.queue_tasks.post(std::move(tasks));
    }

    // get the result
    ctx_server.receive_multi_results(task_ids, [&](std::vector<server_task_result_ptr> & results) {
        for (auto & res : results) {
            GGML_ASSERT(dynamic_cast<server_task_result_embd*>(res.get()) != nullptr);
            responses.push_back(res->to_json());
        }
    }, [&](const json & error_data) {
        res_error(res, error_data);
        error = true;
    }, req.is_connection_closed);

    ctx_server.queue_results.remove_waiting_task_ids(task_ids);

    if (error) {
        return;
    }

    // write JSON response
    json root = oaicompat == OAICOMPAT_TYPE_EMBEDDING
                ? format_embeddings_response_oaicompat(body, responses, use_base64)
                : json(responses);
    res_ok(res, root);
}

void Scheduler::handle_embeddings(const Request & req, Response & res) {
    handle_embeddings_impl(req, res, OAICOMPAT_TYPE_NONE);
}

void Scheduler::handle_embeddings_oai(const Request & req, Response & res) {
    handle_embeddings_impl(req, res, OAICOMPAT_TYPE_EMBEDDING);
}

void Scheduler::res_error(Response & res, const json & error_data) {
    json final_response {{"error", error_data}};
    res.write(res.id,safe_json_to_str(final_response));
    res.success= false;
};

void Scheduler::res_ok(Response & res, const json & data) {
    res.write(res.id,safe_json_to_str(data));
    res.success= true;
};
