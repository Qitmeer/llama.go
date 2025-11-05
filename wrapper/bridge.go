package wrapper

/*
#include <stdlib.h>
#include "core.h"
*/
import "C"

import (
	"fmt"
	"math"
	"sync"
	"unsafe"

	econfig "github.com/Qitmeer/llama.go/app/embedding/config"
	"github.com/Qitmeer/llama.go/config"
)

var (
	mu         sync.Mutex
	channels   = make(map[int]chan any)
	nextChanID = 1
)

func LlamaStartInteractive(cfg *config.Config) error {
	if !cfg.HasModel() {
		return fmt.Errorf("No model")
	}
	ip := C.CString(cfg.Prompt)
	defer C.free(unsafe.Pointer(ip))

	cfgArgs := assemblyArgs(cfg)
	ca := C.CString(cfgArgs)
	defer C.free(unsafe.Pointer(ca))

	ret := C.llama_interactive_start(ca, ip)
	if !bool(ret) {
		return fmt.Errorf("Llama interactive error")
	}
	return nil
}

func LlamaStopInteractive() error {
	ret := C.llama_interactive_stop()
	if !bool(ret) {
		return fmt.Errorf("Llama interactive error")
	}
	return nil
}

func LlamaGenerate(id int, jsStr string) error {
	if len(jsStr) <= 0 {
		return fmt.Errorf("json string")
	}
	js := C.CString(jsStr)
	defer C.free(unsafe.Pointer(js))

	ret := C.llama_gen(C.int(id), js)
	if !bool(ret.ret) {
		return fmt.Errorf("Llama run error")
	}
	return nil
}

func LlamaChat(id int, jsStr string) error {
	if len(jsStr) <= 0 {
		return fmt.Errorf("json string")
	}
	js := C.CString(jsStr)
	defer C.free(unsafe.Pointer(js))

	ret := C.llama_chat(C.int(id), js)
	if !bool(ret.ret) {
		return fmt.Errorf("Llama run error")
	}
	return nil
}

func LlamaStart(cfg *config.Config) error {
	if !cfg.HasModel() {
		return fmt.Errorf("No model")
	}
	cfgArgs := assemblyArgs(cfg)
	ca := C.CString(cfgArgs)
	defer C.free(unsafe.Pointer(ca))

	ret := C.llama_start(ca)
	if !bool(ret) {
		return fmt.Errorf("Llama start error")
	}
	return nil
}

func LlamaStop() error {
	ret := C.llama_stop()
	if !bool(ret) {
		return fmt.Errorf("Llama stop error")
	}
	return nil
}

func LlamaEmbedding(cfg *config.Config, prompts string, embdOutputFormat string) (string, error) {
	if len(prompts) <= 0 {
		return "", fmt.Errorf("No prompt")
	}
	ip := C.CString(prompts)
	defer C.free(unsafe.Pointer(ip))

	cfgArgs := assemblyArgs(cfg)
	cfgArgs = fmt.Sprintf("%s --embd-normalize %d", cfgArgs, econfig.Conf.EmbdNormalize)
	if len(embdOutputFormat) > 0 {
		cfgArgs = fmt.Sprintf("%s --embd-output-format %s", cfgArgs, embdOutputFormat)
	} else {
		cfgArgs = fmt.Sprintf("%s --embd-output-format %s", cfgArgs, econfig.Conf.EmbdOutputFormat)
	}
	cfgArgs = fmt.Sprintf("%s --embd-separator %s", cfgArgs, econfig.Conf.EmbdSeparator)

	ca := C.CString(cfgArgs)
	defer C.free(unsafe.Pointer(ca))

	ret := C.llama_embedding(ca, ip)
	if !bool(ret.ret) {
		return "", fmt.Errorf("Llama run error")
	}

	content := C.GoString(ret.content)
	C.free(unsafe.Pointer(ret.content))
	return content, nil
}

func WhisperGenerate(cfg *config.Config, input string) (string, error) {
	if !cfg.HasModel() {
		return "", fmt.Errorf("No model")
	}
	if len(input) <= 0 {
		return "", fmt.Errorf("No input")
	}
	model := C.CString(cfg.ModelPath())
	defer C.free(unsafe.Pointer(model))

	ip := C.CString(input)
	defer C.free(unsafe.Pointer(ip))

	ret := C.whisper_gen(model, ip)
	if !bool(ret.ret) {
		return "", fmt.Errorf("Llama run error")
	}

	content := C.GoString(ret.content)
	C.free(unsafe.Pointer(ret.content))
	return content, nil
}

func NewChan() (int, chan any) {
	mu.Lock()
	defer mu.Unlock()
	id := nextChanID
	_, ok := channels[id]
	if ok {
		return 0, nil
	}
	ch := make(chan any)
	channels[id] = ch
	nextChanID++
	return id, ch
}

//export PushToChan
func PushToChan(id C.int, val *C.char) {
	str := C.GoString(val)
	mu.Lock()
	ch, ok := channels[int(id)]
	mu.Unlock()
	if ok {
		ch <- str
	}
}

//export CloseChan
func CloseChan(id C.int) {
	mu.Lock()
	ch, ok := channels[int(id)]
	if ok {
		close(ch)
		delete(channels, int(id))
	}
	mu.Unlock()
}

func GetCommonParams() CommonParams {
	ret := C.get_common_params()
	return CommonParams{EndpointProps: bool(ret.endpoint_props)}
}

func GetProps() (string, error) {
	ret := C.get_props()
	if !bool(ret.ret) {
		return "", fmt.Errorf("Llama run error")
	}

	content := C.GoString(ret.content)
	C.free(unsafe.Pointer(ret.content))
	return content, nil
}
func GetSlots() (string, error) {
	ret := C.get_slots()
	if !bool(ret.ret) {
		return "", fmt.Errorf("Llama run error")
	}

	content := C.GoString(ret.content)
	C.free(unsafe.Pointer(ret.content))
	return content, nil
}

func assemblyArgs(cfg *config.Config) string {
	cfgArgs := "llama"
	if len(cfg.ModelPath()) > 0 {
		cfgArgs = fmt.Sprintf("%s --model %s", cfgArgs, cfg.ModelPath())
	}
	if cfg.CtxSize != config.DefaultContextSize {
		cfgArgs = fmt.Sprintf("%s --ctx-size %d", cfgArgs, cfg.CtxSize)
	}
	if cfg.NGpuLayers != config.DefaultNGpuLayers {
		cfgArgs = fmt.Sprintf("%s --n-gpu-layers %d", cfgArgs, cfg.NGpuLayers)
	}
	if cfg.Seed != math.MaxUint32 {
		cfgArgs = fmt.Sprintf("%s --seed %d", cfgArgs, cfg.Seed)
	}
	if cfg.BatchSize != 2048 {
		cfgArgs = fmt.Sprintf("%s --batch-size %d", cfgArgs, cfg.BatchSize)
	}
	if cfg.UBatchSize != 512 {
		cfgArgs = fmt.Sprintf("%s --ubatch-size %d", cfgArgs, cfg.UBatchSize)
	}
	if cfg.Jinja {
		cfgArgs = fmt.Sprintf("%s --jinja", cfgArgs)
	}
	if len(cfg.ChatTemplate) > 0 {
		cfgArgs = fmt.Sprintf("%s --chat-template %s", cfgArgs, cfg.ChatTemplate)
	}
	if len(cfg.ChatTemplateFile) > 0 {
		cfgArgs = fmt.Sprintf("%s --chat-template-file %s", cfgArgs, cfg.ChatTemplateFile)
	}
	if len(cfg.ChatTemplateKwargs) > 0 {
		cfgArgs = fmt.Sprintf("%s --chat-template-kwargs %s", cfgArgs, cfg.ChatTemplateKwargs)
	}
	if len(cfg.Pooling) > 0 {
		cfgArgs = fmt.Sprintf("%s --pooling %s", cfgArgs, cfg.Pooling)
	}
	return cfgArgs
}
