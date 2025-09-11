package wrapper

/*
#include <stdlib.h>
#include "core.h"
*/
import "C"

import (
	"fmt"
	"sync"
	"unsafe"

	"github.com/Qitmeer/llama.go/config"
)

var (
	mu         sync.Mutex
	channels   = make(map[int]chan any)
	nextChanID = 1
)

func LlamaInteractive(cfg *config.Config) error {
	if !cfg.Interactive {
		return fmt.Errorf("Not config interactive")
	}
	if !cfg.HasModel() {
		return fmt.Errorf("No model")
	}
	ip := C.CString(cfg.Prompt)
	defer C.free(unsafe.Pointer(ip))

	cfgArgs := fmt.Sprintf("llama -i --model %s --ctx-size %d --n-gpu-layers %d --n-predict %d --seed %d",
		cfg.ModelPath(), cfg.CtxSize, cfg.NGpuLayers, cfg.NPredict, cfg.Seed)
	ca := C.CString(cfgArgs)
	defer C.free(unsafe.Pointer(ca))

	ret := C.llama_interactive(ca, ip)
	if !bool(ret) {
		return fmt.Errorf("Llama interactive error")
	}
	return nil
}

func LlamaGenerate(id int, jsStr string) error {
	if len(jsStr) <= 0 {
		return fmt.Errorf("json string")
	}
	fmt.Println(jsStr)
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
	cfgArgs := fmt.Sprintf("llama --model %s --ctx-size %d --n-gpu-layers %d --n-predict %d --seed %d",
		cfg.ModelPath(), cfg.CtxSize, cfg.NGpuLayers, cfg.NPredict, cfg.Seed)
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

func LlamaEmbedding(cfg *config.Config, model string, prompts string, embdOutputFormat string) (string, error) {
	if len(model) <= 0 {
		return "", fmt.Errorf("No model")
	}
	if len(prompts) <= 0 {
		return "", fmt.Errorf("No prompt")
	}
	ip := C.CString(prompts)
	defer C.free(unsafe.Pointer(ip))

	cfgArgs := fmt.Sprintf("llama --model %s --ctx-size %d --n-gpu-layers %d --n-predict %d --seed %d --embd-normalize %d --batch-size %d --ubatch-size %d",
		model, cfg.CtxSize, cfg.NGpuLayers, cfg.NPredict, cfg.Seed, cfg.EmbdNormalize, cfg.BatchSize, cfg.UBatchSize)
	if len(cfg.Pooling) > 0 {
		cfgArgs = fmt.Sprintf("%s --pooling %s", cfgArgs, cfg.Pooling)
	}
	if len(embdOutputFormat) > 0 {
		cfgArgs = fmt.Sprintf("%s --embd-output-format %s", cfgArgs, embdOutputFormat)
	}
	if len(cfg.EmbdSeparator) > 0 {
		cfgArgs = fmt.Sprintf("%s --embd-separator %s", cfgArgs, cfg.EmbdSeparator)
	}
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
