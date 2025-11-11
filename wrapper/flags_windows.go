//go:build windows
// +build windows

package wrapper

/*
#cgo CFLAGS: -std=c11 -D__USE_MINGW_ANSI_STDIO=0
#cgo CXXFLAGS: -std=c++17 -D__USE_MINGW_ANSI_STDIO=0
#cgo CFLAGS: -I${SRCDIR}/../core/include
#cgo CXXFLAGS: -I${SRCDIR}/../core/include
#cgo LDFLAGS: -L${SRCDIR}/../build/lib -Wl,--start-group -lllama_core -lcommon -lllama -lwhisper -lwhisper-common -lmtmd -l:ggml.a -l:ggml-base.a -l:ggml-cpu.a -lstdc++ -lws2_32 -lwinpthread -lpthread -lmingwex -lmingw32 -Wl,--end-group
*/
import "C"
