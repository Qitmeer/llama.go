//go:build windows
// +build windows

package wrapper

/*
#cgo CFLAGS: -std=c11
#cgo CXXFLAGS: -std=c++17
#cgo CFLAGS: -I${SRCDIR}/../core/include
#cgo CXXFLAGS: -I${SRCDIR}/../core/include
#cgo LDFLAGS: -L${SRCDIR}/../build/lib -lllama_core -lllama -lcommon -lwhisper -lwhisper-common -lmtmd -l:ggml.a -l:ggml-base.a -l:ggml-cpu.a -lstdc++
*/
import "C"
