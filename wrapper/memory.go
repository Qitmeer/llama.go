package wrapper

/*
#include "../core/include/process.h"
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"runtime"
	"unsafe"
)

// LoadFromMemory loads a model from a memory buffer
func LoadFromMemory(modelData []byte, args string) error {
	if len(modelData) == 0 {
		return fmt.Errorf("empty model data")
	}

	// Pin memory to prevent GC from moving it
	var pin runtime.Pinner
	pin.Pin(&modelData[0])
	defer pin.Unpin()

	// Build configuration arguments
	cargs := C.CString(args)
	defer C.free(unsafe.Pointer(cargs))

	// Call the C function to load from memory
	ret := C.llama_start_from_memory(
		unsafe.Pointer(&modelData[0]),
		C.size_t(len(modelData)),
		cargs,
	)

	if !ret {
		return fmt.Errorf("failed to load model from memory")
	}

	return nil
}

// LoadFromMmap loads a model from memory-mapped data
func LoadFromMmap(addr uintptr, data []byte, args string) error {
	if len(data) == 0 {
		return fmt.Errorf("empty mmap data")
	}

	// Build configuration arguments
	cargs := C.CString(args)
	defer C.free(unsafe.Pointer(cargs))

	// Get pointer from data slice (safer than converting uintptr)
	// This avoids govet warning about unsafe.Pointer(uintptr)
	dataPtr := unsafe.Pointer(&data[0])

	// Call the C function to load from mmap
	ret := C.llama_start_from_mmap(
		dataPtr,
		C.size_t(len(data)),
		cargs,
	)

	if !ret {
		return fmt.Errorf("failed to load model from mmap")
	}

	return nil
}
