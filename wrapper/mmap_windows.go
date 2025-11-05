//go:build windows
// +build windows

package wrapper

import (
	"fmt"
	"io"
	"os"
	"syscall"
	"unsafe"
)

var (
	kernel32              = syscall.NewLazyDLL("kernel32.dll")
	procCreateFileMapping = kernel32.NewProc("CreateFileMappingW")
	procMapViewOfFile     = kernel32.NewProc("MapViewOfFile")
	procUnmapViewOfFile   = kernel32.NewProc("UnmapViewOfFile")
)

const (
	FILE_MAP_READ  = 0x0004
	PAGE_READONLY  = 0x02
	INVALID_HANDLE = ^uintptr(0)
)

// MmapModel maps a model file into memory using Windows memory mapping
func MmapModel(path string) (addr uintptr, data []byte, err error) {
	// Open the model file
	file, err := os.Open(path)
	if err != nil {
		return 0, nil, fmt.Errorf("failed to open model file: %w", err)
	}
	defer func() {
		if err != nil {
			file.Close()
		}
	}()

	// Get file statistics to determine size
	stat, err := file.Stat()
	if err != nil {
		return 0, nil, fmt.Errorf("failed to stat model file: %w", err)
	}

	size := int(stat.Size())
	if size == 0 {
		return 0, nil, fmt.Errorf("model file is empty")
	}

	// Get Windows handle from file
	handle := syscall.Handle(file.Fd())

	// Create file mapping
	high := uint32(size >> 32)
	low := uint32(size & 0xffffffff)

	mapHandle, _, err := procCreateFileMapping.Call(
		uintptr(handle),
		0, // No security attributes
		PAGE_READONLY,
		uintptr(high),
		uintptr(low),
		0, // No name
	)

	if mapHandle == 0 || mapHandle == INVALID_HANDLE {
		return 0, nil, fmt.Errorf("CreateFileMapping failed: %w", err)
	}
	defer syscall.CloseHandle(syscall.Handle(mapHandle))

	// Map view of file
	mappedAddr, _, err := procMapViewOfFile.Call(
		mapHandle,
		FILE_MAP_READ,
		0, // Offset high
		0, // Offset low
		uintptr(size),
	)

	if mappedAddr == 0 {
		return 0, nil, fmt.Errorf("MapViewOfFile failed: %w", err)
	}

	// Create a byte slice from the mapped memory using unsafe.Slice (Go 1.17+)
	// This is the modern, safe way to create a slice from a pointer
	mappedData := unsafe.Slice((*byte)(unsafe.Pointer(mappedAddr)), size)

	// Verify GGUF magic number
	if len(mappedData) < 4 {
		procUnmapViewOfFile.Call(mappedAddr)
		return 0, nil, fmt.Errorf("model file too small to be valid GGUF")
	}

	// Check for GGUF magic: 'G' 'G' 'U' 'F'
	if mappedData[0] != 'G' || mappedData[1] != 'G' || mappedData[2] != 'U' || mappedData[3] != 'F' {
		procUnmapViewOfFile.Call(mappedAddr)
		return 0, nil, fmt.Errorf("invalid GGUF magic number: got %x %x %x %x",
			mappedData[0], mappedData[1], mappedData[2], mappedData[3])
	}

	return mappedAddr, mappedData, nil
}

// UnmapModel unmaps a previously memory-mapped model
func UnmapModel(data []byte) error {
	if len(data) == 0 {
		return nil
	}

	addr := uintptr(unsafe.Pointer(&data[0]))
	ret, _, err := procUnmapViewOfFile.Call(addr)

	if ret == 0 {
		return fmt.Errorf("UnmapViewOfFile failed: %w", err)
	}

	return nil
}

// MmapModelAtOffset maps a model file at a specific offset
func MmapModelAtOffset(fd int, offset int64, size int) (addr uintptr, data []byte, err error) {
	if size <= 0 {
		return 0, nil, fmt.Errorf("invalid size: %d", size)
	}

	// Get Windows handle
	handle := syscall.Handle(fd)

	// Get system allocation granularity (usually 64KB on Windows)
	// For simplicity, use 65536 (64KB) as it's the standard on Windows
	const allocationGranularity = 65536
	
	// Calculate aligned offset
	alignedOffset := (offset / allocationGranularity) * allocationGranularity
	adjustment := int(offset - alignedOffset)
	mapSize := size + adjustment

	// Calculate file mapping parameters
	totalSize := alignedOffset + int64(mapSize)
	high := uint32(totalSize >> 32)
	low := uint32(totalSize & 0xffffffff)

	// Create file mapping for the entire needed range
	mapHandle, _, err := procCreateFileMapping.Call(
		uintptr(handle),
		0, // No security attributes
		PAGE_READONLY,
		uintptr(high),
		uintptr(low),
		0, // No name
	)

	if mapHandle == 0 || mapHandle == INVALID_HANDLE {
		return 0, nil, fmt.Errorf("CreateFileMapping failed: %w", err)
	}
	defer syscall.CloseHandle(syscall.Handle(mapHandle))

	// Calculate offset for MapViewOfFile (must be aligned)
	offsetHigh := uint32(alignedOffset >> 32)
	offsetLow := uint32(alignedOffset & 0xffffffff)

	// Map view of file at the aligned offset
	mappedAddr, _, err := procMapViewOfFile.Call(
		mapHandle,
		FILE_MAP_READ,
		uintptr(offsetHigh),
		uintptr(offsetLow),
		uintptr(mapSize),
	)

	if mappedAddr == 0 {
		return 0, nil, fmt.Errorf("MapViewOfFile at offset failed: %w", err)
	}

	// Adjust the pointer to the actual model data start
	modelAddr := mappedAddr + uintptr(adjustment)

	// Create a byte slice from the adjusted memory using unsafe.Slice (Go 1.17+)
	modelData := unsafe.Slice((*byte)(unsafe.Pointer(modelAddr)), size)

	// Return the adjusted data
	return modelAddr, modelData, nil
}

// LoadSelfContainedModel loads a model that's embedded in the current executable
func LoadSelfContainedModel(offset int64, size int) (addr uintptr, data []byte, err error) {
	// Open the current executable
	executable, err := os.Executable()
	if err != nil {
		return 0, nil, fmt.Errorf("failed to get executable path: %w", err)
	}

	file, err := os.Open(executable)
	if err != nil {
		return 0, nil, fmt.Errorf("failed to open executable: %w", err)
	}
	defer file.Close()

	// Check if model is too large for mmap (>2GB on some systems)
	const maxMmapSize = 2 * 1024 * 1024 * 1024 // 2GB limit for safety
	if size > maxMmapSize {
		fmt.Printf("Model size %.2f GB exceeds safe mmap limit, using standard memory loading\n",
			float64(size)/(1024*1024*1024))

		// Seek to model start
		_, err = file.Seek(offset, 0)
		if err != nil {
			return 0, nil, fmt.Errorf("failed to seek to model start: %w", err)
		}

		// Read model data - use io.ReadFull for large files
		modelData := make([]byte, size)
		n, err := io.ReadFull(file, modelData)
		if err != nil {
			return 0, nil, fmt.Errorf("failed to read model data: %w", err)
		}
		if n != size {
			return 0, nil, fmt.Errorf("incomplete read: expected %d bytes, got %d", size, n)
		}

		// Verify GGUF header in the loaded data
		if len(modelData) >= 4 {
			if modelData[0] == 'G' && modelData[1] == 'G' && modelData[2] == 'U' && modelData[3] == 'F' {
				fmt.Printf("✅ Valid GGUF header found in loaded data\n")
			} else {
				fmt.Printf("❌ Invalid magic in loaded data: %02x %02x %02x %02x (expected GGUF)\n",
					modelData[0], modelData[1], modelData[2], modelData[3])
			}
		}

		fmt.Printf("Successfully loaded self-contained model into memory\n")

		// Return the address and data (not actually mmap'd)
		dataAddr := uintptr(unsafe.Pointer(&modelData[0]))
		return dataAddr, modelData, nil
	}

	// Try mmap for smaller models
	mappedAddr, mappedData, err := MmapModelAtOffset(int(file.Fd()), offset, size)
	if err != nil {
		// Fallback to standard memory loading if mmap fails
		fmt.Printf("mmap failed (%v), falling back to standard memory loading\n", err)

		// Seek to model start
		_, err = file.Seek(offset, 0)
		if err != nil {
			return 0, nil, fmt.Errorf("failed to seek to model start: %w", err)
		}

		// Read model data - use io.ReadFull for large files
		modelData := make([]byte, size)
		n, err := io.ReadFull(file, modelData)
		if err != nil {
			return 0, nil, fmt.Errorf("failed to read model data: %w", err)
		}
		if n != size {
			return 0, nil, fmt.Errorf("incomplete read: expected %d bytes, got %d", size, n)
		}

		// Verify GGUF header in the loaded data
		if len(modelData) >= 4 {
			if modelData[0] == 'G' && modelData[1] == 'G' && modelData[2] == 'U' && modelData[3] == 'F' {
				fmt.Printf("✅ Valid GGUF header found in loaded data\n")
			} else {
				fmt.Printf("❌ Invalid magic in loaded data: %02x %02x %02x %02x (expected GGUF)\n",
					modelData[0], modelData[1], modelData[2], modelData[3])
			}
		}

		fmt.Printf("Successfully loaded self-contained model into memory\n")

		// Return the address and data (not actually mmap'd)
		dataAddr := uintptr(unsafe.Pointer(&modelData[0]))
		return dataAddr, modelData, nil
	}

	fmt.Printf("Successfully mapped self-contained model at address 0x%x\n", mappedAddr)

	// Debug: Check the first bytes to verify GGUF header
	if len(mappedData) >= 16 {
		fmt.Printf("Debug: First 16 bytes of mapped data: ")
		for i := 0; i < 16 && i < len(mappedData); i++ {
			fmt.Printf("%02x ", mappedData[i])
		}
		fmt.Println()

		// Check for GGUF magic
		if len(mappedData) >= 4 {
			if mappedData[0] == 'G' && mappedData[1] == 'G' && mappedData[2] == 'U' && mappedData[3] == 'F' {
				fmt.Printf("Debug: ✅ Valid GGUF header found at mapped address\n")
			} else {
				fmt.Printf("Debug: ⚠️ Invalid magic: %02x%02x%02x%02x (expected 47475546 for GGUF)\n",
					mappedData[0], mappedData[1], mappedData[2], mappedData[3])
			}
		}
	}

	return mappedAddr, mappedData, nil
}
