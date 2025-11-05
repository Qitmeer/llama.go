//go:build linux
// +build linux

package wrapper

import (
	"fmt"
	"io"
	"os"
	"syscall"
	"unsafe"
)

// MmapModel maps a model file into memory using mmap on Linux
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

	// Get file descriptor
	fd := int(file.Fd())

	// Perform mmap with proper alignment
	pageSize := syscall.Getpagesize()
	offset := int64(0)

	// Ensure offset is page-aligned
	pageAlignedOffset := (offset / int64(pageSize)) * int64(pageSize)
	adjustment := int(offset - pageAlignedOffset)
	mapSize := size + adjustment

	// Map the file into memory
	// Use PROT_READ for read-only access, MAP_PRIVATE for copy-on-write semantics
	mappedData, err := syscall.Mmap(fd, pageAlignedOffset, mapSize,
		syscall.PROT_READ, syscall.MAP_PRIVATE)
	if err != nil {
		return 0, nil, fmt.Errorf("mmap failed: %w", err)
	}

	// Adjust for any offset within the page
	modelData := mappedData[adjustment:]

	// Get the address of the mapped region
	dataAddr := uintptr(unsafe.Pointer(&modelData[0]))

	// Verify GGUF magic number
	if len(modelData) < 4 {
		syscall.Munmap(mappedData)
		return 0, nil, fmt.Errorf("model file too small to be valid GGUF")
	}

	// Check for GGUF magic: 'G' 'G' 'U' 'F' (0x46554747 in little-endian)
	if modelData[0] != 'G' || modelData[1] != 'G' || modelData[2] != 'U' || modelData[3] != 'F' {
		syscall.Munmap(mappedData)
		return 0, nil, fmt.Errorf("invalid GGUF magic number: got %x %x %x %x",
			modelData[0], modelData[1], modelData[2], modelData[3])
	}

	return dataAddr, mappedData, nil
}

// UnmapModel unmaps a previously memory-mapped model
func UnmapModel(data []byte) error {
	if len(data) == 0 {
		return nil
	}

	err := syscall.Munmap(data)
	if err != nil {
		return fmt.Errorf("munmap failed: %w", err)
	}

	return nil
}

// MmapModelAtOffset maps a model file at a specific offset
func MmapModelAtOffset(fd int, offset int64, size int) (addr uintptr, data []byte, err error) {
	if size <= 0 {
		return 0, nil, fmt.Errorf("invalid size: %d", size)
	}

	// Get page size for alignment
	pageSize := int64(syscall.Getpagesize())

	// Calculate page-aligned offset
	pageAlignedOffset := (offset / pageSize) * pageSize
	adjustment := int(offset - pageAlignedOffset)
	mapSize := size + adjustment

	// Perform the mmap
	mappedData, err := syscall.Mmap(fd, pageAlignedOffset, mapSize,
		syscall.PROT_READ, syscall.MAP_PRIVATE)
	if err != nil {
		return 0, nil, fmt.Errorf("mmap at offset failed: %w", err)
	}

	// Adjust to the actual model data start
	modelData := mappedData[adjustment:]
	dataAddr := uintptr(unsafe.Pointer(&modelData[0]))

	// Return modelData, not mappedData
	return dataAddr, modelData, nil
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
