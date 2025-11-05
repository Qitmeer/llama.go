package main

import (
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"runtime"
	"strings"

	"github.com/Qitmeer/llama.go/wrapper"
)

// checkForEmbeddedModel checks if the current executable has an embedded model
func checkForEmbeddedModel() (offset int64, size int) {
	executable, err := os.Executable()
	if err != nil {
		return 0, 0
	}

	file, err := os.Open(executable)
	if err != nil {
		return 0, 0
	}
	defer file.Close()

	stat, err := file.Stat()
	if err != nil {
		return 0, 0
	}

	fileSize := stat.Size()
	if fileSize < 16 {
		return 0, 0
	}

	// Read the last 16 bytes (offset + size)
	_, err = file.Seek(-16, 2)
	if err != nil {
		return 0, 0
	}

	metadata := make([]byte, 16)
	n, err := file.Read(metadata)
	if err != nil || n != 16 {
		return 0, 0
	}

	// Read offset and size (little-endian)
	modelOffset := int64(binary.LittleEndian.Uint64(metadata[0:8]))
	modelSize := int64(binary.LittleEndian.Uint64(metadata[8:16]))

	// Sanity check
	expectedSize := modelOffset + modelSize + 16
	if modelOffset <= 0 || modelSize <= 0 || expectedSize != fileSize {
		return 0, 0
	}

	return modelOffset, int(modelSize)
}

func main() {
	runtime.GOMAXPROCS(runtime.NumCPU())

	var modelPath string
	var prompt string
	var useMemory bool
	var useMmap bool

	flag.StringVar(&modelPath, "model", "", "Path to the GGUF model file (optional if embedded)")
	flag.StringVar(&prompt, "p", "Hello! How can I help you today?", "Prompt to execute")
	flag.BoolVar(&useMemory, "memory", false, "Load model from memory buffer")
	flag.BoolVar(&useMmap, "mmap", true, "Use memory-mapped file loading (default)")
	flag.Parse()

	// If memory is explicitly requested, disable mmap
	if useMemory {
		useMmap = false
	}

	// Check for embedded model first
	embeddedOffset, embeddedSize := checkForEmbeddedModel()
	hasEmbedded := embeddedOffset > 0 && embeddedSize > 0

	if hasEmbedded {
		fmt.Printf("Found embedded model at offset %d, size %.2f MB\n", embeddedOffset, float64(embeddedSize)/(1024*1024))
	} else if modelPath == "" {
		fmt.Println("Error: No embedded model found and no -model flag specified")
		fmt.Println("Usage: modelembed [-model <path>] [-memory] [-mmap] -p <prompt>")
		os.Exit(1)
	}

	fmt.Println("=== Memory Loading Example ===")
	fmt.Printf("System: %s/%s\n", runtime.GOOS, runtime.GOARCH)
	fmt.Printf("CPU cores: %d\n\n", runtime.NumCPU())

	// Load model
	var args string
	if hasEmbedded {
		fmt.Println("Loading embedded model...")
		addr, data, err := wrapper.LoadSelfContainedModel(embeddedOffset, embeddedSize)
		if err != nil {
			log.Fatalf("Failed to load embedded model: %v", err)
		}

		// Use dummy path - file doesn't need to exist with patched arg.cpp
		const maxMmapSize = 2 * 1024 * 1024 * 1024 // 2GB
		const dummyPath = "embedded-model.gguf"

		if embeddedSize > maxMmapSize {
			fmt.Printf("✓ Model loaded into memory: %.2f GB\n", float64(embeddedSize)/(1024*1024*1024))
			args = fmt.Sprintf("llama --model %s --log-disable", dummyPath)
			err = wrapper.LoadFromMemory(data[:embeddedSize], args)
			if err != nil {
				log.Fatalf("Failed to load from memory: %v", err)
			}
		} else {
			defer wrapper.UnmapModel(data)
			fmt.Printf("✓ Model mapped at address: 0x%x, size: %.2f MB\n", addr, float64(embeddedSize)/(1024*1024))
			args = fmt.Sprintf("llama --model %s --log-disable", dummyPath)
			err = wrapper.LoadFromMmap(addr, data, args)
			if err != nil {
				log.Fatalf("Failed to load from mmap: %v", err)
			}
		}
	} else if useMmap {
		fmt.Printf("Loading model using mmap: %s\n", modelPath)
		addr, data, err := wrapper.MmapModel(modelPath)
		if err != nil {
			log.Fatalf("Failed to mmap model: %v", err)
		}
		defer wrapper.UnmapModel(data)

		fmt.Printf("✓ Model mapped at address: 0x%x, size: %.2f MB\n", addr, float64(len(data))/(1024*1024))
		args = fmt.Sprintf("llama --model %s --log-disable", modelPath)
		err = wrapper.LoadFromMmap(addr, data, args)
		if err != nil {
			log.Fatalf("Failed to load from mmap: %v", err)
		}
	} else {
		fmt.Printf("Loading model into memory: %s\n", modelPath)
		modelData, err := os.ReadFile(modelPath)
		if err != nil {
			log.Fatalf("Failed to read model file: %v", err)
		}

		fmt.Printf("✓ Model size: %.2f MB\n", float64(len(modelData))/(1024*1024))

		// Verify GGUF magic
		if len(modelData) < 4 || modelData[0] != 'G' || modelData[1] != 'G' ||
			modelData[2] != 'U' || modelData[3] != 'F' {
			log.Fatal("Invalid GGUF file format")
		}
		fmt.Println("✓ GGUF magic verified")

		args = fmt.Sprintf("llama --model %s --log-disable", modelPath)
		err = wrapper.LoadFromMemory(modelData, args)
		if err != nil {
			log.Fatalf("Failed to load from memory: %v", err)
		}
	}

	fmt.Println("\n✓ Model loaded successfully!")

	// Execute inference
	if prompt != "" {
		fmt.Printf("\nPrompt: %s\n", prompt)
		fmt.Println("───────────────────────────────")
		executeInference(prompt)
	}
}

// executeInference performs inference using the loaded model
func executeInference(prompt string) {
	// Create a channel to receive responses
	id, ch := wrapper.NewChan()
	if ch == nil {
		log.Fatal("Failed to create response channel")
	}

	// Prepare OpenAI-compatible JSON request
	jsonReq := fmt.Sprintf(`{"prompt":%q,"n_predict":128,"temperature":0.7,"top_k":40,"top_p":0.9,"stream":true}`, prompt)

	// Start inference in a goroutine
	go func() {
		err := wrapper.LlamaGenerate(id, jsonReq)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error during inference: %v\n", err)
		}
	}()

	// Parse and display response
	fmt.Print("Response: ")
	var fullResponse strings.Builder
	for content := range ch {
		if str, ok := content.(string); ok {
			// Parse SSE format: "data: {json}\n\n"
			if strings.HasPrefix(str, "data: ") {
				jsonData := strings.TrimPrefix(str, "data: ")
				jsonData = strings.TrimSpace(jsonData)

				// Skip [DONE] message
				if jsonData == "[DONE]" {
					continue
				}

				// Parse JSON to extract text
				var response struct {
					Choices []struct {
						Text string `json:"text"`
					} `json:"choices"`
				}

				if err := json.Unmarshal([]byte(jsonData), &response); err == nil {
					if len(response.Choices) > 0 {
						text := response.Choices[0].Text
						fullResponse.WriteString(text)
						fmt.Print(text)
					}
				}
			}
		}
	}

	fmt.Printf("\n───────────────────────────────\n")
	fmt.Printf("Generated %d characters\n", fullResponse.Len())
}
