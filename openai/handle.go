package openai

import (
	"errors"
	"io"
	"log"
	"math"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/Qitmeer/llama.go/config"
	"github.com/Qitmeer/llama.go/wrapper"
	"github.com/devalexandre/langsmithgo"
	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
)

func (s *Service) GenerateHandler(c *gin.Context) {
	var req GenerateRequest
	if err := c.ShouldBindJSON(&req); errors.Is(err, io.EOF) {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "missing request body"})
		return
	} else if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// load the model
	if req.Prompt == "" {
		c.JSON(http.StatusOK, GenerateResponse{
			Model:      req.Model,
			CreatedAt:  time.Now().UTC(),
			Done:       true,
			DoneReason: "load",
		})
		return
	}
	resp, err := wrapper.LlamaProcess(req.Prompt)
	if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	var r GenerateResponse
	r.Thinking = ""
	r.Response = resp

	c.JSON(http.StatusOK, r)
	return
}

func (s *Service) EmbedHandler(c *gin.Context) {
	checkpointStart := time.Now()
	var req EmbedRequest
	err := c.ShouldBindJSON(&req)
	switch {
	case errors.Is(err, io.EOF):
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "missing request body"})
		return
	case err != nil:
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	var input []string

	switch i := req.Input.(type) {
	case string:
		if len(i) > 0 {
			input = append(input, i)
		}
	case []any:
		for _, v := range i {
			if _, ok := v.(string); !ok {
				c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "invalid input type"})
				return
			}
			input = append(input, v.(string))
		}
	default:
		if req.Input != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "invalid input type"})
			return
		}
	}
	cfg := &config.Config{
		Prompt: req.Input.(string),
	}
	embeddingstr, err := wrapper.LlamaEmbedding(cfg)
	if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	embedding := []float32{}
	arr := strings.Split(embeddingstr, ",")
	for _, v := range arr {
		f, _ := strconv.ParseFloat(v, 10)
		embedding = append(embedding, float32(f))
	}
	checkpointLoaded := time.Now()
	resp := EmbedResponse{
		Model:           req.Model,
		Embeddings:      [][]float32{embedding},
		TotalDuration:   time.Since(checkpointStart),
		LoadDuration:    checkpointLoaded.Sub(checkpointStart),
		PromptEvalCount: 1,
	}
	c.JSON(http.StatusOK, resp)
}

func normalize(vec []float32) []float32 {
	var sum float32
	for _, v := range vec {
		sum += v * v
	}

	norm := float32(0.0)
	if sum > 0 {
		norm = float32(1.0 / math.Sqrt(float64(sum)))
	}

	for i := range vec {
		vec[i] *= norm
	}
	return vec
}

func (s *Service) EmbeddingsHandler(c *gin.Context) {
	var req EmbeddingRequest
	if err := c.ShouldBindJSON(&req); errors.Is(err, io.EOF) {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "missing request body"})
		return
	} else if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// an empty request loads the model
	if req.Prompt == "" {
		c.JSON(http.StatusOK, EmbeddingResponse{Embedding: []float64{}})
		return
	}
	cfg := &config.Config{
		Prompt: req.Prompt,
	}
	embedding, err := wrapper.LlamaEmbedding(cfg)
	if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	var e []float64
	for _, v := range embedding {
		e = append(e, float64(v))
	}

	resp := EmbeddingResponse{
		Embedding: e,
	}
	c.JSON(http.StatusOK, resp)
}
func filterThinkTags(msgs []ReqMessage, model string) []ReqMessage {
	if strings.Contains(strings.ToLower(model), "qwen3") || strings.Contains(strings.ToLower(model), "deepseek-r1") {
		finalUserIndex := -1
		for i, msg := range msgs {
			if msg.Role == "user" {
				finalUserIndex = i
			}
		}

		for i, msg := range msgs {
			if msg.Role == "assistant" && i < finalUserIndex {
				// TODO(drifkin): this is from before we added proper thinking support.
				// However, even if thinking is not enabled (and therefore we shouldn't
				// change the user output), we should probably perform this filtering
				// for all thinking models (not just qwen3 & deepseek-r1) since it tends
				// to save tokens and improve quality.
				thinkingState := &Parser{
					OpeningTag: "<think>",
					ClosingTag: "</think>",
				}
				_, content := thinkingState.AddContent(msg.Content)
				msgs[i].Content = content
			}
		}
	}
	return msgs
}

func (s *Service) ChatHandler(c *gin.Context) {
	os.Setenv("LANGSMITH_API_KEY", "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")
	os.Setenv("LANGSMITH_PROJECT_NAME", "MyProject")
	smith, err := langsmithgo.NewClient()
	if err != nil {
		log.Fatalf("langsmithgo.NewClient error: %v", err)
	}

	var req ChatRequest
	if err := c.ShouldBindJSON(&req); errors.Is(err, io.EOF) {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "missing request body"})
		return
	} else if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	if len(req.Messages) == 0 {
		c.JSON(http.StatusOK, ChatResponse{
			Model:      req.Model,
			CreatedAt:  time.Now().UTC(),
			Message:    ReqMessage{Role: "assistant"},
			Done:       true,
			DoneReason: "load",
		})
		return
	}
	prompt := ""
	msgs := req.Messages
	if req.Messages[0].Role != "system" {
		msgs = append([]ReqMessage{{Role: "system", Content: "Qitmeer AI Agent"}}, msgs...)
	}
	msgs = filterThinkTags(msgs, req.Model)
	for _, v := range msgs {
		prompt += v.Role + ":" + v.Content + "."
	}
	runID := uuid.New().String()
	err = smith.Run(&langsmithgo.RunPayload{
		RunID:   runID,
		Name:    "Qitmeer-Chain",
		RunType: langsmithgo.Chain,
		Inputs: map[string]interface{}{
			"prompt": prompt,
		},
	})
	if err != nil {
		log.Fatalf("smith.Run error: %v", err)
	}
	log.Println(prompt, "\n------------------------------------------------\n", req.Messages, "\n------------------------------------------------")
	var resp ChatResponse
	res, err := wrapper.LlamaProcess(prompt)
	if err != nil {
		c.JSON(http.StatusBadGateway, gin.H{"error": err.Error()})
		return
	}
	err = smith.PatchRun(runID, &langsmithgo.RunPayload{
		Outputs: map[string]interface{}{
			"output": res,
		},
	})
	if err != nil {
		log.Fatalf("smith.PatchRun error: %v", err)
	}
	resp.Message.Content = res
	resp.Message.Thinking = ""
	c.JSON(http.StatusOK, resp)
	return
}
func (s *Service) ListHandler(c *gin.Context) {
	// tag should never be masked
	models := make([]ListModelResponse, 0)
	models = append(models, ListModelResponse{
		Model:      "test",
		Name:       "test",
		Size:       1000,
		Digest:     "",
		ModifiedAt: time.Now(),
		Details:    ModelDetails{},
	})

	c.JSON(http.StatusOK, ListResponse{Models: models})
}
func (s *Service) ShowHandler(c *gin.Context) {
	var req ShowRequest
	err := c.ShouldBindJSON(&req)
	switch {
	case errors.Is(err, io.EOF):
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "missing request body"})
		return
	case err != nil:
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	if req.Model != "" {
		// noop
	} else if req.Name != "" {
		req.Model = req.Name
	} else {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "model is required"})
		return
	}

	c.JSON(http.StatusOK, map[string]interface{}{})
}
