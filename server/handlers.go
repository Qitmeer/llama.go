package server

import (
	"bytes"
	"cmp"
	"encoding/json"
	"errors"
	"fmt"
	"github.com/Qitmeer/llama.go/wrapper"
	"github.com/ethereum/go-ethereum/log"
	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/template"
	"github.com/ollama/ollama/types/model"
	"io"
	"net/http"
	"slices"
	"strings"
	"time"
)

func (s *Service) PsHandler(c *gin.Context) {
	models := []api.ProcessModelResponse{}
	slices.SortStableFunc(models, func(i, j api.ProcessModelResponse) int {
		// longest duration remaining listed first
		return cmp.Compare(j.ExpiresAt.Unix(), i.ExpiresAt.Unix())
	})

	c.JSON(http.StatusOK, api.ProcessResponse{Models: models})
}

func (s *Service) GenerateHandler(c *gin.Context) {
	checkpointStart := time.Now()
	var req api.GenerateRequest
	if err := c.ShouldBindJSON(&req); errors.Is(err, io.EOF) {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "missing request body"})
		return
	} else if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	if len(req.Model) > 0 {
		if s.cfg.Model != req.Model {
			c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model '%s' not found", req.Model)})
			return
		}
	}

	// expire the runner
	if req.Prompt == "" && req.KeepAlive != nil && int(req.KeepAlive.Seconds()) == 0 {
		c.JSON(http.StatusOK, api.GenerateResponse{
			Model:      req.Model,
			CreatedAt:  time.Now().UTC(),
			Response:   "Not currently supported",
			Done:       true,
			DoneReason: "unload",
		})
		return
	}

	if req.Raw && (req.Template != "" || req.System != "" || len(req.Context) > 0) {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "raw mode does not support template, system, or context"})
		return
	}

	caps := []model.Capability{model.CapabilityCompletion}
	if req.Suffix != "" {
		caps = append(caps, model.CapabilityInsert)
	}
	if req.Think != nil && *req.Think {
		caps = append(caps, model.CapabilityThinking)
		// TODO(drifkin): consider adding a warning if it's false and the model
		// doesn't support thinking. It's not strictly required, but it can be a
		// hint that the user is on an older qwen3/r1 model that doesn't have an
		// updated template supporting thinking
	}

	checkpointLoaded := time.Now()

	// load the model
	if req.Prompt == "" {
		c.JSON(http.StatusOK, api.GenerateResponse{
			Model:      req.Model,
			CreatedAt:  time.Now().UTC(),
			Done:       true,
			DoneReason: "load",
		})
		return
	}

	images := make([]ImageData, len(req.Images))
	for i := range req.Images {
		images[i] = ImageData{ID: i, Data: req.Images[i]}
	}

	prompt := req.Prompt
	if !req.Raw {
		tmpl := s.tmpl
		if req.Template != "" {
			tm, err := template.Parse(req.Template)
			if err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}
			tmpl = tm
		}

		var values template.Values
		if req.Suffix != "" {
			values.Prompt = prompt
			values.Suffix = req.Suffix
		} else {
			var msgs []api.Message
			if req.System != "" {
				msgs = append(msgs, api.Message{Role: "system", Content: req.System})
			}

			for _, i := range images {
				imgPrompt := ""
				msgs = append(msgs, api.Message{Role: "user", Content: fmt.Sprintf("[img-%d]"+imgPrompt, i.ID)})
			}

			values.Messages = append(msgs, api.Message{Role: "user", Content: req.Prompt})
		}

		values.Think = req.Think != nil && *req.Think
		values.IsThinkSet = req.Think != nil

		var b bytes.Buffer
		if err := tmpl.Execute(&b, values); err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		prompt = b.String()
	}
	content, err := wrapper.LlamaGenerate(prompt)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	res := api.GenerateResponse{
		Model:     req.Model,
		CreatedAt: time.Now().UTC(),
		Response:  content,
		Done:      true,
	}
	res.TotalDuration = time.Since(checkpointStart)
	res.LoadDuration = checkpointLoaded.Sub(checkpointStart)
	if req.Stream == nil || !*req.Stream {
		c.JSON(http.StatusOK, res)
		return
	}

	c.Header("Content-Type", "application/x-ndjson")
	c.Stream(func(w io.Writer) bool {
		bts, err := json.Marshal(res)
		if err != nil {
			log.Info(fmt.Sprintf("streamResponse: json.Marshal failed with %s", err))
			return false
		}

		// Delineate chunks with new-line delimiter
		bts = append(bts, '\n')
		if _, err := w.Write(bts); err != nil {
			log.Info(fmt.Sprintf("streamResponse: w.Write failed with %s", err))
			return false
		}
		return true
	})
}

func (s *Service) ChatHandler(c *gin.Context) {
	checkpointStart := time.Now()

	var req api.ChatRequest
	if err := c.ShouldBindJSON(&req); errors.Is(err, io.EOF) {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "missing request body"})
		return
	} else if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// expire the runner
	if len(req.Messages) == 0 && req.KeepAlive != nil && int(req.KeepAlive.Seconds()) == 0 {
		c.JSON(http.StatusOK, api.ChatResponse{
			Model:      req.Model,
			CreatedAt:  time.Now().UTC(),
			Message:    api.Message{Role: "assistant", Content: "Not currently supported"},
			Done:       true,
			DoneReason: "unload",
		})
		return
	}

	caps := []model.Capability{model.CapabilityCompletion}
	if len(req.Tools) > 0 {
		caps = append(caps, model.CapabilityTools)
	}
	if req.Think != nil && *req.Think {
		caps = append(caps, model.CapabilityThinking)
	}

	checkpointLoaded := time.Now()

	if len(req.Messages) == 0 {
		c.JSON(http.StatusOK, api.ChatResponse{
			Model:      req.Model,
			CreatedAt:  time.Now().UTC(),
			Message:    api.Message{Role: "assistant"},
			Done:       true,
			DoneReason: "load",
		})
		return
	}

	content, err := wrapper.LlamaChat(req.Messages)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	res := api.ChatResponse{
		Model:     req.Model,
		CreatedAt: time.Now().UTC(),
		Message:   api.Message{Role: "assistant", Content: content},
		Done:      true,
	}

	res.TotalDuration = time.Since(checkpointStart)
	res.LoadDuration = checkpointLoaded.Sub(checkpointStart)
	if req.Stream == nil && !*req.Stream {
		c.JSON(http.StatusOK, res)
		return
	}

	c.Header("Content-Type", "application/x-ndjson")
	c.Stream(func(w io.Writer) bool {
		bts, err := json.Marshal(res)
		if err != nil {
			log.Info(fmt.Sprintf("streamResponse: json.Marshal failed with %s", err))
			return false
		}

		// Delineate chunks with new-line delimiter
		bts = append(bts, '\n')
		if _, err := w.Write(bts); err != nil {
			log.Info(fmt.Sprintf("streamResponse: w.Write failed with %s", err))
			return false
		}
		return true
	})
}

func (s *Service) EmbedHandler(c *gin.Context) {
	checkpointStart := time.Now()
	var req api.EmbedRequest
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

	checkpointLoaded := time.Now()

	if len(input) == 0 {
		c.JSON(http.StatusOK, api.EmbedResponse{Model: req.Model, Embeddings: [][]float32{}})
		return
	}

	prompts := ""
	for k, i := range input {
		if k > 0 {
			prompts += s.cfg.EmbdSeparator
		}
		prompts += i
	}

	ret, err := wrapper.LlamaEmbedding(s.cfg, s.cfg.Model, prompts, "array")
	if err != nil {
		c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": strings.TrimSpace(err.Error())})
		return
	}
	var embeddings [][]float32
	err = json.Unmarshal([]byte(ret), &embeddings)
	if err != nil {
		c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": strings.TrimSpace(err.Error())})
		return
	}
	if len(embeddings) != len(input) {
		c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("%d != %d", len(embeddings), len(input))})
		return
	}
	resp := api.EmbedResponse{
		Model:           req.Model,
		Embeddings:      embeddings,
		TotalDuration:   time.Since(checkpointStart),
		LoadDuration:    checkpointLoaded.Sub(checkpointStart),
		PromptEvalCount: len(input),
	}
	c.JSON(http.StatusOK, resp)
}

func (s *Service) EmbeddingsHandler(c *gin.Context) {
	var req api.EmbeddingRequest
	if err := c.ShouldBindJSON(&req); errors.Is(err, io.EOF) {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "missing request body"})
		return
	} else if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// an empty request loads the model
	if req.Prompt == "" {
		c.JSON(http.StatusOK, api.EmbeddingResponse{Embedding: []float64{}})
		return
	}

	ret, err := wrapper.LlamaEmbedding(s.cfg, s.cfg.Model, req.Prompt, "array")
	if err != nil {
		c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": strings.TrimSpace(err.Error())})
		return
	}
	var embeddings [][]float64
	err = json.Unmarshal([]byte(ret), &embeddings)
	if err != nil {
		c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": strings.TrimSpace(err.Error())})
		return
	}
	resp := api.EmbeddingResponse{
		Embedding: embeddings[0],
	}
	c.JSON(http.StatusOK, resp)
}

func (s *Service) ListHandler(c *gin.Context) {
	models := []api.ListModelResponse{}

	slices.SortStableFunc(models, func(i, j api.ListModelResponse) int {
		// most recently modified first
		return cmp.Compare(j.ModifiedAt.Unix(), i.ModifiedAt.Unix())
	})
	c.JSON(http.StatusOK, api.ListResponse{Models: models})
}

func (s *Service) ShowHandler(c *gin.Context) {
	var req api.ShowRequest
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
	c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "No support"})
}
