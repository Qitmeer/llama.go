package routes

import (
	"github.com/Qitmeer/llama.go/config"
	"github.com/Qitmeer/llama.go/version"
	"github.com/ethereum/go-ethereum/log"
	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/openai"
	"github.com/ollama/ollama/template"
	"net/http"
)

type API struct {
	cfg *config.Config

	tmpl *template.Template
}

func New(cfg *config.Config) *API {
	log.Info("New API ...")
	ser := API{cfg: cfg}
	return &ser
}

func (s *API) Start() error {
	tmpl, err := template.Parse("{{- range .Messages }}<|im_start|>{{ .Role }}\n{{ .Content }}<|im_end|>\n{{ end }}<|im_start|>assistant")
	if err != nil {
		log.Error(err.Error())
		return err
	}
	s.tmpl = tmpl

	return nil
}

func (s *API) Setup(r *gin.Engine) {
	// General
	r.HEAD("/api/version", func(c *gin.Context) { c.JSON(http.StatusOK, gin.H{"version": version.String()}) })
	r.GET("/api/version", func(c *gin.Context) { c.JSON(http.StatusOK, gin.H{"version": version.String()}) })

	r.HEAD("/api/tags", s.ListHandler)
	r.GET("/api/tags", s.ListHandler)
	r.HEAD("/api/models", s.ListHandler)
	r.GET("/api/models", s.ListHandler)
	r.POST("/api/show", s.ShowHandler)
	r.GET("/api/ps", s.PsHandler)
	r.GET("/api/props", s.PropsHandler)
	r.GET("/props", s.PropsHandler)

	r.POST("/api/generate", s.GenerateHandler)
	r.POST("/api/chat", s.ChatHandler)
	r.POST("/api/embed", s.EmbedHandler)
	r.POST("/api/embeddings", s.EmbeddingsHandler)

	// Inference (OpenAI compatibility)
	//r.POST("/v1/chat/completions", openai.ChatMiddleware(), s.ChatHandler)
	//r.POST("/v1/completions", openai.CompletionsMiddleware(), s.GenerateHandler)
	r.POST("/v1/completions", s.GenerateHandler)
	r.POST("/v1/chat/completions", s.ChatHandler)

	r.POST("/v1/embeddings", openai.EmbeddingsMiddleware(), s.EmbedHandler)
	r.GET("/v1/models", openai.ListMiddleware(), s.ListHandler)
	r.GET("/v1/models/:model", openai.RetrieveMiddleware(), s.ShowHandler)

	// webui index
	r.GET("/", s.IndexHandler)
	r.HEAD("/", s.IndexHandler)
	r.GET("/index.html", s.IndexHandler)
	r.HEAD("/index.html", s.IndexHandler)
}
