package openai

import (
	"errors"
	"fmt"
	"net"
	"net/http"

	"github.com/Qitmeer/llama.go/config"
	"github.com/Qitmeer/llama.go/grpc/cmds"
	"github.com/Qitmeer/llama.go/version"
	"github.com/ethereum/go-ethereum/log"
	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
	"github.com/urfave/cli/v2"
)

type Service struct {
	ctx      *cli.Context
	cfg      *config.Config
	quit     chan struct{}
	srvr     *http.Server
	hw       *cmds.HelloWorld
	generate *cmds.Generate
}

func New(ctx *cli.Context, cfg *config.Config) *Service {
	log.Info("New openai Service...")
	ser := Service{ctx: ctx, cfg: cfg}
	return &ser
}

func (ser *Service) Start() error {
	log.Info("Start openai Service...")

	go ser.Register()
	return nil
}

func (ser *Service) Stop() {
	log.Info("Stop openai Service...")
	ser.srvr.Close()
}

func (ser *Service) Register() {
	log.Info(fmt.Sprintf("Register:%s", config.DefaultOpenAIEndpoint))
	corsConfig := cors.DefaultConfig()
	corsConfig.AllowWildcard = true
	corsConfig.AllowBrowserExtensions = true
	corsConfig.AllowHeaders = []string{
		"Authorization",
		"Content-Type",
		"User-Agent",
		"Accept",
		"X-Requested-With",

		// OpenAI compatibility headers
		"OpenAI-Beta",
		"x-stainless-arch",
		"x-stainless-async",
		"x-stainless-custom-poll-interval",
		"x-stainless-helper-method",
		"x-stainless-lang",
		"x-stainless-os",
		"x-stainless-package-version",
		"x-stainless-poll-helper",
		"x-stainless-retry-count",
		"x-stainless-runtime",
		"x-stainless-runtime-version",
		"x-stainless-timeout",
	}
	// corsConfig.AllowOrigins = envconfig.AllowedOrigins()

	r := gin.Default()
	r.HandleMethodNotAllowed = true

	// General
	r.HEAD("/", func(c *gin.Context) { c.String(http.StatusOK, "llama.go is running") })
	r.GET("/", func(c *gin.Context) { c.String(http.StatusOK, "llama.go is running") })
	r.HEAD("/api/version", func(c *gin.Context) { c.JSON(http.StatusOK, gin.H{"version": version.String}) })
	r.GET("/api/version", func(c *gin.Context) { c.JSON(http.StatusOK, gin.H{"version": version.String}) })

	// Inference (OpenAI compatibility)
	r.POST("/v1/chat/completions", ChatMiddleware(), ser.ChatHandler)
	r.POST("/v1/completions", CompletionsMiddleware(), ser.GenerateHandler)
	r.POST("/v1/embeddings", EmbeddingsMiddleware(), ser.EmbedHandler)
	r.GET("/v1/models", ListMiddleware(), ser.ListHandler)
	r.GET("/v1/models/:model", RetrieveMiddleware(), ser.ShowHandler)

	http.Handle("/", r)
	ser.srvr = &http.Server{
		// Use http.DefaultServeMux so we get net/http/pprof for
		// free.
		//
		// TODO(bmizerany): Decide if we want to make this
		// configurable so it is not exposed by default, or allow
		// users to bind it to a different port. This was a quick
		// and easy way to get pprof, but it may not be the best
		// way.
		Handler: nil,
	}
	listener, err := net.Listen("tcp", config.DefaultOpenAIEndpoint)
	if err != nil {
		log.Error("Listen error", "err", err)
		return
	}

	err = ser.srvr.Serve(listener)
	// If server is closed from the signal handler, wait for the ctx to be done
	// otherwise error out quickly
	if !errors.Is(err, http.ErrServerClosed) {
		log.Error("failed to serve", "err", err)
		return
	}
}
