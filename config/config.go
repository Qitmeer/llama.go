// Copyright (c) 2017-2025 The qitmeer developers

package config

import (
	"fmt"
	"math"
	"path/filepath"
	"runtime"

	"github.com/ethereum/go-ethereum/log"
	"github.com/urfave/cli/v2"
)

const (
	defaultLogLevel       = "info"
	DefaultGrpcEndpoint   = "localhost:50051"
	DefaultOpenAIEndpoint = "localhost:50052"
	defaultNPredict       = 512
)

var (
	defaultHomeDir     = "."
	defaultSwaggerFile = filepath.Join(defaultHomeDir, "swagger.json")
)

var (
	Conf = &Config{}

	LogLevel = &cli.StringFlag{
		Name:        "log-level",
		Aliases:     []string{"l"},
		Usage:       "Logging level {trace, debug, info, warn, error}",
		Value:       defaultLogLevel,
		Destination: &Conf.LogLevel,
	}

	Model = &cli.StringFlag{
		Name:        "model",
		Aliases:     []string{"m"},
		Usage:       "Specify the path to the LLaMA model file",
		Destination: &Conf.Model,
	}

	CtxSize = &cli.IntFlag{
		Name:        "ctx-size",
		Aliases:     []string{"c"},
		Usage:       "Set the size of the prompt context. The default is 4096, but if a LLaMA model was built with a longer context, increasing this value will provide better results for longer input/inference",
		Value:       4096,
		Destination: &Conf.CtxSize,
	}

	Prompt = &cli.StringFlag{
		Name:        "prompt",
		Aliases:     []string{"p"},
		Usage:       "Provide a prompt directly as a command-line option.",
		Destination: &Conf.Prompt,
	}

	NGpuLayers = &cli.IntFlag{
		Name:        "n-gpu-layers",
		Aliases:     []string{"ngl"},
		Usage:       "When compiled with GPU support, this option allows offloading some layers to the GPU for computation. Generally results in increased performance.",
		Value:       -1,
		Destination: &Conf.NGpuLayers,
	}

	NPredict = &cli.IntFlag{
		Name:        "n-predict",
		Aliases:     []string{"n"},
		Usage:       "Set the number of tokens to predict when generating text. Adjusting this value can influence the length of the generated text.",
		Value:       defaultNPredict,
		Destination: &Conf.NPredict,
	}

	Interactive = &cli.BoolFlag{
		Name:        "interactive",
		Aliases:     []string{"i"},
		Usage:       "Run the program in interactive mode, allowing you to provide input directly and receive real-time responses",
		Value:       false,
		Destination: &Conf.Interactive,
	}

	Seed = &cli.UintFlag{
		Name:        "seed",
		Aliases:     []string{"s"},
		Usage:       "Set the random number generator (RNG) seed (default: -1, -1 = random seed).",
		Value:       math.MaxUint32,
		Destination: &Conf.Seed,
	}

	Pooling = &cli.StringFlag{
		Name:        "pooling",
		Aliases:     []string{"o"},
		Usage:       "pooling type for embeddings, use model default if unspecified {none,mean,cls,last,rank}",
		Value:       "mean",
		Destination: &Conf.Pooling,
	}

	EmbdNormalize = &cli.IntFlag{
		Name:        "embd-normalize",
		Aliases:     []string{"N"},
		Usage:       "normalisation for embeddings (default: %d) (-1=none, 0=max absolute int16, 1=taxicab, 2=euclidean, >2=p-norm)",
		Value:       2,
		Destination: &Conf.EmbdNormalize,
	}

	EmbdOutputFormat = &cli.StringFlag{
		Name:        "embd-output-format",
		Aliases:     []string{"FORMAT"},
		Usage:       "empty = default, \"array\" = [[],[]...], \"json\" = openai style, \"json+\" = same \"json\" + cosine similarity matrix",
		Value:       "json",
		Destination: &Conf.EmbdOutputFormat,
	}

	EmbdSeparator = &cli.StringFlag{
		Name:        "embd-separator",
		Aliases:     []string{"STRING"},
		Usage:       "separator of embeddings (default \\n) for example \"<#sep#>\\",
		Value:       "\n",
		Destination: &Conf.EmbdSeparator,
	}

	BatchSize = &cli.IntFlag{
		Name:        "batch-size",
		Aliases:     []string{"b"},
		Usage:       "logical maximum batch size",
		Value:       2048,
		Destination: &Conf.BatchSize,
	}

	UBatchSize = &cli.IntFlag{
		Name:        "ubatch-size",
		Aliases:     []string{"ub"},
		Usage:       "physical maximum batch size",
		Value:       512,
		Destination: &Conf.UBatchSize,
	}

	OutputFile = &cli.StringFlag{
		Name:        "output-file",
		Aliases:     []string{"of"},
		Usage:       "output file",
		Destination: &Conf.OutputFile,
	}

	AppFlags = []cli.Flag{
		LogLevel,
		Model,
		CtxSize,
		Prompt,
		NGpuLayers,
		NPredict,
		Interactive,
		Seed,
		Pooling,
		EmbdNormalize,
		EmbdOutputFormat,
		EmbdSeparator,
		BatchSize,
		UBatchSize,
		OutputFile,
	}
)

type Config struct {
	LogLevel         string
	Model            string
	CtxSize          int
	Prompt           string
	NGpuLayers       int
	NPredict         int
	Interactive      bool
	Seed             uint
	Pooling          string
	EmbdNormalize    int
	EmbdOutputFormat string
	EmbdSeparator    string
	BatchSize        int
	UBatchSize       int
	OutputFile       string
}

func (c *Config) Load() error {
	log.Debug("Try to load config")
	if len(c.Model) <= 0 {
		return fmt.Errorf("No config model")
	}
	return nil
}

func (c *Config) IsLonely() bool {
	return len(c.Prompt) > 0 || c.Interactive
}

func defaultNGpuLayers() int {
	switch runtime.GOOS {
	case "darwin":
		return -1
	}
	return 0
}
