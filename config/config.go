// Copyright (c) 2017-2025 The qitmeer developers

package config

import (
	"fmt"
	"github.com/ethereum/go-ethereum/log"
	"github.com/urfave/cli/v2"
	"math"
	"path/filepath"
	"runtime"
)

const (
	defaultLogLevel     = "info"
	DefaultGrpcEndpoint = "localhost:50051"
	defaultNPredict     = 512
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

	NBatch = &cli.IntFlag{
		Name:        "n-batch",
		Usage:       "Logical batch size for prompt processing",
		Value:       512,
		Destination: &Conf.NBatch,
	}

	NKeep = &cli.IntFlag{
		Name:        "n-keep",
		Usage:       "Number of tokens to keep from initial prompt",
		Value:       0,
		Destination: &Conf.NKeep,
	}

	NParallel = &cli.IntFlag{
		Name:        "n-parallel",
		Usage:       "Number of parallel sequences to decode",
		Value:       1,
		Destination: &Conf.NParallel,
	}

	GrpAttnN = &cli.IntFlag{
		Name:        "grp-attn-n",
		Usage:       "Group-attention factor",
		Value:       1,
		Destination: &Conf.GrpAttnN,
	}

	GrpAttnW = &cli.IntFlag{
		Name:        "grp-attn-w",
		Usage:       "Group-attention width",
		Value:       512,
		Destination: &Conf.GrpAttnW,
	}

	NPrint = &cli.IntFlag{
		Name:        "n-print",
		Usage:       "Print token count every n tokens (-1 = disabled)",
		Value:       -1,
		Destination: &Conf.NPrint,
	}

	RopeFreqBase = &cli.Float64Flag{
		Name:        "rope-freq-base",
		Usage:       "RoPE base frequency",
		Value:       10000.0,
		Destination: &Conf.RopeFreqBase,
	}

	RopeFreqScale = &cli.Float64Flag{
		Name:        "rope-freq-scale",
		Usage:       "RoPE frequency scaling factor",
		Value:       1.0,
		Destination: &Conf.RopeFreqScale,
	}

	YarnExtFactor = &cli.Float64Flag{
		Name:        "yarn-ext-factor",
		Usage:       "YaRN extrapolation mix factor",
		Value:       1.0,
		Destination: &Conf.YarnExtFactor,
	}

	YarnAttnFactor = &cli.Float64Flag{
		Name:        "yarn-attn-factor",
		Usage:       "YaRN magnitude scaling factor",
		Value:       1.0,
		Destination: &Conf.YarnAttnFactor,
	}

	YarnBetaFast = &cli.Float64Flag{
		Name:        "yarn-beta-fast",
		Usage:       "YaRN low correction dim",
		Value:       32.0,
		Destination: &Conf.YarnBetaFast,
	}

	YarnBetaSlow = &cli.Float64Flag{
		Name:        "yarn-beta-slow",
		Usage:       "YaRN high correction dim",
		Value:       1.0,
		Destination: &Conf.YarnBetaSlow,
	}

	YarnOrigCtx = &cli.IntFlag{
		Name:        "yarn-orig-ctx",
		Usage:       "YaRN original context length",
		Value:       0,
		Destination: &Conf.YarnOrigCtx,
	}

	DefragThold = &cli.Float64Flag{
		Name:        "defrag-thold",
		Usage:       "KV cache defragmentation threshold",
		Value:       -1.0,
		Destination: &Conf.DefragThold,
	}

	MainGpu = &cli.IntFlag{
		Name:        "main-gpu",
		Usage:       "The GPU that is used for scratch and small tensors",
		Value:       0,
		Destination: &Conf.MainGpu,
	}

	Temperature = &cli.Float64Flag{
		Name:        "temperature",
		Usage:       "Temperature for sampling",
		Value:       0.7,
		Destination: &Conf.Temperature,
	}

	TopK = &cli.IntFlag{
		Name:        "top-k",
		Usage:       "Top-k sampling (0 = disabled)",
		Value:       40,
		Destination: &Conf.TopK,
	}

	TopP = &cli.Float64Flag{
		Name:        "top-p",
		Usage:       "Top-p sampling (1.0 = disabled)",
		Value:       0.9,
		Destination: &Conf.TopP,
	}

	MinP = &cli.Float64Flag{
		Name:        "min-p",
		Usage:       "Min-p sampling (0.0 = disabled)",
		Value:       0.0,
		Destination: &Conf.MinP,
	}

	TopNSigma = &cli.Float64Flag{
		Name:        "top-n-sigma",
		Usage:       "Top-n-sigma sampling (-1.0 = disabled)",
		Value:       -1.0,
		Destination: &Conf.TopNSigma,
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
		NBatch,
		NKeep,
		NParallel,
		GrpAttnN,
		GrpAttnW,
		NPrint,
		RopeFreqBase,
		RopeFreqScale,
		YarnExtFactor,
		YarnAttnFactor,
		YarnBetaFast,
		YarnBetaSlow,
		YarnOrigCtx,
		DefragThold,
		MainGpu,
		Temperature,
		TopK,
		TopP,
		MinP,
		TopNSigma,
	}
)

type Config struct {
	LogLevel    string
	Model       string
	CtxSize     int
	Prompt      string
	NGpuLayers  int
	NPredict    int
	Interactive bool
	Seed        uint
	NBatch        int
	NKeep         int
	NParallel     int
	GrpAttnN      int
	GrpAttnW      int
	NPrint        int
	RopeFreqBase  float64
	RopeFreqScale float64
	YarnExtFactor float64
	YarnAttnFactor float64
	YarnBetaFast  float64
	YarnBetaSlow  float64
	YarnOrigCtx   int
	DefragThold   float64
	MainGpu       int
	Temperature   float64
	TopK          int
	TopP          float64
	MinP          float64
	TopNSigma     float64
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
