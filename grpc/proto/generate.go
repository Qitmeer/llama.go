package proto

import (
	"context"
	"fmt"

	"github.com/Qitmeer/llama.go/config"
	"github.com/Qitmeer/llama.go/wrapper"
	"github.com/ethereum/go-ethereum/log"
	"google.golang.org/grpc"
)

type Generate struct {
	// client
	conn   *grpc.ClientConn
	client GenerateClient
	cfg    *config.Config
}

func NewGenerate(s *grpc.Server, cfg *config.Config) *Generate {
	log.Trace("NewGenerate")
	hw := &Generate{cfg: cfg}

	RegisterGenerateServer(s, hw)
	return hw
}

func (k *Generate) Generate(ctx context.Context, in *GenerateRequest) (*GenerateResponse, error) {
	k.cfg.Prompt = in.Prompt
	k.cfg.CtxSize = int(in.NCtx)
	k.cfg.NBatch = int(in.NBatch)
	k.cfg.NKeep = int(in.NKeep)
	k.cfg.NParallel = int(in.NParallel)
	k.cfg.NGpuLayers = int(in.NGpuLayers)
	k.cfg.MainGpu = int(in.MainGpu)
	k.cfg.Temperature = float64(in.Temperature)
	k.cfg.TopK = int(in.TopK)
	k.cfg.TopP = float64(in.TopP)
	k.cfg.MinP = float64(in.MinP)
	k.cfg.Seed = uint(in.Seed)

	content, err := wrapper.LlamaProcess(k.cfg)
	if err != nil {
		return nil, err
	}
	return &GenerateResponse{Content: content}, nil
}

func (Generate) mustEmbedUnimplementedGenerateServer() {}

func (k *Generate) Client() GenerateClient {

	if k.client == nil {
		// Set up a connection to the gRPC server.
		conn, err := grpc.Dial(config.DefaultGrpcEndpoint, grpc.WithInsecure())
		if err != nil {
			log.Error(fmt.Sprintf("did not connect: %v", err))
			return nil
		}
		k.client = NewGenerateClient(conn)

		log.Trace("New GenerateClient")
	}

	return k.client
}

func (k *Generate) Close() {
	log.Trace("Close Generate")
	if k.conn != nil {
		k.conn.Close()
	}
}
