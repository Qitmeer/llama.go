package cmds

import (
	"github.com/Qitmeer/llama.go/config"
	"github.com/Qitmeer/llama.go/grpc/proto"
	"google.golang.org/grpc"
)

func NewGenerate(s *grpc.Server, cfg *config.Config) *proto.Generate {
	return proto.NewGenerate(s, cfg)
}
