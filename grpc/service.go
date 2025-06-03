package grpc

import (
	"fmt"
	"net"

	"github.com/Qitmeer/llama.go/config"
	"github.com/Qitmeer/llama.go/grpc/cmds"
	"github.com/Qitmeer/llama.go/grpc/proto"
	"github.com/ethereum/go-ethereum/log"
	"github.com/urfave/cli/v2"
	"google.golang.org/grpc"
	"google.golang.org/grpc/reflection"
)

type Service struct {
	ctx *cli.Context
	gs  *grpc.Server
	cfg *config.Config

	quit chan struct{}

	hw       *cmds.HelloWorld
	generate *proto.Generate
}

func New(ctx *cli.Context, cfg *config.Config) *Service {
	log.Info("New grpc Service...")
	ser := Service{ctx: ctx, cfg: cfg}
	return &ser
}

func (ser *Service) Start() error {
	log.Info("Start grpc Service...")

	go ser.Register()
	go ser.gateway()
	return nil
}

func (ser *Service) Stop() {
	ser.gs.Stop()
	log.Info("Stop grpc Service...")
}

func (ser *Service) Register() {
	log.Info(fmt.Sprintf("Register:%s", config.DefaultGrpcEndpoint))
	lis, err := net.Listen("tcp", config.DefaultGrpcEndpoint)
	if err != nil {
		log.Error(fmt.Sprintf("failed to listen: %v", err))
		return
	}
	ser.gs = grpc.NewServer()

	// register all cmd service
	ser.hw = cmds.NewHelloWorld(ser.gs)
	ser.generate = cmds.NewGenerate(ser.gs, ser.cfg)

	// Register reflection service on gRPC server.
	reflection.Register(ser.gs)
	if err := ser.gs.Serve(lis); err != nil {
		log.Error("failed to serve", "err", err)
		return
	}
}

func (ser *Service) HelloWorld() *cmds.HelloWorld {
	return ser.hw
}

func (ser *Service) Generate() *proto.Generate {
	return ser.generate
}
