package grpc

import (
	"context"
	"net/http"

	"github.com/Qitmeer/llama.go/config"
	"github.com/Qitmeer/llama.go/grpc/proto"
	"github.com/ethereum/go-ethereum/log"
	"github.com/grpc-ecosystem/grpc-gateway/runtime"
	v2runtime "github.com/grpc-ecosystem/grpc-gateway/v2/runtime"
	"google.golang.org/grpc"
)

const DefaultGatewayAddr = "localhost:8081"

func (ser *Service) gateway() {
	ctx, cancel := context.WithCancel(ser.ctx.Context)
	defer cancel()

	log.Info("start grpc gateway", "addr", DefaultGatewayAddr)
	mux := runtime.NewServeMux()
	opts := []grpc.DialOption{grpc.WithInsecure()}
	err := proto.RegisterHelloWorldHandlerFromEndpoint(ctx, mux, config.DefaultGrpcEndpoint, opts)
	if err != nil {
		log.Error(err.Error())
		return
	}
	mux2 := v2runtime.NewServeMux()
	err = proto.RegisterGenerateHandlerFromEndpoint(ctx, mux2, config.DefaultGrpcEndpoint, opts)
	if err != nil {
		log.Error(err.Error())
		return
	}
	err = http.ListenAndServe(DefaultGatewayAddr, cors(mux.ServeHTTP))
	if err != nil {
		log.Error(err.Error())
		return
	}
}

func cors(f http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Add("Access-Control-Allow-Headers", "Content-Type,AccessToken,X-CSRF-Token, Authorization, Token")
		w.Header().Add("Access-Control-Allow-Credentials", "true")
		w.Header().Add("Access-Control-Allow-Methods", "POST, GET, OPTIONS, PUT, DELETE")
		w.Header().Set("content-type", "application/json;charset=UTF-8")
		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		f(w, r)
	}
}
