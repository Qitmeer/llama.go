// Copyright (c) 2017-2025 The qitmeer developers

package app

import (
	"sync"
	"time"

	"github.com/Qitmeer/llama.go/config"
	"github.com/Qitmeer/llama.go/grpc"
	"github.com/Qitmeer/llama.go/openai"
	"github.com/Qitmeer/llama.go/wrapper"
	"github.com/ethereum/go-ethereum/log"
	"github.com/urfave/cli/v2"
)

type App struct {
	ctx       *cli.Context
	cfg       *config.Config
	grSer     *grpc.Service
	openaiSer *openai.Service
	wg        sync.WaitGroup
}

func NewApp(ctx *cli.Context, cfg *config.Config) *App {
	app := &App{
		ctx:       ctx,
		cfg:       cfg,
		grSer:     grpc.New(ctx, cfg),
		openaiSer: openai.New(ctx, cfg),
	}
	return app
}

func (a *App) Start() error {
	err := initLog(a.cfg)
	if err != nil {
		return err
	}
	log.Info("Start App")
	err = a.cfg.Load()
	if err != nil {
		return err
	}
	if a.cfg.Interactive {
		log.Debug("Run Interactive")
		return wrapper.LlamaInteractive(a.cfg)
	} else if a.cfg.IsLonely() {
		log.Debug("Not support")
		a.wg.Add(1)
		go a.startLLama()
		time.Sleep(time.Second)
		content, err := wrapper.LlamaProcess(a.cfg.Prompt)
		if err != nil {
			return err
		}
		log.Info(content)
		return nil
	} else {
		a.wg.Add(1)
		go a.startLLama()
	}
	a.openaiSer.Start()
	return a.grSer.Start()
}

func (a *App) startLLama() {
	defer a.wg.Done()

	err := wrapper.LlamaStart(a.cfg)
	if err != nil {
		log.Error(err.Error())
	}
}

func (a *App) Stop() error {
	log.Info("Stop App")
	if !a.cfg.Interactive && !a.cfg.IsLonely() {
		a.grSer.Stop()
		a.openaiSer.Stop()
	}
	if !a.cfg.Interactive {
		err := wrapper.LlamaStop()
		if err != nil {
			log.Error(err.Error())
		}
	}
	a.wg.Wait()
	return nil
}
