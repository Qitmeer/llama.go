// Copyright (c) 2017-2025 The qitmeer developers

package app

import (
	"github.com/Qitmeer/llama.go/config"
	"github.com/Qitmeer/llama.go/server"
	"github.com/Qitmeer/llama.go/wrapper"
	"github.com/ethereum/go-ethereum/log"
	"github.com/urfave/cli/v2"
	"sync"
)

type App struct {
	ctx *cli.Context
	cfg *config.Config
	ser *server.Service
	wg  sync.WaitGroup
}

func NewApp(ctx *cli.Context, cfg *config.Config) *App {
	app := &App{
		ctx: ctx,
		cfg: cfg,
		ser: server.New(ctx, cfg),
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
		return nil
	} else {
		a.wg.Add(1)
		go a.startLLama()
	}
	return a.ser.Start()
}

func (a *App) startLLama() {
	defer a.wg.Done()

	err := wrapper.LlamaStart(a.cfg)
	if err != nil {
		log.Error(err.Error())
	}
	log.Info("Started llama")
}

func (a *App) Stop() error {
	log.Info("Stop App")
	if !a.cfg.Interactive && !a.cfg.IsLonely() {
		err := a.ser.Stop()
		if err != nil {
			log.Error(err.Error())
		}
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
