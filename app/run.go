// Copyright (c) 2017-2025 The qitmeer developers

package app

import (
	"github.com/Qitmeer/llama.go/config"
	"github.com/Qitmeer/llama.go/system"
	"github.com/Qitmeer/llama.go/system/limits"
	"github.com/Qitmeer/llama.go/version"
	"github.com/urfave/cli/v2"
	"os"
)

func Run() error {
	app := &cli.App{
		Name:    "",
		Version: version.String(),
		Authors: []*cli.Author{
			&cli.Author{
				Name: "Qitmeer",
			},
		},
		Copyright:            "(c) 2025 Qitmeer",
		Usage:                "Llama",
		Flags:                config.AppFlags,
		EnableBashCompletion: true,
		Commands:             commands(),
		Action: func(c *cli.Context) error {
			err := limits.SetLimits()
			if err != nil {
				return err
			}
			interrupt := system.InterruptListener()

			a := NewApp(c, config.Conf)
			err = a.Start()
			if err != nil {
				return err
			}
			if !config.Conf.IsLonely() {
				<-interrupt
			}
			return a.Stop()
		},
	}

	return app.Run(os.Args)
}
