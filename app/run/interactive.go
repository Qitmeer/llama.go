package run

import (
	"errors"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/Qitmeer/llama.go/api"
	"github.com/Qitmeer/llama.go/common/readline"
	"github.com/urfave/cli/v2"
)

type MultilineState int

const (
	MultilineNone MultilineState = iota
	MultilinePrompt
	MultilineSystem
)

func generateInteractive(pctx *cli.Context, opts runOptions) error {
	usage := func() {
		fmt.Fprintln(os.Stderr, "Available Commands:")
		fmt.Fprintln(os.Stderr, "  /bye            Exit")
		fmt.Fprintln(os.Stderr, "  /?, /help       Help for a command")
		fmt.Fprintln(os.Stderr, "")
		fmt.Fprintln(os.Stderr, "Use \"\"\" to begin a multi-line message.")
		fmt.Fprintln(os.Stderr, "")
	}

	scanner, err := readline.New(readline.Prompt{
		Prompt:         ">>> ",
		AltPrompt:      "... ",
		Placeholder:    "Send a message (/? for help)",
		AltPlaceholder: `Use """ to end multi-line input`,
	})
	if err != nil {
		return err
	}

	fmt.Print(readline.StartBracketedPaste)
	defer fmt.Printf(readline.EndBracketedPaste)

	var sb strings.Builder
	var multiline MultilineState

	for {
		line, err := scanner.Readline()
		switch {
		case errors.Is(err, io.EOF):
			fmt.Println()
			return nil
		case errors.Is(err, readline.ErrInterrupt):
			if line == "" {
				fmt.Println("\nUse Ctrl + d or /bye to exit.")
			}

			scanner.Prompt.UseAlt = false
			sb.Reset()

			continue
		case err != nil:
			return err
		}

		switch {
		case multiline != MultilineNone:
			// check if there's a multiline terminating string
			before, ok := strings.CutSuffix(line, `"""`)
			sb.WriteString(before)
			if !ok {
				fmt.Fprintln(&sb)
				continue
			}

			switch multiline {
			case MultilineSystem:
				opts.System = sb.String()
				opts.Messages = append(opts.Messages, api.Message{Role: "system", Content: opts.System})
				fmt.Println("Set system message.")
				sb.Reset()
			}

			multiline = MultilineNone
			scanner.Prompt.UseAlt = false
		case strings.HasPrefix(line, `"""`):
			line := strings.TrimPrefix(line, `"""`)
			line, ok := strings.CutSuffix(line, `"""`)
			sb.WriteString(line)
			if !ok {
				// no multiline terminating string; need more input
				fmt.Fprintln(&sb)
				multiline = MultilinePrompt
				scanner.Prompt.UseAlt = true
			}
		case scanner.Pasting:
			fmt.Fprintln(&sb, line)
			continue

		case strings.HasPrefix(line, "/help"), strings.HasPrefix(line, "/?"):
			usage()
		case strings.HasPrefix(line, "/exit"), strings.HasPrefix(line, "/bye"):
			return nil
		default:
			sb.WriteString(line)
		}

		if sb.Len() > 0 && multiline == MultilineNone {
			newMessage := api.Message{Role: "user", Content: sb.String()}

			opts.Messages = append(opts.Messages, newMessage)

			assistant, err := chat(pctx, opts)
			if err != nil {
				if strings.Contains(err.Error(), "does not support thinking") ||
					strings.Contains(err.Error(), "invalid think value") {
					fmt.Printf("error: %v\n", err)
					sb.Reset()
					continue
				}
				return err
			}
			if assistant != nil {
				opts.Messages = append(opts.Messages, *assistant)
			}

			sb.Reset()
		}
	}
}
