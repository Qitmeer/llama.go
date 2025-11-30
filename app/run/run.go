package run

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"github.com/Qitmeer/llama.go/api"
	"github.com/Qitmeer/llama.go/app/pull"
	"github.com/Qitmeer/llama.go/common/progress"
	"github.com/Qitmeer/llama.go/common/readline"
	"github.com/Qitmeer/llama.go/config"
	"github.com/Qitmeer/llama.go/model"
	"github.com/mattn/go-runewidth"
	"github.com/urfave/cli/v2"
	"golang.org/x/term"
	"io"
	"net/http"
	"os"
	"os/signal"
	"slices"
	"strings"
	"syscall"
	"time"
)

type generateContextKey string

type runOptions struct {
	Model        string
	ParentModel  string
	Prompt       string
	Messages     []api.Message
	WordWrap     bool
	Format       string
	System       string
	Images       []api.ImageData
	Options      map[string]any
	MultiModal   bool
	KeepAlive    *api.Duration
	Think        *api.ThinkValue
	HideThinking bool
	ShowConnect  bool
}

func (r runOptions) Copy() runOptions {
	var messages []api.Message
	if r.Messages != nil {
		messages = make([]api.Message, len(r.Messages))
		copy(messages, r.Messages)
	}

	var images []api.ImageData
	if r.Images != nil {
		images = make([]api.ImageData, len(r.Images))
		copy(images, r.Images)
	}

	var opts map[string]any
	if r.Options != nil {
		opts = make(map[string]any, len(r.Options))
		for k, v := range r.Options {
			opts[k] = v
		}
	}

	var think *api.ThinkValue
	if r.Think != nil {
		cThink := *r.Think
		think = &cThink
	}

	return runOptions{
		Model:        r.Model,
		ParentModel:  r.ParentModel,
		Prompt:       r.Prompt,
		Messages:     messages,
		WordWrap:     r.WordWrap,
		Format:       r.Format,
		System:       r.System,
		Images:       images,
		Options:      opts,
		MultiModal:   r.MultiModal,
		KeepAlive:    r.KeepAlive,
		Think:        think,
		HideThinking: r.HideThinking,
		ShowConnect:  r.ShowConnect,
	}
}

type displayResponseState struct {
	lineLength int
	wordBuffer string
}

func displayResponse(content string, wordWrap bool, state *displayResponseState) {
	termWidth, _, _ := term.GetSize(int(os.Stdout.Fd()))
	if wordWrap && termWidth >= 10 {
		for _, ch := range content {
			if state.lineLength+1 > termWidth-5 {
				if runewidth.StringWidth(state.wordBuffer) > termWidth-10 {
					fmt.Printf("%s%c", state.wordBuffer, ch)
					state.wordBuffer = ""
					state.lineLength = 0
					continue
				}

				// backtrack the length of the last word and clear to the end of the line
				a := runewidth.StringWidth(state.wordBuffer)
				if a > 0 {
					fmt.Printf("\x1b[%dD", a)
				}
				fmt.Printf("\x1b[K\n")
				fmt.Printf("%s%c", state.wordBuffer, ch)
				chWidth := runewidth.RuneWidth(ch)

				state.lineLength = runewidth.StringWidth(state.wordBuffer) + chWidth
			} else {
				fmt.Print(string(ch))
				state.lineLength += runewidth.RuneWidth(ch)
				if runewidth.RuneWidth(ch) >= 2 {
					state.wordBuffer = ""
					continue
				}

				switch ch {
				case ' ', '\t':
					state.wordBuffer = ""
				case '\n', '\r':
					state.lineLength = 0
					state.wordBuffer = ""
				default:
					state.wordBuffer += string(ch)
				}
			}
		}
	} else {
		fmt.Printf("%s%s", state.wordBuffer, content)
		if len(state.wordBuffer) > 0 {
			state.wordBuffer = ""
		}
	}
}

func thinkingOutputOpeningText(plainText bool) string {
	text := "Thinking...\n"

	if plainText {
		return text
	}

	return readline.ColorGrey + readline.ColorBold + text + readline.ColorDefault + readline.ColorGrey
}

func thinkingOutputClosingText(plainText bool) string {
	text := "...done thinking.\n\n"

	if plainText {
		return text
	}

	return readline.ColorGrey + readline.ColorBold + text + readline.ColorDefault
}

func RunHandler(ctx *cli.Context) error {
	interactive := true

	opts := runOptions{
		Model:       config.Conf.Model,
		WordWrap:    os.Getenv("TERM") == "xterm-256color",
		Options:     map[string]any{},
		ShowConnect: true,
	}

	opts.Format = Conf.Format

	if len(Conf.Think) > 0 {
		thinkStr := Conf.Think

		// Handle different values for --think
		switch thinkStr {
		case "", "true":
			// --think or --think=true
			opts.Think = &api.ThinkValue{Value: true}
		case "false":
			opts.Think = &api.ThinkValue{Value: false}
		case "high", "medium", "low":
			opts.Think = &api.ThinkValue{Value: thinkStr}
		default:
			return fmt.Errorf("invalid value for --think: %q (must be true, false, high, medium, or low)", thinkStr)
		}
	} else {
		opts.Think = nil
	}

	opts.HideThinking = Conf.Hidethinking

	if len(Conf.Keepalive) > 0 {
		d, err := time.ParseDuration(Conf.Keepalive)
		if err != nil {
			return err
		}
		opts.KeepAlive = &api.Duration{Duration: d}
	}

	// Handle positional arguments (prompt as first argument)
	var promptFromArgs string
	if ctx.Args().Len() > 0 {
		promptFromArgs = strings.Join(ctx.Args().Slice(), " ")
	}

	prompts := []string{}

	// Add prompt from positional arguments if provided
	if promptFromArgs != "" {
		prompts = append(prompts, promptFromArgs)
	}

	// Add prompt from --prompt flag if provided (for backward compatibility)
	if Conf.Prompt != "" {
		prompts = append(prompts, Conf.Prompt)
	}

	// prepend stdin to the prompt if provided
	if !term.IsTerminal(int(os.Stdin.Fd())) {
		in, err := io.ReadAll(os.Stdin)
		if err != nil {
			return err
		}

		prompts = append([]string{string(in)}, prompts...)
		opts.ShowConnect = false
		opts.WordWrap = false
		interactive = false
	}

	opts.Prompt = strings.Join(prompts, " ")
	if len(prompts) > 0 {
		interactive = false
	}
	// Be quiet if we're redirecting to a pipe or file
	if !term.IsTerminal(int(os.Stdout.Fd())) {
		interactive = false
	}

	opts.WordWrap = !Conf.Nowordwrap

	// Fill out the rest of the options based on information about the
	// model.
	client := api.DefaultClient()

	name := config.Conf.Model
	info, err := func() (*api.ShowResponse, error) {
		showReq := &api.ShowRequest{Model: name}
		info, err := client.Show(ctx.Context, showReq)
		var se api.StatusError
		if errors.As(err, &se) && se.StatusCode == http.StatusNotFound {
			if err := pull.PullHandler(ctx); err != nil {
				return nil, err
			}
			return client.Show(ctx.Context, &api.ShowRequest{Model: name})
		}
		return info, err
	}()
	if err != nil {
		return err
	}

	opts.Think, err = inferThinkingOption(&info.Capabilities, &opts, len(Conf.Think) > 0)
	if err != nil {
		return err
	}

	opts.MultiModal = slices.Contains(info.Capabilities, model.CapabilityVision)

	if len(info.ProjectorInfo) != 0 {
		opts.MultiModal = true
	}
	for k := range info.ModelInfo {
		if strings.Contains(k, ".vision.") {
			opts.MultiModal = true
			break
		}
	}

	opts.ParentModel = info.Details.ParentModel

	if interactive {
		return generateInteractive(ctx, opts)
	}
	return generate(ctx, opts)
}

func generate(pctx *cli.Context, opts runOptions) error {
	client := api.DefaultClient()
	p := progress.NewProgress(os.Stderr)
	defer p.StopAndClear()

	spinner := progress.NewSpinner("")
	p.Add("", spinner)

	var latest api.GenerateResponse

	generateContext, ok := pctx.Context.Value(generateContextKey("context")).([]int)
	if !ok {
		generateContext = []int{}
	}

	ctx, cancel := context.WithCancel(pctx.Context)
	defer cancel()

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT)

	go func() {
		<-sigChan
		cancel()
	}()

	var state *displayResponseState = &displayResponseState{}
	var thinkingContent strings.Builder
	var thinkTagOpened bool = false
	var thinkTagClosed bool = false

	plainText := !term.IsTerminal(int(os.Stdout.Fd()))

	fn := func(response api.GenerateResponse) error {
		latest = response
		content := response.Content()

		if content != "" || !opts.HideThinking {
			p.StopAndClear()
		}

		if response.Thinking != "" && !opts.HideThinking {
			if !thinkTagOpened {
				fmt.Print(thinkingOutputOpeningText(plainText))
				thinkTagOpened = true
				thinkTagClosed = false
			}
			thinkingContent.WriteString(response.Thinking)
			displayResponse(response.Thinking, opts.WordWrap, state)
		}

		if thinkTagOpened && !thinkTagClosed && (content != "" || len(response.ToolCalls) > 0) {
			if !strings.HasSuffix(thinkingContent.String(), "\n") {
				fmt.Println()
			}
			fmt.Print(thinkingOutputClosingText(plainText))
			thinkTagOpened = false
			thinkTagClosed = true
			state = &displayResponseState{}
		}

		displayResponse(content, opts.WordWrap, state)

		if response.ToolCalls != nil {
			toolCalls := response.ToolCalls
			if len(toolCalls) > 0 {
				fmt.Print(renderToolCalls(toolCalls, plainText))
			}
		}

		return nil
	}

	if opts.MultiModal {
		var err error
		opts.Prompt, opts.Images, err = extractFileData(opts.Prompt)
		if err != nil {
			return err
		}
	}

	if opts.Format == "json" {
		opts.Format = `"` + opts.Format + `"`
	}

	request := api.GenerateRequest{
		Model:     opts.Model,
		Prompt:    opts.Prompt,
		Context:   generateContext,
		Images:    opts.Images,
		Format:    json.RawMessage(opts.Format),
		System:    opts.System,
		Options:   opts.Options,
		KeepAlive: opts.KeepAlive,
		Think:     opts.Think,
	}

	if err := client.Generate(ctx, &request, fn); err != nil {
		if errors.Is(err, context.Canceled) {
			return nil
		}
		return err
	}

	if opts.Prompt != "" {
		fmt.Println()
		fmt.Println()
	}

	if !latest.Done() {
		return nil
	}

	if Conf.Verbose {
		latest.Summary()
	}

	pctx.Context = context.WithValue(pctx.Context, generateContextKey("context"), latest.Context)
	return nil
}

func inferThinkingOption(caps *[]model.Capability, runOpts *runOptions, explicitlySetByUser bool) (*api.ThinkValue, error) {
	if explicitlySetByUser {
		return runOpts.Think, nil
	}

	if caps == nil {
		client := api.DefaultClient()
		ret, err := client.Show(context.Background(), &api.ShowRequest{
			Model: runOpts.Model,
		})
		if err != nil {
			return nil, err
		}
		caps = &ret.Capabilities
	}

	thinkingSupported := false
	for _, cap := range *caps {
		if cap == model.CapabilityThinking {
			thinkingSupported = true
		}
	}

	if thinkingSupported {
		return &api.ThinkValue{Value: true}, nil
	}

	return nil, nil
}

func renderToolCalls(toolCalls []api.ToolCall, plainText bool) string {
	out := ""
	formatExplanation := ""
	formatValues := ""
	if !plainText {
		formatExplanation = readline.ColorGrey + readline.ColorBold
		formatValues = readline.ColorDefault
		out += formatExplanation
	}
	for i, toolCall := range toolCalls {
		argsAsJSON, err := json.Marshal(toolCall.Function.Arguments)
		if err != nil {
			return ""
		}
		if i > 0 {
			out += "\n"
		}
		// all tool calls are unexpected since we don't currently support registering any in the CLI
		out += fmt.Sprintf("  Model called a non-existent function '%s()' with arguments: %s", formatValues+toolCall.Function.Name+formatExplanation, formatValues+string(argsAsJSON)+formatExplanation)
	}
	if !plainText {
		out += readline.ColorDefault
	}
	return out
}

func chat(pctx *cli.Context, opts runOptions) (*api.Message, error) {
	client := api.DefaultClient()
	p := progress.NewProgress(os.Stderr)
	defer p.StopAndClear()

	spinner := progress.NewSpinner("")
	p.Add("", spinner)

	cancelCtx, cancel := context.WithCancel(pctx.Context)
	defer cancel()

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT)

	go func() {
		<-sigChan
		cancel()
	}()

	var state *displayResponseState = &displayResponseState{}
	var thinkingContent strings.Builder
	var latest api.ChatResponse
	var fullResponse strings.Builder
	var thinkTagOpened bool = false
	var thinkTagClosed bool = false

	role := "assistant"

	fn := func(response api.ChatResponse) error {
		if response.Message.Content != "" || !opts.HideThinking {
			p.StopAndClear()
		}

		latest = response

		role = response.Message.Role
		if response.Message.Thinking != "" && !opts.HideThinking {
			if !thinkTagOpened {
				fmt.Print(thinkingOutputOpeningText(false))
				thinkTagOpened = true
				thinkTagClosed = false
			}
			thinkingContent.WriteString(response.Message.Thinking)
			displayResponse(response.Message.Thinking, opts.WordWrap, state)
		}

		content := response.Message.Content
		if thinkTagOpened && !thinkTagClosed && (content != "" || len(response.Message.ToolCalls) > 0) {
			if !strings.HasSuffix(thinkingContent.String(), "\n") {
				fmt.Println()
			}
			fmt.Print(thinkingOutputClosingText(false))
			thinkTagOpened = false
			thinkTagClosed = true
			state = &displayResponseState{}
		}
		// purposefully not putting thinking blocks in the response, which would
		// only be needed if we later added tool calling to the cli (they get
		// filtered out anyway since current models don't expect them unless you're
		// about to finish some tool calls)
		fullResponse.WriteString(content)

		if response.Message.ToolCalls != nil {
			toolCalls := response.Message.ToolCalls
			if len(toolCalls) > 0 {
				fmt.Print(renderToolCalls(toolCalls, false))
			}
		}

		displayResponse(content, opts.WordWrap, state)

		return nil
	}

	if opts.Format == "json" {
		opts.Format = `"` + opts.Format + `"`
	}

	req := &api.ChatRequest{
		Model:    opts.Model,
		Messages: opts.Messages,
		Format:   json.RawMessage(opts.Format),
		Options:  opts.Options,
		Think:    opts.Think,
	}

	if opts.KeepAlive != nil {
		req.KeepAlive = opts.KeepAlive
	}

	if err := client.Chat(cancelCtx, req, fn); err != nil {
		if errors.Is(err, context.Canceled) {
			return nil, nil
		}

		// this error should ideally be wrapped properly by the client
		if strings.Contains(err.Error(), "upstream error") {
			p.StopAndClear()
			fmt.Println("An error occurred while processing your message. Please try again.")
			fmt.Println()
			return nil, nil
		}
		return nil, err
	}

	if len(opts.Messages) > 0 {
		fmt.Println()
		fmt.Println()
	}

	if Conf.Verbose {
		latest.Summary()
	}

	return &api.Message{Role: role, Content: fullResponse.String()}, nil
}
