// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// ra25 runs a simple random-associator four-layer leabra network
// that uses the standard supervised learning paradigm to learn
// mappings between 25 random input / output patterns
// defined over 5x5 input / output layers (i.e., 25 units)

// This is an attempt to change the ra25 into a training model by Diheng and Anna.
package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"strconv"
	"time"

	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/env"
	"github.com/emer/emergent/netview"
	"github.com/emer/emergent/params"
	"github.com/emer/emergent/patgen"
	"github.com/emer/emergent/prjn"
	"github.com/emer/emergent/relpos"
	"github.com/emer/etable/agg"
	"github.com/emer/etable/eplot"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	_ "github.com/emer/etable/etview" // include to get gui views
	"github.com/emer/etable/split"
	"github.com/emer/leabra/leabra"
	"github.com/goki/gi/gi"
	"github.com/goki/gi/gimain"
	"github.com/goki/gi/giv"
	"github.com/goki/ki/ki"
	"github.com/goki/ki/kit"
)

// this is the stub main for gogi that calls our actual mainrun function, at end of file
func main() {
	gimain.Main(func() {
		mainrun()
	})
}

// LogPrec is precision for saving float values in logs
const LogPrec = 4

// ParamSets is the default set of parameters -- Base is always applied, and others can be optionally
// selected to apply on top of that
var ParamSets = params.Sets{
	{Name: "Base", Desc: "these are the best params", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Prjn", Desc: "norm and momentum on works better, but wt bal is not better for smaller nets",
				Params: params.Params{
					"Prjn.Learn.Norm.On":     "true",
					"Prjn.Learn.Momentum.On": "true",
					"Prjn.Learn.WtBal.On":    "true",
				}},
			{Sel: "Layer", Desc: "using default 1.8 inhib for all of network -- can explore",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "1.8",
				}},
			{Sel: ".Back", Desc: "top-down back-projections MUST have lower relative weight scale, otherwise network hallucinates",
				Params: params.Params{
					"Prjn.WtScale.Rel": "0.2",
				}},
			{Sel: "#Output", Desc: "output definitely needs lower inhib -- true for smaller layers in general",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "1.4",
				}},
			{Sel: "#Hidden1", Desc: "output definitely needs higher inhib -- true for smaller layers in general",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "1.9",
				}},
		},
		"Sim": &params.Sheet{ // sim params apply to sim object
			{Sel: "Sim", Desc: "best params always finish in this time",
				Params: params.Params{
					"Sim.MaxEpcs": "50",
				}},
		},
	}},
	{Name: "UnBalIn", Desc: "Unbalent BLA input, while Negative > Positive", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "#PoToHidden1", Desc: "Weaker Positive input",
				Params: params.Params{
					"Prjn.WtScale.Abs": "0.7",
				}},
		},
	}},
	{Name: "UnBalLearn", Desc: "Unbalent BLA learning, while Negative > Positive", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "#NeToHidden1", Desc: "Higher learning rate for negative input",
				Params: params.Params{
					"Prjn.Learn.Lrate": "0.08",
				}},
		},
	}},
	{Name: "DefaultInhib", Desc: "output uses default inhib instead of lower", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "#Output", Desc: "go back to default",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "1.8",
				}},
		},
		"Sim": &params.Sheet{ // sim params apply to sim object
			{Sel: "Sim", Desc: "takes longer -- generally doesn't finish..",
				Params: params.Params{
					"Sim.MaxEpcs": "100",
				}},
		},
	}},
	{Name: "NoMomentum", Desc: "no momentum or normalization", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Prjn", Desc: "no norm or momentum",
				Params: params.Params{
					"Prjn.Learn.Norm.On":     "false",
					"Prjn.Learn.Momentum.On": "false",
				}},
		},
	}},
	{Name: "WtBalOn", Desc: "try with weight bal on", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Prjn", Desc: "weight bal on",
				Params: params.Params{
					"Prjn.Learn.WtBal.On": "true",
				}},
		},
	}},
	{Name: "Sleep", Desc: "these are the sleep params", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Prjn", Desc: "norm and momentum on works better, but wt bal is not better for smaller nets",
				Params: params.Params{
					"Prjn.Learn.Norm.On":     "true",
					"Prjn.Learn.Momentum.On": "true",
					"Prjn.Learn.WtBal.On":    "true",
				}},
			{Sel: ".Back", Desc: "top-down back-projections MUST have lower relative weight scale, otherwise network hallucinates",
				Params: params.Params{
					"Prjn.WtScale.Rel": "0.8",
				}},
			{Sel: "Layer", Desc: "using higher inhib for all of network during sleep-- can explore",
				Params: params.Params{
					"Layer.Inhib.Layer.GiBase": "1.8",
				}},
			{Sel: "Layer", Desc: "using higher inhib for all of network during sleep-- can explore",
				Params: params.Params{
					"Layer.Inhib.Layer.FB": "1.2",
				}},
		},
		"Sim": &params.Sheet{ // sim params apply to sim object
			{Sel: "Sim", Desc: "best params always finish in this time",
				Params: params.Params{
					"Sim.MaxEpcs": "50",
				}},
		},
	}},
}

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around
// as arguments to methods, and provides the core GUI interface (note the view tags
// for the fields which provide hints to how things should be displayed).
type Sim struct {
	Net          *leabra.Network   `view:"no-inline"`
	Pats         *etable.Table     `view:"no-inline" desc:"the training patterns to use"`
	SlpCycLog    *etable.Table     `view:"no-inline" desc:"sleeping cycle-level log data"`
	TrnEpcLog    *etable.Table     `view:"no-inline" desc:"training epoch-level log data"`
	TstEpcLog    *etable.Table     `view:"no-inline" desc:"testing epoch-level log data"`
	TstTrlLog    *etable.Table     `view:"no-inline" desc:"testing trial-level log data"`
	TstErrLog    *etable.Table     `view:"no-inline" desc:"log of all test trials where errors were made"`
	TstErrStats  *etable.Table     `view:"no-inline" desc:"stats on test trials where errors were made"`
	TstCycLog    *etable.Table     `view:"no-inline" desc:"testing cycle-level log data"`
	RunLog       *etable.Table     `view:"no-inline" desc:"summary log of each run"`
	RunStats     *etable.Table     `view:"no-inline" desc:"aggregate stats on all runs"`
	Params       params.Sets       `view:"no-inline" desc:"full collection of param sets"`
	ParamSet     string            `desc:"which set of *additional* parameters to use -- always applies Base and optionaly this next if set"`
	Tag          string            `desc:"extra tag string to add to any file names output from sim (e.g., weights files, log files)"`
	MaxRuns      int               `desc:"maximum number of model runs to perform"`
	MaxEpcs      int               `desc:"maximum number of epochs to run per model run"`
	MaxSlpCyc    int               `desc:"maximum number of cycle to sleep for a trial"`
	TrainEnv     env.FixedTable    `desc:"Training environment -- contains everything about iterating over input / output patterns over training"`
	SleepEnv     env.FixedTable    `desc:"Sleep environment -- contains everything about iterating over sleep trials"` // added by DH
	TestEnv      env.FixedTable    `desc:"Testing environment -- manages iterating over testing"`
	Time         leabra.Time       `desc:"leabra timing parameters and state"`
	ViewOn       bool              `desc:"whether to update the network view while running"`
	Sleep        bool              `desc:"Sleep or not"`
	InhibOscil   bool              `desc:"whether to implement inhibition oscillation"`
	TrainUpdt    leabra.TimeScales `desc:"at what time scale to update the display during training?  Anything longer than Epoch updates at Epoch in this model"`
	SleepUpdt    leabra.TimeScales `desc:"at what time scale to update the display during sleep? Anything longer than Epoch updates at Epoch in this model"` // added by DH
	TestUpdt     leabra.TimeScales `desc:"at what time scale to update the display during testing?  Anything longer than Epoch updates at Epoch in this model"`
	TestInterval int               `desc:"how often to run through all the test patterns, in terms of training epochs"`

	// statistics: note use float64 as that is best for etable.Table
	TrlSSE     float64 `inactive:"+" desc:"current trial's sum squared error"`
	TrlAvgSSE  float64 `inactive:"+" desc:"current trial's average sum squared error"`
	TrlCosDiff float64 `inactive:"+" desc:"current trial's cosine difference"`
	EpcSSE     float64 `inactive:"+" desc:"last epoch's total sum squared error"`
	EpcAvgSSE  float64 `inactive:"+" desc:"last epoch's average sum squared error (average over trials, and over units within layer)"`
	EpcPctErr  float64 `inactive:"+" desc:"last epoch's percent of trials that had SSE > 0 (subject to .5 unit-wise tolerance)"`
	EpcPctCor  float64 `inactive:"+" desc:"last epoch's percent of trials that had SSE == 0 (subject to .5 unit-wise tolerance)"`
	EpcCosDiff float64 `inactive:"+" desc:"last epoch's average cosine difference for output layer (a normalized error measure, maximum of 1 when the minus phase exactly matches the plus)"`
	FirstZero  int     `inactive:"+" desc:"epoch at when SSE first went to zero"`
	AvgLaySim  float64 `inactive:"+" desc:"Average layer similarity between current cycle and previous cycle"`

	// internal state - view:"-"
	SumSSE       float64          `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	SumAvgSSE    float64          `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	SumCosDiff   float64          `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	CntErr       int              `view:"-" inactive:"+" desc:"sum of errs to increment as we go through epoch"`
	Win          *gi.Window       `view:"-" desc:"main GUI window"`
	NetView      *netview.NetView `view:"-" desc:"the network viewer"`
	ToolBar      *gi.ToolBar      `view:"-" desc:"the master toolbar"`
	SlpCycPlot   *eplot.Plot2D    `view:"-" desc:"the sleeping cycle plot"`
	TrnEpcPlot   *eplot.Plot2D    `view:"-" desc:"the training epoch plot"`
	TstEpcPlot   *eplot.Plot2D    `view:"-" desc:"the testing epoch plot"`
	TstTrlPlot   *eplot.Plot2D    `view:"-" desc:"the test-trial plot"`
	TstCycPlot   *eplot.Plot2D    `view:"-" desc:"the test-cycle plot"`
	RunPlot      *eplot.Plot2D    `view:"-" desc:"the run plot"`
	TrnEpcFile   *os.File         `view:"-" desc:"log file"`
	RunFile      *os.File         `view:"-" desc:"log file"`
	SaveWts      bool             `view:"-" desc:"for command-line run only, auto-save final weights after each run"`
	NoGui        bool             `view:"-" desc:"if true, runing in no GUI mode"`
	LogSetParams bool             `view:"-" desc:"if true, print message for all params that are set"`
	IsRunning    bool             `view:"-" desc:"true if sim is running"`
	StopNow      bool             `view:"-" desc:"flag to stop running"`
	RndSeed      int64            `view:"-" desc:"the current random seed"`
}

// this registers this Sim Type and gives it properties that e.g.,
// prompt for filename for save methods.
var KiT_Sim = kit.Types.AddType(&Sim{}, SimProps)

// TheSim is the overall state for this simulation
var TheSim Sim

// New creates new blank elements and initializes defaults
func (ss *Sim) New() {
	ss.Net = &leabra.Network{}
	ss.Pats = &etable.Table{}
	ss.TrnEpcLog = &etable.Table{}
	ss.SlpCycLog = &etable.Table{}
	ss.TstEpcLog = &etable.Table{}
	ss.TstTrlLog = &etable.Table{}
	ss.TstCycLog = &etable.Table{}
	ss.RunLog = &etable.Table{}
	ss.RunStats = &etable.Table{}
	ss.Params = ParamSets
	ss.RndSeed = 1
	ss.ViewOn = true
	ss.Sleep = true
	ss.InhibOscil = true
	ss.TrainUpdt = leabra.FastSpike
	ss.TestUpdt = leabra.Cycle
	ss.TestInterval = 5
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Configs

// Config configures all the elements using the standard functions
func (ss *Sim) Config() {
	//ss.ConfigPats()
	ss.OpenPats()
	ss.ConfigEnv()
	ss.ConfigNet(ss.Net)
	ss.ConfigSlpCycLog(ss.SlpCycLog)
	ss.ConfigTrnEpcLog(ss.TrnEpcLog)
	ss.ConfigTstEpcLog(ss.TstEpcLog)
	ss.ConfigTstTrlLog(ss.TstTrlLog)
	ss.ConfigTstCycLog(ss.TstCycLog)
	ss.ConfigRunLog(ss.RunLog)
}

func (ss *Sim) ConfigEnv() {
	if ss.MaxRuns == 0 { // allow user override
		ss.MaxRuns = 10
	}
	if ss.MaxEpcs == 0 { // allow user override
		ss.MaxEpcs = 50
	}
	if ss.MaxSlpCyc == 0 { // allow user override
		ss.MaxSlpCyc = 800
	}

	ss.TrainEnv.Nm = "TrainEnv"
	ss.TrainEnv.Dsc = "training params and state"
	ss.TrainEnv.Table = etable.NewIdxView(ss.Pats)
	ss.TrainEnv.Validate()
	ss.TrainEnv.Run.Max = ss.MaxRuns // note: we are not setting epoch max -- do that manually

	ss.SleepEnv.Nm = "SleepEnv"
	ss.SleepEnv.Dsc = "sleep params and state"
	ss.SleepEnv.Table = etable.NewIdxView(ss.Pats)
	ss.SleepEnv.Validate()
	// ss.SleepEnv.Cycle.Max = ss.MaxSlpCyc // note: we are not setting epoch max -- do that manually // Not sure if this is needed.

	ss.TestEnv.Nm = "TestEnv"
	ss.TestEnv.Dsc = "testing params and state"
	ss.TestEnv.Table = etable.NewIdxView(ss.Pats)
	ss.TestEnv.Sequential = true
	ss.TestEnv.Validate()

	// note: to create a train / test split of pats, do this:
	// all := etable.NewIdxView(ss.Pats)
	// splits, _ := split.Permuted(all, []float64{.8, .2}, []string{"Train", "Test"})
	// ss.TrainEnv.Table = splits.Splits[0]
	// ss.TestEnv.Table = splits.Splits[1]

	ss.TrainEnv.Init(0)
	ss.SleepEnv.Init(0)
	ss.TestEnv.Init(0)
}

func (ss *Sim) ConfigNet(net *leabra.Network) {
	net.InitName(net, "SUMMER")
	inLay := net.AddLayer2D("Input", 5, 5, emer.Input)
	blaNeInLay := net.AddLayer2D("Ne", 3, 1, emer.Input)
	blaPoInLay := net.AddLayer2D("Po", 3, 1, emer.Input)
	hid1Lay := net.AddLayer2D("Hidden1", 12, 12, emer.Hidden)
	outLay := net.AddLayer2D("Output", 5, 5, emer.Target)
	blaNeOutLay := net.AddLayer2D("Ne_Out", 3, 1, emer.Target)
	blaPoOutLay := net.AddLayer2D("Po_Out", 3, 1, emer.Target)

	// use this to position layers relative to each other
	// default is Above, YAlign = Front, XAlign = Center
	blaNeInLay.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "Input", YAlign: relpos.Front, Space: 2})
	blaPoInLay.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "Input", YAlign: relpos.Front, Space: 3})
	blaNeOutLay.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "Output", YAlign: relpos.Front, Space: 2})
	blaPoOutLay.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "Output", YAlign: relpos.Front, Space: 3})

	net.ConnectLayers(inLay, hid1Lay, prjn.NewFull(), emer.Forward)
	net.ConnectLayers(blaNeInLay, hid1Lay, prjn.NewFull(), emer.Forward)
	net.ConnectLayers(blaPoInLay, hid1Lay, prjn.NewFull(), emer.Forward)
	net.ConnectLayers(hid1Lay, outLay, prjn.NewFull(), emer.Forward)
	net.ConnectLayers(hid1Lay, blaNeOutLay, prjn.NewFull(), emer.Forward)
	net.ConnectLayers(hid1Lay, blaPoOutLay, prjn.NewFull(), emer.Forward)

	// note: see emergent/prjn module for all the options on how to connect
	// NewFull returns a new prjn.Full connectivity pattern
	net.ConnectLayers(hid1Lay, inLay, prjn.NewFull(), emer.Back)
	// i.SetOff(true)
	net.ConnectLayers(hid1Lay, blaNeInLay, prjn.NewFull(), emer.Back)
	// n.SetOff(true)
	net.ConnectLayers(hid1Lay, blaPoInLay, prjn.NewFull(), emer.Back)
	// p.SetOff(true)
	net.ConnectLayers(outLay, hid1Lay, prjn.NewFull(), emer.Back)
	net.ConnectLayers(blaNeOutLay, hid1Lay, prjn.NewFull(), emer.Back)
	net.ConnectLayers(blaPoOutLay, hid1Lay, prjn.NewFull(), emer.Back)

	// note: can set these to do parallel threaded computation across multiple cpus
	// not worth it for this small of a model, but definitely helps for larger ones
	// if Thread {
	// 	hid2Lay.SetThread(1)
	// 	outLay.SetThread(1)
	// }

	// note: if you wanted to change a layer type from e.g., Target to Compare, do this:
	// outLay.SetType(emer.Compare)
	// that would mean that the output layer doesn't reflect target values in plus phase
	// and thus removes error-driven learning -- but stats are still computed.

	net.Defaults()
	ss.SetParams("Network", ss.LogSetParams) // only set Network params
	err := net.Build()
	if err != nil {
		log.Println(err)
		return
	}
	net.InitWts()
}

////////////////////////////////////////////////////////////////////////////////
// 	    Init, utils

// Init restarts the run, and initializes everything, including network weights
// and resets the epoch log table
func (ss *Sim) Init() {
	rand.Seed(ss.RndSeed)
	ss.ConfigEnv() // re-config env just in case a different set of patterns was
	// selected or patterns have been modified etc
	ss.StopNow = false
	ss.SetParams("", ss.LogSetParams) // all sheets
	ss.NewRun()
	ss.UpdateView("train")
}

// NewRndSeed gets a new random seed based on current time -- otherwise uses
// the same random seed for every run
func (ss *Sim) NewRndSeed() {
	ss.RndSeed = time.Now().UnixNano()
}

// DONE Counters returns a string of the current counter state
// use tabs to achieve a reasonable formatting overall
// and add a few tabs at the end to allow for expansion..
// Redefined by DH to add the sleep state
func (ss *Sim) Counters(state string) string {
	switch state {
	case "train":
		return fmt.Sprintf("Run:\t%d\tEpoch:\t%d\tTrial:\t%d\tName:\t%v\t\t\t", ss.TrainEnv.Run.Cur, ss.TrainEnv.Epoch.Cur, ss.TrainEnv.Trial.Cur, ss.TrainEnv.TrialName)
	case "test":
		return fmt.Sprintf("Run:\t%d\tEpoch:\t%d\tTrial:\t%d\tName:\t%v\t\t\t", ss.TrainEnv.Run.Cur, ss.TrainEnv.Epoch.Cur, ss.TestEnv.Trial.Cur, ss.TestEnv.TrialName)
	case "sleep":
		return fmt.Sprintf("Run:\t%d\tEpoch:\t%d\tSleep_Trial:\t%d\t\t\t", ss.TrainEnv.Run.Cur, ss.TrainEnv.Epoch.Cur, ss.SleepEnv.Trial.Cur)
	}
	return ""
}

func (ss *Sim) UpdateView(state string) {
	if ss.NetView != nil {
		// note: essential to use Go version of update when called from another goroutine
		ss.NetView.GoUpdate(ss.Counters(state)) // note: using counters is significantly slower..
	}
}

// TODO SleepCycInit handles all initialization at start of new sleep trials, including computing
// netinput scaling from running average activation etc.
// Added by DH
func (ss *Sim) SleepCycInit() {
	// Set all layers to be hidden
	// Set all layers into random activation
	fmt.Println("Now I am going to reset the layers.... May cause some damages here.")
	// Need to connect hidden back to input.
	//ss.SetInBackPrjnOff(false)

	// Set the parameters
	ss.SetParamsSet("Sleep", "", true)

	ss.Net.Sleep(&ss.Time)

	// Set all layers to be random activation and no clamping.
	for _, ly := range ss.Net.Layers {
		ly.SetType(emer.Hidden)
		//ly.Act.Clamp.Hard = false
		//	fmt.Println("Here is a sanity check, the type of layer now should be 0, and it is:%d", int(ly.Type()))
		for ni := range ly.(*leabra.Layer).Neurons {
			nrn := &ly.(*leabra.Layer).Neurons[ni]
			if nrn.IsOff() {
				continue
			}
			//	fmt.Println("Layer: %v, Neuron: %d, Original activation: %d", ly.Label(), ni, nrn.Act)
			nrn.Act = rand.Float32()
			//fmt.Println("Layer: %v, Neuron: %d, Random activation: %d", ly.Label(), ni, nrn.Act)
		}
	}
	fmt.Println("I reset the network layers! Hope everything is still fine....")
	// Set all the parameters to sleep mode - need to replicate the SRAvgCaiSynDepConSpec file from the older version
	// TODO Not yet done.

	//	fmt.Scanln()
	ss.UpdateView("sleep")
}

// TODO BackToWake set the model back to training model
// Added by DH
func (ss *Sim) BackToWake() {
	// Set the input and output layers back to normal.
	inLay := ss.Net.LayerByName("Input").(*leabra.Layer)
	blaNeInLay := ss.Net.LayerByName("Ne").(*leabra.Layer)
	blaPoInLay := ss.Net.LayerByName("Po").(*leabra.Layer)
	outLay := ss.Net.LayerByName("Output").(*leabra.Layer)
	blaNeOutLay := ss.Net.LayerByName("Ne_Out").(*leabra.Layer)
	blaPoOutLay := ss.Net.LayerByName("Po_Out").(*leabra.Layer)

	inLay.SetType(emer.Input)
	blaNeInLay.SetType(emer.Input)
	blaPoInLay.SetType(emer.Input)
	outLay.SetType(emer.Target)
	blaNeOutLay.SetType(emer.Target)
	blaPoOutLay.SetType(emer.Target)

	// Turn the back prjn from hidden to input off.
	//ss.SetInBackPrjnOff(true)

	// Set the parameters
	ss.SetParamsSet("Base", "", true)

	// If Inhibition oscillation is on, set it back to base
	if ss.InhibOscil {
		ss.Net.InhibOscilMute(&ss.Time)
	}

	// Set all parameters back to wake
	ss.Net.Wake(&ss.Time)

	//fmt.Println("All layers should be back to normal. Here is a sanity check, the type of inLay is: %d", int(inLay.Type()))
	//fmt.Println("All layers should be back to normal. Here is a sanity check, the type of outLay is: %d", int(outLay.Type()))
}

////////////////////////////////////////////////////////////////////////////////
// 	    Running the Network, starting bottom-up..

// AlphaCyc runs one alpha-cycle (100 msec, 4 quarters)			 of processing.
// External inputs must have already been applied prior to calling,
// using ApplyExt method on relevant layers (see TrainTrial, TestTrial).
// If train is true, then learning DWt or WtFmDWt calls are made.
// Handles netview updating within scope of AlphaCycle
func (ss *Sim) AlphaCyc(state string) {
	// ss.Win.PollEvents() // this can be used instead of running in a separate goroutine
	viewUpdt := ss.TrainUpdt
	switch state {
	case "test":
		viewUpdt = ss.TestUpdt
	case "sleep":
		viewUpdt = ss.SleepUpdt
	}
	ss.Net.AlphaCycInit()
	ss.Time.AlphaCycStart()
	for qtr := 0; qtr < 4; qtr++ {
		for cyc := 0; cyc < ss.Time.CycPerQtr; cyc++ {
			ss.Net.Cycle(&ss.Time, false)
			//			ss.Net.Cycle(&ss.Time, true) // For syndep
			if state == "test" {
				ss.LogTstCyc(ss.TstCycLog, ss.Time.Cycle)
			}
			ss.Time.CycleInc()
			if ss.ViewOn {
				switch viewUpdt {
				case leabra.Cycle:
					ss.UpdateView(state)
				case leabra.FastSpike:
					if (cyc+1)%10 == 0 {
						ss.UpdateView(state)
					}
				}
			}
		}
		ss.Net.QuarterFinal(&ss.Time)
		ss.Time.QuarterInc()
		if ss.ViewOn {
			switch viewUpdt {
			case leabra.Quarter:
				ss.UpdateView(state)
			case leabra.Phase:
				if qtr >= 2 {
					ss.UpdateView(state)
				}
			}
		}
	}

	if state == "train" {
		ss.Net.DWt()
		ss.Net.WtFmDWt()
	}
	if ss.ViewOn && viewUpdt == leabra.AlphaCycle {
		ss.UpdateView(state)
	}
	if state == "test" {
		ss.TstCycPlot.GoUpdate() // make sure up-to-date at end
	}
}

// This is a function called to print the hidden network activities, as a monitor.
func (ss *Sim) MonSlpCyc() {
	hid1Lay := ss.Net.LayerByName("Hidden1").(*leabra.Layer)
	for ni := range hid1Lay.Neurons {
		nrn := &hid1Lay.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		fmt.Println("Layer: hidden1, Neuron: %d, Original activation: %d", ni, nrn.Act)
	}
	fmt.Scanln()
}

// TODO the sleepCyc is a modification from the AlphaCycle, Added by DH based on Anna's design.
// The Sleep Mode structure:
// - SleepTrial is a modification of the TrainTrial
// - SleepCyc is a modification of AlphaCyc, similar to the original SleepTrial program by Anna
// - SleepCyc is also built upon Cycle. How? Don't know yet.

func (ss *Sim) SleepCyc(WakeReplay bool) {
	// fmt.Println("I am in the SleepCyc!!!! Can't believe it!!!")
	//ss.MaxSlpCyc = 50
	viewUpdt := ss.SleepUpdt
	//fmt.Scanln()
	ss.SleepCycInit()
	//fmt.Scanln()
	fmt.Println("Sleep mode officially starts here.")
	ss.Time.SleepCycStart()
	for cyc := 0; cyc < ss.MaxSlpCyc; cyc++ {
		// Need to init the network here. How? Don't know yet. It was the SetToSleep program in Anna's version.
		// Need to set the network to sleep mode, meaning set the input and output to be "hidden"
		//	fmt.Println("%d real sleep cyc. Wish me luck!", cyc)
		if (cyc+1)%10 == 0 {
			ss.Net.InitGInc()
		}
		if ss.InhibOscil {
			ss.Net.InhibOscil(&ss.Time, cyc)
		}
		ss.Net.Cycle(&ss.Time, true)
		//fmt.Scanln()
		//	fmt.Println("Sleep cyc works? Now what?")
		ss.Time.CycleInc()
		//	fmt.Println("I think everything works through but I am not sure.")
		// Logging the SlpCycLog
		ss.LogSlpCyc(ss.SlpCycLog, ss.Time.Cycle)
		if ss.ViewOn {
			switch viewUpdt {
			case leabra.Cycle:
				//			fmt.Scanln()
				ss.UpdateView("sleep")
			case leabra.FastSpike:
				if (cyc+1)%10 == 0 {
					//					fmt.Println("Should be seeing some flashing in the netview at this point.")
					ss.UpdateView("sleep")
					//ss.MonSlpCyc()
				}
			case leabra.Quarter:
				if (cyc+1)%25 == 0 {
					//				fmt.Println("Should be seeing some flashing in the netview at this point.")
					ss.UpdateView("sleep")
				}
			case leabra.Phase:
				if (cyc+1)%100 == 0 {
					//			fmt.Println("Should be seeing some flashing in the netview at this point.")
					ss.UpdateView("sleep")
				}
			}
		}
		// In the AlphaCyc(), we have quarters, but during sleep, I did not add quarters - maybe later?
	}
	//ss.Net.MonChge(&ss.Time)
	if ss.ViewOn {
		//fmt.Println("Should be seeing some flashing in the netview at this point.")
		//fmt.Scanln()
		ss.UpdateView("sleep") // Update at the end of each sleep trials
	}
}

// ApplyInputs applies input patterns from given environment.
// It is good practice to have this be a separate method with appropriate
// args so that it can be used for various different contexts
// (training, testing, etc).
func (ss *Sim) ApplyInputs(en env.Env) {
	ss.Net.InitExt() // clear any existing inputs -- not strictly necessary if always
	// going to the same layers, but good practice and cheap anyway

	inLay := ss.Net.LayerByName("Input").(*leabra.Layer)
	blaNeInLay := ss.Net.LayerByName("Ne").(*leabra.Layer)
	blaPoInLay := ss.Net.LayerByName("Po").(*leabra.Layer)
	outLay := ss.Net.LayerByName("Output").(*leabra.Layer)
	blaNeOutLay := ss.Net.LayerByName("Ne_Out").(*leabra.Layer)
	blaPoOutLay := ss.Net.LayerByName("Po_Out").(*leabra.Layer)

	inPats_In := en.State(inLay.Nm)
	inPats_Bla_Ne := en.State(blaNeInLay.Nm)
	inPats_Bla_Po := en.State(blaPoInLay.Nm)
	if (inPats_In != nil) || (inPats_Bla_Ne != nil) || (inPats_Bla_Po != nil) {
		inLay.ApplyExt(inPats_In)
		blaNeInLay.ApplyExt(inPats_Bla_Ne)
		blaPoInLay.ApplyExt(inPats_Bla_Po)
	}
	outPats_Out := en.State(outLay.Nm)
	outPats_Bla_Ne := en.State(blaNeOutLay.Nm)
	outPats_Bla_Po := en.State(blaPoOutLay.Nm)
	if (inPats_In != nil) || (inPats_Bla_Ne != nil) || (inPats_Bla_Po != nil) {
		outLay.ApplyExt(outPats_Out)
		blaNeOutLay.ApplyExt(outPats_Bla_Ne)
		blaPoOutLay.ApplyExt(outPats_Bla_Po)
	}
}

// TODO SleepTrial runs one trial of sleep
// Similar to the original SetToSleep program by Anna.

func (ss *Sim) SleepTrial() {
	//fmt.Println("I am here in the SleepTrial, everything means still fine.")
	ss.SleepEnv.Step() // Should it step or not?
	//fmt.Println("So I believe the system make one step forward... Historical!!!!")
	// Query counters FIRST
	_, _, chg := ss.SleepEnv.Counter(env.Epoch)
	if chg {
		//	fmt.Println("About to update view, not sure what will happen.")
		if ss.ViewOn && ss.SleepUpdt > leabra.AlphaCycle {
			ss.UpdateView("sleep")
		}
		//ss.LogTstEpc(ss.TstEpcLog)
		return
	}
	//fmt.Println("I survived the mysterious counters... So what is next?")
	ss.SleepCyc(true)        // Need to implement this
	ss.SlpCycPlot.GoUpdate() // make sure up-to-date at end
	ss.TrialStats(true)      // I think this is necessary, but need to check.
	ss.BackToWake()
}

// TrainTrial runs one trial of training using TrainEnv
func (ss *Sim) TrainTrial() {
	ss.TrainEnv.Step() // the Env encapsulates and manages all counter state

	// Key to query counters FIRST because current state is in NEXT epoch
	// if epoch counter has changed
	epc, _, chg := ss.TrainEnv.Counter(env.Epoch)
	if chg {
		ss.LogTrnEpc(ss.TrnEpcLog)
		if ss.ViewOn && ss.TrainUpdt > leabra.AlphaCycle {
			ss.UpdateView("train")
		}
		if epc%ss.TestInterval == 0 { // note: epc is *next* so won't trigger first time
			ss.TestAll()
		}
		if epc >= ss.MaxEpcs { // done with training..
			ss.RunEnd()
			if ss.TrainEnv.Run.Incr() { // we are done!
				ss.StopNow = true
				return
			} else {
				ss.NewRun()
				return
			}
		}
	}

	// TODO Added by DH: Here should be the good place to check if we should start a sleep
	if ss.Sleep {
		if (epc > 1) && (ss.EpcSSE < 0.2) {
			// Save trained weights first
			fnm := ss.WeightsFileName()
			fmt.Printf("Saving Weights to: %v\n", fnm)
			ss.Net.SaveWtsJSON(gi.FileName(fnm))
			ss.Net.InitExt() // clear any existing inputs -- not strictly necessary if always
			//fmt.Println("I stepped into the sleeping black hole...")
			ss.SleepTrial()
		}
	}

	ss.ApplyInputs(&ss.TrainEnv)
	ss.AlphaCyc("train") // train
	ss.TrialStats(true)  // accumulate
}

// RunEnd is called at the end of a run -- save weights, record final log, etc here
func (ss *Sim) RunEnd() {
	ss.LogRun(ss.RunLog)
	if ss.SaveWts {
		fnm := ss.WeightsFileName()
		fmt.Printf("Saving Weights to: %v\n", fnm)
		ss.Net.SaveWtsJSON(gi.FileName(fnm))
	}
}

// NewRun intializes a new run of the model, using the TrainEnv.Run counter
// for the new run value
func (ss *Sim) NewRun() {
	run := ss.TrainEnv.Run.Cur
	ss.TrainEnv.Init(run)
	ss.TestEnv.Init(run)
	ss.SleepEnv.Init(run)
	ss.Time.Reset()
	ss.Net.InitWts()
	ss.InitStats()
	ss.TrnEpcLog.SetNumRows(0)
	ss.TstEpcLog.SetNumRows(0)
}

// intializes the network properties
func (ss *Sim) SetInBackPrjnOff(off bool) {
	// Need to connect hidden back to input.
	inLay := ss.Net.LayerByName("Input").(*leabra.Layer)
	blaNeInLay := ss.Net.LayerByName("Ne").(*leabra.Layer)
	blaPoInLay := ss.Net.LayerByName("Po").(*leabra.Layer)

	// Turn on all the RcvPrjns
	for _, p := range inLay.RcvPrjns {
		p.SetOff(off)
	}
	for _, p := range blaPoInLay.RcvPrjns {
		p.SetOff(off)
	}
	for _, p := range blaNeInLay.RcvPrjns {
		p.SetOff(off)
	}
}

// InitStats initializes all the statistics, especially important for the
// cumulative epoch stats -- called at start of new run
func (ss *Sim) InitStats() {
	// accumulators
	ss.SumSSE = 0
	ss.SumAvgSSE = 0
	ss.SumCosDiff = 0
	ss.CntErr = 0
	ss.FirstZero = -1
	// clear rest just to make Sim look initialized
	ss.TrlSSE = 0
	ss.TrlAvgSSE = 0
	ss.EpcSSE = 0
	ss.EpcAvgSSE = 0
	ss.EpcPctErr = 0
	ss.EpcCosDiff = 0
	ss.AvgLaySim = 0
}

// TrialStats computes the trial-level statistics and adds them to the epoch accumulators if
// accum is true.  Note that we're accumulating stats here on the Sim side so the
// core algorithm side remains as simple as possible, and doesn't need to worry about
// different time-scales over which stats could be accumulated etc.
// You can also aggregate directly from log data, as is done for testing stats
func (ss *Sim) TrialStats(accum bool) (sse, avgsse, cosdiff float64) {
	outLay := ss.Net.LayerByName("Output").(*leabra.Layer)
	ss.TrlCosDiff = float64(outLay.CosDiff.Cos)
	ss.TrlSSE, ss.TrlAvgSSE = outLay.MSE(0.5) // 0.5 = per-unit tolerance -- right side of .5
	if accum {
		ss.SumSSE += ss.TrlSSE
		ss.SumAvgSSE += ss.TrlAvgSSE
		ss.SumCosDiff += ss.TrlCosDiff
		if ss.TrlSSE != 0 {
			ss.CntErr++
		}
	}
	return
}

// TrainEpoch runs training trials for remainder of this epoch
func (ss *Sim) TrainEpoch() {
	ss.StopNow = false
	curEpc := ss.TrainEnv.Epoch.Cur
	for {
		ss.TrainTrial()
		if ss.StopNow || ss.TrainEnv.Epoch.Cur != curEpc {
			break
		}
	}
	ss.Stopped()
}

// TrainRun runs training trials for remainder of run
func (ss *Sim) TrainRun() {
	ss.StopNow = false
	curRun := ss.TrainEnv.Run.Cur
	for {
		ss.TrainTrial()
		if ss.StopNow || ss.TrainEnv.Run.Cur != curRun {
			break
		}
	}
	ss.Stopped()
}

// Train runs the full training from this point onward
func (ss *Sim) Train() {
	ss.StopNow = false
	//ss.SetInBackPrjnOff(true)
	for {
		ss.TrainTrial()
		if ss.StopNow {
			break
		}
	}
	ss.Stopped()
}

// Stop tells the sim to stop running
func (ss *Sim) Stop() {
	ss.StopNow = true
}

// Stopped is called when a run method stops running -- updates the IsRunning flag and toolbar
func (ss *Sim) Stopped() {
	ss.IsRunning = false
	if ss.Win != nil {
		vp := ss.Win.WinViewport2D()
		vp.BlockUpdates()
		if ss.ToolBar != nil {
			ss.ToolBar.UpdateActions()
		}
		vp.UnblockUpdates()
		vp.SetNeedsFullRender()
	}
}

// SaveWeights saves the network weights -- when called with giv.CallMethod
// it will auto-prompt for filename
func (ss *Sim) SaveWeights(filename gi.FileName) {
	ss.Net.SaveWtsJSON(filename)
}

////////////////////////////////////////////////////////////////////////////////////////////
// Testing

// TestTrial runs one trial of testing -- always sequentially presented inputs
func (ss *Sim) TestTrial() {
	ss.TestEnv.Step()

	// Query counters FIRST
	_, _, chg := ss.TestEnv.Counter(env.Epoch)
	if chg {
		if ss.ViewOn && ss.TestUpdt > leabra.AlphaCycle {
			ss.UpdateView("test")
		}
		ss.LogTstEpc(ss.TstEpcLog)
		return
	}

	ss.ApplyInputs(&ss.TestEnv)
	ss.AlphaCyc("test")  // !train
	ss.TrialStats(false) // !accumulate
	ss.LogTstTrl(ss.TstTrlLog)
}

// TestItem tests given item which is at given index in test item list
func (ss *Sim) TestItem(idx int) {
	cur := ss.TestEnv.Trial.Cur
	ss.TestEnv.Trial.Cur = idx
	ss.TestEnv.SetTrialName()
	ss.ApplyInputs(&ss.TestEnv)
	ss.AlphaCyc("test")  // !train
	ss.TrialStats(false) // !accumulate
	ss.TestEnv.Trial.Cur = cur
}

// TestAll runs through the full set of testing items
func (ss *Sim) TestAll() {
	ss.TestEnv.Init(ss.TrainEnv.Run.Cur)
	for {
		ss.TestTrial()
		_, _, chg := ss.TestEnv.Counter(env.Epoch)
		if chg || ss.StopNow {
			break
		}
	}
}

// RunTestAll runs through the full set of testing items, has stop running = false at end -- for gui
func (ss *Sim) RunTestAll() {
	ss.StopNow = false
	ss.TestAll()
	ss.Stopped()
}

/////////////////////////////////////////////////////////////////////////
//   Params setting

// ParamsName returns name of current set of parameters
func (ss *Sim) ParamsName() string {
	if ss.ParamSet == "" {
		return "Base"
	}
	return ss.ParamSet
}

// SetParams sets the params for "Base" and then current ParamSet.
// If sheet is empty, then it applies all avail sheets (e.g., Network, Sim)
// otherwise just the named sheet
// if setMsg = true then we output a message for each param that was set.
func (ss *Sim) SetParams(sheet string, setMsg bool) error {
	if sheet == "" {
		// this is important for catching typos and ensuring that all sheets can be used
		ss.Params.ValidateSheets([]string{"Network", "Sim"})
	}
	err := ss.SetParamsSet("Base", sheet, setMsg)
	if ss.ParamSet != "" && ss.ParamSet != "Base" {
		err = ss.SetParamsSet(ss.ParamSet, sheet, setMsg)
	}
	return err
}

// SetParamsSet sets the params for given params.Set name.
// If sheet is empty, then it applies all avail sheets (e.g., Network, Sim)
// otherwise just the named sheet
// if setMsg = true then we output a message for each param that was set.
func (ss *Sim) SetParamsSet(setNm string, sheet string, setMsg bool) error {
	pset, err := ss.Params.SetByNameTry(setNm)
	if err != nil {
		return err
	}
	if sheet == "" || sheet == "Network" {
		netp, ok := pset.Sheets["Network"]
		if ok {
			ss.Net.ApplyParams(netp, setMsg)
		}
	}

	if sheet == "" || sheet == "Sim" {
		simp, ok := pset.Sheets["Sim"]
		if ok {
			simp.Apply(ss, setMsg)
		}
	}
	// note: if you have more complex environments with parameters, definitely add
	// sheets for them, e.g., "TrainEnv", "TestEnv" etc
	return err
}

func (ss *Sim) ConfigPats() {
	dt := ss.Pats
	dt.SetMetaData("name", "TrainPats")
	dt.SetMetaData("desc", "Training patterns")
	dt.SetFromSchema(etable.Schema{
		{"Name", etensor.STRING, nil, nil},
		{"Ne", etensor.FLOAT32, []int{1, 1}, []string{"Y", "X"}},
		{"Po", etensor.FLOAT32, []int{1, 1}, []string{"Y", "X"}},
		{"Input", etensor.FLOAT32, []int{5, 5}, []string{"Y", "X"}},
		{"Output", etensor.FLOAT32, []int{5, 5}, []string{"Y", "X"}},
		{"Ne_Out", etensor.FLOAT32, []int{1, 1}, []string{"Y", "X"}},
		{"Po_Out", etensor.FLOAT32, []int{1, 1}, []string{"Y", "X"}},
	}, 25)

	patgen.PermutedBinaryRows(dt.Cols[1], 6, 1, 0)
	patgen.PermutedBinaryRows(dt.Cols[2], 6, 1, 0)
	dt.SaveCSV("summer_5x5_25_gen.dat", ',', true)
}

func (ss *Sim) OpenPats() {
	dt := ss.Pats
	dt.SetMetaData("name", "TrainPats")
	dt.SetMetaData("desc", "Training patterns")
	//	err := dt.OpenCSV("summer_5x5_25.dat", etable.Tab)
	err := dt.OpenCSV("./examples/summer/summer_5x5_25.dat", etable.Tab)
	if err != nil {
		log.Println(err)
	}
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Logging

// RunName returns a name for this run that combines Tag and Params -- add this to
// any file names that are saved.
func (ss *Sim) RunName() string {
	if ss.Tag != "" {
		return ss.Tag + "_" + ss.ParamsName()
	} else {
		return ss.ParamsName()
	}
}

// RunEpochName returns a string with the run and epoch numbers with leading zeros, suitable
// for using in weights file names.  Uses 3, 5 digits for each.
func (ss *Sim) RunEpochName(run, epc int) string {
	return fmt.Sprintf("%03d_%05d", run, epc)
}

// WeightsFileName returns default current weights file name
func (ss *Sim) WeightsFileName() string {
	return ss.Net.Nm + "_" + ss.RunName() + "_" + ss.RunEpochName(ss.TrainEnv.Run.Cur, ss.TrainEnv.Epoch.Cur) + ".wts"
}

// LogFileName returns default log file name
func (ss *Sim) LogFileName(lognm string) string {
	return ss.Net.Nm + "_" + ss.RunName() + "_" + lognm + ".csv"
}

//////////////////////////////////////////////
//  SlpCycLog

// LogSlpCyc adds data from current sleep cycle to the SlpCycLog table.
// computes cycle averages prior to logging.
func (ss *Sim) LogSlpCyc(dt *etable.Table, cyc int) {
	if dt.Rows <= cyc {
		dt.SetNumRows(cyc + 1)
	}

	inLay := ss.Net.LayerByName("Input").(*leabra.Layer)
	blaNeInLay := ss.Net.LayerByName("Ne").(*leabra.Layer)
	blaPoInLay := ss.Net.LayerByName("Po").(*leabra.Layer)
	hid1Lay := ss.Net.LayerByName("Hidden1").(*leabra.Layer)
	outLay := ss.Net.LayerByName("Output").(*leabra.Layer)
	blaNeOutLay := ss.Net.LayerByName("Ne_Out").(*leabra.Layer)
	blaPoOutLay := ss.Net.LayerByName("Po_Out").(*leabra.Layer)

	ss.AvgLaySim = (inLay.Sim + blaNeInLay.Sim + blaPoInLay.Sim + hid1Lay.Sim + outLay.Sim + blaPoOutLay.Sim + blaNeOutLay.Sim) / 7

	dt.SetCellFloat("Cycle", cyc, float64(cyc))
	dt.SetCellFloat("AvgLaySim", cyc, float64(ss.AvgLaySim))
	dt.SetCellFloat("Input LaySim", cyc, float64(inLay.Sim))
	dt.SetCellFloat("BlaNeIn LaySim", cyc, float64(blaNeInLay.Sim))
	dt.SetCellFloat("BlaPoIn LaySim", cyc, float64(blaPoInLay.Sim))
	dt.SetCellFloat("Hidden1 LaySim", cyc, float64(hid1Lay.Sim))
	dt.SetCellFloat("Output LaySim", cyc, float64(outLay.Sim))
	dt.SetCellFloat("BlaNeOut LaySim", cyc, float64(blaNeOutLay.Sim))
	dt.SetCellFloat("BlaPoOut LaySim", cyc, float64(blaPoOutLay.Sim))

	if cyc%10 == 0 { // too slow to do every cyc
		// note: essential to use Go version of update when called from another goroutine
		ss.SlpCycPlot.GoUpdate()
	}
}

func (ss *Sim) ConfigSlpCycLog(dt *etable.Table) {
	dt.SetMetaData("name", "SlpCycLog")
	dt.SetMetaData("desc", "Record of activity etc over one sleep trial by cycle")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	np := 800 // max cycles
	dt.SetFromSchema(etable.Schema{
		{"Cycle", etensor.INT64, nil, nil},
		{"AvgLaySim", etensor.FLOAT64, nil, nil},
		{"Input LaySim", etensor.FLOAT64, nil, nil},
		{"BlaNeIn LaySim", etensor.FLOAT64, nil, nil},
		{"BlaPoIn LaySim", etensor.FLOAT64, nil, nil},
		{"Hidden1 LaySim", etensor.FLOAT64, nil, nil},
		{"Output LaySim", etensor.FLOAT64, nil, nil},
		{"BlaNeOut LaySim", etensor.FLOAT64, nil, nil},
		{"BlaPoOut LaySim", etensor.FLOAT64, nil, nil},
	}, np)
}

func (ss *Sim) ConfigSlpCycPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Leabra Random Associator 25 Sleep Cycle Plot"
	plt.Params.XAxisCol = "Cycle"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Cycle", false, true, 0, false, 0)
	plt.SetColParams("AvgLaySim", true, true, -1, true, 1)
	plt.SetColParams("Input LaySim", true, true, -1, true, 1)
	plt.SetColParams("BlaNeIn LaySim", true, true, -1, true, 1)
	plt.SetColParams("BlaPoIn LaySim", true, true, -1, true, 1)
	plt.SetColParams("Hid1 LaySim", true, true, -1, true, 1)
	plt.SetColParams("Output LaySim", true, true, -1, true, 1)
	plt.SetColParams("BlaNeOut LaySim", true, true, -1, true, 1)
	plt.SetColParams("BlaPoOut LaySim", true, true, -1, true, 1)
	return plt
}

//////////////////////////////////////////////
//  TrnEpcLog

// LogTrnEpc adds data from current epoch to the TrnEpcLog table.
// computes epoch averages prior to logging.
func (ss *Sim) LogTrnEpc(dt *etable.Table) {
	row := dt.Rows
	ss.TrnEpcLog.SetNumRows(row + 1)

	hid1Lay := ss.Net.LayerByName("Hidden1").(*leabra.Layer)
	outLay := ss.Net.LayerByName("Output").(*leabra.Layer)
	blaNeOutLay := ss.Net.LayerByName("Ne_Out").(*leabra.Layer)
	blaPoOutLay := ss.Net.LayerByName("Po_Out").(*leabra.Layer)

	epc := ss.TrainEnv.Epoch.Prv           // this is triggered by increment so use previous value
	nt := float64(ss.TrainEnv.Table.Len()) // number of trials in view

	ss.EpcSSE = ss.SumSSE / nt
	ss.SumSSE = 0
	ss.EpcAvgSSE = ss.SumAvgSSE / nt
	ss.SumAvgSSE = 0
	ss.EpcPctErr = float64(ss.CntErr) / nt
	ss.CntErr = 0
	ss.EpcPctCor = 1 - ss.EpcPctErr
	ss.EpcCosDiff = ss.SumCosDiff / nt
	ss.SumCosDiff = 0
	if ss.FirstZero < 0 && ss.EpcPctErr == 0 {
		ss.FirstZero = epc
	}

	dt.SetCellFloat("Run", row, float64(ss.TrainEnv.Run.Cur))
	dt.SetCellFloat("Epoch", row, float64(epc))
	dt.SetCellFloat("SSE", row, ss.EpcSSE)
	dt.SetCellFloat("AvgSSE", row, ss.EpcAvgSSE)
	dt.SetCellFloat("PctErr", row, ss.EpcPctErr)
	dt.SetCellFloat("PctCor", row, ss.EpcPctCor)
	dt.SetCellFloat("CosDiff", row, ss.EpcCosDiff)
	dt.SetCellFloat("Hid1 ActAvg", row, float64(hid1Lay.Pools[0].ActAvg.ActPAvgEff))
	dt.SetCellFloat("Out ActAvg", row, float64(outLay.Pools[0].ActAvg.ActPAvgEff))
	dt.SetCellFloat("BlaNeOut ActAvg", row, float64(blaNeOutLay.Pools[0].ActAvg.ActPAvgEff))
	dt.SetCellFloat("BlaPoOut ActAvg", row, float64(blaPoOutLay.Pools[0].ActAvg.ActPAvgEff))

	// note: essential to use Go version of update when called from another goroutine
	ss.TrnEpcPlot.GoUpdate()
	if ss.TrnEpcFile != nil {
		if ss.TrainEnv.Run.Cur == 0 && epc == 0 {
			dt.WriteCSVHeaders(ss.TrnEpcFile, etable.Tab)
		}
		dt.WriteCSVRow(ss.TrnEpcFile, row, etable.Tab, true)
	}
}

func (ss *Sim) ConfigTrnEpcLog(dt *etable.Table) {
	dt.SetMetaData("name", "TrnEpcLog")
	dt.SetMetaData("desc", "Record of performance over epochs of training")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	dt.SetFromSchema(etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Epoch", etensor.INT64, nil, nil},
		{"SSE", etensor.FLOAT64, nil, nil},
		{"AvgSSE", etensor.FLOAT64, nil, nil},
		{"PctErr", etensor.FLOAT64, nil, nil},
		{"PctCor", etensor.FLOAT64, nil, nil},
		{"CosDiff", etensor.FLOAT64, nil, nil},
		{"Hid1 ActAvg", etensor.FLOAT64, nil, nil},
		{"Out ActAvg", etensor.FLOAT64, nil, nil},
		{"BlaNeOut ActAvg", etensor.FLOAT64, nil, nil},
		{"BlaPoOut ActAvg", etensor.FLOAT64, nil, nil},
	}, 0)
}

func (ss *Sim) ConfigTrnEpcPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Leabra Random Associator 25 Epoch Plot"
	plt.Params.XAxisCol = "Epoch"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", false, true, 0, false, 0)
	plt.SetColParams("Epoch", false, true, 0, false, 0)
	plt.SetColParams("SSE", false, true, 0, false, 0)
	plt.SetColParams("AvgSSE", false, true, 0, false, 0)
	plt.SetColParams("PctErr", true, true, 0, true, 1) // default plot
	plt.SetColParams("PctCor", true, true, 0, true, 1) // default plot
	plt.SetColParams("CosDiff", false, true, 0, true, 1)
	plt.SetColParams("Hid1 ActAvg", false, true, 0, true, .5)
	plt.SetColParams("Out ActAvg", false, true, 0, true, .5)
	plt.SetColParams("BlaNeOut ActAvg", false, true, 0, true, .5)
	plt.SetColParams("BlaPoOut ActAvg", false, true, 0, true, .5)
	return plt
}

//////////////////////////////////////////////
//  TstTrlLog

// LogTstTrl adds data from current trial to the TstTrlLog table.
// log always contains number of testing items
func (ss *Sim) LogTstTrl(dt *etable.Table) {
	inLay := ss.Net.LayerByName("Input").(*leabra.Layer)
	blaNeInLay := ss.Net.LayerByName("Ne").(*leabra.Layer)
	blaPoInLay := ss.Net.LayerByName("Po").(*leabra.Layer)
	hid1Lay := ss.Net.LayerByName("Hidden1").(*leabra.Layer)
	outLay := ss.Net.LayerByName("Output").(*leabra.Layer)
	blaNeOutLay := ss.Net.LayerByName("Ne_Out").(*leabra.Layer)
	blaPoOutLay := ss.Net.LayerByName("Po_Out").(*leabra.Layer)

	epc := ss.TrainEnv.Epoch.Prv // this is triggered by increment so use previous value

	trl := ss.TestEnv.Trial.Cur

	dt.SetCellFloat("Run", trl, float64(ss.TrainEnv.Run.Cur))
	dt.SetCellFloat("Epoch", trl, float64(epc))
	dt.SetCellFloat("Trial", trl, float64(trl))
	dt.SetCellString("TrialName", trl, ss.TestEnv.TrialName)
	dt.SetCellFloat("SSE", trl, ss.TrlSSE)
	dt.SetCellFloat("AvgSSE", trl, ss.TrlAvgSSE)
	dt.SetCellFloat("CosDiff", trl, ss.TrlCosDiff)
	dt.SetCellFloat("Hid1 ActM.Avg", trl, float64(hid1Lay.Pools[0].ActM.Avg))
	dt.SetCellFloat("Out ActM.Avg", trl, float64(outLay.Pools[0].ActM.Avg))
	dt.SetCellFloat("BlaNeOut ActM.Avg", trl, float64(blaNeOutLay.Pools[0].ActM.Avg))
	dt.SetCellFloat("BlaPoOut ActM.Avg", trl, float64(blaPoOutLay.Pools[0].ActM.Avg))

	dt.SetCellTensor("InAct", trl, inLay.UnitValsTensor("Act"))
	dt.SetCellTensor("BlaNeInAct", trl, blaNeInLay.UnitValsTensor("Act"))
	dt.SetCellTensor("BlaPoInAct", trl, blaPoInLay.UnitValsTensor("Act"))
	dt.SetCellTensor("OutActM", trl, outLay.UnitValsTensor("ActM"))
	dt.SetCellTensor("OutActP", trl, outLay.UnitValsTensor("ActP"))
	dt.SetCellTensor("BlaNeOutAct", trl, blaNeOutLay.UnitValsTensor("Act"))
	dt.SetCellTensor("BlaPoOutAct", trl, blaPoOutLay.UnitValsTensor("Act"))

	// note: essential to use Go version of update when called from another goroutine
	ss.TstTrlPlot.GoUpdate()
}

func (ss *Sim) ConfigTstTrlLog(dt *etable.Table) {
	inLay := ss.Net.LayerByName("Input").(*leabra.Layer)
	blaNeInLay := ss.Net.LayerByName("Ne").(*leabra.Layer)
	blaPoInLay := ss.Net.LayerByName("Po").(*leabra.Layer)
	outLay := ss.Net.LayerByName("Output").(*leabra.Layer)
	blaNeOutLay := ss.Net.LayerByName("Ne_Out").(*leabra.Layer)
	blaPoOutLay := ss.Net.LayerByName("Po_Out").(*leabra.Layer)

	dt.SetMetaData("name", "TstTrlLog")
	dt.SetMetaData("desc", "Record of testing per input pattern")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	nt := ss.TestEnv.Table.Len() // number in view
	dt.SetFromSchema(etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Epoch", etensor.INT64, nil, nil},
		{"Trial", etensor.INT64, nil, nil},
		{"TrialName", etensor.STRING, nil, nil},
		{"SSE", etensor.FLOAT64, nil, nil},
		{"AvgSSE", etensor.FLOAT64, nil, nil},
		{"CosDiff", etensor.FLOAT64, nil, nil},
		{"Hid1 ActM.Avg", etensor.FLOAT64, nil, nil},
		{"Out ActM.Avg", etensor.FLOAT64, nil, nil},
		{"BlaNeOut ActM.Avg", etensor.FLOAT64, nil, nil},
		{"BlaPoOut ActM.Avg", etensor.FLOAT64, nil, nil},
		{"InAct", etensor.FLOAT64, inLay.Shp.Shp, nil},
		{"BlaNeInAct", etensor.FLOAT64, blaNeInLay.Shp.Shp, nil},
		{"BlaPoInAct", etensor.FLOAT64, blaPoInLay.Shp.Shp, nil},
		{"BlaNeOutAct", etensor.FLOAT64, blaNeOutLay.Shp.Shp, nil},
		{"BlaPoOutAct", etensor.FLOAT64, blaPoOutLay.Shp.Shp, nil},
		{"OutActM", etensor.FLOAT64, outLay.Shp.Shp, nil},
		{"OutActP", etensor.FLOAT64, outLay.Shp.Shp, nil},
	}, nt)
}

func (ss *Sim) ConfigTstTrlPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Leabra Random Associator 25 Test Trial Plot"
	plt.Params.XAxisCol = "Trial"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", false, true, 0, false, 0)
	plt.SetColParams("Epoch", false, true, 0, false, 0)
	plt.SetColParams("Trial", false, true, 0, false, 0)
	plt.SetColParams("TrialName", false, true, 0, false, 0)
	plt.SetColParams("SSE", false, true, 0, false, 0)
	plt.SetColParams("AvgSSE", true, true, 0, false, 0)
	plt.SetColParams("CosDiff", true, true, 0, true, 1)
	plt.SetColParams("Hid1 ActM.Avg", true, true, 0, true, .5)
	plt.SetColParams("Out ActM.Avg", true, true, 0, true, .5)
	plt.SetColParams("BlaNeOut ActM.Avg", true, true, 0, true, .5)
	plt.SetColParams("BlaPoOut ActM.Avg", true, true, 0, true, .5)

	plt.SetColParams("InAct", false, true, 0, true, 1)
	plt.SetColParams("BlaNeInAct", false, true, 0, true, 1)
	plt.SetColParams("BlaPoInAct", false, true, 0, true, 1)
	plt.SetColParams("OutActM", false, true, 0, true, 1)
	plt.SetColParams("OutActP", false, true, 0, true, 1)
	plt.SetColParams("BlaNeOutAct", false, true, 0, true, 1)
	plt.SetColParams("BlaPoOutAct", false, true, 0, true, 1)
	return plt
}

//////////////////////////////////////////////
//  TstEpcLog

func (ss *Sim) LogTstEpc(dt *etable.Table) {
	row := dt.Rows
	dt.SetNumRows(row + 1)

	trl := ss.TstTrlLog
	tix := etable.NewIdxView(trl)
	epc := ss.TrainEnv.Epoch.Prv // ?

	// note: this shows how to use agg methods to compute summary data from another
	// data table, instead of incrementing on the Sim
	dt.SetCellFloat("Run", row, float64(ss.TrainEnv.Run.Cur))
	dt.SetCellFloat("Epoch", row, float64(epc))
	dt.SetCellFloat("SSE", row, agg.Sum(tix, "SSE")[0])
	dt.SetCellFloat("AvgSSE", row, agg.Mean(tix, "AvgSSE")[0])
	dt.SetCellFloat("PctErr", row, agg.PropIf(tix, "SSE", func(idx int, val float64) bool {
		return val > 0
	})[0])
	dt.SetCellFloat("PctCor", row, agg.PropIf(tix, "SSE", func(idx int, val float64) bool {
		return val == 0
	})[0])
	dt.SetCellFloat("CosDiff", row, agg.Mean(tix, "CosDiff")[0])

	trlix := etable.NewIdxView(trl)
	trlix.Filter(func(et *etable.Table, row int) bool {
		return et.CellFloat("SSE", row) > 0 // include error trials
	})
	ss.TstErrLog = trlix.NewTable()

	allsp := split.All(trlix)
	split.Agg(allsp, "SSE", agg.AggSum)
	split.Agg(allsp, "AvgSSE", agg.AggMean)
	split.Agg(allsp, "InAct", agg.AggMean)
	split.Agg(allsp, "BlaNeInAct", agg.AggMean)
	split.Agg(allsp, "BlaPoInAct", agg.AggMean)
	split.Agg(allsp, "OutActM", agg.AggMean)
	split.Agg(allsp, "OutActP", agg.AggMean)
	split.Agg(allsp, "BlaNeOutAct", agg.AggMean)
	split.Agg(allsp, "BlaPoOutAct", agg.AggMean)

	ss.TstErrStats = allsp.AggsToTable(false)

	// note: essential to use Go version of update when called from another goroutine
	ss.TstEpcPlot.GoUpdate()
}

func (ss *Sim) ConfigTstEpcLog(dt *etable.Table) {
	dt.SetMetaData("name", "TstEpcLog")
	dt.SetMetaData("desc", "Summary stats for testing trials")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	dt.SetFromSchema(etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Epoch", etensor.INT64, nil, nil},
		{"SSE", etensor.FLOAT64, nil, nil},
		{"AvgSSE", etensor.FLOAT64, nil, nil},
		{"PctErr", etensor.FLOAT64, nil, nil},
		{"PctCor", etensor.FLOAT64, nil, nil},
		{"CosDiff", etensor.FLOAT64, nil, nil},
	}, 0)
}

func (ss *Sim) ConfigTstEpcPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Leabra Random Associator 25 Testing Epoch Plot"
	plt.Params.XAxisCol = "Epoch"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", false, true, 0, false, 0)
	plt.SetColParams("Epoch", false, true, 0, false, 0)
	plt.SetColParams("SSE", false, true, 0, false, 0)
	plt.SetColParams("AvgSSE", false, true, 0, false, 0)
	plt.SetColParams("PctErr", true, true, 0, true, 1) // default plot
	plt.SetColParams("PctCor", true, true, 0, true, 1) // default plot
	plt.SetColParams("CosDiff", false, true, 0, true, 1)
	return plt
}

//////////////////////////////////////////////
//  TstCycLog

// LogTstCyc adds data from current trial to the TstCycLog table.
// log just has 100 cycles, is overwritten
func (ss *Sim) LogTstCyc(dt *etable.Table, cyc int) {
	if dt.Rows <= cyc {
		dt.SetNumRows(cyc + 1)
	}

	hid1Lay := ss.Net.LayerByName("Hidden1").(*leabra.Layer)
	outLay := ss.Net.LayerByName("Output").(*leabra.Layer)
	blaNeOutLay := ss.Net.LayerByName("Ne_Out").(*leabra.Layer)
	blaPoOutLay := ss.Net.LayerByName("Po_Out").(*leabra.Layer)

	dt.SetCellFloat("Cycle", cyc, float64(cyc))
	dt.SetCellFloat("Hid1 Ge.Avg", cyc, float64(hid1Lay.Pools[0].Ge.Avg))
	dt.SetCellFloat("Out Ge.Avg", cyc, float64(outLay.Pools[0].Ge.Avg))
	dt.SetCellFloat("BlaNeOut Ge.Avg", cyc, float64(blaNeOutLay.Pools[0].Ge.Avg))
	dt.SetCellFloat("BlaPoOut Ge.Avg", cyc, float64(blaPoOutLay.Pools[0].Ge.Avg))
	dt.SetCellFloat("Hid1 Act.Avg", cyc, float64(hid1Lay.Pools[0].Act.Avg))
	dt.SetCellFloat("Out Act.Avg", cyc, float64(outLay.Pools[0].Act.Avg))
	dt.SetCellFloat("BlaNeOut Act.Avg", cyc, float64(blaNeOutLay.Pools[0].Act.Avg))
	dt.SetCellFloat("BlaPoOut Act.Avg", cyc, float64(blaPoOutLay.Pools[0].Act.Avg))

	if cyc%10 == 0 { // too slow to do every cyc
		// note: essential to use Go version of update when called from another goroutine
		ss.TstCycPlot.GoUpdate()
	}
}

func (ss *Sim) ConfigTstCycLog(dt *etable.Table) {
	dt.SetMetaData("name", "TstCycLog")
	dt.SetMetaData("desc", "Record of activity etc over one trial by cycle")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	np := 100 // max cycles
	dt.SetFromSchema(etable.Schema{
		{"Cycle", etensor.INT64, nil, nil},
		{"Hid1 Ge.Avg", etensor.FLOAT64, nil, nil},
		{"Out Ge.Avg", etensor.FLOAT64, nil, nil},
		{"BlaNeOut Ge.Avg", etensor.FLOAT64, nil, nil},
		{"BlaPoOut Ge.Avg", etensor.FLOAT64, nil, nil},
		{"Hid1 Act.Avg", etensor.FLOAT64, nil, nil},
		{"Out Act.Avg", etensor.FLOAT64, nil, nil},
		{"BlaNeOut Act.Avg", etensor.FLOAT64, nil, nil},
		{"BlaPoOut Act.Avg", etensor.FLOAT64, nil, nil},
	}, np)
}

func (ss *Sim) ConfigTstCycPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Leabra Random Associator 25 Test Cycle Plot"
	plt.Params.XAxisCol = "Cycle"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Cycle", false, true, 0, false, 0)
	plt.SetColParams("Hid1 Ge.Avg", true, true, 0, true, .5)
	plt.SetColParams("Out Ge.Avg", true, true, 0, true, .5)
	plt.SetColParams("BlaNeOut Ge.Avg", true, true, 0, true, .5)
	plt.SetColParams("BlaPoOut Ge.Avg", true, true, 0, true, .5)
	plt.SetColParams("Hid1 Act.Avg", true, true, 0, true, .5)
	plt.SetColParams("Out Act.Avg", true, true, 0, true, .5)
	plt.SetColParams("BlaNeOut Act.Avg", true, true, 0, true, .5)
	plt.SetColParams("BlaPoOut Act.Avg", true, true, 0, true, .5)
	return plt
}

//////////////////////////////////////////////
//  RunLog

// LogRun adds data from current run to the RunLog table.
func (ss *Sim) LogRun(dt *etable.Table) {
	run := ss.TrainEnv.Run.Cur // this is NOT triggered by increment yet -- use Cur
	row := dt.Rows
	ss.RunLog.SetNumRows(row + 1)

	epclog := ss.TrnEpcLog
	// compute mean over last N epochs for run level
	nlast := 10
	epcix := etable.NewIdxView(epclog)
	epcix.Idxs = epcix.Idxs[epcix.Len()-nlast-1:]

	params := ss.ParamsName()

	dt.SetCellFloat("Run", row, float64(run))
	dt.SetCellString("Params", row, params)
	dt.SetCellFloat("FirstZero", row, float64(ss.FirstZero))
	dt.SetCellFloat("SSE", row, agg.Mean(epcix, "SSE")[0])
	dt.SetCellFloat("AvgSSE", row, agg.Mean(epcix, "AvgSSE")[0])
	dt.SetCellFloat("PctErr", row, agg.Mean(epcix, "PctErr")[0])
	dt.SetCellFloat("PctCor", row, agg.Mean(epcix, "PctCor")[0])
	dt.SetCellFloat("CosDiff", row, agg.Mean(epcix, "CosDiff")[0])

	runix := etable.NewIdxView(dt)
	spl := split.GroupBy(runix, []string{"Params"})
	split.Desc(spl, "FirstZero")
	split.Desc(spl, "PctCor")
	ss.RunStats = spl.AggsToTable(false)

	// note: essential to use Go version of update when called from another goroutine
	ss.RunPlot.GoUpdate()
	if ss.RunFile != nil {
		if row == 0 {
			dt.WriteCSVHeaders(ss.RunFile, etable.Tab)
		}
		dt.WriteCSVRow(ss.RunFile, row, etable.Tab, true)
	}
}

func (ss *Sim) ConfigRunLog(dt *etable.Table) {
	dt.SetMetaData("name", "RunLog")
	dt.SetMetaData("desc", "Record of performance at end of training")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	dt.SetFromSchema(etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Params", etensor.STRING, nil, nil},
		{"FirstZero", etensor.FLOAT64, nil, nil},
		{"SSE", etensor.FLOAT64, nil, nil},
		{"AvgSSE", etensor.FLOAT64, nil, nil},
		{"PctErr", etensor.FLOAT64, nil, nil},
		{"PctCor", etensor.FLOAT64, nil, nil},
		{"CosDiff", etensor.FLOAT64, nil, nil},
	}, 0)
}

func (ss *Sim) ConfigRunPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Leabra Random Associator 25 Run Plot"
	plt.Params.XAxisCol = "Run"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", false, true, 0, false, 0)
	plt.SetColParams("FirstZero", true, true, 0, false, 0) // default plot
	plt.SetColParams("SSE", false, true, 0, false, 0)
	plt.SetColParams("AvgSSE", false, true, 0, false, 0)
	plt.SetColParams("PctErr", false, true, 0, true, 1)
	plt.SetColParams("PctCor", false, true, 0, true, 1)
	plt.SetColParams("CosDiff", false, true, 0, true, 1)
	return plt
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Gui

// ConfigGui configures the GoGi gui interface for this simulation,
func (ss *Sim) ConfigGui() *gi.Window {
	width := 1600
	height := 1200

	gi.SetAppName("summer")
	gi.SetAppAbout(`This demonstrates a basic Leabra model. See <a href="https://github.com/emer/emergent">emergent on GitHub</a>.</p>`)

	win := gi.NewWindow2D("summer", "Leabra Random Associator", width, height, true)
	ss.Win = win

	vp := win.WinViewport2D()
	updt := vp.UpdateStart()

	mfr := win.SetMainFrame()

	tbar := gi.AddNewToolBar(mfr, "tbar")
	tbar.SetStretchMaxWidth()
	ss.ToolBar = tbar

	split := gi.AddNewSplitView(mfr, "split")
	split.Dim = gi.X
	split.SetStretchMaxWidth()
	split.SetStretchMaxHeight()

	sv := giv.AddNewStructView(split, "sv")
	sv.SetStruct(ss)

	tv := gi.AddNewTabView(split, "tv")

	nv := tv.AddNewTab(netview.KiT_NetView, "NetView").(*netview.NetView)
	nv.Var = "Act"
	// nv.Params.ColorMap = "Jet" // default is ColdHot
	// which fares pretty well in terms of discussion here:
	// https://matplotlib.org/tutorials/colors/colormaps.html
	nv.SetNet(ss.Net)
	ss.NetView = nv

	plt := tv.AddNewTab(eplot.KiT_Plot2D, "TrnEpcPlot").(*eplot.Plot2D)
	ss.TrnEpcPlot = ss.ConfigTrnEpcPlot(plt, ss.TrnEpcLog)

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "TstTrlPlot").(*eplot.Plot2D)
	ss.TstTrlPlot = ss.ConfigTstTrlPlot(plt, ss.TstTrlLog)

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "TstCycPlot").(*eplot.Plot2D)
	ss.TstCycPlot = ss.ConfigTstCycPlot(plt, ss.TstCycLog)

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "TstEpcPlot").(*eplot.Plot2D)
	ss.TstEpcPlot = ss.ConfigTstEpcPlot(plt, ss.TstEpcLog)

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "SlpCycPlot").(*eplot.Plot2D)
	ss.SlpCycPlot = ss.ConfigSlpCycPlot(plt, ss.SlpCycLog)

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "RunPlot").(*eplot.Plot2D)
	ss.RunPlot = ss.ConfigRunPlot(plt, ss.RunLog)

	split.SetSplits(.3, .7)

	tbar.AddAction(gi.ActOpts{Label: "Init", Icon: "update", Tooltip: "Initialize everything including network weights, and start over.  Also applies current params.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.Init()
		vp.SetNeedsFullRender()
	})

	tbar.AddAction(gi.ActOpts{Label: "Train", Icon: "run", Tooltip: "Starts the network training, picking up from wherever it may have left off.  If not stopped, training will complete the specified number of Runs through the full number of Epochs of training, with testing automatically occuring at the specified interval.",
		UpdateFunc: func(act *gi.Action) {
			act.SetActiveStateUpdt(!ss.IsRunning)
		}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			// ss.Train()
			go ss.Train()
		}
	})

	tbar.AddAction(gi.ActOpts{Label: "Stop", Icon: "stop", Tooltip: "Interrupts running.  Hitting Train again will pick back up where it left off.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.Stop()
	})

	tbar.AddAction(gi.ActOpts{Label: "Step Trial", Icon: "step-fwd", Tooltip: "Advances one training trial at a time.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			ss.TrainTrial()
			ss.IsRunning = false
			vp.SetNeedsFullRender()
		}
	})

	tbar.AddAction(gi.ActOpts{Label: "Step Epoch", Icon: "fast-fwd", Tooltip: "Advances one epoch (complete set of training patterns) at a time.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			go ss.TrainEpoch()
		}
	})

	tbar.AddAction(gi.ActOpts{Label: "Step Run", Icon: "fast-fwd", Tooltip: "Advances one full training Run at a time.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			go ss.TrainRun()
		}
	})

	tbar.AddSeparator("test")

	tbar.AddAction(gi.ActOpts{Label: "Test Trial", Icon: "step-fwd", Tooltip: "Runs the next testing trial.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			ss.TestTrial()
			ss.IsRunning = false
			vp.SetNeedsFullRender()
		}
	})

	tbar.AddAction(gi.ActOpts{Label: "Test Item", Icon: "step-fwd", Tooltip: "Prompts for a specific input pattern name to run, and runs it in testing mode.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		gi.StringPromptDialog(vp, "", "Test Item",
			gi.DlgOpts{Title: "Test Item", Prompt: "Enter the Name of a given input pattern to test (case insensitive, contains given string."},
			win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
				dlg := send.(*gi.Dialog)
				if sig == int64(gi.DialogAccepted) {
					val := gi.StringPromptDialogValue(dlg)
					idxs := ss.TestEnv.Table.RowsByString("Name", val, true, true) // contains, ignoreCase
					if len(idxs) == 0 {
						gi.PromptDialog(nil, gi.DlgOpts{Title: "Name Not Found", Prompt: "No patterns found containing: " + val}, true, false, nil, nil)
					} else {
						if !ss.IsRunning {
							ss.IsRunning = true
							fmt.Printf("testing index: %v\n", idxs[0])
							ss.TestItem(idxs[0])
							ss.IsRunning = false
							vp.SetNeedsFullRender()
						}
					}
				}
			})
	})

	tbar.AddAction(gi.ActOpts{Label: "Test All", Icon: "fast-fwd", Tooltip: "Tests all of the testing trials.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			go ss.RunTestAll()
		}
	})

	tbar.AddSeparator("log")

	tbar.AddAction(gi.ActOpts{Label: "Reset RunLog", Icon: "reset", Tooltip: "Reset the accumulated log of all Runs, which are tagged with the ParamSet used"}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			ss.RunLog.SetNumRows(0)
			ss.RunPlot.Update()
		})

	tbar.AddSeparator("misc")

	tbar.AddAction(gi.ActOpts{Label: "New Seed", Icon: "new", Tooltip: "Generate a new initial random seed to get different results.  By default, Init re-establishes the same initial seed every time."}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			ss.NewRndSeed()
		})

	tbar.AddAction(gi.ActOpts{Label: "README", Icon: "file-markdown", Tooltip: "Opens your browser on the README file that contains instructions for how to run this model."}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			gi.OpenURL("https://github.com/emer/leabra/blob/master/examples/ra25/README.md")
		})

	vp.UpdateEndNoSig(updt)

	// main menu
	appnm := gi.AppName()
	mmen := win.MainMenu
	mmen.ConfigMenus([]string{appnm, "File", "Edit", "Window"})

	amen := win.MainMenu.ChildByName(appnm, 0).(*gi.Action)
	amen.Menu.AddAppMenu(win)

	emen := win.MainMenu.ChildByName("Edit", 1).(*gi.Action)
	emen.Menu.AddCopyCutPaste(win)

	// note: Command in shortcuts is automatically translated into Control for
	// Linux, Windows or Meta for MacOS
	// fmen := win.MainMenu.ChildByName("File", 0).(*gi.Action)
	// fmen.Menu.AddAction(gi.ActOpts{Label: "Open", Shortcut: "Command+O"},
	// 	win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
	// 		FileViewOpenSVG(vp)
	// 	})
	// fmen.Menu.AddSeparator("csep")
	// fmen.Menu.AddAction(gi.ActOpts{Label: "Close Window", Shortcut: "Command+W"},
	// 	win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
	// 		win.Close()
	// 	})

	inQuitPrompt := false
	gi.SetQuitReqFunc(func() {
		if inQuitPrompt {
			return
		}
		inQuitPrompt = true
		gi.PromptDialog(vp, gi.DlgOpts{Title: "Really Quit?",
			Prompt: "Are you <i>sure</i> you want to quit and lose any unsaved params, weights, logs, etc?"}, true, true,
			win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
				if sig == int64(gi.DialogAccepted) {
					gi.Quit()
				} else {
					inQuitPrompt = false
				}
			})
	})

	// gi.SetQuitCleanFunc(func() {
	// 	fmt.Printf("Doing final Quit cleanup here..\n")
	// })

	inClosePrompt := false
	win.SetCloseReqFunc(func(w *gi.Window) {
		if inClosePrompt {
			return
		}
		inClosePrompt = true
		gi.PromptDialog(vp, gi.DlgOpts{Title: "Really Close Window?",
			Prompt: "Are you <i>sure</i> you want to close the window?  This will Quit the App as well, losing all unsaved params, weights, logs, etc"}, true, true,
			win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
				if sig == int64(gi.DialogAccepted) {
					gi.Quit()
				} else {
					inClosePrompt = false
				}
			})
	})

	win.SetCloseCleanFunc(func(w *gi.Window) {
		go gi.Quit() // once main window is closed, quit
	})

	win.MainMenuUpdated()
	return win
}

// These props register Save methods so they can be used
var SimProps = ki.Props{
	"CallMethods": ki.PropSlice{
		{"SaveWeights", ki.Props{
			"desc": "save network weights to file",
			"icon": "file-save",
			"Args": ki.PropSlice{
				{"File Name", ki.Props{
					"ext": ".wts",
				}},
			},
		}},
		{"SaveParams", ki.Props{
			"desc": "save parameters to file",
			"icon": "file-save",
			"Args": ki.PropSlice{
				{"File Name", ki.Props{
					"ext": ".params",
				}},
			},
		}},
	},
}

func (ss *Sim) CmdArgs() {
	ss.NoGui = true
	var nogui bool
	var saveEpcLog bool
	var saveRunLog bool
	flag.StringVar(&ss.ParamSet, "params", "", "ParamSet name to use -- must be valid name as listed in compiled-in params or loaded params")
	flag.StringVar(&ss.Tag, "tag", "", "extra tag to add to file names saved from this run")
	flag.IntVar(&ss.MaxRuns, "runs", 10, "number of runs to do (note that MaxEpcs is in paramset)")
	flag.BoolVar(&ss.LogSetParams, "setparams", false, "if true, print a record of each parameter that is set")
	flag.BoolVar(&ss.SaveWts, "wts", false, "if true, save final weights after each run")
	flag.BoolVar(&saveEpcLog, "epclog", true, "if true, save train epoch log to file")
	flag.BoolVar(&saveRunLog, "runlog", true, "if true, save run epoch log to file")
	flag.BoolVar(&nogui, "nogui", true, "if not passing any other args and want to run nogui, use nogui")
	flag.Parse()
	ss.Init()

	if ss.ParamSet != "" {
		fmt.Printf("Using ParamSet: %s\n", ss.ParamSet)
	}

	if saveEpcLog {
		var err error
		fnm := ss.LogFileName("epc")
		ss.TrnEpcFile, err = os.Create(fnm)
		if err != nil {
			log.Println(err)
			ss.TrnEpcFile = nil
		} else {
			fmt.Printf("Saving epoch log to: %v\n", fnm)
			defer ss.TrnEpcFile.Close()
		}
	}
	if saveRunLog {
		var err error
		fnm := ss.LogFileName("run")
		ss.RunFile, err = os.Create(fnm)
		if err != nil {
			log.Println(err)
			ss.RunFile = nil
		} else {
			fmt.Printf("Saving run log to: %v\n", fnm)
			defer ss.RunFile.Close()
		}
	}
	if ss.SaveWts {
		fmt.Printf("Saving final weights per run\n")
	}
	fmt.Printf("Running %d Runs\n", ss.MaxRuns)
	ss.Train()
}

func mainrun() {
	TheSim.New()
	TheSim.Config()

	if len(os.Args) > 1 {
		TheSim.CmdArgs() // simple assumption is that any args = no gui -- could add explicit arg if you want
	} else {
		// gi.Update2DTrace = true
		TheSim.Init()
		win := TheSim.ConfigGui()
		win.StartEventLoop()
	}
}
