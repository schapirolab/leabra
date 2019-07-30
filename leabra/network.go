// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package leabra

import (
	//"fmt"
	"github.com/emer/emergent/emer"
	"github.com/goki/ki/ki"
	"github.com/goki/ki/kit"
)

// leabra.Network has parameters for running a basic rate-coded Leabra network
type Network struct {
	NetworkStru
	WtBalInterval int `def:"10" desc:"how frequently to update the weight balance average weight factor -- relatively expensive"`
	WtBalCtr      int `inactive:"+" desc:"counter for how long it has been since last WtBal"`
}

var KiT_Network = kit.Types.AddType(&Network{}, NetworkProps)

// NewLayer returns new layer of proper type
func (nt *Network) NewLayer() emer.Layer {
	return &Layer{}
}

// NewPrjn returns new prjn of proper type
func (nt *Network) NewPrjn() emer.Prjn {
	return &Prjn{}
}

// Defaults sets all the default parameters for all layers and projections
func (nt *Network) Defaults() {
	nt.WtBalInterval = 10
	nt.WtBalCtr = 0
	for li, ly := range nt.Layers {
		ly.Defaults()
		ly.SetIndex(li)
	}
}

// UpdateParams updates all the derived parameters if any have changed, for all layers
// and projections
func (nt *Network) UpdateParams() {
	for _, ly := range nt.Layers {
		ly.UpdateParams()
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  Init methods

// InitWts initializes synaptic weights and all other associated long-term state variables
// including running-average state values (e.g., layer running average activations etc)
func (nt *Network) InitWts() {
	nt.WtBalCtr = 0
	for _, ly := range nt.Layers {
		if ly.IsOff() {
			continue
		}
		ly.(LeabraLayer).InitWts()
	}
	// separate pass to enforce symmetry
	for _, ly := range nt.Layers {
		if ly.IsOff() {
			continue
		}
		ly.(LeabraLayer).InitWtSym()
	}
}

// InitEffWt
func (nt *Network) InitSdEffWt() {
	// initEffwt
	for _, ly := range nt.Layers {
		if ly.IsOff() {
			continue
		}
		ly.(LeabraLayer).InitSdEffWt()
	}
}

// InitGInc is a wrapper function added by DH to call the layer level of InitGInc
func (nt *Network) InitGInc() {
	for _, ly := range nt.Layers {
		if ly.IsOff() {
			continue
		}
		ly.(LeabraLayer).InitGInc()
	}
}

// InitActs fully initializes activation state -- not automatically called
func (nt *Network) InitActs() {
	for _, ly := range nt.Layers {
		if ly.IsOff() {
			continue
		}
		ly.(LeabraLayer).InitActs()
	}
}

// InitExt initializes external input state -- call prior to applying external inputs to layers
func (nt *Network) InitExt() {
	for _, ly := range nt.Layers {
		if ly.IsOff() {
			continue
		}
		ly.(LeabraLayer).InitExt()
	}
}

// AlphaCycInit handles all initialization at start of new input pattern, including computing
// netinput scaling from running average activation etc.
func (nt *Network) AlphaCycInit() {
	for _, ly := range nt.Layers {
		if ly.IsOff() {
			continue
		}
		ly.(LeabraLayer).AlphaCycInit()
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  Act methods

// Cycle runs one cycle of activation updating:
// * Sends Ge increments from sending to receiving layers
// * Average and Max Ge stats
// * Inhibition based on Ge stats and Act Stats (computed at end of Cycle)
// * Activation from Ge, Gi, and Gl
// * Average and Max Act stats
// This basic version doesn't use the time info, but more specialized types do, and we
// want to keep a consistent API for end-user code.
func (nt *Network) Cycle(ltime *Time, sleep bool) {
	if sleep {
		nt.CaUpdt(ltime) // Added Synaptic depression by DH.
		nt.CalSynDep(ltime)
		//nt.InitGInc()
	}
	nt.SendGDelta(ltime, sleep) // also does integ
	nt.AvgMaxGe(ltime)
	nt.InhibFmGeAct(ltime)
	nt.ActFmG(ltime)
	nt.AvgMaxAct(ltime)
}

// SendGeDelta sends change in activation since last sent, if above thresholds
// and integrates sent deltas into GeRaw and time-integrated Ge values
func (nt *Network) SendGDelta(ltime *Time, sleep bool) {
	nt.ThrLayFun(func(ly LeabraLayer) { ly.SendGDelta(ltime, sleep) }, "SendGDelta")
	nt.ThrLayFun(func(ly LeabraLayer) { ly.GFmInc(ltime) }, "GFmInc")
}

// MonChge is a monitor
func (nt *Network) MonChge(ltime *Time) {
	nt.ThrLayFun(func(ly LeabraLayer) { ly.MonChge(ltime) }, "MonChge")
}

// CaUpdt computes the synaptic depression variable.
func (nt *Network) CaUpdt(ltime *Time) {
	nt.ThrLayFun(func(ly LeabraLayer) { ly.CaUpdt(ltime) }, "CaUpdt")
}

// CalSynDep computes the synaptic depression variable.
func (nt *Network) CalSynDep(ltime *Time) {
	nt.ThrLayFun(func(ly LeabraLayer) { ly.CalSynDep(ltime) }, "CalSynDep")
}

// AvgMaxGe computes the average and max Ge stats, used in inhibition
func (nt *Network) AvgMaxGe(ltime *Time) {
	nt.ThrLayFun(func(ly LeabraLayer) { ly.AvgMaxGe(ltime) }, "AvgMaxGe")
}

// InhibiFmGeAct computes inhibition Gi from Ge and Act stats within relevant Pools
func (nt *Network) InhibFmGeAct(ltime *Time) {
	nt.ThrLayFun(func(ly LeabraLayer) { ly.InhibFmGeAct(ltime) }, "InhibFmGeAct")
}

// ActFmG computes rate-code activation from Ge, Gi, Gl conductances
func (nt *Network) ActFmG(ltime *Time) {
	nt.ThrLayFun(func(ly LeabraLayer) { ly.ActFmG(ltime) }, "ActFmG   ")
}

// AvgMaxGe computes the average and max Ge stats, used in inhibition
func (nt *Network) AvgMaxAct(ltime *Time) {
	nt.ThrLayFun(func(ly LeabraLayer) { ly.AvgMaxAct(ltime) }, "AvgMaxAct")
}

// QuarterFinal does updating after end of a quarter
func (nt *Network) QuarterFinal(ltime *Time) {
	nt.ThrLayFun(func(ly LeabraLayer) { ly.QuarterFinal(ltime) }, "QuarterFinal")
}

//////////////////////////////////////////////////////////////////////////////////////
//  Learn methods

// DWt computes the weight change (learning) based on current running-average activation values
func (nt *Network) DWt() {
	nt.ThrLayFun(func(ly LeabraLayer) { ly.DWt() }, "DWt     ")
}

// WtFmDWt updates the weights from delta-weight changes.
// Also calls WtBalFmWt every WtBalInterval times
func (nt *Network) WtFmDWt() {
	nt.ThrLayFun(func(ly LeabraLayer) { ly.WtFmDWt() }, "WtFmDWt")
	nt.WtBalCtr++
	if nt.WtBalCtr >= nt.WtBalInterval {
		nt.WtBalCtr = 0
		nt.WtBalFmWt()
	}
}

// WtBalFmWt updates the weight balance factors based on average recv weights
func (nt *Network) WtBalFmWt() {
	nt.ThrLayFun(func(ly LeabraLayer) { ly.WtBalFmWt() }, "WtBalFmWt")
}

//////////////////////////////////////////////////////////////////////////////////////
//  Network props for gui

var NetworkProps = ki.Props{
	"ToolBar": ki.PropSlice{
		{"SaveWtsJSON", ki.Props{
			"label": "Save Wts...",
			"icon":  "file-save",
			"desc":  "Save json-formatted weights",
			"Args": ki.PropSlice{
				{"Weights File Name", ki.Props{
					"default-field": "WtsFile",
					"ext":           ".wts",
				}},
			},
		}},
		{"OpenWtsJSON", ki.Props{
			"label": "Open Wts...",
			"icon":  "file-open",
			"desc":  "Open json-formatted weights",
			"Args": ki.PropSlice{
				{"Weights File Name", ki.Props{
					"default-field": "WtsFile",
					"ext":           ".wts",
				}},
			},
		}},
		{"sep-file", ki.BlankProp{}},
		{"Build", ki.Props{
			"icon": "update",
			"desc": "build the network's neurons and synapses according to current params",
		}},
		{"InitWts", ki.Props{
			"icon": "update",
			"desc": "initialize the network weight values according to prjn parameters",
		}},
		{"InitActs", ki.Props{
			"icon": "update",
			"desc": "initialize the network activation values",
		}},
		{"sep-act", ki.BlankProp{}},
		{"AddLayer", ki.Props{
			"label": "Add Layer...",
			"icon":  "new",
			"desc":  "add a new layer to network",
			"Args": ki.PropSlice{
				{"Layer Name", ki.Props{}},
				{"Layer Shape", ki.Props{
					"desc": "shape of layer, typically 2D (Y, X) or 4D (Pools Y, Pools X, Units Y, Units X)",
				}},
				{"Layer Type", ki.Props{
					"desc": "type of layer -- used for determining how inputs are applied",
				}},
			},
		}},
		{"ConnectLayerNames", ki.Props{
			"label": "Connect Layers...",
			"icon":  "new",
			"desc":  "add a new connection between layers in the network",
			"Args": ki.PropSlice{
				{"Send Layer Name", ki.Props{}},
				{"Recv Layer Name", ki.Props{}},
				{"Pattern", ki.Props{
					"desc": "pattern to connect with",
				}},
				{"Prjn Type", ki.Props{
					"desc": "type of projection -- direction, or other more specialized factors",
				}},
			},
		}},
	},
}
