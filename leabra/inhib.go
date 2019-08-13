// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package leabra

import (
	"github.com/chewxy/math32"
)

// leabra.InhibParams contains all the inhibition computation params and functions for basic Leabra
// This is included in leabra.Layer to support computation.
// This also includes other misc layer-level params such as running-average activation in the layer
// which is used for netinput rescaling and potentially for adapting inhibition over time
type InhibParams struct {
	Layer  FFFBParams      `view:"inline" desc:"inhibition across the entire layer"`
	Pool   FFFBParams      `view:"inline" desc:"inhibition across sub-pools of units, for layers with 4D shape"`
	Self   SelfInhibParams `view:"inline" desc:"neuron self-inhibition parameters -- can be beneficial for producing more graded, linear response -- not typically used in cortical networks"`
	ActAvg ActAvgParams    `view:"inline" desc:"running-average activation computation values -- for overall estimates of layer activation levels, used in netinput scaling"`
}

func (ip *InhibParams) Update() {
	ip.Layer.Update()
	ip.Pool.Update()
	ip.Self.Update()
	ip.ActAvg.Update()
}

func (ip *InhibParams) Defaults() {
	ip.Layer.Defaults()
	ip.Pool.Defaults()
	ip.Self.Defaults()
	ip.ActAvg.Defaults()
}

// FFFBParams parameterizes feedforward (FF) and feedback (FB) inhibition (FFFB)
// based on average (or maximum) netinput (FF) and activation (FB)
type FFFBParams struct {
	On       bool    `desc:"enable this level of inhibition"`
	Gi       float32 `min:"0" def:"1.8" desc:"[1.5-2.3 typical, can go lower or higher as needed] overall inhibition gain -- this is main parameter to adjust to change overall activation levels -- it scales both the the ff and fb factors uniformly"`
	GiBase   float32 `min:"0" def:"1.8" desc:"[1.5-2.3 typical, can go lower or higher as needed] the baseline overall inhibition gain -- this is main parameter to adjust to change overall activation levels -- it scales both the the ff and fb factors uniformly"`
	GiOscPer int     `min:"1" def:"100" desc:"[100 to 360 typical, can go lower or higher as needed] number of cycles for a complete inhibition oscillation period -- this is the frequency parameter to adjust to change how fast the inhibition oscillation would go -- it scales both the the ff and fb factors uniformly"`
	GiOscMax float32 `min:"0" def:"1.8" desc:"[must higher than 1 typically, used as a scalar of base] A percentage of the base. the peak of inhibition oscillation -- this is main parameter to adjust to change the maximum of the oscillation -- it scales both the the ff and fb factors uniformly"`
	GiOscMin float32 `min:"0" def:"1.8" desc:"[0.6-0.8 typical, not higher than 1] A percentage of the base. the tout of inhibition oscillation -- this is main parameter to adjust to change the tout of the oscillation -- it scales both the the ff and fb factors uniformly"`
	FF       float32 `viewif:"On" min:"0" def:"1" desc:"overall inhibitory contribution from feedforward inhibition -- multiplies average netinput (i.e., synaptic drive into layer) -- this anticipates upcoming changes in excitation, but if set too high, it can make activity slow to emerge -- see also ff0 for a zero-point for this value"`
	FB       float32 `viewif:"On" min:"0" def:"1" desc:"overall inhibitory contribution from feedback inhibition -- multiplies average activation -- this reacts to layer activation levels and works more like a thermostat (turning up when the 'heat' in the layer is too high)"`
	FBTau    float32 `viewif:"On" min:"0" def:"1.4,3,5" desc:"time constant in cycles, which should be milliseconds typically (roughly, how long it takes for value to change significantly -- 1.4x the half-life) for integrating feedback inhibitory values -- prevents oscillations that otherwise occur -- the fast default of 1.4 should be used for most cases but sometimes a slower value (3 or higher) can be more robust, especially when inhibition is strong or inputs are more rapidly changing"`
	MaxVsAvg float32 `viewif:"On" def:"0,0.5,1" desc:"what proportion of the maximum vs. average netinput to use in the feedforward inhibition computation -- 0 = all average, 1 = all max, and values in between = proportional mix between average and max (ff_netin = avg + ff_max_vs_avg * (max - avg)) -- including more max can be beneficial especially in situations where the average can vary significantly but the activity should not -- max is more robust in many situations but less flexible and sensitive to the overall distribution -- max is better for cases more closely approximating single or strictly fixed winner-take-all behavior -- 0.5 is a good compromise in many cases and generally requires a reduction of .1 or slightly more (up to .3-.5) from the gi value for 0"`
	FF0      float32 `viewif:"On" def:"0.1" desc:"feedforward zero point for average netinput -- below this level, no FF inhibition is computed based on avg netinput, and this value is subtraced from the ff inhib contribution above this value -- the 0.1 default should be good for most cases (and helps FF_FB produce k-winner-take-all dynamics), but if average netinputs are lower than typical, you may need to lower it"`

	FBDt float32 `inactive:"+" view:"-" json:"-" xml:"-" desc:"rate = 1 / tau"`
}

func (fb *FFFBParams) Update() {
	fb.FBDt = 1 / fb.FBTau
}

func (fb *FFFBParams) Sleep() {
	fb.GiBase = fb.Gi
}

func (fb *FFFBParams) Wake() {
	fb.InhibOscilMute()
}

func (fb *FFFBParams) Defaults() {
	fb.Gi = 1.8
	fb.GiBase = fb.Gi
	fb.GiOscPer = 25
	fb.GiOscMax = 1.03
	fb.GiOscMin = 0.97
	fb.FF = 1
	fb.FB = 1
	fb.FBTau = 1.4
	fb.MaxVsAvg = 0
	fb.FF0 = 0.1
	fb.Update()
}

// FFInhib returns the feedforward inhibition value based on average and max excitatory conductance within
// relevant scope
func (fb *FFFBParams) FFInhib(avgGe, maxGe float32) float32 {
	ffNetin := avgGe + fb.MaxVsAvg*(maxGe-avgGe)
	var ffi float32
	if ffNetin > fb.FF0 {
		ffi = fb.FF * (ffNetin - fb.FF0)
	}
	return ffi
}

// FBInhib computes feedback inhibition value as function of average activation
func (fb *FFFBParams) FBInhib(avgAct float32) float32 {
	fbi := fb.FB * avgAct
	return fbi
}

// FBUpdt updates feedback inhibition using time-integration rate constant
func (fb *FFFBParams) FBUpdt(fbi *float32, newFbi float32) {
	*fbi += fb.FBDt * (newFbi - *fbi)
}

// Inhib is full inhibition computation for given pool activity levels and inhib state
func (fb *FFFBParams) Inhib(avgGe, maxGe, avgAct float32, inh *FFFBInhib) {
	if !fb.On {
		inh.Init()
		return
	}

	ffi := fb.FFInhib(avgGe, maxGe)
	fbi := fb.FBInhib(avgAct)

	inh.FFi = ffi
	fb.FBUpdt(&inh.FBi, fbi)

	inh.Gi = fb.Gi * (ffi + inh.FBi)
	inh.GiOrig = inh.Gi
}

// InhibOscil updates the inhibition oscillation based on the sine function.
func (fb *FFFBParams) InhibOscil(step int) {
	per := float32(step % fb.GiOscPer) / float32(fb.GiOscPer) * 2 * math32.Pi
	scal := float32(math32.Sin(per))
	fscal := float32(1.0)
	if scal > 0 {
		fscal = scal * (fb.GiOscMax - 1) + 1
	} else {
		fscal = scal * (1 - fb.GiOscMin) + 1
	}
	fb.Gi = fb.GiBase * fscal
}

// InhibOscilMute set the Gi back to GiBase.
func (fb *FFFBParams) InhibOscilMute() {
	fb.Gi = fb.GiBase
}

///////////////////////////////////////////////////////////////////////
//  SelfInhibParams

// SelfInhibParams defines parameters for Neuron self-inhibition -- activation of the neuron directly feeds back
// to produce a proportional additional contribution to Gi
type SelfInhibParams struct {
	On  bool    `desc:"enable neuron self-inhibition"`
	Gi  float32 `viewif:"On" def:"0.4" desc:"strength of individual neuron self feedback inhibition -- can produce proportional activation behavior in individual units for specialized cases (e.g., scalar val or BG units), but not so good for typical hidden layers"`
	Tau float32 `viewif:"On" def:"1.4" desc:"time constant in cycles, which should be milliseconds typically (roughly, how long it takes for value to change significantly -- 1.4x the half-life) for integrating unit self feedback inhibitory values -- prevents oscillations that otherwise occur -- relatively rapid 1.4 typically works, but may need to go longer if oscillations are a problem"`
	Dt  float32 `inactive:"+" view:"-" json:"-" xml:"-" desc:"rate = 1 / tau"`
}

func (si *SelfInhibParams) Update() {
	si.Dt = 1 / si.Tau
}

func (si *SelfInhibParams) Defaults() {
	si.On = false
	si.Gi = 0.4
	si.Tau = 1.4
	si.Update()
}

// Inhib updates the self inhibition value based on current unit activation
func (si *SelfInhibParams) Inhib(self *float32, act float32) {
	if si.On {
		*self += si.Dt * (si.Gi*act - *self)
	} else {
		*self = 0
	}
}

///////////////////////////////////////////////////////////////////////
//  ActAvgParams

// ActAvgParams represents expected average activity levels in the layer.
// Used for computing running-average computation that is then used for netinput scaling.
// Also specifies time constant for updating average
// and for the target value for adapting inhibition in inhib_adapt.
type ActAvgParams struct {
	Init      float32 `min:"0" desc:"[typically 0.1 - 0.2] initial estimated average activity level in the layer (see also UseFirst option -- if that is off then it is used as a starting point for running average actual activity level, ActMAvg and ActPAvg) -- ActPAvg is used primarily for automatic netinput scaling, to balance out layers that have different activity levels -- thus it is important that init be relatively accurate -- good idea to update from recorded ActPAvg levels"`
	Fixed     bool    `def:"false" desc:"if true, then the Init value is used as a constant for ActPAvgEff (the effective value used for netinput rescaling), instead of using the actual running average activation"`
	UseExtAct bool    `def:"false" desc:"if true, then use the activation level computed from the external inputs to this layer (avg of targ or ext unit vars) -- this will only be applied to layers with Input or Target / Compare layer types, and falls back on the targ_init value if external inputs are not available or have a zero average -- implies fixed behavior"`
	UseFirst  bool    `viewif:"Fixed=false" def:"true" desc:"use the first actual average value to override targ_init value -- actual value is likely to be a better estimate than our guess"`
	Tau       float32 `viewif:"Fixed=false" def:"100" min:"1" desc:"time constant in trials for integrating time-average values at the layer level -- used for computing Pool.ActAvg.ActsMAvg, ActsPAvg"`
	Adjust    float32 `viewif:"Fixed=false" def:"1" desc:"adjustment multiplier on the computed ActPAvg value that is used to compute ActPAvgEff, which is actually used for netinput rescaling -- if based on connectivity patterns or other factors the actual running-average value is resulting in netinputs that are too high or low, then this can be used to adjust the effective average activity value -- reducing the average activity with a factor < 1 will increase netinput scaling (stronger net inputs from layers that receive from this layer), and vice-versa for increasing (decreases net inputs)"`

	Dt float32 `inactive:"+" view:"-" json:"-" xml:"-" desc:"rate = 1 / tau"`
}

func (aa *ActAvgParams) Update() {
	aa.Dt = 1 / aa.Tau
}

func (aa *ActAvgParams) Defaults() {
	aa.Init = 0.15
	aa.Fixed = false
	aa.UseExtAct = false
	aa.UseFirst = true
	aa.Tau = 100
	aa.Adjust = 1
	aa.Update()
}

// EffInit returns the initial value applied during InitWts for the AvgPAvgEff effective layer activity
func (aa *ActAvgParams) EffInit() float32 {
	if aa.Fixed {
		return aa.Init
	}
	return aa.Adjust * aa.Init
}

// AvgFmAct updates the running-average activation given average activity level in layer
func (aa *ActAvgParams) AvgFmAct(avg *float32, act float32) {
	if act == 0 {
		return
	}
	if aa.UseFirst && *avg == aa.Init {
		*avg += 0.5 * (act - *avg)
	} else {
		*avg += aa.Dt * (act - *avg)
	}
}

// EffFmAvg updates the effective value from the running-average value
func (aa *ActAvgParams) EffFmAvg(eff *float32, avg float32) {
	if aa.Fixed {
		*eff = aa.Init
	} else {
		*eff = aa.Adjust * avg
	}
}
