// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package leabra

import (
	"fmt"
	"reflect"
)

// leabra.Synapse holds state for the synaptic connection between neurons
type Synapse struct {
	Wt                float32 `desc:"synaptic weight value -- sigmoid contrast-enhanced"`
	LWt               float32 `desc:"linear (underlying) weight value -- learns according to the lrate specified in the connection spec -- this is converted into the effective weight value, Wt, via sigmoidal contrast enhancement (see WtSigParams)"`
	DWt               float32 `desc:"change in synaptic weight, from learning"`
	Norm              float32 `desc:"DWt normalization factor -- reset to max of abs value of DWt, decays slowly down over time -- serves as an estimate of variance in weight changes over time"`
	Moment            float32 `desc:"momentum -- time-integrated DWt changes, to accumulate a consistent direction of weight change and cancel out dithering contradictory changes"`
	SRAvgDp           float32 `desc:"Synaptic Depression scaling variable based on sender-receiver neuron average activation (represented by the inverse of sum of co-activation)"`
	Cai               float32 `desc:"cai intacelluarl calcium. Default to be 0."`
	Effwt             float32 `desc:"Maybe it is needed. I don't know yet. Default to be the same as Wt."`
	Ca_inc            float32 `desc:" #DEF_0.2 time constant for increases in Ca_i (from NMDA etc currents) -- default base value is .01 per cycle -- multiply by network->ct_learn.syndep_int to get this value (default = 20)"`
	Ca_dec            float32 `#DEF_0.2 time constant for decreases in Ca_i (from Ca pumps pushing Ca back out into the synapse) -- default base value is .01 per cycle -- multiply by network->ct_learn.syndep_int to get this value (default = 20)`
	sd_ca_thr         float32 `desc:"#DEF_0.2 synaptic depression ca threshold: only when ca_i has increased by this amount (thus synaptic ca depleted) does it affect firing rates and thus synaptic depression"`
	sd_ca_gain        float32 `desc:"#DEF_0.3 multiplier on cai value for computing synaptic depression -- modulates overall level of depression independent of rate parameters"`
	sd_ca_thr_rescale float32 `desc:"#READ_ONLY rescaling factor taking into account sd_ca_gain and sd_ca_thr (= sd_ca_gain/(1 - sd_ca_thr))"`
}

var SynapseVars = []string{"Wt", "LWt", "DWt", "Norm", "Moment", "SRAvgDp", "Cai", "Effwt"}

var SynapseVarsMap map[string]int

func init() {
	SynapseVarsMap = make(map[string]int, len(SynapseVars))
	for i, v := range SynapseVars {
		SynapseVarsMap[v] = i
	}
}

func (sy *Synapse) VarNames() []string {
	return SynapseVars
}

func (sy *Synapse) VarByName(varNm string) (float32, bool) {
	i, ok := SynapseVarsMap[varNm]
	if !ok {
		return 0, false
	}
	// todo: would be ideal to avoid having to use reflect here..
	v := reflect.ValueOf(sy)
	return v.Elem().Field(i).Interface().(float32), true
}

func (sy *Synapse) SetVarByName(varNm string, val float64) bool {
	i, ok := SynapseVarsMap[varNm]
	if !ok {
		return false
	}
	// todo: would be ideal to avoid having to use reflect here..
	v := reflect.ValueOf(sy)
	v.Elem().Field(i).SetFloat(val)
	return true
}

func (sy *Synapse) SynDep() float32 {
	cao_thr := float32(1.0)
	if sy.Cai > sy.sd_ca_thr {
		cao_thr = 1.0 - sy.sd_ca_thr_rescale*(sy.Cai-sy.sd_ca_thr)
		if cao_thr*cao_thr < 0.7 {
			fmt.Println("SynDep happened, syndep is %s", cao_thr*cao_thr)
		}
	}
	return cao_thr * cao_thr

}

// CaUpdt calculated the Cai for each synapses.
func (sy *Synapse) CaUpdt(ru_act float32, su_act float32) {
	drive := ru_act * su_act
	// orgl := sy.Cai
	sy.Cai += sy.Ca_inc*(1.0-sy.Cai)*drive - sy.Ca_dec*sy.Cai
	//	if orgl != sy.Cai {
	//		fmt.Println("Synaptic Cai has been updated, previously %s, now %s", orgl, sy.Cai)
	//	}
}
