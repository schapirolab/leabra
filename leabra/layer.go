// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package leabra

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"math/rand"
	"strings"

	"github.com/chewxy/math32"
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/erand"
	"github.com/emer/etable/etensor"
	"github.com/goki/ki/bitflag"
	"github.com/goki/ki/indent"
	"github.com/goki/ki/ints"
	"github.com/goki/ki/ki"
	"github.com/goki/ki/kit"
	"gonum.org/v1/gonum/stat"
)

// leabra.Layer has parameters for running a basic rate-coded Leabra layer
type Layer struct {
	LayerStru
	Act     ActParams       `desc:"Activation parameters and methods for computing activations"`
	Inhib   InhibParams     `desc:"Inhibition parameters and methods for computing layer-level inhibition"`
	Learn   LearnNeurParams `desc:"Learning parameters and methods that operate at the neuron level"`
	Neurons []Neuron        `desc:"slice of neurons for this layer -- flat list of len = Shp.Len(). You must iterate over index and use pointer to modify values."`
	Pools   []Pool          `desc:"inhibition and other pooled, aggregate state variables -- flat list has at least of 1 for layer, and one for each sub-pool (unit group) if shape supports that (4D).  You must iterate over index and use pointer to modify values."`
	CosDiff CosDiffStats    `desc:"cosine difference between ActM, ActP stats"`
	Sim     float64         `desc:"Similarity between current cycle and previous cycle."`
}

var KiT_Layer = kit.Types.AddType(&Layer{}, LayerProps)

// AsLeabra returns this layer as a leabra.Layer -- all derived layers must redefine
// this to return the base Layer type, so that the LeabraLayer interface does not
// need to include accessors to all the basic stuff
func (ly *Layer) AsLeabra() *Layer {
	return ly
}

func (ly *Layer) Defaults() {
	ly.Act.Defaults()
	ly.Inhib.Defaults()
	ly.Learn.Defaults()
	ly.Inhib.Layer.On = true
	for _, pj := range ly.RcvPrjns {
		pj.Defaults()
	}
}

// CalLaySim calculate the similarity of the PrevState and CurState of activation.
func (ly *Layer) CalLaySim(ltime *Time) {
	var PrevState []float64
	var CurState []float64
	for _, n := range ly.Neurons {
		PrevState = append(PrevState, float64(n.ActSent))
		CurState = append(CurState, float64(n.Act))
	}
	ly.Sim = stat.Correlation(PrevState, CurState, nil)
}

// UpdateParams updates all params given any changes that might have been made to individual values
// including those in the receiving projections of this layer
func (ly *Layer) UpdateParams() {
	ly.Act.Update()
	ly.Inhib.Update()
	ly.Learn.Update()
	for _, pj := range ly.RcvPrjns {
		pj.UpdateParams()
	}
}

// JsonToParams reformates json output to suitable params display output
func JsonToParams(b []byte) string {
	br := strings.Replace(string(b), `"`, ``, -1)
	br = strings.Replace(br, ",\n", "", -1)
	br = strings.Replace(br, "{\n", "{", -1)
	br = strings.Replace(br, "} ", "}\n  ", -1)
	br = strings.Replace(br, "\n }", " }", -1)
	br = strings.Replace(br, "\n  }\n", " }", -1)
	return br[1:] + "\n"
}

// AllParams returns a listing of all parameters in the Layer
func (ly *Layer) AllParams() string {
	str := "/////////////////////////////////////////////////\nLayer: " + ly.Nm + "\n"
	b, _ := json.MarshalIndent(&ly.Act, "", " ")
	str += "Act: {\n " + JsonToParams(b)
	b, _ = json.MarshalIndent(&ly.Inhib, "", " ")
	str += "Inhib: {\n " + JsonToParams(b)
	b, _ = json.MarshalIndent(&ly.Learn, "", " ")
	str += "Learn: {\n " + JsonToParams(b)
	for _, pj := range ly.RcvPrjns {
		pstr := pj.AllParams()
		str += pstr
	}
	return str
}

// UnitVarNames returns a list of variable names available on the units in this layer
func (ly *Layer) UnitVarNames() []string {
	return NeuronVars
}

// UnitVals is emer.Layer interface method to return values of given variable
func (ly *Layer) UnitVals(varNm string) []float32 {
	uv, _ := ly.LeabraLay.UnitValsTry(varNm)
	return uv
}

// UnitValsTry is emer.Layer interface method to return values of given variable
func (ly *Layer) UnitValsTry(varNm string) ([]float32, error) {
	vidx, err := NeuronVarByName(varNm)
	if err != nil {
		return nil, err
	}
	vs := make([]float32, len(ly.Neurons))
	for i := range ly.Neurons {
		nrn := &ly.Neurons[i]
		vs[i] = nrn.VarByIndex(vidx)
	}
	return vs, nil
}

// UnitValsTensor returns values of given variable name on unit
// for each unit in the layer, as a float32 tensor in same shape as layer units.
func (ly *Layer) UnitValsTensor(varNm string) etensor.Tensor {
	uv, _ := ly.LeabraLay.UnitValsTensorTry(varNm)
	return uv
}

// UnitValsTensorTry returns values of given variable name on unit
// for each unit in the layer, as a float32 tensor in same shape as layer units.
func (ly *Layer) UnitValsTensorTry(varNm string) (etensor.Tensor, error) {
	vls, err := ly.UnitValsTry(varNm)
	if err != nil {
		return nil, err
	}
	return etensor.NewFloat32Shape(&ly.Shp, vls), nil
}

// UnitVal returns value of given variable name on given unit,
// using shape-based dimensional index
func (ly *Layer) UnitVal(varNm string, idx []int) float32 {
	uv, _ := ly.LeabraLay.UnitValTry(varNm, idx)
	return uv
}

// UnitValTry returns value of given variable name on given unit,
// using shape-based dimensional index
func (ly *Layer) UnitValTry(varNm string, idx []int) (float32, error) {
	fidx := ly.Shp.Offset(idx)
	nn := len(ly.Neurons)
	if fidx < 0 || fidx >= nn {
		return 0, fmt.Errorf("Layer UnitVal index: %v out of range, N = %v", fidx, nn)
	}
	nrn := &ly.Neurons[fidx]
	return nrn.VarByName(varNm)
}

// UnitVal1D returns value of given variable name on given unit,
// using 1-dimensional index.
func (ly *Layer) UnitVal1D(varNm string, idx int) float32 {
	uv, _ := ly.LeabraLay.UnitVal1DTry(varNm, idx)
	return uv
}

// UnitVal1DTry returns value of given variable name on given unit,
// using 1-dimensional index.
func (ly *Layer) UnitVal1DTry(varNm string, idx int) (float32, error) {
	nn := len(ly.Neurons)
	if idx < 0 || idx >= nn {
		return 0, fmt.Errorf("Layer UnitVal1D index: %v out of range, N = %v", idx, nn)
	}
	nrn := &ly.Neurons[idx]
	return nrn.VarByName(varNm)
}

// Pool returns pool at given index
func (ly *Layer) Pool(idx int) *Pool {
	return &(ly.Pools[idx])
}

// PoolTry returns pool at given index, returns error if index is out of range
func (ly *Layer) PoolTry(idx int) (*Pool, error) {
	np := len(ly.Pools)
	if idx < 0 || idx >= np {
		return nil, fmt.Errorf("Layer Pool index: %v out of range, N = %v", idx, np)
	}
	return &(ly.Pools[idx]), nil
}

//////////////////////////////////////////////////////////////////////////////////////
//  Build

// BuildSubPools initializes neuron start / end indexes for sub-pools
func (ly *Layer) BuildSubPools() {
	if !ly.Is4D() {
		return
	}
	sh := ly.Shp.Shapes()
	spy := sh[0]
	spx := sh[1]
	lastOff := 0
	pi := 0 // will incr to 1 by time used, for first pool
	for py := 0; py < spy; py++ {
		for px := 0; px < spx; px++ {
			idx := []int{py, px, 0, 0}
			off := ly.Shp.Offset(idx)
			if off == 0 {
				continue
			}
			pl := &ly.Pools[pi]
			pl.StIdx = lastOff
			pl.EdIdx = off
			for ni := pl.StIdx; ni < pl.EdIdx; ni++ {
				nrn := &ly.Neurons[ni]
				nrn.SubPool = int32(pi)
			}
			pi++
			lastOff = off
		}
	}
}

// BuildPools builds the inhibitory pools structures -- nu = number of units in layer
func (ly *Layer) BuildPools(nu int) error {
	np := 1
	if ly.Inhib.Pool.On {
		np += ly.NPools()
	}
	ly.Pools = make([]Pool, np)
	lpl := &ly.Pools[0]
	lpl.StIdx = 0
	lpl.EdIdx = nu
	if np > 1 {
		ly.BuildSubPools()
	}
	return nil
}

// BuildPrjns builds the projections, recv-side
func (ly *Layer) BuildPrjns() error {
	emsg := ""
	for _, pj := range ly.RcvPrjns {
		if pj.IsOff() {
			continue
		}
		err := pj.Build()
		if err != nil {
			emsg += err.Error() + "\n"
		}
	}
	if emsg != "" {
		return errors.New(emsg)
	}
	return nil
}

// Build constructs the layer state, including calling Build on the projections
// you MUST have properly configured the Inhib.Pool.On setting by this point
// to properly allocate Pools for the unit groups if necessary.
func (ly *Layer) Build() error {
	nu := ly.Shp.Len()
	if nu == 0 {
		return fmt.Errorf("Build Layer %v: no units specified in Shape", ly.Nm)
	}
	ly.Neurons = make([]Neuron, nu)
	err := ly.BuildPools(nu)
	if err != nil {
		return err
	}
	err = ly.BuildPrjns()
	return err
}

// WriteWtsJSON writes the weights from this layer from the receiver-side perspective
// in a JSON text format.  We build in the indentation logic to make it much faster and
// more efficient.
func (ly *Layer) WriteWtsJSON(w io.Writer, depth int) {
	w.Write(indent.TabBytes(depth))
	w.Write([]byte("{\n"))
	depth++
	w.Write(indent.TabBytes(depth))
	w.Write([]byte(fmt.Sprintf("\"%v\": [\n", ly.Nm)))
	// todo: save average activity state
	depth++
	for _, pj := range ly.RcvPrjns {
		if pj.IsOff() {
			continue
		}
		pj.WriteWtsJSON(w, depth)
	}
	depth--
	w.Write(indent.TabBytes(depth))
	w.Write([]byte("],\n"))
	depth--
	w.Write(indent.TabBytes(depth))
	w.Write([]byte("},\n"))
}

// ReadWtsJSON reads the weights from this layer from the receiver-side perspective
// in a JSON text format.
func (ly *Layer) ReadWtsJSON(r io.Reader) error {
	return nil
}

// VarRange returns the min / max values for given variable
// todo: support r. s. projection values
func (ly *Layer) VarRange(varNm string) (min, max float32, err error) {
	sz := len(ly.Neurons)
	if sz == 0 {
		return
	}
	vidx := 0
	vidx, err = NeuronVarByName(varNm)
	if err != nil {
		return
	}

	v0 := ly.Neurons[0].VarByIndex(vidx)
	min = v0
	max = v0
	for i := 1; i < sz; i++ {
		vl := ly.Neurons[i].VarByIndex(vidx)
		if vl < min {
			min = vl
		}
		if vl > max {
			max = vl
		}
	}
	return
}

// note: all basic computation can be performed on layer-level and prjn level

//////////////////////////////////////////////////////////////////////////////////////
//  Init methods

// InitWts initializes the weight values in the network, i.e., resetting learning
// Also calls InitActs
func (ly *Layer) InitWts() {
	for _, p := range ly.SndPrjns {
		if p.IsOff() {
			continue
		}
		p.(LeabraPrjn).InitWts()
	}
	for pi := range ly.Pools {
		pl := &ly.Pools[pi]
		pl.ActAvg.ActMAvg = ly.Inhib.ActAvg.Init
		pl.ActAvg.ActPAvg = ly.Inhib.ActAvg.Init
		pl.ActAvg.ActPAvgEff = ly.Inhib.ActAvg.EffInit()
	}
	ly.LeabraLay.InitActAvg()
	ly.LeabraLay.InitActs()
	ly.CosDiff.Init()
}

// InitSdEffWt initializes the Effwt and Cai for each synapse.
func (ly *Layer) InitSdEffWt() {
	for _, p := range ly.SndPrjns {
		if p.IsOff() {
			continue
		}
		p.(LeabraPrjn).InitSdEffWt()
	}
}

// InitActAvg initializes the running-average activation values that drive learning.
func (ly *Layer) InitActAvg() {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		ly.Learn.InitActAvg(nrn)
	}
}

// InitActs fully initializes activation state -- only called automatically during InitWts
func (ly *Layer) InitActs() {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		ly.Act.InitActs(nrn)
	}
}

// InitWtsSym initializes the weight symmetry -- higher layers copy weights from lower layers
func (ly *Layer) InitWtSym() {
	for _, p := range ly.SndPrjns {
		if p.IsOff() {
			continue
		}
		// key ordering constraint on which way weights are copied
		if p.RecvLay().Index() < p.SendLay().Index() {
			continue
		}
		rpj, has := ly.RecipToSendPrjn(p)
		if !has {
			continue
		}
		p.(LeabraPrjn).InitWtSym(rpj.(LeabraPrjn))
	}
}

// InitExt initializes external input state -- called prior to apply ext
func (ly *Layer) InitExt() {
	msk := bitflag.Mask32(int(NeurHasExt), int(NeurHasTarg), int(NeurHasCmpr))
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		nrn.Ext = 0
		nrn.Targ = 0
		nrn.ClearMask(msk)
	}
}

// ApplyExtFlags gets the clear mask and set mask for updating neuron flags
// based on layer type, and whether input should be applied to Targ (else Ext)
func (ly *Layer) ApplyExtFlags() (clrmsk, setmsk int32, toTarg bool) {
	clrmsk = bitflag.Mask32(int(NeurHasExt), int(NeurHasTarg), int(NeurHasCmpr))
	toTarg = false
	if ly.Typ == emer.Target {
		setmsk = bitflag.Mask32(int(NeurHasTarg))
		toTarg = true
	} else if ly.Typ == emer.Compare {
		setmsk = bitflag.Mask32(int(NeurHasCmpr))
		toTarg = true
	} else {
		setmsk = bitflag.Mask32(int(NeurHasExt))
	}
	return
}

// ApplyExt applies external input in the form of an etensor.Float32.  If
// dimensionality of tensor matches that of layer, and is 2D or 4D, then each dimension
// is iterated separately, so any mismatch preserves dimensional structure.
// If the layer is a Target or Compare layer type, then it goes in Targ
// otherwise it goes in Ext
func (ly *Layer) ApplyExt(ext etensor.Tensor) {
	if ext.NumDims() != ly.Shp.NumDims() || !(ext.NumDims() == 2 || ext.NumDims() == 4) {
		ly.LeabraLay.ApplyExt1D(ext.Floats())
		return
	}
	clrmsk, setmsk, toTarg := ly.ApplyExtFlags()
	if ext.NumDims() == 2 {
		ymx := ints.MinInt(ext.Dim(0), ly.Shp.Dim(0))
		xmx := ints.MinInt(ext.Dim(1), ly.Shp.Dim(1))
		for y := 0; y < ymx; y++ {
			for x := 0; x < xmx; x++ {
				idx := []int{y, x}
				vl := float32(ext.FloatVal(idx))
				i := ly.Shp.Offset(idx)
				nrn := &ly.Neurons[i]
				if nrn.IsOff() {
					continue
				}
				if toTarg {
					nrn.Targ = vl
				} else {
					nrn.Ext = vl
				}
				nrn.ClearMask(clrmsk)
				nrn.SetMask(setmsk)
			}
		}
		return
	}
	ypmx := ints.MinInt(ext.Dim(0), ly.Shp.Dim(0))
	xpmx := ints.MinInt(ext.Dim(1), ly.Shp.Dim(1))
	ynmx := ints.MinInt(ext.Dim(2), ly.Shp.Dim(2))
	xnmx := ints.MinInt(ext.Dim(3), ly.Shp.Dim(3))
	for yp := 0; yp < ypmx; yp++ {
		for xp := 0; xp < xpmx; xp++ {
			for yn := 0; yn < ynmx; yn++ {
				for xn := 0; xn < xnmx; xn++ {
					idx := []int{yp, xp, yn, xn}
					vl := float32(ext.FloatVal(idx))
					i := ly.Shp.Offset(idx)
					nrn := &ly.Neurons[i]
					if nrn.IsOff() {
						continue
					}
					if toTarg {
						nrn.Targ = vl
					} else {
						nrn.Ext = vl
					}
					nrn.ClearMask(clrmsk)
					nrn.SetMask(setmsk)
				}
			}
		}
	}
}

// ApplyExt1D applies external input in the form of a flat 1-dimensional slice of floats
// If the layer is a Target or Compare layer type, then it goes in Targ
// otherwise it goes in Ext
func (ly *Layer) ApplyExt1D(ext []float64) {
	clrmsk, setmsk, toTarg := ly.ApplyExtFlags()
	mx := ints.MinInt(len(ext), len(ly.Neurons))
	for i := 0; i < mx; i++ {
		nrn := &ly.Neurons[i]
		if nrn.IsOff() {
			continue
		}
		vl := float32(ext[i])
		if toTarg {
			nrn.Targ = vl
		} else {
			nrn.Ext = vl
		}
		nrn.ClearMask(clrmsk)
		nrn.SetMask(setmsk)
	}
}

// ApplyExt1D32 applies external input in the form of a flat 1-dimensional slice of float32s.
// If the layer is a Target or Compare layer type, then it goes in Targ
// otherwise it goes in Ext
func (ly *Layer) ApplyExt1D32(ext []float32) {
	clrmsk, setmsk, toTarg := ly.ApplyExtFlags()
	mx := ints.MinInt(len(ext), len(ly.Neurons))
	for i := 0; i < mx; i++ {
		nrn := &ly.Neurons[i]
		if nrn.IsOff() {
			continue
		}
		vl := ext[i]
		if toTarg {
			nrn.Targ = vl
		} else {
			nrn.Ext = vl
		}
		nrn.ClearMask(clrmsk)
		nrn.SetMask(setmsk)
	}
}

// AlphaCycInit handles all initialization at start of new input pattern, including computing
// input scaling from running average activation etc.
// should already have presented the external input to the network at this point.
func (ly *Layer) AlphaCycInit() {
	ly.LeabraLay.AvgLFmAvgM()
	for pi := range ly.Pools {
		pl := &ly.Pools[pi]
		ly.Inhib.ActAvg.AvgFmAct(&pl.ActAvg.ActMAvg, pl.ActM.Avg)
		ly.Inhib.ActAvg.AvgFmAct(&pl.ActAvg.ActPAvg, pl.ActP.Avg)
		ly.Inhib.ActAvg.EffFmAvg(&pl.ActAvg.ActPAvgEff, pl.ActAvg.ActPAvg)
	}
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		nrn.ActQ0 = nrn.ActP
	}
	ly.LeabraLay.GScaleFmAvgAct()
	if ly.Act.Noise.Type != NoNoise && ly.Act.Noise.Fixed && ly.Act.Noise.Dist != erand.Mean {
		ly.LeabraLay.GenNoise()
	}
	ly.LeabraLay.DecayState(ly.Act.Init.Decay)
	if ly.Act.Clamp.Hard && ly.Typ == emer.Input {
		ly.LeabraLay.HardClamp()
	}
}

// AvgLFmAvgM updates AvgL long-term running average activation that drives BCM Hebbian learning
func (ly *Layer) AvgLFmAvgM() {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		ly.Learn.AvgLFmAvgM(nrn)
		if ly.Learn.AvgL.ErrMod {
			nrn.AvgLLrn *= ly.CosDiff.ModAvgLLrn
		}
	}
}

// GScaleFmAvgAct computes the scaling factor for synaptic input conductances G,
// based on sending layer average activation.
// This attempts to automatically adjust for overall differences in raw activity
// coming into the units to achieve a general target of around .5 to 1
// for the integrated Ge value.
func (ly *Layer) GScaleFmAvgAct() {
	totGeRel := float32(0)
	totGiRel := float32(0)
	for _, p := range ly.RcvPrjns {
		if p.IsOff() {
			continue
		}
		pj := p.(LeabraPrjn).AsLeabra()
		slay := p.SendLay().(LeabraLayer).AsLeabra()
		slpl := slay.Pools[0]
		savg := slpl.ActAvg.ActPAvgEff
		snu := len(slay.Neurons)
		ncon := pj.RConNAvgMax.Avg
		pj.GScale = pj.WtScale.FullScale(savg, float32(snu), ncon)
		if pj.Typ == emer.Inhib {
			totGiRel += pj.WtScale.Rel
		} else {
			totGeRel += pj.WtScale.Rel
		}
	}

	for _, p := range ly.RcvPrjns {
		if p.IsOff() {
			continue
		}
		pj := p.(LeabraPrjn).AsLeabra()
		if pj.Typ == emer.Inhib {
			if totGiRel > 0 {
				pj.GScale /= totGiRel
			}
		} else {
			if totGeRel > 0 {
				pj.GScale /= totGeRel
			}
		}
	}
}

// GenNoise generates random noise for all neurons
func (ly *Layer) GenNoise() {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		nrn.Noise = float32(ly.Act.Noise.Gen(-1))
	}
}

// DecayState decays activation state by given proportion (default is on ly.Act.Init.Decay)
func (ly *Layer) DecayState(decay float32) {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		ly.Act.DecayState(nrn, decay)
	}
	for pi := range ly.Pools { // decaying average act is essential for inhib
		pl := &ly.Pools[pi]
		pl.Act.Max -= decay * pl.Act.Max
		pl.Act.Avg -= decay * pl.Act.Avg
		pl.Inhib.FFi -= decay * pl.Inhib.FFi
		pl.Inhib.FBi -= decay * pl.Inhib.FBi
		pl.Inhib.Gi -= decay * pl.Inhib.Gi
	}
}

// HardClamp hard-clamps the activations in the layer -- called during AlphaCycInit for hard-clamped Input layers
func (ly *Layer) HardClamp() {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		ly.Act.HardClamp(nrn)
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  Cycle

// InitGinc initializes the Ge excitatory and Gi inhibitory conductance accumulation states
// including ActSent and G*Raw values.
// called at start of trial always, and can be called optionally
// when delta-based Ge computation needs to be updated (e.g., weights
// might have changed strength)
func (ly *Layer) InitGInc() {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		ly.Act.InitGInc(nrn)
	}
	for _, p := range ly.RcvPrjns {
		if p.IsOff() {
			continue
		}
		p.(LeabraPrjn).InitGInc()
	}
}

// SendGDelta sends change in activation since last sent, to increment recv
// synaptic conductances G, if above thresholds
func (ly *Layer) SendGDelta(ltime *Time, sleep bool) {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		if nrn.Act > ly.Act.OptThresh.Send {
			delta := nrn.Act - nrn.ActSent
			if math32.Abs(delta) > ly.Act.OptThresh.Delta {
				for _, sp := range ly.SndPrjns {
					if sp.IsOff() {
						continue
					}
					sp.(LeabraPrjn).SendGDelta(ni, delta, sleep)
				}
				nrn.ActSent = nrn.Act
			}
		} else if nrn.ActSent > ly.Act.OptThresh.Send {
			delta := -nrn.ActSent // un-send the last above-threshold activation to get back to 0
			for _, sp := range ly.SndPrjns {
				if sp.IsOff() {
					continue
				}
				sp.(LeabraPrjn).SendGDelta(ni, delta, sleep)
			}
			nrn.ActSent = 0
		}
	}
}

// GFmInc integrates new synaptic conductances from increments sent during last SendGDelta.
func (ly *Layer) GFmInc(ltime *Time) {
	for _, p := range ly.RcvPrjns {
		if p.IsOff() {
			continue
		}
		p.(LeabraPrjn).RecvGInc()
	}
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		ly.Act.GeGiFmInc(nrn)
	}
}

// CaUpdt computes the Sender-Receiver co-activation based synaptic depression, added by DH.
//func (ly *Layer) CaUpdt(ltime *Time) {
//	for ni := range ly.Neurons {
//		nrn := &ly.Neurons[ni]
//		if nrn.IsOff() {
//			continue
//		}
//		for _, sp := range ly.SndPrjns {
//			if sp.IsOff() {
//				continue
//			}
//			sp.(LeabraPrjn).CaUpdt(ni, nrn.Act)
//		}
//	}
//}

// CalSynDep computes the Sender-Receiver co-activation based synaptic depression, added by DH.
func (ly *Layer) CalSynDep(ltime *Time) {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		for _, sp := range ly.SndPrjns {
			if sp.IsOff() {
				continue
			}
			sp.(LeabraPrjn).CaUpdt(ni, nrn.Act)
			sp.(LeabraPrjn).CalSynDep(ni)
			//		sp.(LeabraPrjn).MonChge(ni)
		}
	}
}

// MonChge is a monitor
func (ly *Layer) MonChge(ltime *Time) {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		for _, sp := range ly.SndPrjns {
			if sp.IsOff() {
				continue
			}
			sp.(LeabraPrjn).MonChge(ni)
		}
	}
}

// Sleep set the parameter to be sleep related
func (ly *Layer) Sleep(ltime *Time) {
	ly.Inhib.Layer.Sleep()
	ly.Act.OptThresh.Sleep()
}

// Wake set the parameter to be Wake related
func (ly *Layer) Wake(ltime *Time) {
	ly.Inhib.Layer.Wake()
	ly.Act.OptThresh.Wake()
}

// InhibOscil computes the layer level inhibition oscillation scaling factor.
func (ly *Layer) InhibOscil(ltime *Time, step int) {
	ly.Inhib.Layer.InhibOscil(step)
}

// InhibOscilMute set the layer inhibition back to base
func (ly *Layer) InhibOscilMute(ltime *Time) {
	ly.Inhib.Layer.InhibOscilMute()
}

// AvgMaxGe computes the average and max Ge stats, used in inhibition
func (ly *Layer) AvgMaxGe(ltime *Time) {
	for pi := range ly.Pools {
		pl := &ly.Pools[pi]
		pl.Ge.Init()
		for ni := pl.StIdx; ni < pl.EdIdx; ni++ {
			nrn := &ly.Neurons[ni]
			pl.Ge.UpdateVal(nrn.Ge, ni)
		}
		pl.Ge.CalcAvg()
	}
}

// InhibiFmGeAct computes inhibition Gi from Ge and Act averages within relevant Pools
func (ly *Layer) InhibFmGeAct(ltime *Time) {
	lpl := &ly.Pools[0]
	ly.Inhib.Layer.Inhib(lpl.Ge.Avg, lpl.Ge.Max, lpl.Act.Avg, &lpl.Inhib)
	np := len(ly.Pools)
	if np > 1 {
		for pi := 1; pi < np; pi++ {
			pl := &ly.Pools[pi]
			ly.Inhib.Pool.Inhib(pl.Ge.Avg, pl.Ge.Max, pl.Act.Avg, &pl.Inhib)
			pl.Inhib.Gi = math32.Max(pl.Inhib.Gi, lpl.Inhib.Gi)
			for ni := pl.StIdx; ni < pl.EdIdx; ni++ {
				nrn := &ly.Neurons[ni]
				if nrn.IsOff() {
					continue
				}
				ly.Inhib.Self.Inhib(&nrn.GiSelf, nrn.Act)
				nrn.Gi = pl.Inhib.Gi + nrn.GiSelf + nrn.GiSyn
			}
		}
	} else {
		for ni := lpl.StIdx; ni < lpl.EdIdx; ni++ {
			nrn := &ly.Neurons[ni]
			if nrn.IsOff() {
				continue
			}
			ly.Inhib.Self.Inhib(&nrn.GiSelf, nrn.Act)
			nrn.Gi = lpl.Inhib.Gi + nrn.GiSelf + nrn.GiSyn
		}
	}
}

// ActFmG computes rate-code activation from Ge, Gi, Gl conductances
// and updates learning running-average activations from that Act
func (ly *Layer) ActFmG(ltime *Time) {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		ly.Act.VmFmG(nrn)
		ly.Act.ActFmG(nrn)
		ly.Learn.AvgsFmAct(nrn)
	}
}

// AvgMaxAct computes the average and max Act stats, used in inhibition
func (ly *Layer) AvgMaxAct(ltime *Time) {
	for pi := range ly.Pools {
		pl := &ly.Pools[pi]
		pl.Act.Init()
		for ni := pl.StIdx; ni < pl.EdIdx; ni++ {
			nrn := &ly.Neurons[ni]
			if nrn.IsOff() {
				continue
			}
			pl.Act.UpdateVal(nrn.Act, ni)
		}
		pl.Act.CalcAvg()
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  Quarter

// QuarterFinal does updating after end of a quarter
func (ly *Layer) QuarterFinal(ltime *Time) {
	for pi := range ly.Pools {
		pl := &ly.Pools[pi]
		switch ltime.Quarter {
		case 2:
			pl.ActM = pl.Act
		case 3:
			pl.ActP = pl.Act
		}
	}
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		switch ltime.Quarter {
		case 0:
			nrn.ActQ1 = nrn.Act
		case 1:
			nrn.ActQ2 = nrn.Act
		case 2:
			nrn.ActM = nrn.Act
			if nrn.HasFlag(NeurHasTarg) { // will be clamped in plus phase
				nrn.Ext = nrn.Targ
				nrn.SetFlag(NeurHasExt)
			}
		case 3:
			nrn.ActP = nrn.Act
			nrn.ActDif = nrn.ActP - nrn.ActM
			nrn.ActAvg += ly.Act.Dt.AvgDt * (nrn.Act - nrn.ActAvg)
		}
	}
	if ltime.Quarter == 3 {
		ly.LeabraLay.CosDiffFmActs()
	}
}

// CosDiffFmActs computes the cosine difference in activation state between minus and plus phases.
// this is also used for modulating the amount of BCM hebbian learning
func (ly *Layer) CosDiffFmActs() {
	lpl := &ly.Pools[0]
	avgM := lpl.ActM.Avg
	avgP := lpl.ActP.Avg
	cosv := float32(0)
	ssm := float32(0)
	ssp := float32(0)
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		ap := nrn.ActP - avgP // zero mean
		am := nrn.ActM - avgM
		cosv += ap * am
		ssm += am * am
		ssp += ap * ap
	}

	dist := math32.Sqrt(ssm * ssp)
	if dist != 0 {
		cosv /= dist
	}
	ly.CosDiff.Cos = cosv

	ly.Learn.CosDiff.AvgVarFmCos(&ly.CosDiff.Avg, &ly.CosDiff.Var, ly.CosDiff.Cos)

	if ly.Typ != emer.Hidden {
		ly.CosDiff.AvgLrn = 0 // no BCM for non-hidden layers
		ly.CosDiff.ModAvgLLrn = 0
	} else {
		ly.CosDiff.AvgLrn = 1 - ly.CosDiff.Avg
		ly.CosDiff.ModAvgLLrn = ly.Learn.AvgL.ErrModFmLayErr(ly.CosDiff.AvgLrn)
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  Learning

// DWt computes the weight change (learning) -- calls DWt method on sending projections
func (ly *Layer) DWt() {
	for _, p := range ly.SndPrjns {
		if p.IsOff() {
			continue
		}
		p.(LeabraPrjn).DWt()
	}
}

// WtFmDWt updates the weights from delta-weight changes -- on the sending projections
func (ly *Layer) WtFmDWt() {
	for _, p := range ly.SndPrjns {
		if p.IsOff() {
			continue
		}
		p.(LeabraPrjn).WtFmDWt()
	}
}

// WtBalFmWt computes the Weight Balance factors based on average recv weights
func (ly *Layer) WtBalFmWt() {
	for _, p := range ly.RcvPrjns {
		if p.IsOff() {
			continue
		}
		p.(LeabraPrjn).WtBalFmWt()
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  Stats

// note: use float64 for stats as that is best for logging

// MSE returns the sum-squared-error and mean-squared-error
// over the layer, in terms of ActP - ActM (valid even on non-target layers FWIW).
// Uses the given tolerance per-unit to count an error at all
// (e.g., .5 = activity just has to be on the right side of .5).
func (ly *Layer) MSE(tol float32) (sse, mse float64) {
	nn := len(ly.Neurons)
	if nn == 0 {
		return 0, 0
	}
	sse = 0.0
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		d := nrn.ActP - nrn.ActM
		if math32.Abs(d) < tol {
			continue
		}
		sse += float64(d * d)
	}
	return sse, sse / float64(nn)
}

// SSE returns the sum-squared-error over the layer, in terms of ActP - ActM
// (valid even on non-target layers FWIW).
// Uses the given tolerance per-unit to count an error at all
// (e.g., .5 = activity just has to be on the right side of .5).
// Use this in Python which only allows single return values.
func (ly *Layer) SSE(tol float32) float64 {
	sse, _ := ly.MSE(tol)
	return sse
}

//////////////////////////////////////////////////////////////////////////////////////
//  Lesion

// UnLesionNeurons unlesions (clears the Off flag) for all neurons in the layer
func (ly *Layer) UnLesionNeurons() {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		nrn.ClearFlag(NeurOff)
	}
}

// LesionNeurons lesions (sets the Off flag) for given proportion (0-1) of neurons in layer
// returns number of neurons lesioned.  Emits error if prop > 1 as indication that percent
// might have been passed
func (ly *Layer) LesionNeurons(prop float32) int {
	ly.UnLesionNeurons()
	if prop > 1 {
		log.Printf("LesionNeurons got a proportion > 1 -- must be 0-1 as *proportion* (not percent) of neurons to lesion: %v\n", prop)
		return 0
	}
	nn := len(ly.Neurons)
	if nn == 0 {
		return 0
	}
	p := rand.Perm(nn)
	nl := int(prop * float32(nn))
	for i := 0; i < nl; i++ {
		nrn := &ly.Neurons[p[i]]
		nrn.SetFlag(NeurOff)
	}
	return nl
}

//////////////////////////////////////////////////////////////////////////////////////
//  Layer props for gui

var LayerProps = ki.Props{
	"ToolBar": ki.PropSlice{
		{"Defaults", ki.Props{
			"icon": "reset",
			"desc": "return all parameters to their intial default values",
		}},
		{"InitWts", ki.Props{
			"icon": "update",
			"desc": "initialize the layer's weight values according to prjn parameters, for all *sending* projections out of this layer",
		}},
		{"InitActs", ki.Props{
			"icon": "update",
			"desc": "initialize the layer's activation values",
		}},
		{"sep-act", ki.BlankProp{}},
		{"LesionNeurons", ki.Props{
			"icon": "close",
			"desc": "Lesion (set the Off flag) for given proportion of neurons in the layer (number must be 0 -- 1, NOT percent!)",
			"Args": ki.PropSlice{
				{"Proportion", ki.Props{
					"desc": "proportion (0 -- 1) of neurons to lesion",
				}},
			},
		}},
		{"UnLesionNeurons", ki.Props{
			"icon": "reset",
			"desc": "Un-Lesion (reset the Off flag) for all neurons in the layer",
		}},
	},
}
