#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include <softlib/SFile.h>
#include "DREAM/Constants.hpp"
#include "DREAM/EquationSystem.hpp"
#include "DREAM/Equations/Fluid/HeatTransportDiffusionNN.hpp"
#include "DREAM/Equations/Scalar/WallCurrentTerms.hpp"
#include "DREAM/Settings/Settings.hpp"
#include "DREAM/TimeStepper/TimeStepperConstant.hpp"

using namespace DREAM;
namespace {
    using Profile = std::vector<real_t>;
    real_t Interpolate(const Profile& coordinate, const Profile& values, real_t point) {
        if (point <= coordinate.front()) return values.front();
        if (point >= coordinate.back()) return values.back();
        const size_t upper = std::upper_bound(coordinate.begin(), coordinate.end(), point)-coordinate.begin();
        const real_t fraction=(point-coordinate[upper-1])/(coordinate[upper]-coordinate[upper-1]);
        return values[upper-1]*(1-fraction)+values[upper]*fraction;
    }
    Profile Smooth(const Profile& values) {
        if (values.size()<5) return values;
        Profile result(values.size());
        const int size=static_cast<int>(values.size());
        for (int index=0; index<size; ++index)
            for (int displacement=-2; displacement<=2; ++displacement) {
                int source=index+displacement;
                if (source<0) source=-source;
                if (source>=size) source=2*(size-1)-source;
                result[index]+=values[source]/5;
            }
        result[0]=values[0];
        return result;
    }
    real_t Peak(const Profile& values) {
        real_t peak=0;
        for (const auto value:values) peak=std::max(peak,std::abs(value));
        return peak;
    }
    std::string ModelPath(std::string supplied) {
        if (!supplied.empty()) return supplied;
        const char *overrideRoot=std::getenv("DREAM_NN_MODEL_ROOT");
        const std::string root=overrideRoot?overrideRoot:DREAM_NN_MODEL_ROOT;
        std::ifstream selection(root+"/DEFAULT_MODEL");
        std::string version;
        std::getline(selection,version);
        if (version.empty() || version.find('/')!=std::string::npos || version.find("..")!=std::string::npos)
            throw SettingsException("Invalid DREAM NN default-model pointer at %s",root.c_str());
        return root+"/"+version+"/transport.onnx";
    }
}

struct HeatTransportDiffusionNN::Implementation {
    EquationSystem *equations;
    FVM::RadialGrid *radial;
    FVM::UnknownQuantityHandler *unknowns;
    IonHandler *ions;
    Ort::Env environment{ORT_LOGGING_LEVEL_WARNING,"dream_nn_transport"};
    Ort::SessionOptions options;
    std::unique_ptr<Ort::Session> session;
    std::string modelPath;
    len_t nr,nmodel,window,mainIon=0;
    real_t cadence,Tref,psiScale,psiOffset,minorRadius;
    real_t previousTime=0,initialEnergy=0,initialCurrent=0,previousEnergy=0,previousCurrent=0;
    real_t nePeak=0,tePeak=0,psiPeak=0,pressurePeak=0,jPeak=0,quenchTime=0;
    bool ready=false;
    std::deque<std::vector<float>> history;
    Profile base,sampleTimes,savedTimes,savedBase,savedScaled,savedReady,savedCorrections;
    std::vector<float> savedFeatures;
    real_t correctionCount=0;
    struct Snapshot {
        Profile rho,ne,neutral,neon,te,psi,q,current,ni,ti,pressure;
        real_t energy=0,ip=0;
    } initial;

    Implementation(FVM::Grid *grid, EquationSystem *eqsys, const std::string& path, const std::string& model)
        : equations(eqsys),radial(grid->GetRadialGrid()),unknowns(eqsys->GetUnknownHandler()),
          ions(eqsys->GetIonHandler()),nr(grid->GetNr()),base(nr+1,0) {
        auto *settings=eqsys->GetSettings();
        const int_t modelPoints=settings->GetInteger(path+"/nn/n_rho");
        const int_t historyLength=settings->GetInteger(path+"/nn/window_size");
        if (modelPoints<3 || historyLength<2)
            throw SettingsException("NN transport requires positive model grid/history dimensions.");
        nmodel=modelPoints;
        window=historyLength;
        cadence=settings->GetReal(path+"/nn/time_step");
        Tref=settings->GetReal(path+"/nn/T_ref");
        psiScale=settings->GetReal(path+"/nn/psi_scale");
        psiOffset=settings->GetReal(path+"/nn/psi_offset");
        minorRadius=settings->GetReal(path+"/nn/normalization_minor_radius");
        if (minorRadius==0) minorRadius=radial->GetMinorRadius();
        if (nr<3 || !std::isfinite(cadence) || !std::isfinite(Tref) || !std::isfinite(minorRadius) || cadence<=0 || Tref<=0 || minorRadius<=0 ||
            !std::isfinite(radial->GetR0()) || radial->GetR0()<=0 || !std::isfinite(psiScale) || psiScale==0 || !std::isfinite(psiOffset))
            throw SettingsException("NN heat transport requires finite tokamak geometry and valid feature settings.");
        const std::string mainName=settings->GetString(path+"/nn/main_ion");
        bool found=false;
        for (len_t species=0; species<ions->GetNZ(); ++species)
            if (ions->GetName(species)==mainName) { mainIon=species; found=true; }
        if (!found || ions->GetZ(mainIon)!=1)
            throw SettingsException("NN heat transport requires a named hydrogen-isotope main ion: %s",mainName.c_str());
        options.SetIntraOpNumThreads(1); options.SetInterOpNumThreads(1);
        try {
            modelPath=ModelPath(model);
            session.reset(new Ort::Session(environment,modelPath.c_str(),options));
            if (session->GetInputCount()!=1 || session->GetOutputCount()!=2)
                throw SettingsException("NN transport requires one input and two outputs.");
            const auto inputType=session->GetInputTypeInfo(0);
            const auto inputInfo=inputType.GetTensorTypeAndShapeInfo();
            const auto shape=inputInfo.GetShape();
            if (shape!=std::vector<int64_t>{1,static_cast<int64_t>(window),static_cast<int64_t>(nmodel),14} ||
                inputInfo.GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
                throw SettingsException("NN transport graph shape does not match configured history/grid.");
            Ort::AllocatorWithDefaultOptions allocator;
            if (std::string(session->GetInputNameAllocated(0,allocator).get())!="features" ||
                std::string(session->GetOutputNameAllocated(0,allocator).get())!="log10_d" ||
                std::string(session->GetOutputNameAllocated(1,allocator).get())!="diffusion")
                throw SettingsException("NN transport graph input/output names do not match the deployment contract.");
            for (size_t output=0; output<2; ++output) {
                const auto outputType=session->GetOutputTypeInfo(output);
                const auto outputInfo=outputType.GetTensorTypeAndShapeInfo();
                if (outputInfo.GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
                    outputInfo.GetShape()!=std::vector<int64_t>{1,static_cast<int64_t>(nmodel)})
                    throw SettingsException("NN transport requires float32 full-profile outputs.");
            }
        } catch (const Ort::Exception& error) { throw SettingsException("Cannot load NN transport model: %s",error.what()); }
    }

    Snapshot ReadState() {
        Snapshot state;
        state.rho.resize(nr); state.ne.assign(nr,0); state.neutral.assign(nr,0); state.neon.assign(nr,0);
        state.ni.resize(nr); state.ti.resize(nr); state.q.resize(nr); state.current.resize(nr); state.pressure.resize(nr);
        auto read=[&](const std::string& name,len_t length) {
            const auto *data=unknowns->GetUnknownData(unknowns->GetUnknownID(name));
            Profile values(data,data+length);
            for (const auto value:values) if (!std::isfinite(value)) throw SettingsException("NN transport nonfinite input: %s",name.c_str());
            return values;
        };
        state.te=read(OptionConstants::UQTY_T_COLD,nr);
        state.psi=read(OptionConstants::UQTY_POL_FLUX,nr);
        const auto jtot=read(OptionConstants::UQTY_J_TOT,nr);
        const auto ionDensity=read(OptionConstants::UQTY_ION_SPECIES,ions->GetNzs()*nr);
        const auto ionEnergy=read(OptionConstants::UQTY_WI_ENER,ions->GetNZ()*nr);
        const auto coldEnergy=read(OptionConstants::UQTY_W_COLD,nr);
        state.ip=read(OptionConstants::UQTY_I_P,1)[0];
        const real_t psiEdge=read(OptionConstants::UQTY_PSI_EDGE,1)[0];
        const real_t weight=radial->GetR(0)/(radial->GetR(1)-radial->GetR(0));
        const real_t psiAxis=state.psi[0]*(1+weight)-state.psi[1]*weight;
        if (psiEdge==psiAxis) throw SettingsException("NN transport degenerate poloidal flux.");
        for (len_t ir=0; ir<nr; ++ir) {
            const real_t normalized=(state.psi[ir]-psiAxis)/(psiEdge-psiAxis);
            if (normalized<0 || normalized>1) throw SettingsException("NN transport normalized flux outside [0,1].");
            state.rho[ir]=sqrt(normalized);
            if (ir && state.rho[ir]<=state.rho[ir-1]) throw SettingsException("NN transport requires monotone flux surfaces.");
            state.psi[ir]=psiScale*state.psi[ir]+psiOffset;
            const real_t enclosed=TotalPlasmaCurrentFromJTot::EvaluateIpInsideR(ir,radial,jtot.data());
            state.q[ir]=radial->SafetyFactorNormalized(ir,Constants::mu0*enclosed)/radial->GetR0();
            state.current[ir]=jtot[ir]*radial->GetBTorG(ir)*radial->GetFSA_1OverR2(ir)/(radial->GetBmin(ir)*std::pow(radial->GetR0(),3));
            state.energy+=coldEnergy[ir]*radial->GetVpVol(ir)*radial->GetDr(ir);
        }
        for (len_t species=0; species<ions->GetNZ(); ++species) {
            real_t scale=0;
            for (len_t ir=0; ir<nr; ++ir) {
                real_t total=0;
                for (len_t charge=0; charge<=ions->GetZ(species); ++charge)
                    total+=std::max((real_t)0,ionDensity[ions->GetIndex(species,charge)*nr+ir]);
                scale=std::max(scale,total);
            }
            for (len_t ir=0; ir<nr; ++ir) {
                real_t total=0;
                for (len_t charge=0; charge<=ions->GetZ(species); ++charge) {
                    real_t density=ionDensity[ions->GetIndex(species,charge)*nr+ir];
                    if (density < -1e-6*scale) throw SettingsException("NN transport materially negative ion density.");
                    if (density<0) { density=0; correctionCount++; }
                    total+=density; state.ne[ir]+=charge*density;
                    if (ions->GetZ(species)==10) {
                        if (charge==0) state.neutral[ir]+=density;
                        else state.neon[ir]+=density;
                    }
                    if (species==mainIon && charge==1) state.ni[ir]=density;
                }
                const real_t energy=ionEnergy[species*nr+ir];
                if (energy<0 || (total==0 && energy!=0)) throw SettingsException("NN transport invalid ion thermal energy.");
                state.energy+=energy*radial->GetVpVol(ir)*radial->GetDr(ir);
                if (species==mainIon) state.ti[ir]=total>0?(2.0/3)*energy/(Constants::ec*total):0;
            }
        }
        for (len_t ir=0; ir<nr; ++ir) {
            if (!std::isfinite(state.q[ir]) || !std::isfinite(state.current[ir]) || state.te[ir]<0)
                throw SettingsException("NN transport invalid q/current/temperature.");
            state.pressure[ir]=Constants::ec*(state.ne[ir]*state.te[ir]+state.ni[ir]*state.ti[ir]);
        }
        return state;
    }

    std::vector<float> Features(const Snapshot& state,real_t dwth,real_t dip) {
        const auto neutral=Smooth(state.neutral),neon=Smooth(state.neon),te=Smooth(state.te),
            psi=Smooth(state.psi),current=Smooth(state.current),pressure=Smooth(state.pressure);
        std::array<Profile,14> fields;
        for (auto& field:fields) field.resize(nr);
        Profile surfaces1,surfaces2;
        for (int target=1; target<=2; ++target) {
            auto& surfaces=target==1?surfaces1:surfaces2;
            for (len_t ir=0; ir<nr; ++ir) {
                const real_t value=std::abs(state.q[ir]);
                if (std::abs(value-target)<=1e-10) surfaces.push_back(state.rho[ir]);
                if (ir+1<nr) {
                    const real_t next=std::abs(state.q[ir+1]);
                    if ((value-target)*(next-target)<0)
                        surfaces.push_back(state.rho[ir]+(target-value)/(next-value)*(state.rho[ir+1]-state.rho[ir]));
                }
            }
        }
        real_t integral=0;
        for (len_t ir=0; ir<nr; ++ir) {
            const real_t coordinate=state.rho[ir];
            fields[0][ir]=neutral[ir]/nePeak; fields[1][ir]=neon[ir]/nePeak;
            fields[2][ir]=te[ir]/tePeak; fields[3][ir]=psi[ir]/psiPeak; fields[4][ir]=std::abs(state.q[ir]);
            real_t distance=1e30;
            for (const auto surface:surfaces1) distance=std::min(distance,std::abs(coordinate-surface));
            fields[5][ir]=surfaces1.empty()?0:exp(-0.5*std::pow(distance/0.15,2));
            distance=1e30;
            for (const auto surface:surfaces2) if (std::abs(coordinate-surface)<std::abs(distance)) distance=coordinate-surface;
            fields[6][ir]=surfaces2.empty()?0:(distance>=0?1:-1);
            real_t gradient=0;
            if (ir+1==nr) gradient=(pressure[ir]-pressure[ir-1])/(state.rho[ir]-state.rho[ir-1]);
            else if (ir>0) {
                const real_t left=coordinate-state.rho[ir-1],right=state.rho[ir+1]-coordinate;
                gradient=-right/(left*(left+right))*pressure[ir-1]+(right-left)/(left*right)*pressure[ir]
                    +left/(right*(left+right))*pressure[ir+1];
            }
            fields[7][ir]=std::max((real_t)-5,std::min((real_t)5,gradient/pressurePeak));
            fields[8][ir]=state.energy/initialEnergy; fields[9][ir]=dwth*quenchTime*100;
            fields[10][ir]=state.ip/initialCurrent; fields[11][ir]=dip*quenchTime*100;
            const real_t safeq=std::abs(state.q[ir])>1e-6?state.q[ir]:1e-6;
            if (ir>0) {
                const real_t previousq=std::abs(state.q[ir-1])>1e-6?state.q[ir-1]:1e-6;
                integral+=0.5*(std::pow(coordinate,3)/(safeq*safeq)+std::pow(state.rho[ir-1],3)/(previousq*previousq))*(coordinate-state.rho[ir-1]);
            }
            fields[12][ir]=coordinate>1e-6?2*safeq*safeq/std::pow(coordinate,4)*integral:0.5;
            fields[13][ir]=current[ir]/jPeak;
        }
        std::vector<float> features(nmodel*14);
        for (len_t ir=0; ir<nmodel; ++ir)
            for (len_t feature=0; feature<14; ++feature) {
                const real_t value=Interpolate(state.rho,fields[feature],(real_t)ir/(nmodel-1));
                if (!std::isfinite(value)) throw SettingsException("NN transport produced nonfinite features.");
                features[ir*14+feature]=static_cast<float>(value);
            }
        return features;
    }
    void Record(real_t time,const real_t *weights) {
        savedTimes.push_back(time); savedReady.push_back(ready?1:0); savedCorrections.push_back(correctionCount);
        const auto *temperature=unknowns->GetUnknownData(unknowns->GetUnknownID(OptionConstants::UQTY_T_COLD));
        for (len_t ir=0; ir<=nr; ++ir) {
            real_t faceT=0;
            if (ir<nr) faceT+=weights[ir]*temperature[ir];
            if (ir>0) faceT+=(1-weights[ir])*temperature[ir-1];
            savedBase.push_back(base[ir]); savedScaled.push_back(base[ir]*sqrt(std::max(faceT,(real_t)1e-12)/Tref));
        }
    }
};

HeatTransportDiffusionNN::HeatTransportDiffusionNN(FVM::Grid *grid,EquationSystem *equations,const std::string& path,const std::string& model)
    : HeatTransportDiffusionTDependent(grid,nullptr,equations->GetUnknownHandler(),equations->GetSettings()->GetReal(path+"/nn/T_ref")),
      implementation(new Implementation(grid,equations,path,model)) { SetName("HeatTransportDiffusionNN"); }
HeatTransportDiffusionNN::~HeatTransportDiffusionNN()=default;
const real_t *HeatTransportDiffusionNN::GetBaseDiffusion(real_t) { return implementation->base.data(); }

void HeatTransportDiffusionNN::InitializeHistory() {
    auto& state=*implementation;
    if (dynamic_cast<TimeStepperConstant*>(state.equations->GetTimeStepper())==nullptr)
        throw SettingsException("NN heat transport currently requires a constant timestepper.");
    state.initial=state.ReadState();
    state.initialEnergy=state.previousEnergy=state.initial.energy;
    state.initialCurrent=state.previousCurrent=state.initial.ip;
    state.nePeak=Peak(state.initial.ne); state.tePeak=Peak(Smooth(state.initial.te)); state.psiPeak=Peak(Smooth(state.initial.psi));
    const auto initialJ=Smooth(state.initial.current);
    state.jPeak=*std::max_element(initialJ.begin(),initialJ.end(),[](real_t left,real_t right){return std::abs(left)<std::abs(right);});
    Profile pressure(state.nr);
    const auto ne=Smooth(state.initial.ne),te=Smooth(state.initial.te),ni=Smooth(state.initial.ni),ti=Smooth(state.initial.ti);
    for (len_t ir=0; ir<state.nr; ++ir) pressure[ir]=Constants::ec*(ne[ir]*te[ir]+ni[ir]*ti[ir]);
    state.pressurePeak=Peak(Smooth(pressure));
    if (state.initialEnergy<=0 || state.initialCurrent==0 || state.nePeak<=0 || state.tePeak<=0 || state.psiPeak<=0 || state.jPeak==0 || state.pressurePeak<=0)
        throw SettingsException("NN heat transport requires nonzero finite initial feature normalization.");
    Profile absq=state.initial.q;
    for (auto& value:absq) value=std::abs(value);
    const real_t q95=Interpolate(state.initial.rho,absq,sqrt(0.95));
    if (q95<=0) throw SettingsException("NN heat transport requires positive initial q95.");
    state.quenchTime=1e-3*state.minorRadius/q95;
    state.Record(0,deltaRadialFlux);
}

void HeatTransportDiffusionNN::AcceptStep(real_t time) {
    auto& state=*implementation;
    if (state.savedTimes.size()==1) {
        const real_t ratio=state.cadence/time;
        if (std::round(ratio)<1 || std::abs(ratio-std::round(ratio))>1e-8)
            throw SettingsException("NN sampling interval must be an integer multiple of the DREAM timestep.");
    }
    const real_t interval=time-state.previousTime;
    if (interval<state.cadence-1e-12) { state.Record(time,deltaRadialFlux); return; }
    if (std::abs(interval-state.cadence)>1e-12)
        throw SettingsException("NN accepted-state cadence mismatch; adaptive sampling is unsupported.");
    const auto snapshot=state.ReadState();
    const real_t dwth=(snapshot.energy-state.previousEnergy)/(state.initialEnergy*interval);
    const real_t dip=(snapshot.ip-state.previousCurrent)/(state.initialCurrent*interval);
    if (state.history.empty()) {
        state.history.push_back(state.Features(state.initial,dwth,dip)); state.sampleTimes.push_back(0);
        state.savedFeatures.insert(state.savedFeatures.end(),state.history.back().begin(),state.history.back().end());
    }
    state.history.push_back(state.Features(snapshot,dwth,dip)); state.sampleTimes.push_back(time);
    state.savedFeatures.insert(state.savedFeatures.end(),state.history.back().begin(),state.history.back().end());
    if (state.history.size()>state.window) state.history.pop_front();
    if (state.history.size()==state.window) {
        std::vector<float> inputs;
        for (const auto& frame:state.history) inputs.insert(inputs.end(),frame.begin(),frame.end());
        const std::array<int64_t,4> shape{1,static_cast<int64_t>(state.window),static_cast<int64_t>(state.nmodel),14};
        auto memory=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
        auto tensor=Ort::Value::CreateTensor<float>(memory,inputs.data(),inputs.size(),shape.data(),shape.size());
        const char *inputNames[]{"features"}, *outputNames[]{"diffusion"};
        try {
            auto outputs=state.session->Run(Ort::RunOptions{nullptr},inputNames,&tensor,1,outputNames,1);
            if (outputs[0].GetTensorTypeAndShapeInfo().GetShape()!=std::vector<int64_t>{1,static_cast<int64_t>(state.nmodel)})
                throw SettingsException("NN transport output shape mismatch.");
            const auto *values=outputs[0].GetTensorData<float>();
            Profile rhoModel(state.nmodel),predicted(state.nmodel);
            for (len_t ir=0; ir<state.nmodel; ++ir) {
                rhoModel[ir]=(real_t)ir/(state.nmodel-1); predicted[ir]=values[ir];
                if (!std::isfinite(predicted[ir]) || predicted[ir]<0) throw SettingsException("NN transport invalid diffusion prediction.");
            }
            const Profile centers(state.radial->GetR(),state.radial->GetR()+state.nr);
            for (len_t ir=0; ir<=state.nr; ++ir) {
                const real_t rhoFace=ir==0?0:(ir==state.nr?1:Interpolate(centers,snapshot.rho,state.radial->GetR_f(ir)));
                state.base[ir]=Interpolate(rhoModel,predicted,rhoFace);
            }
            state.ready=true;
        } catch (const Ort::Exception& error) { throw SettingsException("NN transport inference failed: %s",error.what()); }
    }
    state.previousTime=time; state.previousEnergy=snapshot.energy; state.previousCurrent=snapshot.ip;
    state.Record(time,deltaRadialFlux);
}

void HeatTransportDiffusionNN::SaveDiagnostics(SFile *file,const std::string& name) {
    auto& state=*implementation;
    file->CreateStruct(name);
    const sfilesize_t count=state.savedTimes.size(),faces=state.nr+1;
    file->WriteString(name+"/model",state.modelPath);
    file->WriteString(name+"/profile_timing","Base profile available after each accepted state, frozen during the following solve; scaled diffusion uses the accepted state's temperature.");
    file->WriteScalar(name+"/T_ref",state.Tref);
    file->WriteScalar(name+"/time_step",state.cadence);
    file->WriteScalar(name+"/psi_scale",state.psiScale);
    file->WriteScalar(name+"/psi_offset",state.psiOffset);
    file->WriteScalar(name+"/normalization_minor_radius",state.minorRadius);
    if (count==0) return;
    file->WriteList(name+"/t",state.savedTimes.data(),count);
    file->WriteList(name+"/ready",state.savedReady.data(),count);
    file->WriteList(name+"/density_corrections",state.savedCorrections.data(),count);
    const sfilesize_t profileDimensions[]{count,faces};
    file->WriteMultiArray(name+"/base_diffusion",state.savedBase.data(),2,profileDimensions);
    file->WriteMultiArray(name+"/scaled_diffusion",state.savedScaled.data(),2,profileDimensions);
    file->WriteList(name+"/sample_times",state.sampleTimes.data(),state.sampleTimes.size());
    if (!state.sampleTimes.empty()) {
        std::vector<real_t> features(state.savedFeatures.begin(),state.savedFeatures.end());
        const sfilesize_t dimensions[]{state.sampleTimes.size(),state.nmodel,14};
        file->WriteMultiArray(name+"/features",features.data(),3,dimensions);
    }
}