/*
 * Native-control mohdm6 imitation. Copyright (C) 2026 OpenMoHAA contributors.
 * SPDX-License-Identifier: GPL-2.0-or-later. No warranty; see COPYING.txt.
 */
#include "imitation_policy.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace imitation {
namespace {
constexpr float pi = 3.14159265358979323846f;
constexpr std::size_t count = 2*features + 3*hidden*features + 3*hidden*hidden + 6*hidden + output*hidden + output + 4;
float Sigmoid(float x) { return 1.f/(1.f+std::exp(-std::clamp(x,-80.f,80.f))); }
std::uint32_t U32(const unsigned char *p) {
    return std::uint32_t(p[0]) | std::uint32_t(p[1])<<8 | std::uint32_t(p[2])<<16 | std::uint32_t(p[3])<<24;
}
int ArgMax(const float *p, int n) { return int(std::max_element(p,p+n)-p); }
}
float AngleDelta(float from, float to) {
    float d=std::fmod(to-from+180.f,360.f);
    if(d<0)d+=360.f;
    return d-180.f;
}
int WeaponClass(std::string name) {
    for(char& c:name)if(c>='A' && c<='Z')c=char(c-'A'+'a');
    if(name.empty())return 0;
    if(name.find("sniper")!=std::string::npos)return 2;
    if(name=="mp40" || name=="thompson")return 1;
    if(name=="stg 44" || name=="mp44" || name=="bar")return 4;
    if(name.find("grenade")!=std::string::npos || name.find("stielhandgranate")!=std::string::npos)return 6;
    if(name.find("shotgun")!=std::string::npos)return 7;
    if(name.find("colt")!=std::string::npos || name.find("p38")!=std::string::npos)return 5;
    if(name.find("garand")!=std::string::npos || name.find("kar 98")!=std::string::npos || name.find("kar98")!=std::string::npos)return 3;
    return 8;
}
Features Encode(const Observation& o) {
    Features f{}; std::size_t n=0;
    auto add=[&](float x){ if(!std::isfinite(x))throw std::invalid_argument("nonfinite imitation observation"); f.at(n++)=std::clamp(x,-16.f,16.f); };
    add(o.position[0]/2048);add(o.position[1]/2048);add(o.position[2]/512);
    for(float x:o.velocity)add(x/400);
    const float yaw=o.yaw*pi/180, pitch=o.pitch*pi/180;
    add(std::sin(yaw));add(std::cos(yaw));add(std::sin(pitch));add(std::cos(pitch));
    add(o.lean/40);add(o.viewHeight/96);add(o.ducked);add(o.prone);add(o.walking);
    add(o.health/100);add(std::max(0.f,o.clipAmmo)/50);add(std::max(0.f,o.reserveAmmo)/300);
    for(int i=0;i<9;++i)add(o.weaponClass==i);
    for(int x:o.previousMove)add(float(x)/127);
    for(unsigned b:{4u,16u,32u,1u,2u,8u})add((o.previousButtons&b)!=0);
    for(float x:o.previousViewDelta)add(std::asinh(x)/5);
    for(int axis=0;axis<2;++axis)for(int wavelength:{1024,512}) {
        const float p=2*pi*o.position[axis]/wavelength;
        add(std::sin(p));add(std::cos(p));
    }
    const bool target=o.target && o.targetDistance>=1 && o.targetDistance<=4096;
    add(target);
    for(float x:o.targetRelative)add(target?x/o.targetDistance:0);
    add(target?o.targetDistance/4096:0);
    for(float x:o.targetVelocity)add(target?x/400:0);
    if(n!=f.size())throw std::logic_error("imitation feature contract mismatch");
    return f;
}
bool Policy::Load(const void *data, std::size_t size, std::string& error) {
    error.clear();
    try {
        static_assert(sizeof(float)==4 && std::numeric_limits<float>::is_iec559,"IEEE float32 required");
        if(!data || size!=36+4*count)throw std::runtime_error("invalid imitation model size");
        const auto *p=static_cast<const unsigned char *>(data);
        if(std::memcmp(p,"OMIM0001",8))throw std::runtime_error("unknown imitation model format");
        const unsigned expected[]={features,hidden,components,output,mapChecksum,tickMsec,static_cast<unsigned>(count)};
        for(int i=0;i<7;++i)if(U32(p+8+4*i)!=expected[i])throw std::runtime_error("incompatible imitation model contract");
        std::vector<float> parsed(count);
        for(std::size_t i=0;i<count;++i) {
            auto bits=U32(p+36+4*i);std::memcpy(&parsed[i],&bits,4);
            if(!std::isfinite(parsed[i]) || std::fabs(parsed[i])>100000)throw std::runtime_error("invalid imitation weights");
            if(i>=features && i<2*features && parsed[i]<.000001f)throw std::runtime_error("invalid observation normalization");
        }
        weights.swap(parsed);return true;
    }catch(const std::exception& e){error=e.what();return false;}
}
Output Policy::Step(const Features& x, Hidden& h) const {
    if(!Loaded())throw std::logic_error("imitation model not loaded");
    Features norm{};
    for(int i=0;i<features;++i) {
        if(!std::isfinite(x[i]))throw std::invalid_argument("nonfinite features");
        norm[i]=std::clamp((x[i]-weights[i])/weights[features+i],-10.f,10.f);
    }
    for(float v:h)if(!std::isfinite(v) || std::fabs(v)>1.00001f)throw std::invalid_argument("invalid recurrent state");
    const float *wi=weights.data()+2*features, *wh=wi+3*hidden*features;
    const float *bi=wh+3*hidden*hidden,*bh=bi+3*hidden,*wo=bh+3*hidden,*bo=wo+output*hidden;
    std::array<float,3*hidden> a{},b{};
    for(int i=0;i<3*hidden;++i) {
        a[i]=bi[i];b[i]=bh[i];
        for(int j=0;j<features;++j)a[i]+=wi[i*features+j]*norm[j];
        for(int j=0;j<hidden;++j)b[i]+=wh[i*hidden+j]*h[j];
    }
    // PyTorch GRU gate order and reset-after-matrix-multiply convention.
    for(int i=0;i<hidden;++i) {
        float r=Sigmoid(a[i]+b[i]),z=Sigmoid(a[hidden+i]+b[hidden+i]);
        float next=std::tanh(a[2*hidden+i]+r*b[2*hidden+i]);
        h[i]=(1-z)*next+z*h[i];
    }
    Output out{};
    for(int i=0;i<output;++i) {
        out[i]=bo[i];for(int j=0;j<hidden;++j)out[i]+=wo[i*hidden+j]*h[j];
        if(!std::isfinite(out[i]))throw std::runtime_error("nonfinite policy output");
    }
    return out;
}
Action Decode(const Output& out) {
    for(float v:out)if(!std::isfinite(v))throw std::invalid_argument("nonfinite action output");
    Action a; a.component=ArgMax(out.data(),components);
    const float *c=out.data()+components+35*a.component;
    int xy=ArgMax(c,9),vertical=ArgMax(c+9,3),lean=ArgMax(c+12,3),bits=ArgMax(c+15,16);
    a.forward=127*(xy/3-1);a.right=127*(xy%3-1);a.up=127*(vertical-1);
    a.buttons=((bits&1)?4:0) | ((bits&2)?1:0) | ((bits&4)?2:0) | ((bits&8)?8:0) | (lean==0?16:lean==2?32:0);
    a.pitchDelta=std::sinh(std::clamp(c[31],-5.19f,5.19f));
    a.yawDelta=std::sinh(std::clamp(c[32],-5.19f,5.19f));
    float denominator=0;for(int i=0;i<components;++i)denominator+=std::exp(out[i]-out[a.component]);
    a.mixtureWeight=1/denominator;
    return a;
}

Action Policy::Choose(const Output& out, const std::array<int,4>& previous,
                      const std::array<float,5>& uniform) const {
    if(!Loaded())throw std::logic_error("imitation model not loaded");
    constexpr int sizes[]={9,3,3,16};
    for(int j=0;j<4;++j)if(previous[j]<0 || previous[j]>=sizes[j])throw std::invalid_argument("invalid previous action");
    for(float u:uniform)if(!std::isfinite(u) || u<0 || u>=1)throw std::invalid_argument("invalid random variate");
    for(float v:out)if(!std::isfinite(v))throw std::invalid_argument("nonfinite policy logits");
    Output adjusted=out;
    auto logSum=[](const float *p,int n) {
        float m=*std::max_element(p,p+n),sum=0;
        for(int i=0;i<n;++i)sum+=std::exp(p[i]-m);
        return m+std::log(sum);
    };
    for(int k=0;k<components;++k) {
        int offset=components+35*k;
        for(int j=0;j<4;++j) {
            const float before=logSum(out.data()+offset,sizes[j]);
            adjusted[offset+previous[j]]+=weights[count-4+j];
            adjusted[k]+=logSum(adjusted.data()+offset,sizes[j])-before;
            offset+=sizes[j];
        }
    }
    auto draw=[&](const float *p,int n,float u) {
        const float denominator=logSum(p,n);float sum=0;
        for(int i=0;i<n;++i) {sum+=std::exp(p[i]-denominator);if(u<sum)return i;}
        return n-1;
    };
    const int chosen=draw(adjusted.data(),components,uniform[0]);
    const float weight=std::exp(adjusted[chosen]-logSum(adjusted.data(),components));
    for(int k=0;k<components;++k)adjusted[k]=k==chosen?100.f:-100.f;
    int offset=components+35*chosen;
    for(int j=0;j<4;++j) {
        const int picked=draw(adjusted.data()+offset,sizes[j],uniform[j+1]);
        for(int i=0;i<sizes[j];++i)adjusted[offset+i]=i==picked?100.f:-100.f;
        offset+=sizes[j];
    }
    Action a=Decode(adjusted);a.mixtureWeight=weight;return a;
}

} // namespace imitation
