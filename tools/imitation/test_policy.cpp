/* Copyright (C) 2026 OpenMoHAA contributors. GPL-2.0-or-later. */
#include "imitation_policy.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <vector>
using Bytes=std::vector<unsigned char>;
void Check(bool b,const char *message){if(!b)throw std::runtime_error(message);}
void U32(Bytes& b,std::uint32_t n){for(int i=0;i<4;++i)b.push_back((n>>(i*8))&255);}
void F32(Bytes& b,float f){std::uint32_t n;std::memcpy(&n,&f,4);U32(b,n);}
void Put(Bytes& b,std::size_t index,float f){std::uint32_t n;std::memcpy(&n,&f,4);for(int i=0;i<4;++i)b.at(36+index*4+i)=(n>>(8*i))&255;}
Bytes Fixture(){
    constexpr int f=imitation::features,h=imitation::hidden,o=imitation::output;
    constexpr int n=2*f+3*h*f+3*h*h+6*h+o*h+o+4;
    Bytes b={'O','M','I','M','0','0','0','1'};
    for(unsigned x:{unsigned(f),unsigned(h),4u,unsigned(o),imitation::mapChecksum,20u,unsigned(n)})U32(b,x);
    for(int i=0;i<n;++i)F32(b,i>=f && i<2*f?1.f:0.f);
    return b;
}
imitation::Observation Known(){
    imitation::Observation o;o.position={1024,-512,256};o.velocity={200,-100,0};o.pitch=10;o.yaw=90;
    o.lean=20;o.viewHeight=82;o.ducked=true;o.walking=true;o.health=80;o.clipAmmo=20;o.reserveAmmo=150;
    o.weaponClass=1;o.previousMove={127,-127,0};o.previousButtons=4|1|32;o.previousViewDelta={2,-4};
    o.target=true;o.targetRelative={100,0,10};o.targetDistance=100;o.targetVelocity={40,-20,0};return o;
}
void CoreTests(){
    auto b=Fixture();imitation::Policy p;std::string error;
    Check(p.Load(b.data(),b.size(),error),error.c_str());
    imitation::Hidden h{},independent{};imitation::Features x{};
    auto out=p.Step(x,h);Check(h==independent,"zero GRU fixture");
    for(std::size_t n:{std::size_t(0),std::size_t(8),std::size_t(35),b.size()-1})Check(!p.Load(b.data(),n,error),"truncated model accepted");
    auto bad=b;bad[0]='X';Check(!p.Load(bad.data(),bad.size(),error),"bad magic accepted");
    bad=b;bad[8]^=1;Check(!p.Load(bad.data(),bad.size(),error),"wrong feature contract accepted");
    bad=b;bad[24]^=1;Check(!p.Load(bad.data(),bad.size(),error),"wrong map accepted");
    bad=b;Put(bad,imitation::features,0);Check(!p.Load(bad.data(),bad.size(),error),"zero normalization accepted");
    bad=b;Put(bad,0,std::nanf(""));Check(!p.Load(bad.data(),bad.size(),error),"NaN accepted");
    Check(p.Loaded() && p.Step(x,h)==out,"failed load changed prior model");
    const int inputBias=2*imitation::features+3*imitation::hidden*imitation::features+3*imitation::hidden*imitation::hidden;
    for(int j=0;j<imitation::hidden;++j)Put(b,inputBias+2*imitation::hidden+j,std::atanh(.5f));
    Check(p.Load(b.data(),b.size(),error),"fixture update");h={};p.Step(x,h);Check(std::abs(h[0]-.25f)<1e-6f,"first GRU state");
    p.Step(x,h);Check(std::abs(h[0]-.375f)<1e-6f,"second GRU state");
    p.Step(x,independent);Check(std::abs(independent[0]-.25f)<1e-6f,"shared hidden states");
    imitation::Output logits{};logits[2]=10;int offset=4+2*35;
    logits[offset+6]=10;logits[offset+9+1]=10;logits[offset+12+2]=10;logits[offset+15+3]=10;
    logits[offset+31]=std::asinh(2.f);logits[offset+32]=std::asinh(-3.f);
    auto a=imitation::Decode(logits);Check(a.forward==127 && a.right==-127 && a.up==0,"XY decode");
    Check(a.buttons==(4u|1u|32u) && std::abs(a.yawDelta+3)<1e-5f,"joint action decode");
    imitation::FireGate gate{true,true,true,true,true};Check(gate.Allows(),"valid fire gate");
    for(int i=0;i<5;++i){bool v[5]={true,true,true,true,true};v[i]=false;Check(!imitation::FireGate{v[0],v[1],v[2],v[3],v[4]}.Allows(),"unsafe firing");}
    auto f=imitation::Encode(Known());Check(f.size()==54 && f[0]==.5f && f[1]==-.25f && f[46]==1,"feature order");
    auto unknown=Known();unknown.target=false;f=imitation::Encode(unknown);for(int i=46;i<54;++i)Check(f[i]==0,"hidden target leakage");
    Check(imitation::WeaponClass("KAR98 - Sniper")==2 && imitation::WeaponClass("Thompson")==1,"weapon classes");
    Check(imitation::AngleDelta(179,-179)==2,"angle wrap");
    const int nf=int((b.size()-36)/4);for(int i=nf-4;i<nf;++i)Put(b,i,2.f);
    Check(p.Load(b.data(),b.size(),error),"calibrated fixture");
    imitation::Output zero{};int stays=0;
    for(int i=0;i<1000;++i){auto chosen=p.Choose(zero,{7,1,1,1},{.1f,(i+.5f)/1000,.5f,.5f,.5f});stays+=(chosen.forward==127 && chosen.right==0);}
    Check(stays>450 && stays<510,"calibration/sampling contract");
    bool rejected=false;try{p.Choose(zero,{9,1,1,1},{0,0,0,0,0});}catch(const std::invalid_argument&){rejected=true;}
    Check(rejected,"invalid prior action accepted");
    std::cout<<"PASS: parser, transactional load, recurrent math, independent states, joint decoding, temporal sampling, features and fire gate\n";
}
int main(int argc,char **argv){
    try{
        CoreTests();
        if(argc==3 && std::string(argv[1])=="--features"){
            auto f=imitation::Encode(Known());std::ofstream o(argv[2],std::ios::binary);o.write(reinterpret_cast<char*>(f.data()),f.size()*4);return 0;
        }
        if(argc==4){
            std::ifstream in(argv[1],std::ios::binary);Bytes b((std::istreambuf_iterator<char>(in)),{});
            imitation::Policy p;std::string error;Check(p.Load(b.data(),b.size(),error),error.c_str());
            std::ifstream fixture(argv[2],std::ios::binary);std::ofstream result(argv[3],std::ios::binary);
            char magic[8];std::uint32_t rows;fixture.read(magic,8);fixture.read(reinterpret_cast<char*>(&rows),4);
            Check(!std::memcmp(magic,"IMCHECK1",8) && rows<=4096,"bad parity fixture");imitation::Hidden h{};
            auto begin=std::chrono::steady_clock::now();
            for(unsigned i=0;i<rows;++i){
                unsigned reset;imitation::Features x;std::array<int,4> previous;std::array<float,5> uniform;
                fixture.read(reinterpret_cast<char*>(&reset),4);fixture.read(reinterpret_cast<char*>(x.data()),x.size()*4);
                fixture.read(reinterpret_cast<char*>(previous.data()),16);fixture.read(reinterpret_cast<char*>(uniform.data()),20);
                Check(bool(fixture),"truncated parity fixture");if(reset)h={};auto out=p.Step(x,h);auto a=p.Choose(out,previous,uniform);
                result.write(reinterpret_cast<char*>(out.data()),out.size()*4);result.write(reinterpret_cast<char*>(h.data()),h.size()*4);
                const float action[]={float(a.forward),float(a.right),float(a.up),float(a.buttons),a.pitchDelta,a.yawDelta,a.mixtureWeight,float(a.component)};
                result.write(reinterpret_cast<const char*>(action),sizeof(action));
            }
            Check(bool(result),"cannot write parity results");
            std::cout<<"Checked "<<rows<<" exported steps in "<<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count()<<" ms (includes IO)\n";
        }
    }catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
