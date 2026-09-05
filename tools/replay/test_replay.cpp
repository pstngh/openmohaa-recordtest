/* Copyright (C) 2026 OpenMoHAA contributors
 * SPDX-License-Identifier: GPL-2.0-or-later. No warranty; see COPYING.txt. */
#include "replay_track.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <set>
#include <stdexcept>
#include <vector>

namespace {
void Check(bool value, const char *message) { if (!value) throw std::runtime_error(message); }
using Bytes = std::vector<unsigned char>;
void U32(Bytes& b, std::uint32_t n) { for (int i = 0; i < 4; ++i) b.push_back((n >> (8*i)) & 255); }
void F32(Bytes& b, float f) { std::uint32_t bits; std::memcpy(&bits, &f, 4); U32(b, bits); }
void Text(Bytes& b, const std::string& s) { U32(b, s.size()); b.insert(b.end(), s.begin(), s.end()); }
void Vec(Bytes& b, const replay::Vec3& v) { for (float x : v) F32(b, x); }
void Set32(Bytes& b, std::size_t offset, std::uint32_t n) {
    for (int i=0; i<4; ++i) b.at(offset+i) = (n >> (8*i)) & 255;
}
Bytes Fixture(std::size_t& frameStart) {
    Bytes b{'O','M','R','P','L','0','0','1'};
    Text(b, "dm/test");
    U32(b, 0xffff1234); U32(b,1); U32(b,8); U32(b,50); U32(b,1);
    Text(b,"life0"); U32(b,3); Vec(b,{10,0,1}); U32(b,150); U32(b,1);
    Text(b, "Thompson"); U32(b,0); U32(b,30); U32(b,90);
    U32(b,3); U32(b,3);
    frameStart=b.size();
    for (int i=0;i<3;++i) {
        U32(b,i*50); Vec(b,{float(10+10*i),0,1}); Vec(b,{200,0,0});
        Vec(b,{0,i==0?179.0f:-179.0f,0}); Vec(b,{-16,-16,0}); Vec(b,{16,16,float(i==0?94:60)});
        Vec(b,{0,0,82}); U32(b,i==0?0:1);
        U32(b,127); U32(b,0); U32(b,0); U32(b,i==0?4:5); U32(b,1); U32(b,0); F32(b,0);
    }
    U32(b,0); U32(b,1); U32(b,50); U32(b,2); U32(b,100); U32(b,3);
    return b;
}
void CoreTests() {
    std::size_t frameStart;
    Bytes bytes=Fixture(frameStart);
    replay::Library lib; std::string error;
    Check(replay::Load(bytes.data(),bytes.size(),lib,error),error.c_str());
    Check(replay::Compatible(lib,"dm/test",0xffff1234,1,8),"matching identity");
    Check(!replay::Compatible(lib,"dm/test",0,1,8),"checksum filter");
    Check(!replay::Compatible(lib,"dm/other",0xffff1234,1,8),"map filter");
    Check(!replay::Compatible(lib,"dm/test",0xffff1234,2,8),"mode filter");
    Check(!replay::Compatible(lib,"dm/test",0xffff1234,1,15),"protocol filter");
    const auto c=lib.clips[0];
    for (auto f:c.frames) {
        const auto s=replay::Sample(c,f.time);
        Check(s.origin==f.origin && s.velocity==f.velocity && s.angles==f.angles && s.mins==f.mins
              && s.maxs==f.maxs && s.buttons==f.buttons,"exact stored samples");
    }
    auto halfway=replay::Sample(c,25);
    Check(halfway.origin[0]==15 && std::fabs(halfway.angles[1]-180)<.001f,"interpolation and angle wrap");
    Check(halfway.maxs[2]==94 && halfway.buttons==4,"posture and actions held, not interpolated");
    Check(replay::Sample(c,1000).origin==c.frames.back().origin,"tail does not extrapolate");
    std::size_t cursor=0;
    Check(replay::DueActions(c,cursor,0).size()==1,"time zero event");
    Check(replay::DueActions(c,cursor,0).empty(),"no duplicate event");
    Check(replay::DueActions(c,cursor,110).size()==2,"crossed events delivered once");
    Check(replay::DueActions(c,cursor,200).empty(),"no duplicate later events");
    for (std::size_t n=0;n<bytes.size();++n) {
        Check(!replay::Load(bytes.data(),n,lib,error),"truncated file accepted");
        Check(lib.map=="dm/test" && lib.clips.size()==1,"failed parse modified caller library");
    }
    for (auto change : std::vector<std::pair<std::size_t,std::uint32_t>>{
            {0,0}, {frameStart+4,0x7fc00000}, {frameStart+80,0x80000000},
            {frameStart+80,128}, {frameStart+100,1}, {frameStart+108,0}, {frameStart+108,125}}) {
        auto bad=bytes; Set32(bad,change.first,change.second);
        Check(!replay::Load(bad.data(),bad.size(),lib,error),"corrupt recording accepted");
    }
    auto bad=bytes;bad.push_back(0);
    Check(!replay::Load(bad.data(),bad.size(),lib,error),"trailing bytes accepted");
    auto repeated=c;
    for (int i=1;i<5;++i) { repeated.id="life"+std::to_string(i); lib.clips.push_back(repeated); }
    replay::SpawnSelector selector(10), equal(10);
    std::size_t last=static_cast<std::size_t>(-1);
    for (int cycle=0;cycle<20;++cycle) {
        std::set<std::size_t> seen;
        for (int n=0;n<5;++n) {
            const auto choice=selector.Select(lib,c.spawn,1);
            Check(choice==equal.Select(lib,c.spawn,1),"seed must reproduce selection sequence");
            Check(choice<5 && choice!=last,"invalid choice or immediate repetition");
            Check(seen.insert(choice).second,"repeat before shuffled pool exhausted");
            last=choice;
        }
    }
    const auto none=static_cast<std::size_t>(-1);
    Check(selector.Select(lib,{500,0,1},8)==none,"wrong spawn selected");
    Check(selector.Select(lib,c.spawn,33)==none,"unsafe tolerance accepted");
    Check(selector.Select(lib,{std::numeric_limits<float>::quiet_NaN(),0,1},8)==none,"NaN spawn accepted");
    selector.Reset(3);
    Check(selector.Select(lib,c.spawn,1,{0,1,2,3,4})==none,"busy clips selected");
    Check(selector.Select(lib,c.spawn,1,{0,1,2,3})==4,"busy pool recovery");
    repeated.spawn[0]+=3; lib.clips.push_back(repeated);
    Check(selector.Select(lib,c.spawn,8)==none,"ambiguous anchors merged");
    // Team and weapon metadata cannot partition a shared spawn pool in either mode.
    lib.clips.pop_back(); // Remove the deliberately ambiguous anchor above.
    for (int mode : {1, 2}) {
        lib.gameType = mode;
        for (std::size_t i = 0; i < lib.clips.size(); ++i) {
            lib.clips[i].team = i % 2 ? 3 : 4;
            lib.clips[i].weapons[0].name = i % 2 ? "MP40" : "Springfield '03 Sniper";
        }
        selector.Reset(10);
        for (int cycle = 0; cycle < 5; ++cycle) {
            std::set<std::size_t> seen;
            for (std::size_t i = 0; i < lib.clips.size(); ++i) {
                const auto choice = selector.Select(lib, c.spawn, 1);
                Check(choice < lib.clips.size(), "cross-team/weapon life excluded");
                Check(seen.insert(choice).second, "separate team bags repeat before shared pool exhaustion");
                Check(replay::Sample(lib.clips[choice], 50).origin == c.frames[1].origin,
                      "loadout-neutral selection changed a movement sample");
            }
        }
        selector.Reset(3);
        Check(selector.Select(lib, c.spawn, 1, {0, 1, 2, 3}) == 4, "cross-team busy exclusion");
        repeated.team = 4;
        lib.clips.push_back(repeated);
        Check(selector.Select(lib, c.spawn, 8) == none, "cross-team ambiguous anchors merged");
        lib.clips.pop_back();
    }
    std::cout << "Core replay tests passed (parser, exact sampling, event timing, spawn pools).\n";
}
void VerifyFile(const char *name) {
    std::ifstream file(name,std::ios::binary);
    Check(bool(file),"cannot open replay library");
    Bytes bytes((std::istreambuf_iterator<char>(file)),{});
    replay::Library lib;std::string error;
    Check(replay::Load(bytes.data(),bytes.size(),lib,error),error.c_str());
    std::size_t count=0;
    for (const auto& clip:lib.clips) for (const auto& frame:clip.frames) {
        const auto sample=replay::Sample(clip,frame.time);
        Check(sample.origin==frame.origin && sample.velocity==frame.velocity && sample.angles==frame.angles,
              "sample drift");
        ++count;
    }
    std::cout << name << ": " << lib.clips.size() << " lives, " << count << " exact samples, zero sample drift.\n";
}
}
int main(int argc,char **argv) {
    try { CoreTests(); for (int i=1;i<argc;++i) VerifyFile(argv[i]); }
    catch (const std::exception& e) { std::cerr << "FAIL: " << e.what() << '\n'; return 1; }
}
