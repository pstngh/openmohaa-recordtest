/* Copyright (C) 2026 OpenMoHAA contributors. SPDX-License-Identifier: GPL-2.0-or-later. */
#include "imitation_runtime.h"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
void Check(bool value,const char *why){if(!value)throw std::runtime_error(why);}
int main(){try{
    imitation::ControlFeedback feedback;
    imitation::Action intent;intent.forward=127;intent.buttons=4|1|2|8;
    feedback.Record(intent,4);
    Check((feedback.intent.buttons&1)!=0,"denied fire erased model intent");
    Check(feedback.sentButtons==4,"guard suppression was bypassed");
    imitation::Observation o;o.previousMove={feedback.intent.forward,0,0};o.previousButtons=feedback.intent.buttons;
    auto f=imitation::Encode(o);
    Check(f[33]==1 && f[34]==1 && f[35]==1,"observation did not preserve requested button history");
    feedback.Reset();Check(!feedback.intent.buttons && !feedback.sentButtons,"reset retained prior controls");
    Check(imitation::Elapsed(false,0,200)==20,"first tick rate");
    Check(imitation::Elapsed(true,200,200)==0,"duplicate tick");
    Check(imitation::Elapsed(true,200,220)==20,"normal tick");
    Check(imitation::Discontinuous(imitation::Elapsed(true,200,100)),"clock reset");
    Check(imitation::Discontinuous(imitation::Elapsed(true,-2000000000,2000000000)),"clock overflow");
    o.previousButtons=16|32;Check(imitation::PreviousCategories(o)[2]==1,"opposing lean flags do not cancel");
    o.previousButtons=16;Check(imitation::PreviousCategories(o)[2]==0,"left lean");
    o.previousButtons=32;Check(imitation::PreviousCategories(o)[2]==2,"right lean");
    imitation::Action action;action.pitchDelta=1;action.yawDelta=2;
    auto v=imitation::CommandView(359,179,action,20);
    Check(std::abs(v[0])<1e-6f && v[1]==-179,"signed pitch/yaw wrap");
    v=imitation::CommandView(0,0,action,40);Check(v[0]==2 && v[1]==4,"camera time scaling");
    v=imitation::CommandView(0,0,action,0);Check(v[0]==0 && v[1]==0,"duplicate view advanced");
    v=imitation::CommandView(87.89f,0,action,20);Check(v[0]==87.890625f,"native pitch clamp parity");
    // ANGLE2SHORT(delta-subtracted command) followed by PM_UpdateViewAngles.
    for(float angle:{-87.f,-1.f,0.f,1.f,87.f})for(int delta:{-16000,0,16000}) {
        int encoded=(int(angle*65536.f/360.f)&65535)-delta;
        short resolved=static_cast<short>(encoded+delta);
        float actual=resolved*360.f/65536.f;
        Check(std::abs(actual-angle)<=360.f/65536.f,"native angle/command round trip");
    }
    std::cout<<"PASS: intended/executed feedback, reset, duplicate ticks, angle timing/wrap/clamp\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
