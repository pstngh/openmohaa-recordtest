/*
 * Native-control mohdm6 imitation. Copyright (C) 2026 OpenMoHAA contributors.
 * SPDX-License-Identifier: GPL-2.0-or-later. No warranty; see COPYING.txt.
 */
#include "g_imitation.h"
#include "imitation_policy.h"
#include "imitation_runtime.h"
#include "g_main.h"
#include "player.h"
#include "weapon.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <map>
#include <memory>
#include <random>
#include <sstream>

namespace {
cvar_t *enabled, *modelPath, *debug, *actions, *reactionMsec, *seed, *sampling;
struct ImitationState {
    std::mt19937 generator;
    std::uint32_t sequence = 0;
    SafePtr<Player> owner;
    imitation::Hidden memory{};
    imitation::ControlFeedback feedback;
    usercmd_t cachedCommand{};
    usereyes_t cachedEyes{};
    bool hasTick = false;
    Vector lastView;
    int lastTime = 0, lastLog = 0, noticed = 0;
    SafePtr<Player> visibleEnemy;
    SafePtr<Weapon> lastWeapon;
    bool primaryDown = false;
};
struct ImitationWorld {
    std::uint32_t spawnCounter = 0;
    imitation::Policy policy;
    std::map<Player*,ImitationState> states;
    bool attempted = false, available = false;
    std::uint32_t modelId = 0;
};
std::unique_ptr<ImitationWorld> data;

bool Enemy(Player *p, Player *other) {
    return other && other!=p && !other->IsDead() && !other->IsSpectator() && !other->hidden()
        && !(other->flags&FL_NOTARGET) && other->getSolidType()!=SOLID_NOT
        && (g_gametype->integer==GT_FFA || other->GetTeam()!=p->GetTeam());
}
struct Target { Player *player = nullptr; Vector relative; float distance = 0; };
Target Observe(Player *p, const Vector& angles, float degrees) {
    Target found; Vector forward,right;
    angles.AngleVectors(&forward,&right);
    const Vector eye=p->EyePosition();
    float bestDot=std::cos(degrees*3.14159265358979323846f/180.f);
    for(int i=1;i<=SentientList.NumObjects();++i) {
        Sentient *candidate=SentientList.ObjectAt(i);
        if(!candidate->IsSubclassOfPlayer())continue;
        Player *other=static_cast<Player*>(candidate);
        if(!Enemy(p,other))continue;
        Vector delta=other->centroid-eye;
        const float distance=delta.length();
        if(distance<1 || distance>4096)continue;
        const float dot=DotProduct(delta,forward)/distance;
        if(dot<bestDot)continue;
        const trace_t sight=G_Trace(eye,vec_zero,vec_zero,other->centroid,p,MASK_OPAQUE,false,"Imitation observation");
        if(sight.fraction<1 && sight.entityNum!=other->entnum)continue;
        bestDot=dot;found.player=other;found.distance=distance;
        // Telemetry uses view-forward/right projections and WORLD vertical delta.
        found.relative=Vector(DotProduct(delta,forward),DotProduct(delta,right),delta[2]);
    }
    return found;
}
bool CanShoot(Player *p, Weapon *weapon, const Vector& view, const Target& target, bool reacted) {
    imitation::FireGate gate;
    gate.visible=Enemy(p,target.player);gate.reacted=reacted;gate.aligned=target.player!=nullptr;
    if(!gate.visible || !weapon)return false;
    Vector forward;view.AngleVectors(&forward);
    const Vector eye=p->EyePosition();
    const trace_t line=G_Trace(eye,vec_zero,vec_zero,eye+forward*target.distance,p,MASK_SHOT,false,"Imitation firing line");
    gate.clear=line.fraction==1 || line.entityNum==target.player->entnum;
    Vector muzzle;weapon->GetMuzzlePosition(muzzle);
    const trace_t barrel=G_Trace(muzzle,vec_zero,vec_zero,target.player->centroid,p,MASK_SHOT,false,"Imitation muzzle line");
    gate.muzzle=weapon->MuzzleClear() && !barrel.startsolid && !barrel.allsolid
        && (barrel.fraction==1 || barrel.entityNum==target.player->entnum);
    return gate.Allows();
}
bool Load() {
    if(data->attempted)return data->available;
    data->attempted=true;
    const auto checksum=static_cast<std::uint32_t>(std::strtoll(gi.Cvar_Get("sv_mapChecksum","0",0)->string,nullptr,10));
    if(level.mapname!="dm/mohdm6" || checksum!=imitation::mapChecksum || g_protocol!=8
        || (g_gametype->integer!=GT_FFA && g_gametype->integer!=GT_TEAM)
        || gi.Cvar_Get("sv_fps","20",0)->integer!=50) {
        gi.Printf("Imitation unavailable: requires dm/mohdm6 checksum 1974169620, protocol 8, FFA/TDM and sv_fps 50. Native bots remain active.\n");
        return false;
    }
    const long length=gi.FS_ReadFile(modelPath->string,nullptr,qtrue);
    if(length<=0 || length>2*1024*1024) {
        gi.Printf("Imitation unavailable: missing/oversized model %s. Native bots remain active.\n",modelPath->string);return false;
    }
    void *bytes=nullptr;
    const long read=gi.FS_ReadFile(modelPath->string,&bytes,qtrue);
    std::string error;
    const bool valid=read==length && data->policy.Load(bytes,length,error);
    if(valid) {
        // Diagnostic fingerprint only, not a security/integrity checksum.
        data->modelId=2166136261u;
        const auto *raw=static_cast<const unsigned char*>(bytes);
        for(long i=0;i<read;++i)data->modelId=(data->modelId^raw[i])*16777619u;
    }
    if(bytes)gi.FS_FreeFile(bytes);
    if(!valid) {
        gi.Printf("Imitation model rejected: %s. Native bots remain active.\n",error.c_str());return false;
    }
    gi.Printf("Imitation: loaded mohdm6 GRU policy %s; native physics, no replay or tactical planner. Experimental, not gameplay-validated.\n",modelPath->string);
    gi.Printf("Imitation runtime feedback-v2: model_id=%08x decoder=%s\n",data->modelId,sampling->integer?"sampled":"MAP");
    data->available=true;return true;
}
} // namespace

void G_ImitationInit() {
    data.reset(new ImitationWorld);
    enabled=gi.Cvar_Get("g_imitation_bots","0",0);
    modelPath=gi.Cvar_Get("g_imitation_model","bots/imitation/mohdm6.omim",0);
    seed=gi.Cvar_Get("g_imitation_seed","1",0);
    sampling=gi.Cvar_Get("g_imitation_sampling","1",0);
    actions=gi.Cvar_Get("g_imitation_actions","1",0);
    debug=gi.Cvar_Get("g_imitation_debug","0",0);
    // The learned policy supplies response timing; extra minimum delay is optional.
    reactionMsec=gi.Cvar_Get("g_imitation_reaction_ms","0",0);
    gi.Cvar_CheckRange(reactionMsec,0,2000,qtrue);
}
void G_ImitationShutdown() {data.reset();}
void G_ImitationForget(Player *p) {if(data)data->states.erase(p);}

bool G_ImitationBuildCommand(Player *p,usercmd_t *command,usereyes_t *eyes) {
    if(!data || !enabled || !p || !p->client)return false;
    if(!enabled->integer) {G_ImitationForget(p);return false;}
    if(p->IsDead() || p->IsSpectator() || p->GetTeam()<TEAM_FREEFORALL || level.intermissiontime) {
        G_ImitationForget(p);return false; // Native lifecycle owns real death/team/respawn.
    }
    if(!p->client->pers.dm_primary[0])return false;
    if(!Load())return false;
    auto& state=data->states[p];
    if(state.owner.Pointer()!=p) {state=ImitationState{};state.owner=p;state.lastView=p->GetViewAngles();
        state.sequence=++data->spawnCounter;
        state.generator.seed(static_cast<std::uint32_t>(seed->integer) ^ (state.sequence*2654435761u) ^ p->entnum);}
    *command={};*eyes={};command->serverTime=level.svsTime;
    Vector current=p->GetViewAngles();
    const std::int64_t elapsed=imitation::Elapsed(state.hasTick,state.lastTime,level.inttime);
    if(state.hasTick && elapsed==0) {
        *command=state.cachedCommand;*eyes=state.cachedEyes;return true;
    }
    const bool reset=imitation::Discontinuous(elapsed);
    const int dt=reset?int(imitation::tickMsec):static_cast<int>(elapsed);
    if(reset) {state.memory={};state.feedback.Reset();state.primaryDown=false;state.lastView=current;}
    // Attachments/script freezes are not training states; release controls rather than teleporting.
    if(p->HasVehicle() || p->GetTurret() || p->GetLadder() || p->m_bFrozen || level.playerfrozen) {
        state.memory={};state.feedback.Reset();state.primaryDown=false;state.lastView=current;state.lastTime=level.inttime;
        for(int i=0;i<3;++i)command->angles[i]=ANGLE2SHORT(current[i])-p->client->ps.delta_angles[i];
        eyes->ofs[2]=p->viewheight;eyes->angles[0]=current[0];eyes->angles[1]=current[1];
        state.hasTick=true;state.cachedCommand=*command;state.cachedEyes=*eyes;return true;
    }
    try {
        imitation::Observation observation;
        for(int i=0;i<3;++i) {observation.position[i]=p->origin[i];observation.velocity[i]=p->velocity[i];}
        observation.pitch=current[0];observation.yaw=current[1];
        observation.lean=p->client->ps.fLeanAngle;observation.viewHeight=p->viewheight;
        observation.ducked=(p->client->ps.pm_flags&PMF_DUCKED)!=0;
        observation.prone=(p->client->ps.pm_flags&PMF_VIEW_PRONE)!=0;
        observation.walking=p->client->ps.walking;observation.health=p->health;
        Weapon *weapon=p->GetActiveWeapon(WEAPON_MAIN);
        if(weapon) {
            observation.weaponClass=imitation::WeaponClass(weapon->getName().c_str());
            observation.clipAmmo=weapon->ClipAmmo(FIRE_PRIMARY);
            observation.reserveAmmo=weapon->AmmoAvailable(FIRE_PRIMARY);
        }
        observation.previousMove={state.feedback.intent.forward,state.feedback.intent.right,state.feedback.intent.up};
        observation.previousButtons=state.feedback.intent.buttons;
        if(!reset && dt>0 && dt<=100 && state.hasTick)for(int i=0;i<2;++i)
            observation.previousViewDelta[i]=imitation::AngleDelta(state.lastView[i],current[i])*20.f/dt;
        const Target observed=Observe(p,current,12);
        observation.target=observed.player!=nullptr;observation.targetDistance=observed.distance;
        if(observed.player)for(int i=0;i<3;++i) {
            observation.targetRelative[i]=observed.relative[i];observation.targetVelocity[i]=observed.player->velocity[i];
        }
        const auto encoded=imitation::Encode(observation);
        const auto logits=data->policy.Step(encoded,state.memory);
        const auto previous=imitation::PreviousCategories(observation);
        std::array<float,5> uniform{};
        for(float& u:uniform)u=float(state.generator()>>8)*(1.f/16777216.f);
        const auto predicted=sampling->integer ? data->policy.Choose(logits,previous,uniform) : imitation::Decode(logits);
        const auto view=imitation::CommandView(current[0],current[1],predicted,dt);
        Vector desired(view[0],view[1],view[2]);
        command->forwardmove=predicted.forward;command->rightmove=predicted.right;command->upmove=predicted.up;
        command->buttons=predicted.buttons;
        // No routine points the camera toward an enemy: only the policy changes desired.
        const Target visible=Observe(p,current,60);
        if(state.visibleEnemy.Pointer()!=visible.player) {state.visibleEnemy=visible.player;state.noticed=level.inttime;}
        const Target aligned=Observe(p,desired,8);
        const bool reacted=aligned.player && (reactionMsec->integer==0 ||
            (state.visibleEnemy.Pointer()==aligned.player && level.inttime-state.noticed>=reactionMsec->integer));
        const bool fire=actions->integer && CanShoot(p,weapon,desired,aligned,reacted);
        if(state.lastWeapon.Pointer()!=weapon) {state.lastWeapon=weapon;state.primaryDown=false;}
        bool primary=(command->buttons&BUTTON_ATTACKLEFT) && fire;
        if(!weapon || p->GetNewActiveWeapon()) primary=false;
        else if(weapon->IsSemiAuto()) primary=primary && !state.primaryDown && weapon->GetState()==WEAPON_READY
            && weapon->ReadyToFire(FIRE_PRIMARY,qfalse);
        state.primaryDown=primary;
        command->buttons&=~BUTTON_ATTACKLEFT;
        if(primary)command->buttons|=BUTTON_ATTACKLEFT;
        // Secondary labels can operate the live sniper scope, not another weapon's melee/launcher.
        if(!actions->integer || !weapon || !weapon->GetZoom() || p->GetNewActiveWeapon())command->buttons&=~BUTTON_ATTACKRIGHT;
        // Native ammo-based reload is a documented safeguard, not learned reload timing.
        if(actions->integer && weapon && !p->GetNewActiveWeapon() && weapon->GetState()==WEAPON_READY
            && !weapon->HasAmmoInClip(FIRE_PRIMARY) && weapon->CheckReload(FIRE_PRIMARY))p->PlayerReload(nullptr);
        if(!actions->integer)command->buttons&=~BUTTON_USE;
        for(int i=0;i<3;++i)command->angles[i]=ANGLE2SHORT(desired[i])-p->client->ps.delta_angles[i];
        eyes->ofs[2]=static_cast<signed char>(std::clamp(p->viewheight,-127,127));
        eyes->angles[0]=desired[0];eyes->angles[1]=desired[1];
        // The model's history is its requested control, not the guard's denied
        // fire request. The actually sent command is cached/logged separately.
        state.feedback.Record(predicted,command->buttons);
        state.lastView=current;state.lastTime=level.inttime;state.hasTick=true;
        state.cachedCommand=*command;state.cachedEyes=*eyes;
        if(debug->integer && level.inttime-state.lastLog>=1000) {
            gi.Printf("imitation_control bot=%d pos=%.2f,%.2f,%.2f cmd=%d,%d,%d look_delta=%.3f,%.3f mode=%d mode_weight=%.3f pitch=%.2f yaw=%.2f visible=%d aligned=%d requested_fire=%d permitted=%d sent_fire=%d\n",
                p->entnum,p->origin[0],p->origin[1],p->origin[2],command->forwardmove,command->rightmove,command->upmove,
                predicted.pitchDelta,predicted.yawDelta,predicted.component,predicted.mixtureWeight,current[0],current[1],
                visible.player!=nullptr,aligned.player!=nullptr,(predicted.buttons&BUTTON_ATTACKLEFT)!=0,fire,primary);
            state.lastLog=level.inttime;
        }
        if(debug->integer>=2) {
            std::ostringstream line;
            line.precision(9);
            line << "imitation_frame bot=" << p->entnum << " time=" << level.inttime
                 << " sequence=" << state.sequence << " model_id=" << data->modelId << " dt=" << dt << " reset=" << reset
                 << " pitch=" << current[0] << " yaw=" << current[1]
                 << " requested=" << predicted.forward << ',' << predicted.right << ',' << predicted.up << ',' << predicted.buttons
                 << " sent_buttons=" << command->buttons << " view_delta=" << predicted.pitchDelta << ',' << predicted.yawDelta
                 << " visible=" << (visible.player!=nullptr) << " aligned=" << (aligned.player!=nullptr)
                 << " permitted=" << fire << " weapon_class=" << observation.weaponClass << " obs=";
            for(std::size_t i=0;i<encoded.size();++i) {if(i)line << ',';line << encoded[i];}
            line << '\n';gi.Printf("%s",line.str().c_str());
        }
        return true;
    }catch(const std::exception& e) {
        gi.Printf("Imitation inference failed for bot %d: %s; returning to native controller.\n",p->entnum,e.what());
        G_ImitationForget(p);data->available=false;return false;
    }
}
