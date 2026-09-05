/*
 * Spawn-indexed replay bots. Part of OpenMoHAA.
 * Copyright (C) 2026 OpenMoHAA contributors
 * SPDX-License-Identifier: GPL-2.0-or-later
 * Distributed without warranty; see COPYING.txt.
 */
#pragma once
#include "../qcommon/q_shared.h"
class Player;

// Added in OPM: replay owns translation, the ordinary Player still owns combat.
void G_ReplayInit();
void G_ReplayShutdown();
void G_ReplaySpawned(Player *player);
void G_ReplayForget(Player *player);
bool G_ReplayBuildCommand(Player *player, usercmd_t *command, usereyes_t *eyes);
bool G_ReplayLocked(Player *player);
bool G_ReplayClientMove(Player *player, const usercmd_t *command);
void G_ReplayRestore(Player *player);
void G_ReplayValidate(Player *player);
