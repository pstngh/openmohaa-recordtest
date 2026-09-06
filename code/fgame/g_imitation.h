/* Copyright (C) 2026 OpenMoHAA contributors
 * SPDX-License-Identifier: GPL-2.0-or-later. No warranty; see COPYING.txt. */
#pragma once
#include "g_local.h"
class Player;
void G_ImitationInit();
void G_ImitationShutdown();
void G_ImitationForget(Player *player);
bool G_ImitationBuildCommand(Player *player, usercmd_t *command, usereyes_t *eyes);
