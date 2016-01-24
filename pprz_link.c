/*
 * Copyright (C) 2015 Volker Strobel
 *
 * This file is part of paparazzi.
 *
 * paparazzi is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2, or (at your option)
 * any later version.
 *
 * paparazzi is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with paparazzi; see the file COPYING.  If not, write to
 * the Free Software Foundation, 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

/**
 * @file gps_trexton.c
 * @brief GPS system based on textons (computer vision)
 *
 * This GPS parses the treXton input and sets the GPS structure to the
 * values received.
 */


#include "subsystems/gps.h"
#include "subsystems/abi.h"


bool_t gps_available;   ///< Is set to TRUE when a new REMOTE_GPS packet is received and parsed

/** GPS initialization */
void gps_impl_init(void) {
  gps.fix = GPS_FIX_NONE;
  gps_available = FALSE;
  gps.gspeed = 700; // To enable course setting
  gps.cacc = 0; // To enable course setting
}


/** Parse the REMOTE_GPS datalink packet */
void parse_gps_trexton()
{
  gps.fix = GPS_FIX_3D;
  gps_available = TRUE;

  // publish new GPS data
  uint32_t now_ts = get_sys_time_usec();
  gps.last_msg_ticks = sys_time.nb_sec_rem;
  gps.last_msg_time = sys_time.nb_sec;
  if (gps.fix == GPS_FIX_3D) {
    gps.last_3dfix_ticks = sys_time.nb_sec_rem;
    gps.last_3dfix_time = sys_time.nb_sec;
  }
  AbiSendMsgGPS(GPS_DATALINK_ID, now_ts, &gps);
}
