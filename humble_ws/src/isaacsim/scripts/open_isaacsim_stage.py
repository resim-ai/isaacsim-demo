# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script to open a USD stage in Isaac Sim (in standard gui mode). This script along with its arguments are automatically passed to Isaac Sim via the ROS2 launch workflow"""

import carb
import argparse
import omni.usd
import asyncio
import omni.client
import omni.kit.async_engine
import omni.timeline
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True, help='The path to USD stage')
    
    # Add the --start-on-play option
    # If --start-on-play is specified, it sets the value to True
    parser.add_argument('--start-on-play', action='store_true',
                        help='If present, to true.')
    
    try:
        options = parser.parse_args()
    except Exception as e:
        carb.log_error(str(e))
        return
    
    print("opening stage")

    omni.kit.async_engine.run_coroutine(open_stage_async(options.path, options.start_on_play))

async def monitor_sim_time_rate():
    """Monitor and print the rate between realtime and sim time every 5 seconds."""
    timeline_interface = omni.timeline.get_timeline_interface()
    
    # Store initial values
    initial_sim_time: float = None
    initial_real_time: float = None

    last_sim_time: float = 0.0
    sim_time_offset = 0.0

    print("Monitoring sim time rate...")
    
    while True:
        await omni.kit.app.get_app().next_update_async()
        # Get current sim time and real time
        current_sim_time: float = timeline_interface.get_current_time()
        current_real_time: float = time.time()
        
        # Initialize on first iteration
        if initial_sim_time is None:
            initial_sim_time = current_sim_time
            initial_real_time = current_real_time
            continue
        
        if current_sim_time < last_sim_time:
            sim_time_offset = last_sim_time - current_sim_time
            print(f"Sim time is going backwards. Setting sim time offset to {sim_time_offset} seconds.")
        
        # Calculate deltas
        sim_time_delta = (current_sim_time + sim_time_offset) - initial_sim_time
        real_time_delta = current_real_time - initial_real_time
        
        # Calculate and print the average rate
        if real_time_delta > 0:
            rate = sim_time_delta / real_time_delta
            print(f"Sim time rate: {rate:.4f}x, sim time: {(current_sim_time + sim_time_offset):.2f}s")
        
        last_sim_time = current_sim_time
        await asyncio.sleep(1.0)


async def open_stage_async(path: str, start_on_play: bool):
    timeline_interface = None
    if start_on_play:
        timeline_interface = omni.timeline.get_timeline_interface()
    
    async def _open_stage_internal(path):
        is_stage_with_session = False
        try:
            import omni.kit.usd.layers as layers
            live_session_name = layers.get_live_session_name_from_shared_link(path)
            is_stage_with_session = live_session_name is not None
        except Exception:
            pass

        if is_stage_with_session:
            # Try to open the stage with specified live session.
            (success, error) = await layers.get_live_syncing().open_stage_with_live_session_async(path)
        else:
            # Otherwise, use normal stage open.
            (success, error) = await omni.usd.get_context().open_stage_async(path)
        
        if not success:
            carb.log_error(f"Failed to open stage {path}: {error}.")
        else:
            if timeline_interface is not None:
                await omni.kit.app.get_app().next_update_async()
                timeline_interface.play()
                await omni.kit.app.get_app().next_update_async()
                print("Stage loaded and simulation is playing.")
            pass
    result, _ = await omni.client.stat_async(path)
    if result == omni.client.Result.OK:
        await _open_stage_internal(path)
        omni.kit.async_engine.run_coroutine(monitor_sim_time_rate())
        return

    broken_url = omni.client.break_url(path)
    if broken_url.scheme == 'omniverse':
        # Attempt to connect to nucleus server before opening stage
        try:
            from omni.kit.widget.nucleus_connector import get_nucleus_connector
            nucleus_connector = get_nucleus_connector()
        except Exception:
            carb.log_warn("Open stage: Could not import Nucleus connector.")
            return

        server_url = omni.client.make_url(scheme='omniverse', host=broken_url.host)
        nucleus_connector.connect(
            broken_url.host, server_url,
            on_success_fn=lambda *_: asyncio.ensure_future(_open_stage_internal(path)),
            on_failed_fn=lambda *_: carb.log_error(f"Open stage: Failed to connect to server '{server_url}'.")
        )
    else:
        carb.log_warn(f"Open stage: Could not open non-existent url '{path}'.")


main()
