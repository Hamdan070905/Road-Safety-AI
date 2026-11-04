"""Configurable movement-risk observations using shared tracked points."""
import math
from collections import defaultdict

VEHICLES={2,3,5,7}


def _inside(point, polygon):
    x,y=point; inside=False
    for i,(x1,y1) in enumerate(polygon):
        x2,y2=polygon[(i+1)%len(polygon)]
        if (y1>y)!=(y2>y) and x < (x2-x1)*(y-y1)/(y2-y1+1e-15)+x1: inside=not inside
    return inside


def _continuous(rows, max_gap=.75):
    return all(b['t']-a['t']<=max_gap for a,b in zip(rows,rows[1:]))


def _event(kind, rows, reason, config_version, metrics=None):
    return {'id':f'{kind}-{rows[0]["id"]}-{round(rows[0]["t"]*1000)}','type':kind,
        'timestamp':round(rows[-1]['t'],3),'start_timestamp':round(rows[0]['t'],3),
        'peak_timestamp':round(rows[-1]['t'],3),'end_timestamp':round(rows[-1]['t'],3),
        'participants':[int(rows[0]['id'])],'participant_classes':[int(rows[0].get('class',2))],
        'participant_class_names':['person' if rows[0].get('class')==0 else 'vehicle'],
        'reason':reason,'metrics':metrics or {},'measurement_mode':'image-space observation',
        'measurement_quality':{'uncertainty':'Track-based candidate; zone calibration, identity continuity, and scene context require review','reliable_identities':True},
        'review':'pending','configuration_version':config_version}


def detect_movement_candidates(samples, config, *, calibrated=False, moving_camera=False):
    """Create candidates only for explicitly configured modules and zones.

    Image zones and direction vectors are normalized source-image coordinates.
    Speed/deceleration use only metric road-plane samples from valid calibration.
    """
    if moving_camera:return []
    tracks=defaultdict(list)
    for p in samples:
        if p.get('class') in VEHICLES:tracks[int(p['id'])].append(p)
    for rows in tracks.values():rows.sort(key=lambda p:p['t'])
    events=[]; version=config.get('version','movement-config-v1')
    wrong=config.get('wrong_way') or {}
    if wrong.get('enabled'):
        dirs=wrong.get('permitted_directions',[]); turn_zones=wrong.get('permitted_turn_zones',[]); min_s=float(wrong.get('min_duration_s',1.5))
        for rows in tracks.values():
            for start in range(len(rows)):
                end=next((i for i in range(start+2,len(rows)) if rows[i]['t']-rows[start]['t']>=min_s),None)
                if end is None:continue
                window=rows[start:end+1]; dt=window[-1]['t']-window[0]['t']; dx=window[-1]['x']-window[0]['x'];dy=window[-1]['y']-window[0]['y'];norm=math.hypot(dx,dy)
                if not _continuous(window):continue
                if norm<.025:continue
                unit=(dx/norm,dy/norm)
                permitted=[(float(d[0]),float(d[1])) for d in dirs if len(d)==2 and math.hypot(float(d[0]),float(d[1]))>0]
                if not permitted:continue
                dots=[unit[0]*x+unit[1]*y for x,y in permitted]
                in_turn=any(_inside((p['x'],p['y']),poly) for p in window for poly in turn_zones if len(poly)>=3)
                if not in_turn and max(dots)<-.75:
                    events.append(_event('wrong_way_candidate',window,'Sustained movement opposite all configured permitted directions; permitted-turn zones were excluded.',version,{'observed_duration_s':round(dt,2),'image_displacement_normalized':round(norm,3),'physical_speed_available':False}))
                    break
    stopped=config.get('stopped_vehicle') or {}
    if stopped.get('enabled') and len(stopped.get('zone',[]))>=3:
        zone=stopped['zone']; dwell=float(stopped.get('dwell_s',10)); queue_suppress=int(stopped.get('queue_suppress_count',3)); queue_zones=stopped.get('queue_exclusion_zones',[]); occupants=[]
        for tid,rows in tracks.items():
            inside=[p for p in rows if _inside((p['x'],p['y']),zone)]
            in_queue=any(_inside((p['x'],p['y']),poly) for p in inside for poly in queue_zones if len(poly)>=3)
            if not in_queue and len(inside)>=3 and _continuous(inside) and inside[-1]['t']-inside[0]['t']>=dwell:
                speeds=[math.hypot(b['x']-a['x'],b['y']-a['y'])/(b['t']-a['t']) for a,b in zip(inside,inside[1:]) if b['t']>a['t']]
                if speeds and sum(speeds)/len(speeds)<float(stopped.get('max_dwell_speed',.008)):occupants.append(inside)
        # Clustered stopped traffic is treated as queue context and suppressed.
        if len(occupants)<queue_suppress:
            for rows in occupants:events.append(_event('stopped_vehicle_candidate',rows,'Vehicle remained nearly stationary in the configured zone; signal queues and other normal stops still need operator context.',version,{'observed_dwell_s':round(rows[-1]['t']-rows[0]['t'],2),'queue_context_count':len(occupants)}))
    crossing=config.get('blocked_crossing') or {}
    if crossing.get('enabled') and len(crossing.get('zone',[]))>=3:
        dwell=float(crossing.get('dwell_s',2.0)); zone=crossing['zone']
        for rows in tracks.values():
            occupancy=[p for p in rows if _inside((p['x'],p['y']),zone)]
            if len(occupancy)>=3 and _continuous(occupancy) and occupancy[-1]['t']-occupancy[0]['t']>=dwell:
                events.append(_event('blocked_crossing_candidate',occupancy,'Vehicle occupied the configured pedestrian-crossing zone beyond its dwell threshold.',version,{'observed_dwell_s':round(occupancy[-1]['t']-occupancy[0]['t'],2)}))
    speed=config.get('speeding') or {}
    decel=config.get('sudden_deceleration') or {}
    if calibrated:
        for rows in tracks.values():
            metric=[p for p in rows if p.get('coordinate_system')=='road_plane' and p.get('units')=='m']
            if len(metric)<3:continue
            for a,b in zip(metric,metric[1:]):
                dt=b['t']-a['t']; distance=math.hypot(b['x']-a['x'],b['y']-a['y'])
                if dt<=0 or dt>.5 or distance/dt>45:continue
                velocity=distance/dt
                if speed.get('enabled') and speed.get('speed_limit_mps') and velocity>float(speed['speed_limit_mps']):
                    events.append(_event('speeding_candidate',[a,b],'Calibrated track speed exceeded the configured speed limit; measurement uncertainty requires review.',version,{'speed_mps':round(velocity,2),'speed_limit_mps':float(speed['speed_limit_mps']),'units':'m/s'}))
            if decel.get('enabled'):
                max_decel=float(decel.get('threshold_mps2',4.0)); velocities=[]
                for a,b in zip(metric,metric[1:]):
                    dt=b['t']-a['t'];distance=math.hypot(b['x']-a['x'],b['y']-a['y'])
                    if 0<dt<=.5 and distance/dt<=45:velocities.append((b,(distance/dt)))
                for (a,v0),(b,v1) in zip(velocities,velocities[1:]):
                    dt=b['t']-a['t']
                    if dt>0 and (v0-v1)/dt>=max_decel:
                        events.append(_event('sudden_deceleration_candidate',[a,b],'Calibrated track showed a sudden speed reduction above the configured threshold.',version,{'estimated_deceleration_mps2':round((v0-v1)/dt,2),'units':'m/s^2'}))
    return events
