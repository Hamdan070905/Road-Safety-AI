"""Explainable constant-velocity footprint screening, with explicit nulls."""
import math


def analyze_pair(a, b, horizon_s=3.0, footprint_radius_m=1.2, clearance_threshold_m=2.0):
    if not a.get('reliable') or not b.get('reliable'):
        return None
    rx=b['x']-a['x']; ry=b['y']-a['y']; vx=b['vx']-a['vx']; vy=b['vy']-a['vy']; vv=vx*vx+vy*vy
    if vv < 1e-8:
        clearance=math.hypot(rx,ry)
        if clearance>clearance_threshold_m: return None
        return {'ttc_s':None,'ttc_unavailable_reason':'Near-zero relative velocity; time to collision cannot be estimated','minimum_predicted_clearance_m':clearance,'time_to_closest_approach_s':None,'pet_s':None,'pet_unavailable_reason':'Conflict-area entry and exit occupancy intervals are not established','footprint_assumption':'Circular road-user footprints; configurable radius, approximate and not class-specific','uncertainty':'Stationary or parallel pair; no reliable TTC estimate'}
    t=-(rx*vx+ry*vy)/vv
    if t < 0 or t > horizon_s: return None
    dx=rx+vx*t; dy=ry+vy*t; clearance=math.hypot(dx,dy)
    if clearance > clearance_threshold_m: return None
    # TTC is earliest predicted overlap of circular footprint assumptions, not CPA time.
    radius=2*footprint_radius_m
    c=rx*rx+ry*ry-radius*radius
    disc=(rx*vx+ry*vy)**2-vv*c
    ttc=None; reason='Predicted footprint overlap is not reached within horizon'
    if c <= 0: ttc=0.0; reason=None
    elif disc >= 0:
        entry=(-(rx*vx+ry*vy)-math.sqrt(disc))/vv
        if 0 <= entry <= horizon_s: ttc=entry; reason=None
    return {'ttc_s':ttc,'ttc_unavailable_reason':reason,'minimum_predicted_clearance_m':clearance,'time_to_closest_approach_s':t,'pet_s':None,'pet_unavailable_reason':'Conflict-area entry and exit occupancy intervals are not established','footprint_assumption':'Circular road-user footprints; configurable radius, approximate and not class-specific','uncertainty':'Constant velocity prediction; excludes turning, occlusion and identity switches'}


def pet_from_intervals(first_exit_s, next_entry_s):
    if first_exit_s is None or next_entry_s is None: return None
    if next_entry_s < first_exit_s: return None
    return next_entry_s-first_exit_s


def merge_events(events, window_s=2.0):
    merged=[]
    for event in sorted(events,key=lambda e:e['timestamp']):
        pair=tuple(sorted(event.get('participants',[])))
        start=event.get('start_timestamp',event['timestamp']); end=event.get('end_timestamp',event['timestamp'])
        prior=next((x for x in reversed(merged) if x.get('type')==event.get('type') and tuple(sorted(x.get('participants',[])))==pair and start-x.get('end_timestamp',x['timestamp'])<=window_s),None)
        if prior:
            prior['end_timestamp']=max(prior.get('end_timestamp',prior['timestamp']),end)
            prior['end_timestamp_s']=prior['end_timestamp']
            old_clear=prior.get('metrics',{}).get('minimum_predicted_clearance_m')
            new_clear=event.get('metrics',{}).get('minimum_predicted_clearance_m')
            if new_clear is not None and (old_clear is None or new_clear < old_clear):
                prior['peak_timestamp']=event.get('peak_timestamp',event['timestamp']); prior['timestamp']=event['timestamp']; prior['metrics']=event.get('metrics',prior.get('metrics',{}))
        else:
            e=dict(event); e['start_timestamp']=start; e.setdefault('peak_timestamp',event['timestamp']); e['end_timestamp']=end; e['start_timestamp_s']=start; e['end_timestamp_s']=end; merged.append(e)
    return merged
