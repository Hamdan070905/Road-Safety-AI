"""Per-identity timestamped track filtering and quality gates."""
import math
from collections import defaultdict


def process_tracks(samples, window=1.0, max_gap=0.75, max_speed=45.0, min_age=0.35):
    grouped = defaultdict(list)
    for p in samples:
        grouped[p['id']].append(p)
    output = []
    for tid, rows in grouped.items():
        rows.sort(key=lambda x: x['t'])
        accepted = []
        for p in rows:
            if accepted:
                prev = accepted[-1]; dt = p['t']-prev['t']
                if dt <= 0 or dt > max_gap:
                    continue
                if math.hypot(p['x']-prev['x'], p['y']-prev['y'])/dt > max_speed:
                    continue
            accepted.append(p)
        if not accepted:
            continue
        last = accepted[-1]; recent = [p for p in accepted if last['t']-p['t'] <= window]
        age = last['t']-accepted[0]['t']
        gaps = sum(1 for a,b in zip(accepted, accepted[1:]) if b['t']-a['t'] > max_gap*.6)
        reliable = age >= min_age and len(recent) >= 3 and recent[-1]['t']-recent[0]['t'] >= .2
        if reliable:
            # least-squares slope on actual timestamps, independently per axis
            ts=[p['t'] for p in recent]; mean=sum(ts)/len(ts); den=sum((t-mean)**2 for t in ts)
            vx=sum((t-mean)*p['x'] for t,p in zip(ts,recent))/den
            vy=sum((t-mean)*p['y'] for t,p in zip(ts,recent))/den
        else: vx=vy=None
        output.append({'id':tid,'t':last['t'],'x':last['x'],'y':last['y'],'vx':vx,'vy':vy,'age_s':age,'gaps':gaps,'reliable':reliable,'coordinate_system':last.get('coordinate_system','normalized_image'),'units':last.get('units','normalized')})
    return output
