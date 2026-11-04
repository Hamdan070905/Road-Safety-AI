"""Conservative, image-space collision candidate screening.

This is a review aid, not an accident classifier. A candidate needs observed
vehicle-box interaction, prior approach, and a movement discontinuity around
the interaction. No physical measurements are produced.
"""
from collections import defaultdict
import math

VEHICLE_CLASSES = {2, 3, 5, 7}
CLASS_NAMES = {2:'car',3:'motorcycle',5:'bus',7:'truck'}


def _iou(a, b):
    x1=max(a[0],b[0]); y1=max(a[1],b[1]); x2=min(a[2],b[2]); y2=min(a[3],b[3])
    inter=max(0.,x2-x1)*max(0.,y2-y1)
    union=max(0.,a[2]-a[0])*max(0.,a[3]-a[1])+max(0.,b[2]-b[0])*max(0.,b[3]-b[1])-inter
    return inter/union if union else 0.


def _speed_change(rows, t, side):
    before=[p for p in rows if t-.65 <= p['t'] < t-.08]
    after=[p for p in rows if t+.08 < p['t'] <= t+.65]
    if len(before)<3 or len(after)<3: return None
    def velocity(samples):
        a,b=samples[0],samples[-1]; dt=b['t']-a['t']
        return ((b['x']-a['x'])/dt,(b['y']-a['y'])/dt) if dt>0 else None
    v0=velocity(before); v1=velocity(after)
    if not v0 or not v1:return None
    s0=math.hypot(*v0); s1=math.hypot(*v1)
    # Image-space scale is unknown. Detect a clear change of heading or speed,
    # but never translate this into acceleration or speed units.
    angle=0.
    if s0>.015 and s1>.015:
        angle=math.degrees(math.acos(max(-1,min(1,(v0[0]*v1[0]+v0[1]*v1[1])/(s0*s1)))))
    ratio=abs(s1-s0)/max(s0,s1,.01)
    return {'before':s0,'after':s1,'heading_change_deg':angle,'relative_speed_change':ratio,
            'changed':angle>=25 or ratio>=.45}


def _join_identity_fragments(tracks, max_gap=.14, max_distance=.075):
    """Conservatively connect non-overlapping same-class ID fragments."""
    ids=sorted(tracks,key=lambda i:tracks[i][0]['t']); links={i:i for i in ids}
    candidates=[]
    for a in ids:
        left=tracks[a]
        for b in ids:
            right=tracks[b]
            if a==b or left[0].get('class')!=right[0].get('class'):continue
            gap=right[0]['t']-left[-1]['t']
            if not 0<=gap<=max_gap:continue
            if left[-1].get('scene_id',0)!=right[0].get('scene_id',0):continue
            # Refuse a handoff only if identities co-occur in the same sampled
            # frame. Adjacent frames across a brief occlusion are not overlap.
            if any(abs(p['t']-q['t'])<.005 for p in left for q in right):continue
            distance=math.hypot(left[-1]['x']-right[0]['x'],left[-1]['y']-right[0]['y'])
            if distance<=max_distance:candidates.append((gap+distance,a,b))
    used_in=set(); used_out=set()
    for _cost,a,b in sorted(candidates):
        if a in used_out or b in used_in:continue
        links[b]=links[a]; used_out.add(a);used_in.add(b)
    groups=defaultdict(list)
    for i in ids:groups[links[i]].extend(tracks[i])
    return {k:sorted(v,key=lambda p:p['t']) for k,v in groups.items()},links


def detect_collision_candidates(samples, *, moving_camera=False, diagnostics=None):
    """Return possible-collision events only when multiple temporal signals agree.

    Input points use normalized centers and normalized ``bbox`` coordinates.
    Moving-camera footage is deliberately withheld because global image motion
    invalidates these image-space movement-change comparisons.
    """
    diagnostics=diagnostics if diagnostics is not None else []
    if moving_camera:
        diagnostics.append({'engine':'image-space collision screen','decision':'withheld','reason':'Moving camera; image motion invalidates this screening geometry'})
        return []
    tracks=defaultdict(list)
    for p in samples:
        if p.get('class') in VEHICLE_CLASSES and p.get('bbox'):
            row=dict(p); row['x']=p.get('image_x_normalized',p['x']); row['y']=p.get('image_y_normalized',p['y'])
            tracks[int(p['id'])].append(row)
    for rows in tracks.values():rows.sort(key=lambda p:p['t'])
    actors,links=_join_identity_fragments(tracks)
    result=[]
    actor_ids=list(actors)
    for i,a_id in enumerate(actor_ids):
        for b_id in actor_ids[i+1:]:
            a,b=actors[a_id],actors[b_id]
            if a[0].get('class') not in VEHICLE_CLASSES or b[0].get('class') not in VEHICLE_CLASSES:continue
            overlaps=[]
            for pa in a:
                for pb in b:
                    if abs(pa['t']-pb['t'])<=.035 and _iou(pa['bbox'],pb['bbox'])>=.08:
                        overlaps.append((max(pa['t'],pb['t']),pa,pb,_iou(pa['bbox'],pb['bbox'])))
            pair_ids_a=sorted({int(p['id']) for p in a});pair_ids_b=sorted({int(p['id']) for p in b})
            if not overlaps:
                diagnostics.append({'pair_a':pair_ids_a,'pair_b':pair_ids_b,'decision':'rejected','reason':'No plausible simultaneous vehicle-box interaction observed'})
                continue
            # Group adjacent box interactions first. A collision may begin as
            # a close interaction and only show its strongest response later.
            clusters=[]
            for hit in overlaps:
                if not clusters or hit[0]-clusters[-1][-1][0]>.20:clusters.append([hit])
                else:clusters[-1].append(hit)
            for cluster in clusters:
                contact_start=cluster[0][0];contact_end=cluster[-1][0];pre=[]
                for qa in a:
                    if not contact_start-.8<=qa['t']<contact_start-.08:continue
                    qb=min((q for q in b if abs(q['t']-qa['t'])<=.035),key=lambda q:abs(q['t']-qa['t']),default=None)
                    if qb:pre.append((math.hypot(qa['x']-qb['x'],qa['y']-qb['y']),qa['t']))
                closing=len(pre)>=3 and pre[-1][0] < pre[0][0]-.012
                responses=[]
                for t,pa,pb,_iou_value in cluster:
                    change_a=_speed_change(a,t,'after');change_b=_speed_change(b,t,'after')
                    for change in (change_a,change_b):
                        # A single smooth braking response is common at signals.
                        # Require a deflection, or a synchronized hard stop by
                        # both involved tracks at a strong observed overlap.
                        synchronized_stop=(change_a and change_b and _iou_value>=.2 and
                            change_a['relative_speed_change']>=.75 and change_b['relative_speed_change']>=.75 and
                            change_a['after']<=change_a['before']*.25 and change_b['after']<=change_b['before']*.25)
                        if change and (change['heading_change_deg']>=25 or synchronized_stop):
                            responses.append((change['relative_speed_change']+change['heading_change_deg']/180,t,change))
                rejection=[]
                if not closing:rejection.append('No decreasing inter-vehicle image-space separation before interaction window')
                if not responses:rejection.append('No abrupt heading deflection or synchronized hard stop after interaction')
                # Choose the strongest observed movement response, while
                # retaining the full overlap interval as event context.
                support=len(cluster);max_iou=max(x[3] for x in cluster)
                if support<2 and max_iou<.2:rejection.append('Box interaction was too brief and weak to support a candidate')
                diagnostic={'pair_a':pair_ids_a,'pair_b':pair_ids_b,'interaction_interval_s':[round(contact_start,3),round(contact_end,3)],'overlap_observations':support,'max_box_iou':round(max_iou,3),'approach_observed':closing,'movement_response_candidates':responses,'decision':'candidate' if not rejection else 'rejected','rejection_reasons':rejection}
                diagnostics.append(diagnostic)
                if rejection:continue
                _score,peak,response=max(responses,key=lambda item:item[0])
                ids_a=sorted({int(p['id']) for p in a if contact_start-.8<=p['t']<=contact_end});ids_b=sorted({int(p['id']) for p in b if contact_start-.8<=p['t']<=contact_end})
                participants=sorted(set(ids_a+ids_b))
                class_by_id={int(p['id']):int(p['class']) for p in a+b if int(p['id']) in participants}
                if not participants:continue
                eid=f"collision-{participants[0]}-{participants[-1]}-{round(peak*1000)}"
                result.append({'id':eid,'type':'possible_collision','timestamp':round(peak,3),
                    'start_timestamp':round(contact_start,3),'peak_timestamp':round(peak,3),'end_timestamp':round(contact_end,3),
                    'participants':participants,'participant_classes':[class_by_id[x] for x in participants],
                    'participant_class_names':[CLASS_NAMES.get(class_by_id[x],'vehicle') for x in participants],
                    'reason':'Possible collision — review required. Observed vehicle-box interaction followed approach and an abrupt image-space movement change.',
                    'signals':{'observed_box_overlap_iou':round(max_iou,3),'box_overlap_at_peak_iou':round(min(cluster,key=lambda x:abs(x[0]-peak))[3],3),'overlap_observations_near_contact':support,
                        'approach_observed':True,'approach_window_s':[round(pre[0][1],3),round(pre[-1][1],3)],'post_interaction_movement_change':response,
                        'identity_fragments':{'participant_a':ids_a,'participant_b':ids_b}},
                    'measurement_mode':'uncalibrated image-space collision screening',
                    'metrics':{'ttc_s':None,'ttc_unavailable_reason':'Collision candidate is based on observed interaction; physical timing/geometry unavailable','pet_s':None,'pet_unavailable_reason':'PET requires calibrated shared-area occupancy','minimum_predicted_clearance_m':None},
                    'measurement_quality':{'uncertainty':'Image-space temporal screening only; overlap, identity fragmentation, and camera perspective require human review','reliable_identities':len(ids_a)==1 and len(ids_b)==1},
                    'review':'pending','configuration_version':'collision-screen-v1'})
    # Consolidate multiple contact frames for the same participant identities.
    merged=[]
    for e in sorted(result,key=lambda x:x['timestamp']):
        prior=next((x for x in reversed(merged) if x['participants']==e['participants'] and e['start_timestamp']<=x['end_timestamp']+.5),None)
        if prior:
            prior['end_timestamp']=max(prior['end_timestamp'],e['end_timestamp'])
            if e['signals']['observed_box_overlap_iou']>prior['signals']['observed_box_overlap_iou']:
                prior['timestamp']=prior['peak_timestamp']=e['timestamp'];prior['signals']=e['signals']
        else:merged.append(e)
    return merged
