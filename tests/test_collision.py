from backend.collision import detect_collision_candidates, _join_identity_fragments


def p(t, ident, x, y, cls=2, w=.11, h=.08):
    return {'t':t,'id':ident,'class':cls,'x':x,'y':y,'bbox':[x-w/2,y-h/2,x+w/2,y+h/2]}


def crossing_tracks(split=False):
    rows=[]
    for i in range(31):
        t=i/10
        # Two vehicles close in, then both trajectories change after contact.
        ax=.25+.125*t if t<=2 else .50+.04*(t-2)
        ay=.50 if t<=2 else .50+.11*(t-2)
        bx=.75-.125*t if t<=2 else .50-.04*(t-2)
        by=.50 if t<=2 else .50-.10*(t-2)
        rows.append(p(t,1,ax,ay))
        bid=2 if not split or t<1.9 else 3
        rows.append(p(t,bid,bx,by))
    return rows


def test_possible_collision_requires_approach_overlap_and_post_response():
    events=detect_collision_candidates(crossing_tracks())
    assert len(events)==1
    assert events[0]['type']=='possible_collision'
    assert events[0]['start_timestamp'] < events[0]['timestamp'] < events[0]['end_timestamp']
    assert events[0]['reason'].startswith('Possible collision — review required')
    assert events[0]['metrics']['ttc_s'] is None


def test_id_switch_is_conservatively_joined_across_brief_fragment_gap():
    rows=[x for x in crossing_tracks(split=True) if not (x['id']==2 and x['t']>=1.9)]
    # Reintroduce the later actor as a new tracker identity: the pre-contact ID
    # is absent after t=1.8, so overlap evidence can only use the new ID.
    assert len(detect_collision_candidates(rows))==1
    assert {2,3}.issubset(detect_collision_candidates(rows)[0]['participants'])


def test_parallel_perspective_overlap_alone_is_not_collision():
    rows=[]
    for i in range(31):
        t=i/10; x=.2+.01*t
        rows += [p(t,1,x,.50),p(t,2,x+.035,.50)]
    assert detect_collision_candidates(rows)==[]


def test_ordinary_queue_overlap_is_not_collision():
    rows=[]
    for i in range(31):
        t=i/10; rows += [p(t,1,.50,.5),p(t,2,.54,.5)]
    diagnostics=[]
    assert detect_collision_candidates(rows,diagnostics=diagnostics)==[]
    assert any(x.get('decision')=='rejected' and x.get('rejection_reasons') for x in diagnostics)


def test_ordinary_braking_into_signal_queue_is_not_collision():
    rows=[]
    for i in range(31):
        t=i/10
        x=min(.40,.20+.14*t) if t<1.4 else .40
        rows += [p(t,1,x,.5),p(t,2,.47,.5)]
    assert detect_collision_candidates(rows)==[]


def test_safe_passing_without_simultaneous_overlap_is_not_collision():
    rows=[]
    for i in range(31):
        t=i/10
        rows += [p(t,1,.2+.02*t,.35),p(t,2,.8-.02*t,.65)]
    assert detect_collision_candidates(rows)==[]


def test_turn_away_without_contact_is_not_collision():
    rows=[]
    for i in range(31):
        t=i/10
        # One road user turns away before reaching the crossing path.
        ax=.35+.08*min(t,1.2); ay=.50-.10*max(0,t-1.2)
        rows += [p(t,1,ax,ay),p(t,2,.62,.68)]
    assert detect_collision_candidates(rows)==[]


def test_occlusion_without_observed_interaction_is_not_collision():
    rows=crossing_tracks()
    rows=[x for x in rows if not (1.8<=x['t']<=2.2)]
    assert detect_collision_candidates(rows)==[]


def test_moving_camera_withholds_image_space_collision_screening():
    assert detect_collision_candidates(crossing_tracks(),moving_camera=True)==[]


def test_identity_fragments_are_not_joined_across_scene_cut():
    before=[p(i/10,1,.5,.5) | {'scene_id':0} for i in range(4)]
    after=[p(.34+i/10,2,.5,.5) | {'scene_id':1} for i in range(4)]
    groups,links=_join_identity_fragments({1:before,2:after})
    assert links[1]!=links[2] and len(groups)==2
