from backend.movement import detect_movement_candidates


def track(t, tid=1, x=.5, y=.5, **extra):
    return {'t':t,'id':tid,'class':2,'x':x,'y':y,**extra}


def poly(): return [[.2,.2],[.8,.2],[.8,.8],[.2,.8]]


def test_wrong_way_persistent_opposite_direction_and_turn_exception():
    samples=[track(i/10,y=.8-.025*i) for i in range(21)]
    cfg={'wrong_way':{'enabled':True,'permitted_directions':[[0,1]],'min_duration_s':1.5}}
    assert any(e['type']=='wrong_way_candidate' for e in detect_movement_candidates(samples,cfg))
    cfg['wrong_way']['permitted_turn_zones']=[poly()]
    assert not any(e['type']=='wrong_way_candidate' for e in detect_movement_candidates(samples,cfg))


def test_wrong_way_allows_configured_bidirectional_road():
    samples=[track(i/10,y=.8-.025*i) for i in range(21)]
    cfg={'wrong_way':{'enabled':True,'permitted_directions':[[0,1],[0,-1]],'min_duration_s':1.5}}
    assert not any(e['type']=='wrong_way_candidate' for e in detect_movement_candidates(samples,cfg))


def test_stopped_vehicle_candidate_and_queue_context_suppression():
    samples=[track(i/10) for i in range(121)]
    cfg={'stopped_vehicle':{'enabled':True,'zone':poly(),'dwell_s':8}}
    assert any(e['type']=='stopped_vehicle_candidate' for e in detect_movement_candidates(samples,cfg))
    queue=[track(i/10,tid,x=.45+tid*.02) for tid in range(3) for i in range(121)]
    assert not any(e['type']=='stopped_vehicle_candidate' for e in detect_movement_candidates(queue,cfg))


def test_blocked_crossing_requires_configured_dwell():
    samples=[track(i/10,x=.5,y=.5) for i in range(31)]
    cfg={'blocked_crossing':{'enabled':True,'zone':poly(),'dwell_s':2}}
    assert any(e['type']=='blocked_crossing_candidate' for e in detect_movement_candidates(samples,cfg))
    cfg['blocked_crossing']['dwell_s']=5
    assert not any(e['type']=='blocked_crossing_candidate' for e in detect_movement_candidates(samples,cfg))


def test_speed_and_deceleration_are_suppressed_without_calibration():
    samples=[track(0,x=0,coordinate_system='road_plane',units='m'),track(.1,x=1,coordinate_system='road_plane',units='m'),track(.2,x=1.5,coordinate_system='road_plane',units='m')]
    cfg={'speeding':{'enabled':True,'speed_limit_mps':4},'sudden_deceleration':{'enabled':True,'threshold_mps2':4}}
    assert detect_movement_candidates(samples,cfg,calibrated=False)==[]
    events=detect_movement_candidates(samples,cfg,calibrated=True)
    assert any(e['type']=='speeding_candidate' for e in events)
    assert any(e['type']=='sudden_deceleration_candidate' for e in events)
