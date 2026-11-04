import pytest
from backend.calibration import estimate_homography, transform_point
from backend.conflict import analyze_pair, pet_from_intervals, merge_events
from backend.trajectory import process_tracks


def test_homography_recovery_and_units():
    src=[[0,0],[10,0],[0,10],[10,10]]; dst=[[1,2],[21,2],[1,32],[21,32]]
    fit=estimate_homography(src,dst)
    assert transform_point(fit['homography'],[5,5]) == pytest.approx([11,17],abs=1e-5)
    assert fit['rms_error_m'] < 1e-5


@pytest.mark.parametrize('src,dst', [([[0,0],[1,1],[2,2],[3,3]],[[0,0],[1,1],[2,2],[3,3]]), ([[0,0],[1,1]],[[0,0],[1,1]])])
def test_invalid_calibration(src,dst):
    with pytest.raises(ValueError): estimate_homography(src,dst)


def test_pet_occupancy_definition():
    assert pet_from_intervals(2.3,3.1)==pytest.approx(.8)
    assert pet_from_intervals(None,3) is None


def test_track_jump_and_occlusion_gating():
    points=[{'id':1,'t':0,'x':0,'y':0},{'id':1,'t':.2,'x':.1,'y':0},{'id':1,'t':.4,'x':.2,'y':0},{'id':1,'t':.6,'x':.3,'y':0},{'id':1,'t':.8,'x':100,'y':0},{'id':1,'t':2,'x':.4,'y':0}]
    row=process_tracks(points)[0]
    assert row['reliable'] and row['vx']==pytest.approx(.5)


def test_pair_zero_relative_velocity_and_unreliable():
    metric=analyze_pair({'x':0,'y':0,'vx':1,'vy':0,'reliable':True},{'x':1,'y':0,'vx':1,'vy':0,'reliable':True})
    assert metric['ttc_s'] is None and 'relative velocity' in metric['ttc_unavailable_reason']
    assert analyze_pair({'x':0,'y':0,'vx':1,'vy':0,'reliable':False},{'x':1,'y':0,'vx':-1,'vy':0,'reliable':True}) is None


def test_duplicate_merge():
    ev=[{'id':'a','timestamp':1,'participants':[1,2]},{'id':'b','timestamp':2,'participants':[2,1]}]
    out=merge_events(ev)
    assert len(out)==1 and out[0]['start_timestamp']==1 and out[0]['end_timestamp']==2

def test_duplicate_merge_preserves_observed_interval_and_peak():
    ev=[{'id':'a','timestamp':2.5,'start_timestamp':2.0,'peak_timestamp':2.5,'end_timestamp':2.7,'participants':[1,2]},
        {'id':'b','timestamp':3.2,'start_timestamp':3.0,'peak_timestamp':3.1,'end_timestamp':3.4,'participants':[2,1]}]
    out=merge_events(ev,window_s=1.0)
    assert len(out)==1
    assert out[0]['start_timestamp']==2.0
    assert out[0]['peak_timestamp']==2.5
    assert out[0]['end_timestamp']==3.4


def test_near_miss_and_collision_types_are_not_merged_together():
    events=[{'id':'a','type':'image_space_candidate','timestamp':1,'participants':[1,2]},
            {'id':'b','type':'possible_collision','timestamp':1.2,'participants':[2,1]}]
    assert [e['type'] for e in merge_events(events,window_s=3)]==['image_space_candidate','possible_collision']
