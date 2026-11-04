"""Image-space interaction screening. No physical distance or crash probability."""
import math

def approach(a, b, horizon=2.0, radius=0.045):
    rx, ry = b['x']-a['x'], b['y']-a['y']
    vx, vy = b['vx']-a['vx'], b['vy']-a['vy']
    vv = vx*vx+vy*vy
    if vv < 1e-8: return None
    t = -(rx*vx+ry*vy)/vv
    if not 0 < t <= horizon: return None
    clearance = math.hypot(rx+vx*t, ry+vy*t)
    if clearance > radius: return None
    return {'closest_approach_seconds': round(t, 3), 'normalized_clearance': round(clearance, 4)}
