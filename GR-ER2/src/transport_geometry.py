"""Local depth-only refinement of a model-selected small block top surface."""
import numpy as np


def refine_block_top(depth, x, y, radius=12):
    """Find a raised, locally flat top near the pointing coordinate.

    Requires the calibrated downward-facing station camera. No object IDs,
    simulator poses or global target search are available to this function.
    Returns (pixel x, pixel y, image-plane depth) or rejects ambiguous surfaces.
    """
    h,w=depth.shape;ix,iy=round(x),round(y)
    x0,x1=max(0,ix-radius),min(w,ix+radius+1)
    y0,y1=max(0,iy-radius),min(h,iy+radius+1)
    patch=depth[y0:y1,x0:x1]
    valid=np.isfinite(patch)&(patch>1.85)&(patch<2.15)
    if valid.sum()<25:raise ValueError('No valid local block surface; use station camera and point at its top')
    values=patch[valid];top=float(np.percentile(values,10))
    if float(np.percentile(values,85))-top<.020:
        raise ValueError('No raised block top near selected point')
    selected=valid&(abs(patch-top)<.006)
    yy,xx=np.nonzero(selected)
    if len(xx)<12 or xx.max()-xx.min()>21 or yy.max()-yy.min()>21:
        raise ValueError('Ambiguous local depth surface; reobserve the block')
    return float(xx.mean()+x0),float(yy.mean()+y0),float(np.median(patch[selected]))
