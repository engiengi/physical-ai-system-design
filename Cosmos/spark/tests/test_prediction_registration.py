import numpy as np
import pytest
from cosmos_spark.prediction import comparison_registration


def test_decoder_crop_preserves_native_pixels_and_excludes_only_bottom_strip():
    observation=np.random.default_rng(8).integers(0,256,(540,640,3),dtype=np.uint8)
    result=comparison_registration(observation,observation[:528])
    assert result['mode']=='verified_initial_frame_top_left_common_region'
    assert (result['width'],result['height'],result['crop_xy'])==(640,528,[0,0])
    assert result['actual_bottom_pixels_excluded']==12 and result['fit_frames']==[0]


@pytest.mark.parametrize('kind',['shifted','ambiguous','unsupported'])
def test_unverified_geometry_is_not_silently_stretched(kind):
    observation=np.random.default_rng(8).integers(0,256,(540,640,3),dtype=np.uint8)
    if kind=='shifted':prediction=observation[12:]
    elif kind=='ambiguous':observation[:]=0;prediction=observation[:528]
    else:prediction=observation[:512]
    with pytest.raises(ValueError):comparison_registration(observation,prediction)


def test_cropped_comparison_requires_preserved_original_observation(tmp_path):
    import json,subprocess
    from cosmos_spark.prediction import PredictionReceiver
    for name,height in [('prediction',528),('policy_view',540)]:
        subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i',f'color=c=blue:s=640x{height}:r=15',
                        '-frames:v','3','-c:v','libx264','-threads','1',str(tmp_path/(name+'.mp4'))],check=True)
    actions=np.zeros((2,8),dtype=np.float32);np.save(tmp_path/'request_0_actions.npy',actions)
    rows=[{'request_id':'request_0','chunk_index':i,'action':row.tolist(),'action_started_at_ns':1} for i,row in enumerate(actions)]
    (tmp_path/'applied_actions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    receiver=PredictionReceiver(tmp_path,2,15)
    receiver.receipt={'request_id':'request_0','saved_at_ns':0,'local_video':str(tmp_path/'prediction.mp4')}
    with pytest.raises(ValueError,match='Original observation required'):receiver.finalize()
