"""Synthetic preprocessing and inference invariants; no recordings or fitted weights."""
import numpy as np
import pytest
import torch
from ai_research_work.har import (
    SensorCNN, HOFGRU, HOFTransformer, TrainingScaler, resample_sensor, hof,
    outputs, fuse_probabilities, predict,
)


def test_parameter_counts_and_embeddings():
    torch.set_num_threads(1)
    for cls, shape, count in [(SensorCNN,(128,6),29883),(HOFGRU,(16,128),39259),(HOFTransformer,(16,128),40763)]:
        model=cls(27).eval()
        assert sum(p.numel() for p in model.parameters()) == count
        result=outputs(model,np.zeros((2,*shape),np.float32),action_ids=list(range(27,0,-1)),labels=[f'label{i}' for i in range(27)])
        assert result['logits'].shape == (2,27)
        assert result['embeddings'].shape == (2,64)
        assert result['action_ids'] == list(range(27,0,-1))
        np.testing.assert_allclose(result['probabilities'].sum(1),1,atol=2e-6)


def test_sensor_index_interpolation_and_training_statistics():
    raw=np.stack([np.arange(3)+i*10 for i in range(6)],axis=1)
    x=resample_sensor(raw)
    expected=np.stack([np.linspace(i*10,i*10+2,128) for i in range(6)],axis=1).astype(np.float32)
    np.testing.assert_array_equal(x,expected)
    scaler=TrainingScaler(np.ones((1,1,6)),np.full((1,1,6),2))
    np.testing.assert_array_equal(scaler.transform(x[None]),(x[None]-1)/2)
    assert not hasattr(scaler,'fit')
    with pytest.raises(ValueError):resample_sensor(np.zeros((1,6)))
    with pytest.raises(ValueError):resample_sensor(np.zeros((10,7)))
    with pytest.raises(ValueError):TrainingScaler(np.zeros((1,1,6)),np.zeros((1,1,6)))


def test_hof_motionless_and_translation():
    frames=np.zeros((3,120,160,3),np.uint8)
    assert np.count_nonzero(hof(frames)) == 0
    frames[0,30:70,30:70]=255;frames[1,30:70,35:75]=255;frames[2,30:70,40:80]=255
    features=hof(frames)
    assert features.shape==(16,128) and features.dtype==np.float32
    assert np.isfinite(features).all() and features.sum()>0
    np.testing.assert_allclose(features.sum(),1,atol=1e-6)
    with pytest.raises(ValueError):hof(frames.astype(np.float32))


def test_single_modality_unchanged_and_no_acceptance():
    p=np.array([[0.,1.]],np.float32)
    record={'action_ids':[8,3],'labels':['eight','three'],'probabilities':p}
    for fusion in ('arithmetic','product'):
        for s,v in [(record,None),(None,record)]:
            result=fuse_probabilities(s,v,fusion=fusion)
            assert result['probabilities'] is p
            assert result['predicted_action_ids']==[3]
            assert result['status']=='prediction_only' and result['acceptance']=='not_evaluated'
    assert predict('does-not-exist',model='hof_gru')['status']=='insufficient_input'
