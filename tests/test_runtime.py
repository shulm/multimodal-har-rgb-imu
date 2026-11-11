import copy
import socket
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from scipy.io import savemat
from ai_research_work.models import SensorModel, VideoModel
from ai_research_work.f1 import Predictor, predict
from ai_research_work.f1.runtime import decide, vector
from ai_research_work.f1.bundle import Bundle, CLASSES, encode_q, decode_q, feature
from ai_research_work.f1.preprocessing import sensor, hof
from ai_research_work.f1.telemetry import append, summarize

@pytest.fixture
def toy():
    pp = {'temperature': 1., 'temperature_provenance': 'synthetic',
          'D1': {'threshold': .5, 'score_definition': 'mean_cosine_distance_top5'},
          'D2': {'threshold': .25, 'score_definition': '1-max_raw_probability'},
          'conformal': {'q': .25}}
    return SimpleNamespace(meta={'patterns': {k:copy.deepcopy(pp) for k in
        ('both','sensor_only','video_only')}, 'bundle_id':'synthetic'}, version='synthetic',
        neighbors=lambda pattern,emb,k,query_id: [{'distance':.2} for _ in range(k)])

def inputs(s=True,v=True):
    p=np.array([.75,.25,0,0,0,0,0,0]); e=np.ones(64)
    return dict(sensor_probs=p if s else None,hof_probs=p if v else None,
                sensor_emb=e if s else None,hof_emb=e if v else None)

def test_missing_no_load_no_network(monkeypatch):
    def forbidden(*a,**k): raise AssertionError('Unexpected I/O')
    monkeypatch.setattr(torch,'load',forbidden)
    monkeypatch.setattr(socket.socket,'connect',forbidden)
    monkeypatch.setattr(Predictor,'load',forbidden)
    assert predict('absent')['status']=='insufficient_input'

@pytest.mark.parametrize('s,v,condition',[(True,True,'both'),(True,False,'sensor_only'),(False,True,'video_only')])
def test_fallback(toy,s,v,condition):
    out=decide(toy,'D2',**inputs(s,v))
    assert out['condition']==condition and out['status']=='accepted'
    np.testing.assert_array_equal(out['probabilities'],inputs()['sensor_probs'])

def test_average_distinct_branches(toy):
    kw=inputs(); kw['hof_probs']=np.array([.25,.75,0,0,0,0,0,0])
    np.testing.assert_array_equal(decide(toy,'D2',**kw)['probabilities'],[.5,.5,0,0,0,0,0,0])

def test_ties_temperature_rejection_and_sets(toy):
    pp=toy.meta['patterns']['both']
    baseline=decide(toy,'D2',**inputs())
    assert baseline['status']=='accepted' and baseline['action_id']==1
    for t in [.25,10]:
        pp['temperature']=t; out=decide(toy,'D2',**inputs())
        assert out['prediction_set']==baseline['prediction_set'] and out['status']=='accepted'
    pp['D2']['threshold']-=1e-10
    out=decide(toy,'D2',**inputs())
    assert out['status']=='unknown' and out['action_id'] is None
    assert out['prediction_set_role']=='diagnostic_after_rejection'
    pp['D2']['threshold']=1.; pp['conformal']['q']=encode_q(float('inf'))
    out=decide(toy,'D2',**inputs())
    assert out['status']=='uncertain' and out['prediction_set']==CLASSES
    pp['conformal']['q']=0
    assert decide(toy,'D2',**inputs())['prediction_set']==[]

def test_d1_independent_of_display(toy):
    a=decide(toy,'D1',**inputs(),explain_k=0)
    b=decide(toy,'D1',**inputs(),explain_k=8)
    assert a['score']==b['score']==.2 and a['neighbors']==[]

@pytest.mark.parametrize('bad',[np.zeros(8),np.ones(7)/7,np.full(8,np.nan),[2,-1,0,0,0,0,0,0],np.full(8,np.inf),np.ones(8)])
def test_bad_probabilities(bad):
    with pytest.raises(ValueError): vector(bad)

@pytest.mark.parametrize('bad',[float('nan'),float('-inf'),-.1,1.1,True])
def test_bad_quantile(bad):
    with pytest.raises(ValueError): encode_q(bad)

def test_shapes_and_parameters():
    torch.set_num_threads(1)
    s,v=SensorModel(),VideoModel(8,128)
    assert sum(p.numel() for m in (s,v) for p in m.parameters())==66672
    assert s(torch.zeros(2,128,6)).shape==(2,8)
    assert v(torch.zeros(2,16,128)).shape==(2,8)

def test_preprocessing_synthetic(tmp_path):
    p=tmp_path/'synthetic.mat'
    x=np.arange(18,dtype=np.float32).reshape(3,6)
    savemat(p,{'d_iner':x}); y=sensor(p)
    assert y.shape==(128,6)
    np.testing.assert_array_equal(y[[0,-1]],x[[0,-1]])
    for bad in [np.zeros((1,6)),np.zeros((4,5)),np.full((4,6),np.nan)]:
        savemat(p,{'d_iner':bad})
        with pytest.raises(ValueError): sensor(p)
    assert np.count_nonzero(hof(np.zeros((2,120,160,3),dtype=np.uint8)))==0

def test_feature_normalization():
    z=feature(np.ones(64),np.full(64,4.))
    assert z.shape==(128,) and np.isclose(np.linalg.norm(z),1)
    np.testing.assert_array_equal(z[:64],z[64:])

def test_telemetry_denominators(tmp_path,toy):
    p=tmp_path/'events.jsonl'; out=decide(toy,'D2',**inputs())
    out['raw_input']='excluded'; out['timing_seconds']={'total':.1}
    append(p,out,review={'action_id':1,'provenance':'synthetic','reviewed_at':'synthetic'})
    append(p,out)
    result=summarize(p)['patterns']['both']
    assert result['n']==2 and result['reviewed_n']==1 and result['reviewed_accepted_accuracy']==1
    assert 'excluded' not in p.read_text()

@pytest.mark.local_artifacts
def test_explicit_local_bundle(request):
    path=request.config.getoption('--bundle')
    if not path: pytest.skip('No explicit trusted local bundle supplied')
    Bundle(path)
