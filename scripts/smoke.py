"""Synthetic offline plumbing only; no trained accuracy claim."""
import socket
def forbidden(*args, **kwargs):
    raise AssertionError('Network access forbidden in smoke')
socket.socket.connect = forbidden
socket.create_connection = forbidden
import numpy as np
import torch
from ai_research_work.models import SensorModel, VideoModel
from ai_research_work.f1 import predict
from ai_research_work.f1.preprocessing import hof
torch.set_num_threads(1)
assert predict('absent')['status'] == 'insufficient_input'
s,v=SensorModel(),VideoModel(8,128)
with torch.inference_mode():
    assert s(torch.zeros(1,128,6)).shape == (1,8)
    assert v(torch.zeros(1,16,128)).shape == (1,8)
assert not hof(np.zeros((2,120,160,3),np.uint8)).any()
print('PASS: no-network synthetic shapes, static HOF and both-missing behavior.')
