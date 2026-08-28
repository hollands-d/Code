import struct
import pandas as pd

from vibration_signal_processing_gui import normalize_accelerometer_dataframe
from vmm_stream import VmmStreamClient


def test_raw_and_converted_equivalence():
    raw=pd.DataFrame({
        'block_seq':[1,1,1],
        'sample_timestamp_us':[1_000_000,1_005_000,1_010_000],
        'sample_period_us':[5000,5000,5000],
        'odr_hz':[200,200,200],
        'fs_g':[2,2,2],
        'status':[0,0,0],
        'x_counts':[16384,0,-16384],
        'y_counts':[0,8192,0],
        'z_counts':[-16384,-16384,-16384],
    })
    out,cols,fs,meta,warnings=normalize_accelerometer_dataframe(raw,200)
    assert warnings==[]
    assert meta['source_format']=='raw_vmm_counts'
    assert abs(fs-200.0)<1e-9
    assert list(out['time_s'])==[0.0,0.005,0.01]
    assert list(out['X_g'])==[1.0,0.0,-1.0]
    assert list(out['Y_g'])==[0.0,0.5,0.0]

    converted=pd.DataFrame({'time_s':[0,.005,.01],'X_g':[1,0,-1],'Y_g':[0,.5,0],'Z_g':[-1,-1,-1]})
    c,_,cfs,cmeta,_=normalize_accelerometer_dataframe(converted,200)
    assert cmeta['source_format']=='converted_g'
    assert abs(cfs-200.0)<1e-9
    import numpy as np
    assert np.allclose(c[['time_s','X_g','Y_g','Z_g']].to_numpy(float), out[['time_s','X_g','Y_g','Z_g']].to_numpy(float))


class FakeSerial:
    def __init__(self):
        self.writes=[]; self.reset_count=0
    def reset_input_buffer(self): self.reset_count+=1
    def write(self,b): self.writes.append(bytes(b)); return len(b)
    def flush(self): pass


def test_renew_does_not_reset_input_buffer():
    client=VmmStreamClient.__new__(VmmStreamClient)
    client._ser=FakeSerial(); client._seq=0
    client.start(timeout_ms=300_000)
    assert client._ser.reset_count==1
    client.renew(timeout_ms=300_000)
    assert client._ser.reset_count==1
    assert len(client._ser.writes)==2


if __name__=='__main__':
    test_raw_and_converted_equivalence()
    test_renew_does_not_reset_input_buffer()
    print('v14 core tests passed')
