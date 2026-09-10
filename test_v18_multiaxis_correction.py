from pathlib import Path
import tkinter as tk
import numpy as np
import pandas as pd
import vibration_signal_processing_gui as m


def proc_from_signal(label, axis, gain=1.0):
    fs=200.0; n=4096; t=np.arange(n)/fs
    rng=np.random.default_rng(1174+ord(axis))
    sig=rng.normal(0,0.002,n)*gain
    df=pd.DataFrame({'time_s':t,'X_g':np.zeros(n),'Y_g':np.zeros(n),'Z_g':np.full(n,-1.0)})
    if axis=='X': df['X_g']=sig
    elif axis=='Y': df['Y_g']=sig
    else: df['Z_g']=-1.0+sig
    p=m.Processor(label); p.set(df,m.Cols('time_s','X_g','Y_g','Z_g'),fs,source=label)
    return p


def main():
    assert m.infer_excitation_axis_from_filename('baseline_X_test.csv') == 'X'
    assert m.infer_excitation_axis_from_filename('Y-axis-reference.csv') == 'Y'
    assert m.infer_excitation_axis_from_filename('reference_axis_Z.csv') == 'Z'

    root=tk.Tk(); root.withdraw(); app=m.App(root)
    for axis in 'XYZ':
        app.baselines[axis]=proc_from_signal(f'baseline_{axis}.csv',axis,1.0)
        app.baseline_use_axis[axis].set(axis)
    # Reader contains all channels; simple independent noise is sufficient for regression.
    n=4096; fs=200.0; t=np.arange(n)/fs; rng=np.random.default_rng(42)
    rdf=pd.DataFrame({'time_s':t,'X_g':rng.normal(0,.0025,n),'Y_g':rng.normal(0,.0018,n),'Z_g':-1+rng.normal(0,.0022,n)})
    app.reader.set(rdf,m.Cols('time_s','X_g','Y_g','Z_g'),fs,source='reader.csv')

    for target in 'XYZ':
        source=app.baseline_use_axis[target].get()
        _,_,_,band,_,_=app._axis_correction_result(target,source)
        assert len(band)==14

    app.baseline_use_axis['X'].set('Y')
    _,_,_,band,_,base=app._axis_correction_result('X','Y')
    assert len(band)==14
    assert base.metadata.get('surrogate_baseline_axis')=='Y'

    out=Path(__file__).with_name('_v18_test_export.csv')
    old_save=m.filedialog.asksaveasfilename; old_info=m.messagebox.showinfo
    try:
        m.filedialog.asksaveasfilename=lambda **kwargs: str(out)
        m.messagebox.showinfo=lambda *args,**kwargs: None
        app.export_correction()
        df=pd.read_csv(out)
        assert len(df)==42
        assert set(df['driven_axis'])==set('XYZ')
        assert all(df.groupby('driven_axis').size()==14)
    finally:
        m.filedialog.asksaveasfilename=old_save; m.messagebox.showinfo=old_info
        for f in [out,out.with_name(out.stem+'_settings.json')]:
            if f.exists(): f.unlink()
        root.destroy()
    print('v18 multi-axis correction tests passed')

if __name__=='__main__': main()
