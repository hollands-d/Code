import tkinter as tk
from pathlib import Path
import numpy as np
import pandas as pd
import vibration_signal_processing_gui as m


def make_proc(label, driven_axis, gain=1.0, shock=False):
    fs=200.0; n=3000; t=np.arange(n)/fs
    rng=np.random.default_rng(500+ord(driven_axis)+(100 if shock else 0))
    sig=rng.normal(0,0.002,n)*gain
    if shock:
        sig[1400:1405]+=np.array([0.2,0.6,1.2,0.5,0.1])
    df=pd.DataFrame({'time_s':t,'X_g':np.zeros(n),'Y_g':np.zeros(n),'Z_g':np.full(n,-1.0)})
    if driven_axis=='X': df['X_g']=sig
    elif driven_axis=='Y': df['Y_g']=sig
    else: df['Z_g']=-1+sig
    p=m.Processor(label); p.set(df,m.Cols('time_s','X_g','Y_g','Z_g'),fs,source=label)
    return p

root=tk.Tk(); root.withdraw(); app=m.App(root)
assert len(app._tab_nav_buttons)==app.tabs.index('end')
assert any(app.tabs.tab(i,'text')=='Shock summary' for i in range(app.tabs.index('end')))
for a in 'XYZ':
    app.baselines[a]=make_proc(f'baseline_{a}.csv',a,1.0); app.baseline_use_axis[a].set(a)
for a,g in [('X',1.2),('Y',.9),('Z',1.1)]:
    app.reader=make_proc(f'reader_{a}.csv',a,g); app.driven_axis.set(a); app.store_current_axis_correction()
assert all(app.stored_correction_axes[a] and len(app.stored_correction_axes[a]['rows'])==14 for a in 'XYZ')
for a in 'XYZ':
    app.shock_baselines[a]=make_proc(f'shock_{a}_baseline.csv',a,1.0,True)
    app.shock_readers[a]=make_proc(f'shock_{a}_reader.csv',a,1.1,True)
app.refresh_shock_summary(); assert len(app.shock_summary_tree.get_children())==3
root.destroy(); print('v40 feature smoke tests passed')
