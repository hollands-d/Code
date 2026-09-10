from pathlib import Path
from unittest.mock import patch
import tkinter as tk
import numpy as np
import pandas as pd
import pytest
import vibration_signal_processing_gui as m
from test_v18_multiaxis_correction import proc_from_signal


def table(ref=100):
    rows=[[None,'Ctl',None,'Ref',None,'Alarm-',None,'Alarm+',None,'CA_Z'],
          ['Hz','Magnitude, (g)^2/Hz']*5]
    for f in np.linspace(2.67,88.2,200):
        rows.append([f,1e-5,f,ref,f,ref/2,f,ref*2,f,1e-10])
    return pd.DataFrame(rows)


def test_formats_and_validation(tmp_path):
    for orientation in ('Lateral','Vertical'):
        path=tmp_path/f'{orientation}.xlsx'; table().to_excel(path,index=False,header=False)
        spec,warnings=m.read_baseline_file(path)
        assert spec.metadata['orientation']==orientation
        assert spec.metadata['selected_source']=='Ctl'
        assert set(spec.traces)=={'Ctl','Ref'}
        assert np.all(spec.psd_g2_per_hz==1e-5)
        assert warnings
    ignored=table(); ignored.iloc[3,4:]='invalid ignored field'
    assert set(m.load_testhouse_table(ignored)[0].traces)=={'Ctl'}
    path=tmp_path/'Lateral.csv'; table().to_csv(path,index=False,header=False)
    assert m.read_baseline_file(path)[0].metadata['selected_source']=='Ctl'
    path=tmp_path/'simple.csv'
    pd.DataFrame({'frequency_hz':[2,10,90],'psd_g2_per_hz':[1,2,3]}).to_csv(path,index=False)
    assert np.array_equal(m.read_baseline_file(path)[0].psd_g2_per_hz,[1,2,3])
    p=proc_from_signal('time','X'); path=tmp_path/'time.csv'; p.original_df.to_csv(path,index=False)
    assert isinstance(m.read_baseline_file(path)[0],m.Processor)
    for row,col,value,match in [(0,1,'Ref','Missing Ctl'),(3,1,-1,'negative'),(3,0,2.67,'duplicate'),(3,1,'bad','non-numeric')]:
        data=table(); data.iloc[row,col]=value
        with pytest.raises(ValueError,match=match): m.load_testhouse_table(data)


def test_baseline_only_then_reader_and_ctl_correction(tmp_path):
    root=tk.Tk(); root.withdraw(); app=m.App(root)
    try:
        lateral=tmp_path/'Lateral.xlsx'; vertical=tmp_path/'Vertical.xlsx'
        table().to_excel(lateral,index=False,header=False); table().to_excel(vertical,index=False,header=False)
        with patch.object(m.filedialog,'askopenfilenames',return_value=[str(lateral),str(vertical)]), patch.object(m.messagebox,'showwarning'), patch.object(m.messagebox,'showerror') as error:
            app.load_baseline_csvs(); error.assert_not_called()
        assert app.baselines['X'] is app.baselines['Y']
        assert app.baselines['Z'].metadata['orientation']=='Vertical'
        summary=app._baseline_summary_text()
        assert 'Not loaded' in summary and 'Ctl' in summary and 'Unavailable until' in summary
        assert app._plot_reference_psd()
        figures=app.fig['PSD comparison'][0].axes
        assert len(figures)==2
        assert all(len(ax.lines)==2 for ax in figures)
        assert any('Ctl = measured' in line.get_label() for line in figures[0].lines)
        assert any('Ref = commanded' in line.get_label() for line in figures[0].lines)
        app.applied_processing_method=m.ISSUE1_METHOD
        app._sync_applied_mode_tabs()
        tab=next(t for t in app.tabs.tabs() if app.tabs.tab(t,'text')=='PSD comparison')
        assert str(app.tabs.tab(tab,'state'))=='normal'
        app.applied_processing_method=m.ISSUE2_METHOD
        with pytest.raises(ValueError,match='H1 and coherence'):
            app._mapped_baseline_processor('X')
        original=app.baselines['X']
        app.reader=proc_from_signal('reader','X')
        app.n.set('256'); app.segs.set('4')
        result=app._axis_correction_from_psd_baseline('X','X',original)
        expected=original.bands().a_rms_g.to_numpy()/app._reader_axis_band_statistics('X')[0].a_rms_g.to_numpy()
        np.testing.assert_allclose(list(result[2].values()),expected)
        changed,_=m.load_testhouse_table(table(ref=1e9))
        np.testing.assert_allclose(list(app._axis_correction_from_psd_baseline('X','X',changed)[2].values()),expected)
        app.refresh()
        assert any(line.get_label()=='Reader X' for line in app.fig['PSD comparison'][0].axes[0].lines)
        assert app.baselines['X'] is original
        assert app._plot_reference_psd()
    finally: root.destroy()
