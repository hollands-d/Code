from pathlib import Path
from unittest.mock import patch
import tkinter as tk
import numpy as np
import pandas as pd
import pytest
import vibration_signal_processing_gui as m
from issue1_processing import ISSUE1_FFT_N, ISSUE1_SAMPLE_RATE_HZ, issue1_amplitude_from_psd


def proc(axis, gain=1.0, source='run.csv', n=3000):
    fs=200.0; t=np.arange(n)/fs
    rng=np.random.default_rng(100+ord(axis))
    # Broadband signal exercises all frequency coordinates.
    primary=rng.normal(0,0.003,n)*gain
    df=pd.DataFrame({'time_s':t,'X_g':rng.normal(0,0.0002,n),'Y_g':rng.normal(0,0.0002,n),'Z_g':rng.normal(0,0.0002,n)})
    df[f'{axis}_g']=primary
    p=m.Processor('Reader' if 'reader' in source else 'Baseline')
    p.set(df,m.Cols('time_s','X_g','Y_g','Z_g'),fs,source,metadata={'excitation_axis':axis},source_df=df)
    return p


def test_time_domain_xlsx_reference_loads(tmp_path):
    p=proc('X',source='baseline.xlsx')
    path=tmp_path/'baseline X.xlsx'; p.original_df.to_excel(path,index=False)
    loaded,_=m.read_baseline_file(path)
    assert isinstance(loaded,m.Processor)
    assert loaded.metadata['baseline_type']=='time-domain Excel'
    np.testing.assert_allclose(loaded.axis('X'),p.axis('X'))


def test_testhouse_parser_ref_is_optional_diagnostic_only():
    from test_testhouse_baseline import table
    spec,_=m.load_testhouse_table(table())
    assert set(spec.traces)=={'Ctl'}
    assert spec.metadata['selected_source']=='Ctl'
    display_spec,_=m.load_testhouse_table(table(), include_ref=True)
    assert set(display_spec.traces)=={'Ctl','Ref'}
    assert display_spec.metadata['selected_source']=='Ctl'
    np.testing.assert_allclose(display_spec.psd_g2_per_hz, display_spec.traces['Ctl'][1])



def test_issue1_psd_conversion_matches_constant_psd_bin_energy():
    f=np.linspace(0,100,20001); p=np.full_like(f,2.5e-6)
    coords,amp=issue1_amplitude_from_psd(f,p,apply_fir_response=False)
    df=ISSUE1_SAMPLE_RATE_HZ/ISSUE1_FFT_N
    active=(coords>=5)&(coords<=80)
    np.testing.assert_allclose(amp[active],np.sqrt(2*2.5e-6*df),rtol=2e-3)


def test_three_reader_runs_and_driven_axis_only_issue2():
    root=tk.Tk(); root.withdraw(); app=m.App(root)
    try:
        for axis in 'XYZ':
            app.baselines[axis]=proc(axis,1.0,f'baseline_{axis}.csv')
            app.reader_runs[axis]=proc(axis,2.0,f'reader_{axis}.csv')
            app.baseline_use_axis[axis].set(axis)
        # Deliberately leave self.reader pointing at X: Y/Z derive from their slots.
        app.reader=app.reader_runs['X']
        for axis in 'XYZ':
            r=app._derive_issue2_translation(axis,axis)
            assert r.driven_axis==axis and len(r.k_values)==14
        app.reader_runs['Y']=None
        with pytest.raises(ValueError,match='Y-excited|no Y reader data'):
            app._derive_issue2_translation('Y','Y')
    finally:
        root.destroy()


def test_issue1_time_and_psd_translation_store(tmp_path):
    root=tk.Tk(); root.withdraw(); app=m.App(root)
    try:
        app.applied_processing_method=m.ISSUE1_METHOD
        app.processing_method.set(m.ISSUE1_METHOD)
        base=proc('X',1.0,'baseline_X.csv'); reader=proc('X',2.0,'reader_X.csv')
        app.baselines['X']=base; app.baseline_use_axis['X'].set('X'); app.reader_runs['X']=reader; app.reader=reader; app.driven_axis.set('X')
        tr=app.derive_translation('X',m.ISSUE1_METHOD)
        assert tr.method==m.ISSUE1_METHOD and tr.reference_domain=='time' and len(tr.k_values)>40
        with patch.object(m.messagebox,'showerror') as err:
            app.store_current_axis_correction(); err.assert_not_called()
        assert app.stored_correction_axes['X']['method']==m.ISSUE1_METHOD

        # Build a PSD reference covering the full Issue-1 range and confirm direct PSD path.
        spec=m.BaselineSpectrum('PSD')
        f=np.linspace(0,100,2001); spec.set(f,np.full_like(f,1e-6),str(tmp_path/'Ctl X.csv'),{'selected_source':'Ctl'})
        app.baselines['X']=spec
        tr2=app.derive_translation('X',m.ISSUE1_METHOD)
        assert tr2.reference_domain=='PSD' and np.all(np.isfinite(tr2.k_values))
    finally:
        root.destroy()


def test_mixed_methods_rejected_on_final_export():
    root=tk.Tk(); root.withdraw(); app=m.App(root)
    try:
        dummy=lambda method,axis: m.TranslationResult(method,axis,np.array([5.0]),np.array([1.0]),'b.csv','r.csv','time',axis,{}, {})
        for axis,method in zip('XYZ',[m.ISSUE1_METHOD,m.ISSUE2_METHOD,m.ISSUE1_METHOD]):
            r=dummy(method,axis); app.stored_correction_axes[axis]={'method':method,'result':r,'rows':r.rows()}
        with patch.object(m.messagebox,'showerror') as error, patch.object(m.filedialog,'asksaveasfilename') as save:
            app.export_stored_xyz_correction()
            error.assert_called_once(); save.assert_not_called()
    finally:
        root.destroy()
