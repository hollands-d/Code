from pathlib import Path
from unittest.mock import patch
import json
import subprocess
import sys
import tkinter as tk
import numpy as np
import pandas as pd
import pytest
import vibration_signal_processing_gui as m
from test_testhouse_baseline import table
from test_v18_multiaxis_correction import proc_from_signal


def test_reader_workbook_ctl_only(tmp_path):
    data=table(); data.iloc[2:,2:]='ignored'
    path=tmp_path/'Office Vertical.xlsx'; data.to_excel(path,index=False,header=False)
    reader,_=m.read_reader_file(path)
    assert set(reader.traces)=={'Ctl'}
    assert reader.metadata['reader_type']=='PSD Excel'
    np.testing.assert_array_equal(reader.psd_g2_per_hz,1e-5)
    data.iloc[0,1]='Ref'; data.to_excel(path,index=False,header=False)
    with pytest.raises(ValueError,match='Ctl'): m.read_reader_file(path)


def test_reader_excel_gui_and_correction(tmp_path):
    # Isolate Tk's process-global state from other GUI regression tests.
    result=subprocess.run([sys.executable,'-c',
        'from pathlib import Path; import sys; from test_reader_excel import _check_reader_excel_gui; _check_reader_excel_gui(Path(sys.argv[1]))',
        str(tmp_path)],capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stdout+result.stderr


def _check_reader_excel_gui(tmp_path):
    root=tk.Tk(); root.withdraw(); app=m.App(root)
    try:
        path=tmp_path/'Office Lateral.xlsx'; data=table(); data.iloc[2:,1]=4e-5
        data.to_excel(path,index=False,header=False)
        baseline,_=m.load_testhouse_table(table(),str(tmp_path/'baseline Lateral.xlsx'))
        app.baselines['X']=baseline; app.baselines['Y']=baseline
        with patch.object(m.filedialog,'askopenfilename',return_value=str(path)), patch.object(m.simpledialog,'askstring',return_value='X'), patch.object(m.messagebox,'showwarning'), patch.object(m.messagebox,'showerror') as error:
            app.load_csv('reader'); error.assert_not_called()
        assert app.baselines['X'] is baseline
        assert app.reader.metadata['excitation_axis']=='X'
        assert app.driven_axis.get()=='X'
        labels=[line.get_label() for line in app.fig['PSD comparison'][0].axes[0].lines]
        assert 'Reader X Ctl' in labels
        assert not any('Reader X Ref' in label for label in labels)
        np.testing.assert_allclose(list(app._axis_correction_result('X','X')[3].values()),.5)
        with pytest.raises(ValueError,match='no Y reader data'): app._reader_axis_band_statistics('Y')
        with pytest.raises(ValueError,match='PSD-only reader'): app._paired_axis_context('X')
        output=tmp_path/'correction.csv'
        with patch.object(m.filedialog,'asksaveasfilename',return_value=str(output)), patch.object(m.messagebox,'showinfo'), patch.object(m.messagebox,'showerror') as error:
            app.export_correction(); error.assert_not_called()
        result=pd.read_csv(output)
        assert len(result)==14 and set(result.driven_axis)=={'X'}
        assert json.loads(output.with_name('correction_settings.json').read_text())['reader_details']['selected_source']=='Ctl'
        # Reader-only display is also supported, and replacing PSD with time samples clears its state.
        app.baselines={a:None for a in 'XYZ'}; assert app._plot_reference_psd()
        time=proc_from_signal('reader','X'); csv=tmp_path/'reader.csv'; time.original_df.to_csv(csv,index=False)
        with patch.object(m.filedialog,'askopenfilename',return_value=str(csv)), patch.object(m.messagebox,'showerror') as error:
            app.load_csv('reader'); error.assert_not_called()
        assert isinstance(app.reader,m.Processor)
    finally: root.destroy()


def test_reader_time_excel_matches_csv(tmp_path):
    proc=proc_from_signal('reader','Z'); path=tmp_path/'reader.xlsx'; proc.original_df.to_excel(path,index=False)
    reader,_=m.read_reader_file(path)
    assert isinstance(reader,m.Processor)
    np.testing.assert_allclose(reader.df[['X_g','Y_g','Z_g']],proc.df[['X_g','Y_g','Z_g']],atol=1e-14)
