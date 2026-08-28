from pathlib import Path
import tkinter as tk
import pandas as pd
import vibration_signal_processing_gui as m


def main():
    assert m.infer_excitation_axis_from_filename('baseline_X_test.csv') == 'X'
    assert m.infer_excitation_axis_from_filename('Y-axis-reference.csv') == 'Y'
    assert m.infer_excitation_axis_from_filename('reference_axis_Z.csv') == 'Z'

    root = tk.Tk()
    root.withdraw()
    app = m.App(root)

    for target in 'XYZ':
        source = app.baseline_use_axis[target].get()
        _, _, _, band, _, _ = app._axis_correction_result(target, source)
        assert len(band) == 14

    app.baseline_use_axis['X'].set('Y')
    _, _, _, band, _, base = app._axis_correction_result('X', 'Y')
    assert len(band) == 14
    assert base.metadata.get('surrogate_baseline_axis') == 'Y'

    out = Path(__file__).with_name('_v18_test_export.csv')
    old_save = m.filedialog.asksaveasfilename
    old_info = m.messagebox.showinfo
    try:
        m.filedialog.asksaveasfilename = lambda **kwargs: str(out)
        m.messagebox.showinfo = lambda *args, **kwargs: None
        app.export_correction()
        df = pd.read_csv(out)
        assert len(df) == 42
        assert set(df['driven_axis']) == set('XYZ')
        assert all(df.groupby('driven_axis').size() == 14)
    finally:
        m.filedialog.asksaveasfilename = old_save
        m.messagebox.showinfo = old_info
        if out.exists():
            out.unlink()
        root.destroy()

    print('v18 multi-axis correction tests passed')


if __name__ == '__main__':
    main()
