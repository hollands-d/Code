"""1174 Vibration Signal Processing Explorer - paired reference/reader edition.

The module contains three layers kept together for this standalone engineering
tool:

* CSV import and canonicalisation into time plus X/Y/Z acceleration in g;
* reusable numerical processing in :class:`Processor` and :class:`PairAnalysis`;
* the Tkinter :class:`App`, which coordinates offline analysis and a background
  VMM serial-capture worker.

Live serial work never updates Tk widgets directly. It places events on
``live_queue`` and the Tk main thread consumes them from ``_poll_live_queue``.
"""
from __future__ import annotations
import csv, json, math, re, tkinter as tk
import queue, threading, time
from tkinter import ttk, filedialog, messagebox, simpledialog
from dataclasses import dataclass, asdict, replace
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.patches import Circle

from vmm_stream import (
    CMD_CONFIGURE, DEFAULT_BAUD, AccelerometerConfig, SampleBlock, VmmStreamClient,
)
from issue1_processing import (
    ISSUE1_ENSEMBLE_COUNT, ISSUE1_FFT_N, ISSUE1_HIGH_HZ, ISSUE1_LOW_HZ,
    ISSUE1_OVERLAP, ISSUE1_SAMPLE_RATE_HZ, Issue1Result, issue1_amplitude_from_psd,
    issue1_ensemble_fft,
    issue1_required_samples,
)


def available_serial_ports():
    """Return (device, description, hwid), with ST-LINK-looking ports first."""
    try:
        from serial.tools import list_ports
    except ImportError:
        return []
    rows=[(p.device,p.description or '',p.hwid or '') for p in list_ports.comports()]
    rows.sort(key=lambda r:(0 if 'STLINK' in (r[1]+' '+r[2]).upper() or 'ST-LINK' in (r[1]+' '+r[2]).upper() else 1,r[0]))
    return rows


class RawVmmCsvLogger:
    """Thread-safe recorder for original VMM int16 samples and block metadata."""

    HEADER=(
        'block_seq','block_timestamp_us','sample_index','sample_timestamp_us',
        'sample_period_us','odr_hz','fs_g','status','x_counts','y_counts','z_counts',
    )

    def __init__(self):
        self._lock=threading.RLock(); self._file=None; self._writer=None
        self.path=None; self.blocks=0; self.samples=0; self._last_flush=0.0

    @property
    def active(self):
        with self._lock: return self._file is not None

    def start(self,path):
        self.stop()
        with self._lock:
            self.path=Path(path)
            self._file=self.path.open('w',newline='',encoding='utf-8')
            self._writer=csv.writer(self._file)
            self._writer.writerow(self.HEADER); self._file.flush()
            self.blocks=0; self.samples=0; self._last_flush=time.monotonic()

    def write_block(self,block):
        with self._lock:
            if self._file is None: return None
            period=int(block.sample_period_us)
            for index,(x,y,z) in enumerate(block.xyz):
                sample_ts=int(block.timestamp_us)+index*period if period>0 else int(block.timestamp_us)
                self._writer.writerow((
                    int(block.seq),int(block.timestamp_us),index,sample_ts,period,
                    int(block.odr_hz),int(block.fs_g),int(block.status),int(x),int(y),int(z),
                ))
            self.blocks+=1; self.samples+=len(block.xyz)
            now=time.monotonic()
            if now-self._last_flush>=1.0:
                self._file.flush(); self._last_flush=now
            return self.blocks,self.samples,self.path

    def stop(self):
        with self._lock:
            result=(self.blocks,self.samples,self.path)
            if self._file is not None:
                self._file.flush(); self._file.close()
            self._file=None; self._writer=None
            return result

class CollapsibleSection(ttk.Frame):
    """Simple arrow-toggle section used to reduce UI clutter."""

    def __init__(self, master, title, expanded=True, body_padding=6, **kwargs):
        super().__init__(master, **kwargs)
        self._title=title
        self._expanded=bool(expanded)
        self._body_padding=body_padding
        self._header=ttk.Button(self, command=self.toggle)
        self._header.pack(fill=tk.X)
        self.body=ttk.Frame(self, padding=self._body_padding)
        self._sync()

    def _sync(self):
        arrow='▾' if self._expanded else '▸'
        self._header.configure(text=f'{arrow} {self._title}')
        if self._expanded:
            if self.body.winfo_manager()!='pack':
                self.body.pack(fill=tk.X, padx=2, pady=(3, 2))
        else:
            if self.body.winfo_manager():
                self.body.pack_forget()

    def toggle(self):
        self._expanded=not self._expanded
        self._sync()

    def expand(self):
        self._expanded=True
        self._sync()

    def collapse(self):
        self._expanded=False
        self._sync()


class ScrollableFrame(ttk.Frame):
    """Canvas-backed scrolling frame for long control sidebars."""

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self._canvas=tk.Canvas(self, highlightthickness=0, borderwidth=0)
        self._scrollbar=ttk.Scrollbar(self, orient=tk.VERTICAL, command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=self._scrollbar.set)
        self._canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.body=ttk.Frame(self._canvas)
        self._window=self._canvas.create_window((0,0), window=self.body, anchor='nw')
        self.body.bind('<Configure>', self._on_body_configure)
        self._canvas.bind('<Configure>', self._on_canvas_configure)
        self.body.bind('<Enter>', self._bind_mousewheel)
        self.body.bind('<Leave>', self._unbind_mousewheel)

    def _on_body_configure(self, _event=None):
        self._canvas.configure(scrollregion=self._canvas.bbox('all'))

    def _on_canvas_configure(self, event):
        self._canvas.itemconfigure(self._window, width=event.width)

    def _on_mousewheel(self, event):
        delta=0
        if hasattr(event, 'delta') and event.delta:
            delta=-1 if event.delta>0 else 1
        elif getattr(event, 'num', None)==4:
            delta=-1
        elif getattr(event, 'num', None)==5:
            delta=1
        if delta:
            self._canvas.yview_scroll(delta, 'units')

    def _bind_mousewheel(self, _event=None):
        for seq in ('<MouseWheel>', '<Button-4>', '<Button-5>'):
            self._canvas.bind_all(seq, self._on_mousewheel)

    def _unbind_mousewheel(self, _event=None):
        for seq in ('<MouseWheel>', '<Button-4>', '<Button-5>'):
            self._canvas.unbind_all(seq)


APP_TITLE="1174 Vibration Signal Processing Explorer — VMM stream v42"
G0=9.80665
DEFAULT_FS=200.0; DEFAULT_N=512; DEFAULT_OVERLAP=0.5; DEFAULT_WELCH_SEGMENTS=4
ISSUE2_METHOD='Issue 2 – PSD / Gordon bands'
ISSUE1_METHOD='Issue 1 – FFT ensemble'
GORDON_FC=np.array([4.0,5.0,6.3,8.0,10.0,12.5,16.0,20.0,25.0,31.5,40.0,50.0,63.0,80.0],float)
OCT_EDGE=2**(1/6)

GORDON_TICK_LABELS = ['4','5','6.3','8','10','12.5','16','20','25','31.5','40','50','63','80']

def trapezoid_integral(y, x):
    """NumPy-version-compatible trapezoidal integration.

    NumPy 2.x removed np.trapz in favour of np.trapezoid. Keep a fallback
    for older NumPy releases so the application works across both.
    """
    fn=getattr(np,'trapezoid',None)
    if fn is not None:
        return fn(y,x)
    return np.trapz(y,x)

def set_gordon_xaxis(ax, rotate=45):
    """Use log spacing but label every Gordon one-third-octave centre frequency."""
    ax.set_xscale('log')
    ax.set_xlim(3.5, 90)
    ax.set_xticks(GORDON_FC)
    ax.set_xticklabels(GORDON_TICK_LABELS, rotation=rotate, ha='right' if rotate else 'center')
    ax.minorticks_off()


def gordon_office_velocity_um_s(fc):
    """Return the Gordon office criterion as RMS velocity at each band centre."""
    # Below 8 Hz the criterion is constant acceleration; from 8 Hz upward it is
    # constant velocity (400 um/s). Convert through acceleration to join them.
    f=np.asarray(fc,float); v=400.0; a8=2*np.pi*8*(v*1e-6)
    return np.where(f<8,a8/(2*np.pi*f)*1e6,v)

def periodogram(x,fs):
    """Return a one-sided, Hann-windowed PSD and its intermediate FFT stages."""
    x=np.asarray(x,float); n=len(x); x0=x-np.mean(x); w=np.hanning(n); xw=x0*w
    X=np.fft.rfft(xw); f=np.fft.rfftfreq(n,1/fs); p=np.abs(X)**2/(fs*np.sum(w*w))
    # A real FFT contains only the non-negative half of the spectrum. Double
    # interior bins to retain the omitted negative-frequency power, excluding
    # DC and (for even n) the Nyquist bin.
    if n%2==0 and len(p)>2:p[1:-1]*=2
    elif n%2 and len(p)>1:p[1:]*=2
    return f,p,x0,w,xw,X

def rolling_mean(x,n):
    """Centred moving average with edge padding and unchanged output length."""
    n=max(1,int(n)); x=np.asarray(x,float)
    if n==1:return x.copy()
    l=n//2; r=n-1-l; return np.convolve(np.pad(x,(l,r),mode='edge'),np.ones(n)/n,mode='valid')

@dataclass
class Cols:
    """Original DataFrame labels assigned to the canonical signal roles."""

    time:str|None=None; x:str|None=None; y:str|None=None; z:str|None=None


def _column_lookup(df, aliases):
    """Case-insensitive exact/substring column lookup used by CSV normalisation."""
    cols=list(df.columns)
    low={str(c).strip().lower():c for c in cols}
    for alias in aliases:
        if alias in low:
            return low[alias]
    for c in cols:
        cl=str(c).strip().lower()
        if any(alias in cl for alias in aliases):
            return c
    return None


def detect_accelerometer_csv_format(df):
    low={str(c).strip().lower() for c in df.columns}
    if {'x_counts','y_counts','z_counts'}.issubset(low):
        return 'raw_vmm_counts'
    # Frequency-domain baseline exported from the laboratory reference data.
    freq_aliases={'frequency_hz','freq_hz','frequency','freq','hz'}
    psd_aliases={'psd_g2_per_hz','psd_g^2/hz','psd_g2_hz','psd','g2_per_hz'}
    if low.intersection(freq_aliases) and low.intersection(psd_aliases):
        return 'baseline_psd'
    x=_column_lookup(df,['x_g','accel_x','acc_x'])
    y=_column_lookup(df,['y_g','accel_y','acc_y'])
    z=_column_lookup(df,['z_g','accel_z','acc_z'])
    if all((x,y,z)):
        return 'converted_g'
    return 'unknown'


def normalize_accelerometer_dataframe(df, fallback_fs=DEFAULT_FS):
    """Return canonical time_s/X_g/Y_g/Z_g data plus source metadata and warnings."""
    source=df.copy().reset_index(drop=True)
    fmt=detect_accelerometer_csv_format(source)
    warnings=[]
    meta={'source_format':fmt,'original_columns':[str(c) for c in source.columns]}

    if fmt=='raw_vmm_counts':
        names={str(c).strip().lower():c for c in source.columns}
        xc,yc,zc=(names['x_counts'],names['y_counts'],names['z_counts'])
        counts=[pd.to_numeric(source[c],errors='coerce').to_numpy(float) for c in (xc,yc,zc)]

        fs_col=names.get('fs_g')
        if fs_col is not None:
            fs_g=pd.to_numeric(source[fs_col],errors='coerce').to_numpy(float)
            valid_fs=np.isfinite(fs_g)&(fs_g>0)
            if not np.all(valid_fs):
                raise ValueError('Raw VMM CSV contains invalid or missing fs_g values.')
            uniq_fs=np.unique(fs_g)
            meta['reported_fs_g']=[float(v) for v in uniq_fs]
            if len(uniq_fs)>1:
                warnings.append('Full scale changes within the raw VMM file; row-by-row scaling was applied.')
        else:
            fs_g=np.full(len(source),float(fallback_fs)*0+2.0)
            meta['reported_fs_g']=[2.0]
            warnings.append('Raw VMM CSV has no fs_g column; assumed ±2 g.')

        out=pd.DataFrame({
            'X_g':counts[0]*fs_g/32768.0,
            'Y_g':counts[1]*fs_g/32768.0,
            'Z_g':counts[2]*fs_g/32768.0,
        })

        ts_col=names.get('sample_timestamp_us')
        period_col=names.get('sample_period_us')
        if ts_col is not None:
            ts=pd.to_numeric(source[ts_col],errors='coerce').to_numpy(float)
            good=np.isfinite(ts)
            if good.sum()<2:
                raise ValueError('Raw VMM sample_timestamp_us does not contain enough valid timestamps.')
            first=ts[np.flatnonzero(good)[0]]
            out.insert(0,'time_s',(ts-first)/1_000_000.0)
        elif period_col is not None:
            per=pd.to_numeric(source[period_col],errors='coerce').to_numpy(float)
            if not np.all(np.isfinite(per)&(per>0)):
                raise ValueError('Raw VMM sample_period_us contains invalid values.')
            t=np.zeros(len(source),float)
            if len(t)>1:
                t[1:]=np.cumsum(per[:-1])/1_000_000.0
            out.insert(0,'time_s',t)
            warnings.append('sample_timestamp_us absent; time reconstructed from sample_period_us.')
        else:
            out.insert(0,'time_s',np.arange(len(source),dtype=float)/float(fallback_fs))
            warnings.append(f'No raw timestamp fields found; time reconstructed using fallback {float(fallback_fs):g} Hz.')

        t=pd.to_numeric(out['time_s'],errors='coerce').to_numpy(float)
        dt=np.diff(t); pos=dt[np.isfinite(dt)&(dt>0)]
        if len(pos):
            med=float(np.median(pos)); estimated_fs=1.0/med
            gap_count=int(np.sum(pos>1.5*med))
        else:
            estimated_fs=float(fallback_fs); gap_count=0
        meta['estimated_fs_hz']=estimated_fs
        meta['timestamp_gap_count']=gap_count
        if gap_count:
            warnings.append(f'{gap_count} timestamp gap(s) detected.')

        odr_col=names.get('odr_hz')
        if odr_col is not None:
            odr=pd.to_numeric(source[odr_col],errors='coerce').dropna().to_numpy(float)
            uniq=np.unique(odr[odr>0]) if len(odr) else np.array([])
            meta['reported_odr_hz']=[float(v) for v in uniq]
            if len(uniq)>1:
                warnings.append('ODR changes within the raw VMM file; timestamps were retained as the time source.')
            if len(uniq)==1 and abs(estimated_fs-float(uniq[0]))>max(0.5,0.02*float(uniq[0])):
                warnings.append(f'Timestamp-derived sample rate {estimated_fs:.3f} Hz differs from reported ODR {float(uniq[0]):g} Hz.')
        else:
            meta['reported_odr_hz']=[]

        status_col=names.get('status')
        fault_count=0
        if status_col is not None:
            status=pd.to_numeric(source[status_col],errors='coerce').fillna(0).astype(np.int64)
            nz=status!=0
            block_col=names.get('block_seq')
            if block_col is not None:
                fault_count=int(source.loc[nz,block_col].nunique())
            else:
                fault_count=int(nz.sum())
        meta['status_fault_count']=fault_count
        if fault_count:
            warnings.append(f'{fault_count} VMM block(s) contain non-zero status flags.')
        return out,Cols('time_s','X_g','Y_g','Z_g'),estimated_fs,meta,warnings

    if fmt=='converted_g':
        tc=_column_lookup(source,['time_s','time','timestamp'])
        xc=_column_lookup(source,['x_g','accel_x','acc_x'])
        yc=_column_lookup(source,['y_g','accel_y','acc_y'])
        zc=_column_lookup(source,['z_g','accel_z','acc_z'])
        out=pd.DataFrame({
            'X_g':pd.to_numeric(source[xc],errors='coerce'),
            'Y_g':pd.to_numeric(source[yc],errors='coerce'),
            'Z_g':pd.to_numeric(source[zc],errors='coerce'),
        })
        if tc is not None:
            t=pd.to_numeric(source[tc],errors='coerce').to_numpy(float)
            good=t[np.isfinite(t)]
            first=float(good[0]) if len(good) else 0.0
            out.insert(0,'time_s',t-first)
            dt=np.diff(t); pos=dt[np.isfinite(dt)&(dt>0)]
            fs=float(1.0/np.median(pos)) if len(pos) else float(fallback_fs)
            meta['timestamp_gap_count']=int(np.sum(pos>1.5*np.median(pos))) if len(pos) else 0
        else:
            fs=float(fallback_fs)
            out.insert(0,'time_s',np.arange(len(source),dtype=float)/fs)
            warnings.append(f'Converted CSV has no usable time column; time reconstructed using {fs:g} Hz.')
            meta['timestamp_gap_count']=0
        meta['estimated_fs_hz']=fs
        meta['reported_fs_g']=[]; meta['reported_odr_hz']=[]; meta['status_fault_count']=0
        return out,Cols('time_s','X_g','Y_g','Z_g'),fs,meta,warnings

    raise ValueError('Unsupported CSV format. Expected converted X_g/Y_g/Z_g data or raw VMM x_counts/y_counts/z_counts data.')


def infer_excitation_axis_from_filename(path):
    """Infer X/Y/Z excitation axis from a baseline filename.

    Strong patterns (axis_x, x_axis, standalone X, suffix X) are preferred so
    ordinary letters inside words such as "dummy" do not create false matches.
    Returns 'X', 'Y', 'Z', or None when the name is ambiguous.
    """
    stem=Path(path).stem.upper()
    candidates=[]
    patterns=[
        r'(?:^|[^A-Z0-9])AXIS[ _-]*([XYZ])(?:$|[^A-Z0-9])',
        r'(?:^|[^A-Z0-9])([XYZ])[ _-]*AXIS(?:$|[^A-Z0-9])',
        r'(?:^|[^A-Z0-9])([XYZ])(?:$|[^A-Z0-9])',
        r'([XYZ])$',
        r'^([XYZ])',
    ]
    for pat in patterns:
        candidates.extend(re.findall(pat,stem))
    uniq=[]
    for c in candidates:
        if c not in uniq:
            uniq.append(c)
    return uniq[0] if len(uniq)==1 else None



class BaselineSpectrum:
    """Frequency-domain reference baseline containing one driven-axis acceleration PSD."""

    def __init__(self,label):
        self.label=label
        self.source=''
        self.metadata={}
        self.freq_hz=np.array([],dtype=float)
        self.psd_g2_per_hz=np.array([],dtype=float)

    @property
    def loaded(self):
        return self.freq_hz.size>1 and self.psd_g2_per_hz.size==self.freq_hz.size

    def set(self,freq_hz,psd_g2_per_hz,source='',metadata=None):
        f=np.asarray(freq_hz,dtype=float)
        p=np.asarray(psd_g2_per_hz,dtype=float)
        good=np.isfinite(f)&np.isfinite(p)&(f>=0)&(p>=0)
        f=f[good]; p=p[good]
        if f.size<2:
            raise ValueError('Baseline PSD must contain at least two valid frequency/PSD rows.')
        order=np.argsort(f); f=f[order]; p=p[order]
        # Collapse duplicate frequencies by averaging their PSD values.
        uf,inv=np.unique(f,return_inverse=True)
        if uf.size!=f.size:
            sums=np.zeros(uf.size,float); counts=np.zeros(uf.size,float)
            np.add.at(sums,inv,p); np.add.at(counts,inv,1)
            f=uf; p=sums/counts
        self.freq_hz=f; self.psd_g2_per_hz=p; self.source=source; self.metadata=dict(metadata or {})

    def bands(self):
        rows=[]
        f=self.freq_hz; p=self.psd_g2_per_hz
        if f.size<2:
            raise ValueError('Baseline PSD is empty.')
        for fc in GORDON_FC:
            lo,hi=fc/OCT_EDGE,fc*OCT_EDGE
            # Include interpolated values at exact band edges when covered by the source spectrum.
            m=(f>lo)&(f<hi)
            fb=f[m]; pb=p[m]
            if f[0]<=lo<=f[-1]:
                fb=np.r_[lo,fb]; pb=np.r_[np.interp(lo,f,p),pb]
            if f[0]<=hi<=f[-1]:
                fb=np.r_[fb,hi]; pb=np.r_[pb,np.interp(hi,f,p)]
            if fb.size>=2:
                ms=float(trapezoid_integral(pb,fb))
            else:
                ms=0.0
            ag=math.sqrt(max(ms,0.0))
            v=ag*G0/(2*np.pi*fc)*1e6
            x=v/(2*np.pi*fc)
            rows.append((fc,ag,v,x))
        return pd.DataFrame(rows,columns=['fc_hz','a_rms_g','v_rms_um_s','x_rms_um'])


@dataclass
class TranslationResult:
    """One driven-axis translation result shared by Issue 1 and Issue 2."""
    method: str
    driven_axis: str
    coordinates_hz: np.ndarray
    k_values: np.ndarray
    reference_source: str
    reader_source: str
    reference_domain: str
    reference_axis: str
    settings: dict
    metadata: dict

    def rows(self):
        coord_name='frequency_hz' if self.method==ISSUE1_METHOD else 'fc_hz'
        rows=[]
        for coordinate,k in zip(self.coordinates_hz,self.k_values):
            row={
                'processing_method':self.method,
                'driven_axis':self.driven_axis,
                'reference_axis':self.reference_axis,
                'reference_source_file':Path(self.reference_source).name if self.reference_source else '',
                'reader_source_file':Path(self.reader_source).name if self.reader_source else '',
                'reference_domain':self.reference_domain,
                coord_name:float(coordinate),
                'K_base_over_reader':float(k),
            }
            rows.append(row)
        return rows


def load_baseline_spectrum_dataframe(df,source=''):
    """Parse a frequency-domain baseline CSV and return BaselineSpectrum plus warnings."""
    freq=_column_lookup(df,['frequency_hz','freq_hz','frequency','freq','hz'])
    psd=_column_lookup(df,['psd_g2_per_hz','psd_g^2/hz','psd_g2_hz','g2_per_hz','psd'])
    if freq is None or psd is None:
        raise ValueError('PSD baseline CSV must contain frequency_hz and psd_g2_per_hz columns.')
    f=pd.to_numeric(df[freq],errors='coerce').to_numpy(float)
    q=pd.to_numeric(df[psd],errors='coerce').to_numpy(float)
    spec=BaselineSpectrum('Baseline spectrum')
    spec.set(f,q,source=source,metadata={'source_format':'baseline_psd','frequency_column':str(freq),'psd_column':str(psd)})
    warnings=[]
    lo_needed=float(GORDON_FC[0]/OCT_EDGE); hi_needed=float(GORDON_FC[-1]*OCT_EDGE)
    if spec.freq_hz[0]>lo_needed or spec.freq_hz[-1]<hi_needed:
        warnings.append(f'PSD frequency coverage is {spec.freq_hz[0]:.3g}-{spec.freq_hz[-1]:.3g} Hz; full Gordon integration ideally covers about {lo_needed:.3g}-{hi_needed:.3g} Hz.')
    return spec,warnings


def baseline_orientation(path):
    name=Path(path).stem.lower()
    lateral='lateral' in name
    vertical='vertical' in name
    return ('Lateral' if lateral else 'Vertical') if lateral != vertical else None


def load_testhouse_table(table, source='', include_ref=False):
    """Read paired Hz/magnitude columns without conflating commanded and measured PSD.

    Ctl is always the authoritative measured baseline used for translation.  Ref may
    optionally be retained for display/diagnostic comparison only.  Alarm and CA
    channels are intentionally ignored.
    """
    names={'ctl':'Ctl'}
    if include_ref:
        names['ref']='Ref'
    header=None
    for i in range(min(30,len(table)-1)):
        labels=[str(v).strip().lower() for v in table.iloc[i]]
        if any(v in names for v in labels):
            header=i; break
    if header is None:
        raise ValueError('Unsupported test-house layout: Missing Ctl frequency/PSD columns.')
    traces={}
    units=[str(v).strip().lower() for v in table.iloc[header+1]]
    for j,value in enumerate(table.iloc[header]):
        key=str(value).strip().lower()
        if key not in names: continue
        name=names.get(key,str(value).strip())
        # Test-house names sit above magnitude; also accept names above the Hz column.
        k=j if units[j] in ('hz','frequency_hz','frequency (hz)') else j-1
        if k<0 or k+1>=len(units) or units[k] not in ('hz','frequency_hz','frequency (hz)') or not ('magnitude' in units[k+1] or 'psd' in units[k+1]):
            raise ValueError(f'Unsupported {name} layout: expected paired Hz and PSD magnitude columns.')
        rows=table.iloc[header+2:,[k,k+1]].dropna(how='all')
        values=rows.apply(pd.to_numeric,errors='coerce').to_numpy(float)
        if len(values)<2 or not np.isfinite(values).all():
            raise ValueError(f'{name}: at least two numeric frequency/PSD rows are required; non-numeric or missing values found.')
        f,q=values.T
        if (f<0).any() or (q<0).any(): raise ValueError(f'{name}: negative frequency or PSD values are invalid.')
        if len(np.unique(f))!=len(f): raise ValueError(f'{name}: duplicate frequencies are not supported.')
        order=np.argsort(f); traces[name]=(f[order],q[order])
    if 'Ctl' not in traces: raise ValueError('Missing Ctl frequency/PSD columns; Ref cannot be used for correction.')
    f,q=traces['Ctl']
    if f[-1]<GORDON_FC[0]/OCT_EDGE or f[0]>GORDON_FC[-1]*OCT_EDGE:
        raise ValueError('Insufficient frequency coverage: Ctl does not overlap the Gordon bands.')
    spec,warnings=load_baseline_spectrum_dataframe(pd.DataFrame({'frequency_hz':f,'psd_g2_per_hz':q}),source)
    spec.traces=traces
    spec.metadata.update({'selected_source':'Ctl','orientation':baseline_orientation(source),
                          'baseline_type':'PSD Excel' if Path(source).suffix.lower()=='.xlsx' else 'PSD CSV',
                          'frequency_range_hz':[float(f[0]),float(f[-1])]})
    return spec,warnings


def _normalise_excel_time_sheet(table,fs,label,path):
    """Try to interpret a worksheet as a normal time-domain accelerometer table."""
    if table.empty:
        return None
    frame=table.iloc[1:].copy(); frame.columns=table.iloc[0].tolist()
    if detect_accelerometer_csv_format(frame)=='unknown':
        return None
    normalized,cols,rate,meta,warnings=normalize_accelerometer_dataframe(frame,fs)
    proc=Processor(label); proc.set(normalized,cols,rate,path,metadata=meta,source_df=frame)
    return proc,warnings


def read_baseline_file(path,fs=200.0,include_ref=True):
    if Path(path).suffix.lower()=='.xlsx':
        sheets=pd.read_excel(path,header=None,sheet_name=None,engine='openpyxl')
        candidates=[(name,t) for name,t in sheets.items() if t.astype(str).apply(lambda col: col.str.strip().str.lower().eq('ctl')).any().any()]
        if candidates:
            if len(candidates)!=1:
                raise ValueError('Unsupported workbook layout: expected exactly one sheet containing Ctl frequency/PSD columns.')
            name,table=candidates[0]
            spec,warnings=load_testhouse_table(table,path,include_ref=include_ref); spec.metadata['worksheet']=name
            return spec,warnings
        time_candidates=[]
        for name,table in sheets.items():
            parsed=_normalise_excel_time_sheet(table,fs,'Baseline',path)
            if parsed is not None: time_candidates.append((name,parsed))
        if len(time_candidates)!=1:
            raise ValueError('Unsupported baseline workbook: expected one Ctl PSD sheet or one time-domain X/Y/Z accelerometer sheet.')
        name,(proc,warnings)=time_candidates[0]
        proc.metadata.update({'worksheet':name,'baseline_type':'time-domain Excel'})
        return proc,warnings
    table=pd.read_csv(path,header=None)
    if table.iloc[:30].astype(str).apply(lambda col: col.str.strip().str.lower().isin(['ctl','ref','alarm-','alarm+'])).any().any():
        return load_testhouse_table(table,path,include_ref=include_ref)
    df=pd.read_csv(path)
    if detect_accelerometer_csv_format(df)=='baseline_psd':
        spec,warnings=load_baseline_spectrum_dataframe(df,path)
        spec.metadata.update({'baseline_type':'PSD CSV','selected_source':'psd_g2_per_hz',
                              'frequency_range_hz':[float(spec.freq_hz[0]),float(spec.freq_hz[-1])]})
        return spec,warnings
    normalized,cols,rate,meta,warnings=normalize_accelerometer_dataframe(df,fs)
    proc=Processor('Baseline'); meta=dict(meta); meta['baseline_type']='time-domain CSV'
    proc.set(normalized,cols,rate,path,metadata=meta,source_df=df)
    return proc,warnings


def read_reader_file(path,fs=DEFAULT_FS):
    """Load reader time samples or a test-house PSD using the existing parsers."""
    if Path(path).suffix.lower()=='.xlsx':
        sheets=pd.read_excel(path,header=None,sheet_name=None,engine='openpyxl')
        spectra=[name for name,t in sheets.items()
                 if t.iloc[:30].astype(str).apply(lambda col: col.str.strip().str.lower().isin(['ctl','ref'])).any().any()]
        if spectra:
            if len(spectra)!=1:
                raise ValueError('Reader workbook contains multiple PSD sheets; use a workbook with one dataset.')
            name=spectra[0]
            proc,warnings=load_testhouse_table(sheets[name],path,include_ref=False)
        else:
            candidates=[]
            for name,table in sheets.items():
                if table.empty: continue
                frame=table.iloc[1:].copy(); frame.columns=table.iloc[0].tolist()
                if detect_accelerometer_csv_format(frame)!='unknown': candidates.append((name,frame))
            if len(candidates)!=1:
                raise ValueError('Unsupported reader workbook: expected one sheet with Ctl/Ref PSD pairs, frequency_hz/psd_g2_per_hz, or reader X/Y/Z acceleration columns.')
            name,df=candidates[0]
            if detect_accelerometer_csv_format(df)=='baseline_psd':
                proc,warnings=load_baseline_spectrum_dataframe(df,path)
                proc.metadata.update({'selected_source':'psd_g2_per_hz','frequency_range_hz':[float(proc.freq_hz[0]),float(proc.freq_hz[-1])]})
            else:
                normalized,cols,rate,meta,warnings=normalize_accelerometer_dataframe(df,fs)
                proc=Processor('Reader'); proc.set(normalized,cols,rate,path,metadata=meta,source_df=df)
        proc.metadata['worksheet']=name
    else:
        proc,warnings=read_baseline_file(path,fs,include_ref=False)
    proc.label='Reader'
    proc.metadata.pop('baseline_type',None)
    proc.metadata['reader_type']=('PSD' if isinstance(proc,BaselineSpectrum) else 'time-domain') + (' Excel' if Path(path).suffix.lower()=='.xlsx' else ' CSV')
    return proc,warnings


class Processor:
    """A single three-axis time history and its signal-processing operations.

    ``original_df`` is immutable working provenance. Cropping changes ``df``
    only, allowing every new crop or export to begin from the imported samples.
    """

    def __init__(self,label):
        self.label=label; self.df=None; self.original_df=None; self.source_df=None; self.cols=Cols(); self.fs=DEFAULT_FS; self.source=''; self.metadata={}
    @property
    def loaded(self): return self.df is not None and all([self.cols.x,self.cols.y,self.cols.z])
    def set(self,df,cols,fs,source='',metadata=None,source_df=None):
        """Load canonical data while retaining both original and source tables."""
        self.original_df=df.copy().reset_index(drop=True)
        self.df=self.original_df.copy()
        self.source_df=(source_df.copy().reset_index(drop=True) if source_df is not None else self.original_df.copy())
        self.cols=cols; self.fs=float(fs); self.source=source; self.metadata=dict(metadata or {})
    def restore_original(self):
        if self.original_df is not None:
            self.df=self.original_df.copy()
    def time(self):
        if self.cols.time:
            t=pd.to_numeric(self.df[self.cols.time],errors='coerce').to_numpy(float)
            if np.isfinite(t).sum()>1:return t
        return np.arange(len(self.df))/self.fs
    def axis(self,a):
        c={'X':self.cols.x,'Y':self.cols.y,'Z':self.cols.z}[a]
        return pd.to_numeric(self.df[c],errors='coerce').to_numpy(float)
    def xyz(self):
        """Return aligned finite time/X/Y/Z arrays, dropping any bad row."""
        t=self.time(); x,y,z=[self.axis(a) for a in 'XYZ']; m=np.isfinite(t)&np.isfinite(x)&np.isfinite(y)&np.isfinite(z)
        return t[m],x[m],y[m],z[m]
    def timebase(self):
        """Summarise measured timing, jitter, and gaps for diagnostics."""
        t,*_=self.xyz(); dt=np.diff(t); g=dt[np.isfinite(dt)&(dt>0)]
        if len(g)==0:return {}
        med=float(np.median(g)); return {'samples':len(t),'duration_s':float(t[-1]-t[0]),'median_fs_hz':1/med,'jitter_std_s':float(np.std(g-med)),'gaps':int(np.sum(g>1.5*med))}
    def welch(self,a,start,n,overlap,segs):
        """Calculate and average ``segs`` overlapping periodograms for one axis."""
        arr=self.axis(a); hop=int(round(n*(1-overlap))); need=n+(segs-1)*hop
        if start<0 or start+need>len(arr): raise ValueError(f'{self.label}: need {need} samples from start {start}')
        ps=[]; xs=[]; starts=[]
        for i in range(segs):
            s=start+i*hop; f,p,*rest=periodogram(arr[s:s+n],self.fs); ps.append(p); xs.append(rest[-1]); starts.append(s)
        return f,np.vstack(ps),np.mean(ps,axis=0),starts
    def stage(self,a,start,n):
        """Expose each processing stage for the GUI's teaching/diagnostic plot."""
        arr=self.axis(a)
        if start<0 or start+n>len(arr):raise ValueError(f'{self.label}: selected block outside data')
        f,p,x0,w,xw,X=periodogram(arr[start:start+n],self.fs)
        return {'raw':arr[start:start+n],'mean_removed':x0,'window':w,'windowed':xw,'freq':f,'fft':X,'fft_mag':np.abs(X),'psd':p}
    def bands(self,f,p):
        """Integrate a PSD into Gordon one-third-octave RMS engineering units."""
        df=f[1]-f[0]; rows=[]
        for fc in GORDON_FC:
            lo,hi=fc/OCT_EDGE,fc*OCT_EDGE; m=(f>=lo)&(f<hi); ms=float(np.sum(p[m])*df) if np.any(m) else 0
            ag=math.sqrt(max(ms,0)); v=ag*G0/(2*np.pi*fc)*1e6; x=v/(2*np.pi*fc)
            rows.append((fc,ag,v,x))
        return pd.DataFrame(rows,columns=['fc_hz','a_rms_g','v_rms_um_s','x_rms_um'])
    def tilt(self,avg_s):
        t,x,y,z=self.xyz(); n=max(1,int(round(avg_s*self.fs))); xa,ya,za=[rolling_mean(q,n) for q in [x,y,z]]
        roll=np.degrees(np.arctan2(ya,za)); pitch=np.degrees(np.arctan2(-xa,np.sqrt(ya*ya+za*za)))
        return t,xa,ya,za,roll,pitch,n

class PairAnalysis:
    """Paired baseline/reader transfer and correction calculations."""

    def __init__(self,base,reader): self.base=base; self.reader=reader
    def check(self):
        if not(self.base.loaded and self.reader.loaded): raise ValueError('Load both baseline and reader datasets.')
        if abs(self.base.fs-self.reader.fs)>1e-6: raise ValueError('Baseline and reader sample rates must match.')
    def paired_band_window(self,start,n,overlap,segs):
        """Compare both sensors over one identical multi-segment time span."""
        self.check(); out=[]
        for a in 'XYZ':
            fb,_,pb,_=self.base.welch(a,start,n,overlap,segs); fr,_,pr,_=self.reader.welch(a,start,n,overlap,segs)
            bb=self.base.bands(fb,pb); rr=self.reader.bands(fr,pr); d=bb.copy();
            d['axis']=a; d['reader_a_rms_g']=rr.a_rms_g; d['reader_v_rms_um_s']=rr.v_rms_um_s; d['reader_x_rms_um']=rr.x_rms_um
            d['H_reader_over_base']=np.divide(rr.a_rms_g,bb.a_rms_g,out=np.full(len(bb),np.nan),where=bb.a_rms_g>1e-12)
            d['K_base_over_reader']=np.divide(bb.a_rms_g,rr.a_rms_g,out=np.full(len(bb),np.nan),where=rr.a_rms_g>1e-12)
            out.append(d)
        return pd.concat(out,ignore_index=True)
    def series(self,n,overlap,segs):
        """Evaluate consecutive non-overlapping analysis spans for model fitting."""
        self.check(); hop=int(round(n*(1-overlap))); span=n+(segs-1)*hop; total=min(len(self.base.df),len(self.reader.df)); starts=list(range(0,total-span+1,span))
        frames=[]
        for wi,s in enumerate(starts):
            d=self.paired_band_window(s,n,overlap,segs); d['window']=wi; d['start_sample']=s; frames.append(d)
        if not frames: raise ValueError('Not enough paired samples for a complete analysis window.')
        return pd.concat(frames,ignore_index=True)
    def model_comparison(self,n,overlap,segs,driven_axis):
        """
        Compare translation models only for the intentionally excited axis.

        A single-axis characterisation run measures X/Y/Z on the reader, but
        only the driven axis is used to derive the base<->reader transfer
        correction. The other two axes remain available as cross-axis
        response diagnostics.
        """
        d=self.series(n,overlap,segs)
        d=d[d.axis==driven_axis].copy()
        wins=np.sort(d.window.unique())
        split=max(1,int(math.ceil(len(wins)*0.6)))
        train_w=wins[:split]
        val_w=wins[split:] if split<len(wins) else wins[-1:]
        train=d[d.window.isin(train_w)].copy()
        val=d[d.window.isin(val_w)].copy()

        r=train.K_base_over_reader.replace([np.inf,-np.inf],np.nan).dropna()
        common=float(r.median())
        band=train.groupby('fc_hz').K_base_over_reader.median().to_dict()

        def score(kind):
            errs=[]
            for _,q in val.iterrows():
                if q.a_rms_g<=1e-12:
                    continue
                k=common if kind=='Single K' else band[q.fc_hz]
                pred=k*q.reader_a_rms_g
                errs.append(abs(pred-q.a_rms_g)/q.a_rms_g*100)
            e=np.array(errs,float)
            if len(e)==0:
                return np.nan,np.nan,np.nan
            return float(np.nanmedian(e)),float(np.nanpercentile(e,95)),float(np.sqrt(np.nanmean(e*e)))

        rows=[]
        for kind in ['Single K','Band-specific K']:
            med,p95,rmse=score(kind)
            rows.append({'model':kind,'median_abs_error_pct':med,'p95_abs_error_pct':p95,'rmse_pct':rmse})
        scores=pd.DataFrame(rows)

        # Use the simplest model where its validation error is acceptable.
        rec='Band-specific K'
        single_p95=scores.loc[scores.model=='Single K','p95_abs_error_pct'].iloc[0]
        if np.isfinite(single_p95) and single_p95<=10:
            rec='Single K'

        return scores,rec,common,band,d

    def h1(self,a,n,overlap):
        """Estimate reader/base H1 transfer function and magnitude-squared coherence."""
        self.check(); xb=self.base.axis(a); yr=self.reader.axis(a); total=min(len(xb),len(yr)); hop=int(round(n*(1-overlap)))
        Gxx=None; Gyx=None; Gyy=None; count=0; w=np.hanning(n); U=np.sum(w*w)
        for s in range(0,total-n+1,hop):
            x=(xb[s:s+n]-np.mean(xb[s:s+n]))*w; y=(yr[s:s+n]-np.mean(yr[s:s+n]))*w
            X=np.fft.rfft(x); Y=np.fft.rfft(y); xx=X*np.conj(X); yx=Y*np.conj(X); yy=Y*np.conj(Y)
            Gxx=xx if Gxx is None else Gxx+xx; Gyx=yx if Gyx is None else Gyx+yx; Gyy=yy if Gyy is None else Gyy+yy; count+=1
        # H1 = cross(reader, base) / auto(base). Coherence indicates which
        # frequencies have enough linear relationship for H1 to be meaningful.
        Gxx/=max(count,1); Gyx/=max(count,1); Gyy/=max(count,1); H=np.divide(Gyx,Gxx,out=np.zeros_like(Gyx),where=np.abs(Gxx)>1e-20)
        coh=np.divide(np.abs(Gyx)**2,Gxx.real*Gyy.real,out=np.zeros(len(Gxx)),where=(Gxx.real*Gyy.real)>1e-20)
        f=np.fft.rfftfreq(n,1/self.base.fs); return f,H,np.clip(coh,0,1)

class App:
    """Tkinter controller for offline analysis and live VMM acquisition."""

    def __init__(self,root):
        self.root=root; root.title(APP_TITLE); root.geometry('1500x930'); root.minsize(1150,720)
        self.base=Processor('Baseline'); self.reader=Processor('Reader'); self.pair=PairAnalysis(self.base,self.reader)
        # Axis-specific baseline/reference datasets. Each file represents the
        # reference measurement for one intentionally excited axis. The reader
        # remains one three-axis CSV. A missing baseline may explicitly reuse a
        # different loaded baseline axis via baseline_use_axis.
        self.baselines={axis:None for axis in 'XYZ'}
        # Characterisation reader captures are retained by deliberately driven
        # axis. self.reader remains the currently selected run for backward
        # compatibility with the existing plotting code.
        self.reader_runs={axis:None for axis in 'XYZ'}
        self.baseline_use_axis={axis:tk.StringVar(value=axis) for axis in 'XYZ'}
        self.lateral_assignment=tk.StringVar(value='X and Y')
        self.fs=tk.DoubleVar(value=DEFAULT_FS); self.n=tk.IntVar(value=DEFAULT_N); self.ov=tk.DoubleVar(value=50); self.segs=tk.IntVar(value=4); self.start=tk.IntVar(value=0); self.stage_block=tk.IntVar(value=1); self.stage_axis=tk.StringVar(value='X'); self.transfer_axis=tk.StringVar(value='X'); self.axis=tk.StringVar(value='X'); self.driven_axis=tk.StringVar(value='X'); self.units=tk.StringVar(value='RMS velocity (µm/s)'); self.auto_crop=tk.BooleanVar(value=True); self.offline_shock_window=tk.IntVar(value=50); self.offline_shock_threshold_g=tk.DoubleVar(value=1.0); self.offline_shock_release_s=tk.DoubleVar(value=0.25); self.offline_shock_axis=tk.StringVar(value='X')
        # Dedicated shock-characterisation files. Each X/Y/Z file still contains
        # all three measured accelerometer channels; the filename identifies the
        # intentionally applied shock direction.
        self.shock_baselines={axis:None for axis in 'XYZ'}
        self.shock_readers={axis:None for axis in 'XYZ'}
        # Custom scrollable tab navigation keeps engineering tab labels readable.
        self._tab_nav_buttons=[]
        self._tab_nav_canvas=None
        self._tab_nav_inner=None
        # Final production-correction assembly. Each axis is deliberately stored
        # from its own driven-axis run rather than inferred from orthogonal channels.
        self.stored_correction_axes={axis:None for axis in 'XYZ'}
        self.stored_correction_status=tk.StringVar(value='Final XYZ correction: X — | Y — | Z —')
        self.status=tk.StringVar(value='Load a baseline and reader dataset, or use the paired demo.')

        # Live LIS2DUX12 acquisition from the specialty hw_test_vmm firmware.
        # This is the standalone COBS+CRC32 VMM stream on the ST-LINK VCP; it
        # deliberately does not use ProtoComms, protobuf, SOM HTTP, or JTAG.
        self.live_port=tk.StringVar(value='COM4')
        self.live_baud=tk.IntVar(value=DEFAULT_BAUD)
        self.live_stream_fs=tk.StringVar(value='Unknown')
        # Desired LIS2DUX12 configuration. These defaults match the current
        # recommended vibration-measurement configuration.
        self.accel_device=tk.StringVar(value='LIS2DUX12')
        self.accel_fs_g=tk.IntVar(value=2)
        self.accel_odr_hz=tk.IntVar(value=200)
        self.accel_mode=tk.StringVar(value='High Performance')
        self.accel_aa_enabled=tk.BooleanVar(value=True)
        self.accel_bw=tk.StringVar(value='ODR/2')
        self.accel_fifo_watermark=tk.IntVar(value=32)
        self.accel_int1_fifo_threshold=tk.BooleanVar(value=True)
        self.accel_stream_timeout_ms=tk.IntVar(value=300_000)
        self.accel_prefilter=tk.StringVar(value='None')
        self.accel_fft_n=tk.IntVar(value=512)
        self.accel_fft_window=tk.StringVar(value='Hann')
        self.accel_fft_overlap=tk.DoubleVar(value=50.0)
        self.accel_psd_averages=tk.IntVar(value=4)
        self.processing_method=tk.StringVar(value=ISSUE2_METHOD)
        self.applied_processing_method=ISSUE2_METHOD
        self.processing_method_details=tk.StringVar(
            value='Issue 2: Welch PSD averaging and Gordon one-third-octave integration.')
        self.issue1_waterfall_axis=tk.StringVar(value='X')
        # Read-only summary of the applied host settings. The editable source
        # remains the accel_* controls; fs/n/ov/segs are the shared applied
        # values consumed by both offline and live processing.
        self.processing_status=tk.StringVar()
        self.accel_config_status=tk.StringVar(value='Requested configuration not yet written to STM32')
        self.accel_readback_status=tk.StringVar(value='Live read-back: waiting for sample block')
        self.accel_program_result=tk.StringVar(value='NOT PROGRAMMED')
        self.accel_effective_config=None
        self.accel_pending_config=None
        self.accel_config_ack_ok=False
        self.accel_config_attempt=0

        self.live_view_seconds=tk.DoubleVar(value=12.0)
        self.live_retention_seconds=tk.DoubleVar(value=120.0)
        # User-facing capture timeout is separate from the finite VMM START timeout.
        self.operator_timeout_enabled=tk.BooleanVar(value=True)
        self.operator_timeout_value=tk.DoubleVar(value=5.0)
        self.operator_timeout_units=tk.StringVar(value='minutes')
        self.operator_timeout_action=tk.StringVar(value='Ask')
        self.live_diagnostics=tk.StringVar(value='Capture diagnostics: stopped')
        self.capture_started_monotonic=None
        self.operator_period_started_monotonic=None
        self.current_capture_indefinite=False
        self.operator_timeout_prompt_open=False
        self.timeout_extensions=0
        self.stream_renewals=0
        self.recovery_attempts=0
        self.last_sample_block_monotonic=None
        self.next_vmm_renewal_monotonic=None
        self.vmm_renew_fraction=0.8
        self.live_link=tk.StringVar(value='Disconnected')
        self.live_capture=tk.StringVar(value='Stopped')
        self.live_fw_version=tk.StringVar(value='VMM v1')
        self.live_client=None
        self.live_last_block_seq=None
        self.live_first_timestamp_us=None
        self.live_stream_time_offset=0.0
        self.live_sequence_gaps=0
        self.live_bad_frames=0
        self.live_total_samples=0
        self.raw_logger=RawVmmCsvLogger()
        self.raw_log_status=tk.StringVar(value='Raw log: stopped')
        self.raw_log_metadata=None
        self.raw_log_settings_path=None
        self.last_raw_log_defaults={'reader_id':'','excited_axis':'X','vibration_level':'','run_number':'1','notes':''}
        self.live_queue=queue.Queue()
        self.live_stop_event=threading.Event()
        self.live_thread=None
        self.live_connect_thread=None
        self.live_analysis_fig={}
        self.live_last_view_draw=0.0
        self.live_last_selected_tab=None
        self.live_redraw_pending=False
        self.live_tilt_zero_roll=None
        self.live_tilt_zero_pitch=None
        self.live_tilt_readout=tk.StringVar(value='Waiting for live samples')
        self.tilt_average_samples=tk.IntVar(value=50)
        self.live_tilt_max_dynamic_g=tk.DoubleVar(value=0.05)
        self.live_tilt_min_gravity_g=tk.DoubleVar(value=0.80)
        self.live_tilt_max_gravity_g=tk.DoubleVar(value=1.20)

        # Shock detector per 1174-Y-056 Proposed Issue 2 section 6.2.2.
        # The document specifies a 50-point (0.25 s at 200 Hz) steady-state
        # average and resultant magnitude of the deviations from that average.
        # It requires a configurable threshold but does not prescribe its value.
        self.live_shock_enabled=tk.BooleanVar(value=True)
        self.live_shock_threshold_g=tk.DoubleVar(value=1.0)
        self.live_shock_window_samples=50
        self.live_shock_readout=tk.StringVar(value='Shock: waiting for 50 samples')
        self.live_shock_peak_g=0.0
        self.live_shock_events=0
        self.live_shock_prev_exceeded=False
        self.live_shock_last_processed_time=None

        self.live_threshold_mode=tk.StringVar(value='Gordon Office')
        self.live_threshold_multiplier=tk.DoubleVar(value=1.0)
        self.live_threshold_value=tk.DoubleVar(value=0.002)
        self.live_threshold_status=tk.StringVar(value='Threshold: waiting for PSD data')

        # Optional live installed-system correction. K = Base / Reader is a
        # multiplicative, frequency-band correction applied after Gordon-band
        # RMS integration; raw time-domain acceleration and tilt are unchanged.
        self.live_k_correction_enabled=tk.BooleanVar(value=False)
        self.live_k_status=tk.StringVar(value='K correction: not loaded')
        self.live_k_factors={axis:None for axis in 'XYZ'}
        self.live_k_coordinates=None
        self.live_k_method=None
        self.live_k_source_files=[]

        self.live_t=[]; self.live_x=[]; self.live_y=[]; self.live_z=[]
        self.live_raw_x=[]; self.live_raw_y=[]; self.live_raw_z=[]
        self.live_last_status=0
        self.live_stream_health=tk.StringVar(value='Stream health: waiting for data')
        self.live_packets=0; self.live_dropped_or_invalid=0

        self._update_processing_status()

        # Tk must be touched only by its main thread. This recurring callback is
        # the bridge from serial worker events to UI state and plots.
        self._build()
        self._sync_processing_method_controls()
        self._sync_applied_mode_tabs()
        self.root.after(100,self._poll_live_queue)
        self.root.protocol('WM_DELETE_WINDOW', self._on_close)
    def _build(self):
        header=ttk.Frame(self.root,padding=(6,6,6,0))
        header.pack(fill=tk.X)

        actions=ttk.Frame(header)
        actions.pack(fill=tk.X)
        for txt,cmd in [
            ('Load baseline/reference file(s)',self.load_baseline_csvs),
            ('Load reader CSV/XLSX',lambda:self.load_csv('reader')),
            ('Load paired demo',self.load_demo_pair),
            ('Export derived correction',self.export_correction),
            ('Store driven-axis K',self.store_current_axis_correction),
            ('Export final XYZ K',self.export_stored_xyz_correction),
            ('Save plot',self.save_plot),
        ]:
            button=ttk.Button(actions,text=txt,command=cmd)
            button.pack(side=tk.LEFT,padx=3,pady=(0,2))
            if txt=='Export derived correction':
                self.btn_export_correction=button

        settings=ttk.Frame(header)
        settings.pack(fill=tk.X,pady=(2,6))
        ttk.Label(
            settings,
            textvariable=self.processing_status,
            font=('Segoe UI',9,'bold'),
        ).pack(side=tk.LEFT,padx=(2,10))
        ttk.Label(settings,textvariable=self.stored_correction_status).pack(side=tk.LEFT,padx=(8,4))

        pan=ttk.Panedwindow(self.root,orient=tk.HORIZONTAL)
        pan.pack(fill=tk.BOTH,expand=True)

        # Keep a very narrow control strip visible at all times so the entire
        # offline-control sidebar can be folded away horizontally. This is
        # independent of the vertical collapsible sections within the sidebar.
        side_toggle=ttk.Frame(pan,padding=(1,4))
        left=ttk.Frame(pan,padding=(6,4,4,6))
        right=ttk.Frame(pan,padding=(2,2,6,6))
        pan.add(side_toggle,weight=0)
        pan.add(left,weight=0)
        pan.add(right,weight=1)
        self.main_pan=pan
        self.main_sidebar_frame=left
        self.main_sidebar_collapsed=False
        self.main_sidebar_toggle=ttk.Button(
            side_toggle,text='◀',width=3,
            command=self._toggle_main_sidebar)
        self.main_sidebar_toggle.pack(side=tk.TOP,fill=tk.X)
        ttk.Label(side_toggle,text='Controls').pack(side=tk.TOP,pady=(3,2))
        self.main_side_toggle_strip=side_toggle

        left_scroll=ScrollableFrame(left)
        left_scroll.pack(fill=tk.BOTH,expand=True)
        sidebar=left_scroll.body

        datasets_section=CollapsibleSection(sidebar,'Datasets',expanded=True)
        datasets_section.pack(fill=tk.X,pady=(0,6))
        self.base_label=ttk.Label(datasets_section.body,text='Baseline: not loaded',wraplength=280,justify=tk.LEFT)
        self.base_label.pack(anchor=tk.W,pady=2)
        self.reader_label=ttk.Label(datasets_section.body,text='Reader: not loaded',wraplength=280,justify=tk.LEFT)
        self.reader_label.pack(anchor=tk.W,pady=2)
        self.auto_crop_check=ttk.Checkbutton(
            datasets_section.body,
            text='Auto-crop paired datasets to common usable duration',
            variable=self.auto_crop,command=self.refresh,
        )
        self.auto_crop_check.pack(anchor=tk.W,pady=(5,1))
        ttk.Label(
            datasets_section.body,
            text='When enabled, the original files are preserved. Analysis is cropped to the common overlapping time/sample range and then to a whole number of complete PSD-analysis spans.',
            wraplength=280,justify=tk.LEFT,
        ).pack(anchor=tk.W,pady=(0,3))

        mapping_section=CollapsibleSection(sidebar,'Correction baseline mapping',expanded=True)
        mapping_section.pack(fill=tk.X,pady=(0,6))
        ttk.Label(mapping_section.body,text='Lateral reference assignment').pack(anchor=tk.W,pady=(0,1))
        ttk.Combobox(
            mapping_section.body,textvariable=self.lateral_assignment,
            values=['X and Y','X only','Y only'],state='readonly'
        ).pack(fill=tk.X,pady=(0,4))
        for target_axis in 'XYZ':
            mr=ttk.Frame(mapping_section.body)
            mr.pack(fill=tk.X,pady=1)
            ttk.Label(mr,text=f'{target_axis} correction uses').pack(side=tk.LEFT)
            mcb=ttk.Combobox(mr,textvariable=self.baseline_use_axis[target_axis],values=list('XYZ'),state='readonly',width=5)
            mcb.pack(side=tk.RIGHT)
            mcb.bind('<<ComboboxSelected>>',lambda e,a=target_axis:self._on_baseline_mapping_changed(a))
        ttk.Label(
            mapping_section.body,
            text='Lateral Ctl → X and Y; Vertical Ctl → Z. Legacy files use X→X, Y→Y, Z→Z. Select another source only as an explicit surrogate.',
            wraplength=280,justify=tk.LEFT,
        ).pack(anchor=tk.W,pady=(3,2))

        analysis_section=CollapsibleSection(sidebar,'Common analysis controls',expanded=True)
        analysis_section.pack(fill=tk.X,pady=(0,6))
        r=ttk.Frame(analysis_section.body)
        r.pack(fill=tk.X,pady=2)
        ttk.Label(r,text='Analysis start sample').pack(side=tk.LEFT)
        ttk.Entry(r,textvariable=self.start,width=9).pack(side=tk.RIGHT)
        ttk.Label(analysis_section.body,text='Gordon units').pack(anchor=tk.W,pady=(7,1))
        self.units_combo=ttk.Combobox(
            analysis_section.body,textvariable=self.units,
            values=['RMS velocity (µm/s)','RMS acceleration (g)','RMS displacement (µm)'],
            state='readonly')
        self.units_combo.pack(fill=tk.X)
        self.units_combo.bind('<<ComboboxSelected>>',self._display_units_changed)

        # Horizontally scrollable tab strip. The native ttk.Notebook headers
        # become compressed once many engineering views are present, so keep
        # full-size labels in a scrollable strip while Notebook manages pages.
        tab_nav=ttk.Frame(right)
        tab_nav.pack(fill=tk.X,pady=(0,2))
        ttk.Button(tab_nav,text='◀',width=3,command=lambda:self._scroll_tab_strip(-1)).pack(side=tk.LEFT,padx=(0,2))
        nav_holder=ttk.Frame(tab_nav)
        nav_holder.pack(side=tk.LEFT,fill=tk.X,expand=True)
        ttk.Button(tab_nav,text='▶',width=3,command=lambda:self._scroll_tab_strip(1)).pack(side=tk.RIGHT,padx=(2,0))
        self._tab_nav_canvas=tk.Canvas(nav_holder,height=30,highlightthickness=0,borderwidth=0)
        self._tab_nav_canvas.pack(fill=tk.X,expand=True)
        self._tab_nav_inner=ttk.Frame(self._tab_nav_canvas)
        self._tab_nav_window=self._tab_nav_canvas.create_window((0,0),window=self._tab_nav_inner,anchor='nw')
        self._tab_nav_inner.bind('<Configure>',lambda e:self._tab_nav_canvas.configure(scrollregion=self._tab_nav_canvas.bbox('all')))
        self._tab_nav_canvas.bind('<MouseWheel>',self._on_tab_nav_mousewheel)
        self._tab_nav_canvas.bind('<Shift-MouseWheel>',self._on_tab_nav_mousewheel)
        style=ttk.Style()
        try:
            style.layout('HiddenTabs.TNotebook.Tab', [])
        except tk.TclError:
            pass
        self.tabs=ttk.Notebook(right,style='HiddenTabs.TNotebook')
        self.tabs.pack(fill=tk.BOTH,expand=True)
        self.tabs.bind('<<NotebookTabChanged>>',lambda e:self._sync_tab_nav_selection())
        self.fig={}

        # General comparison views show all three axes. Axis-specific selectors live only
        # on the tabs where a single-axis diagnostic is genuinely required.
        for name in ['Raw comparison']:
            tab=ttk.Frame(self.tabs)
            self.tabs.add(tab,text=name)
            f=Figure(figsize=(9,6),dpi=100)
            c=FigureCanvasTkAgg(f,master=tab)
            c.get_tk_widget().pack(fill=tk.BOTH,expand=True)
            tb=NavigationToolbar2Tk(c,tab,pack_toolbar=False)
            tb.update()
            tb.pack(fill=tk.X)
            self.fig[name]=(f,c)

        tab=ttk.Frame(self.tabs)
        self.tabs.add(tab,text='Issue 1 FFT ensemble')
        issue1_controls=ttk.Frame(tab,padding=(5,3))
        issue1_controls.pack(fill=tk.X)
        ttk.Label(issue1_controls,text='Individual FFT axis').pack(side=tk.LEFT,padx=(2,2))
        issue1_axis=ttk.Combobox(
            issue1_controls,textvariable=self.issue1_waterfall_axis,
            values=list('XYZ'),state='readonly',width=5)
        issue1_axis.pack(side=tk.LEFT,padx=(0,10))
        issue1_axis.bind('<<ComboboxSelected>>',lambda e:self.plot_issue1_analysis())
        ttk.Button(
            issue1_controls,text='Reprocess Issue 1',
            command=self.plot_issue1_analysis,
        ).pack(side=tk.LEFT,padx=5)
        ttk.Label(
            issue1_controls,
            text='Legacy amplitude ensemble; separate from Welch PSD/Gordon integration.',
        ).pack(side=tk.LEFT,padx=6)
        f=Figure(figsize=(9,6),dpi=100)
        c=FigureCanvasTkAgg(f,master=tab)
        c.get_tk_widget().pack(fill=tk.BOTH,expand=True)
        tb=NavigationToolbar2Tk(c,tab,pack_toolbar=False)
        tb.update(); tb.pack(fill=tk.X)
        self.fig['Issue 1 FFT ensemble']=(f,c)
        ax=f.add_subplot(111); ax.axis('off')
        ax.text(.5,.5,'Select and apply Issue 1, then load a Reader CSV.',
                ha='center',va='center',fontsize=12)
        c.draw_idle()

        # Offline shock analysis follows Raw comparison so recorded reader files
        # can be inspected directly using the Issue-2 time-domain method.
        tab=ttk.Frame(self.tabs)
        self.tabs.add(tab,text='Shock analysis')
        shock_controls=ttk.Frame(tab,padding=(5,3))
        shock_controls.pack(fill=tk.X)
        ttk.Button(shock_controls,text='Load shock baseline CSV(s)',command=lambda:self.load_shock_csvs('baseline')).pack(side=tk.LEFT,padx=(2,4))
        ttk.Button(shock_controls,text='Load shock reader CSV(s)',command=lambda:self.load_shock_csvs('reader')).pack(side=tk.LEFT,padx=4)
        ttk.Label(shock_controls,text='Applied shock axis').pack(side=tk.LEFT,padx=(12,2))
        sh_axis=ttk.Combobox(shock_controls,textvariable=self.offline_shock_axis,values=list('XYZ'),state='readonly',width=5)
        sh_axis.pack(side=tk.LEFT,padx=(0,8)); sh_axis.bind('<<ComboboxSelected>>',lambda e:(self.plot_shock_analysis(),self.plot_shock_translation()))
        ttk.Label(shock_controls,text='Steady-state').pack(side=tk.LEFT,padx=(4,2))
        ttk.Entry(shock_controls,textvariable=self.offline_shock_window,width=6).pack(side=tk.LEFT,padx=(0,2))
        ttk.Label(shock_controls,text='samples').pack(side=tk.LEFT,padx=(0,8))
        ttk.Label(shock_controls,text='Threshold').pack(side=tk.LEFT,padx=(4,2))
        ttk.Entry(shock_controls,textvariable=self.offline_shock_threshold_g,width=6).pack(side=tk.LEFT,padx=(0,2))
        ttk.Label(shock_controls,text='g').pack(side=tk.LEFT,padx=(0,8))
        ttk.Label(shock_controls,text='Event release').pack(side=tk.LEFT,padx=(4,2))
        ttk.Entry(shock_controls,textvariable=self.offline_shock_release_s,width=6).pack(side=tk.LEFT,padx=(0,2))
        ttk.Label(shock_controls,text='s').pack(side=tk.LEFT,padx=(0,8))
        ttk.Button(shock_controls,text='Reprocess shock',command=lambda:(self.plot_shock_analysis(),self.plot_shock_translation())).pack(side=tk.LEFT,padx=5)
        f=Figure(figsize=(9,6),dpi=100)
        c=FigureCanvasTkAgg(f,master=tab)
        c.get_tk_widget().pack(fill=tk.BOTH,expand=True)
        tb=NavigationToolbar2Tk(c,tab,pack_toolbar=False)
        tb.update()
        tb.pack(fill=tk.X)
        self.fig['Shock analysis']=(f,c)

        tab=ttk.Frame(self.tabs)
        self.tabs.add(tab,text='Shock translation')
        shock_trans_controls=ttk.Frame(tab,padding=(5,3))
        shock_trans_controls.pack(fill=tk.X)
        ttk.Label(shock_trans_controls,text='Dedicated SHOCK X/Y/Z files only; peak-response ratios, not H1/FRF.').pack(side=tk.LEFT,padx=(2,8))
        ttk.Button(shock_trans_controls,text='Reprocess shock translation',command=self.plot_shock_translation).pack(side=tk.LEFT,padx=5)
        f=Figure(figsize=(9,6),dpi=100)
        c=FigureCanvasTkAgg(f,master=tab)
        c.get_tk_widget().pack(fill=tk.BOTH,expand=True)
        tb=NavigationToolbar2Tk(c,tab,pack_toolbar=False)
        tb.update()
        tb.pack(fill=tk.X)
        self.fig['Shock translation']=(f,c)

        tab=ttk.Frame(self.tabs)
        self.tabs.add(tab,text='Shock summary')
        shock_summary_controls=ttk.Frame(tab,padding=(5,3))
        shock_summary_controls.pack(fill=tk.X)
        ttk.Button(shock_summary_controls,text='Load shock baseline CSV(s)',command=lambda:self.load_shock_csvs('baseline')).pack(side=tk.LEFT,padx=(2,4))
        ttk.Button(shock_summary_controls,text='Load shock reader CSV(s)',command=lambda:self.load_shock_csvs('reader')).pack(side=tk.LEFT,padx=4)
        ttk.Button(shock_summary_controls,text='Refresh summary',command=self.refresh_shock_summary).pack(side=tk.LEFT,padx=(12,4))
        ttk.Button(shock_summary_controls,text='Export shock results',command=self.export_shock_results).pack(side=tk.LEFT,padx=4)
        ttk.Label(shock_summary_controls,text='One row per deliberately applied X/Y/Z shock direction.').pack(side=tk.LEFT,padx=8)
        shock_summary_frame=ttk.Frame(tab,padding=6)
        shock_summary_frame.pack(fill=tk.BOTH,expand=True)
        cols=('applied','baseline','reader','bpeak','rpeak','kshock','events','dominant')
        self.shock_summary_tree=ttk.Treeview(shock_summary_frame,columns=cols,show='headings',height=6)
        headings={'applied':'Applied','baseline':'Baseline file','reader':'Reader file','bpeak':'Baseline peak S (g)','rpeak':'Reader peak S (g)','kshock':'Kshock B/R','events':'Reader events','dominant':'Dominant axis'}
        widths={'applied':75,'baseline':220,'reader':220,'bpeak':125,'rpeak':125,'kshock':95,'events':95,'dominant':95}
        for col in cols:
            self.shock_summary_tree.heading(col,text=headings[col])
            self.shock_summary_tree.column(col,width=widths[col],minwidth=70,anchor=tk.CENTER if col not in ('baseline','reader') else tk.W)
        ybar=ttk.Scrollbar(shock_summary_frame,orient=tk.VERTICAL,command=self.shock_summary_tree.yview)
        xbar=ttk.Scrollbar(shock_summary_frame,orient=tk.HORIZONTAL,command=self.shock_summary_tree.xview)
        self.shock_summary_tree.configure(yscrollcommand=ybar.set,xscrollcommand=xbar.set)
        self.shock_summary_tree.grid(row=0,column=0,sticky='nsew')
        ybar.grid(row=0,column=1,sticky='ns'); xbar.grid(row=1,column=0,sticky='ew')
        shock_summary_frame.rowconfigure(0,weight=1); shock_summary_frame.columnconfigure(0,weight=1)

        for name in ['PSD comparison','Gordon bands','Translation function']:
            tab=ttk.Frame(self.tabs)
            display_name='Gordon comparison' if name=='Gordon bands' else name
            self.tabs.add(tab,text=display_name)
            f=Figure(figsize=(9,6),dpi=100)
            c=FigureCanvasTkAgg(f,master=tab)
            c.get_tk_widget().pack(fill=tk.BOTH,expand=True)
            tb=NavigationToolbar2Tk(c,tab,pack_toolbar=False)
            tb.update()
            tb.pack(fill=tk.X)
            self.fig[name]=(f,c)

        tab=ttk.Frame(self.tabs)
        self.tabs.add(tab,text='Calculation stages')
        stage_controls=ttk.Frame(tab,padding=(5,3))
        stage_controls.pack(fill=tk.X)
        ttk.Label(stage_controls,text='Axis').pack(side=tk.LEFT,padx=(2,2))
        scb=ttk.Combobox(stage_controls,textvariable=self.stage_axis,values=list('XYZ'),state='readonly',width=5)
        scb.pack(side=tk.LEFT,padx=(0,10))
        scb.bind('<<ComboboxSelected>>',lambda e:self.plot_stage())
        self.stage_block_label=ttk.Label(stage_controls,text='FFT block')
        self.stage_block_label.pack(side=tk.LEFT,padx=(2,2))
        self.stage_block_combo=ttk.Combobox(
            stage_controls,textvariable=self.stage_block,
            values=[1,2,3,4],state='readonly',width=5)
        self.stage_block_combo.pack(side=tk.LEFT,padx=(0,10))
        self.stage_block_combo.bind('<<ComboboxSelected>>',lambda e:self.plot_stage())
        self.stage_explanation=ttk.Label(
            stage_controls,
            text='Blocks are the overlapping segments used in the PSD average.')
        self.stage_explanation.pack(side=tk.LEFT,padx=4)
        f=Figure(figsize=(9,6),dpi=100)
        c=FigureCanvasTkAgg(f,master=tab)
        c.get_tk_widget().pack(fill=tk.BOTH,expand=True)
        tb=NavigationToolbar2Tk(c,tab,pack_toolbar=False)
        tb.update()
        tb.pack(fill=tk.X)
        self.fig['Selected stage']=(f,c)

        tab=ttk.Frame(self.tabs)
        self.tabs.add(tab,text='Transfer function')
        transfer_controls=ttk.Frame(tab,padding=(5,3))
        transfer_controls.pack(fill=tk.X)
        ttk.Label(transfer_controls,text='Driven/reference axis').pack(side=tk.LEFT,padx=(2,2))
        tcb=ttk.Combobox(transfer_controls,textvariable=self.transfer_axis,values=list('XYZ'),state='readonly',width=5)
        tcb.pack(side=tk.LEFT,padx=(0,10))
        tcb.bind('<<ComboboxSelected>>',lambda e:self.plot_transfer())
        ttk.Label(transfer_controls,text='H1/coherence are calculated for the selected same-axis baseline ↔ reader pair.').pack(side=tk.LEFT,padx=4)
        f=Figure(figsize=(9,6),dpi=100)
        c=FigureCanvasTkAgg(f,master=tab)
        c.get_tk_widget().pack(fill=tk.BOTH,expand=True)
        tb=NavigationToolbar2Tk(c,tab,pack_toolbar=False)
        tb.update()
        tb.pack(fill=tk.X)
        self.fig['Transfer function']=(f,c)
        self._build_live_tab()
        self._build_live_analysis_tabs()
        self._build_help_tab()
        self._build_tab_navigation_buttons()
        self.refresh_shock_summary()
        ttk.Label(self.root,textvariable=self.status,relief=tk.SUNKEN,anchor=tk.W).pack(fill=tk.X,side=tk.BOTTOM)
    def _toggle_main_sidebar(self):
        """Collapse/restore the complete offline control sidebar horizontally."""
        try:
            if not self.main_sidebar_collapsed:
                self.main_pan.forget(self.main_sidebar_frame)
                self.main_sidebar_collapsed=True
                self.main_sidebar_toggle.configure(text='▶')
            else:
                # Reinsert after the permanently visible narrow toggle strip.
                self.main_pan.insert(1,self.main_sidebar_frame,weight=0)
                self.main_sidebar_collapsed=False
                self.main_sidebar_toggle.configure(text='◀')
        except tk.TclError:
            # If the pane state is unexpectedly out of sync, restore a usable
            # expanded layout rather than leaving the controls inaccessible.
            try:
                panes=list(self.main_pan.panes())
                if str(self.main_sidebar_frame) not in panes:
                    self.main_pan.insert(1,self.main_sidebar_frame,weight=0)
            except Exception:
                pass
            self.main_sidebar_collapsed=False
            self.main_sidebar_toggle.configure(text='◀')

    def _build_tab_navigation_buttons(self):
        if self._tab_nav_inner is None:
            return
        for child in self._tab_nav_inner.winfo_children():
            child.destroy()
        self._tab_nav_buttons=[]
        for i in range(self.tabs.index('end')):
            name=self.tabs.tab(i,'text')
            btn=ttk.Button(self._tab_nav_inner,text=name,command=lambda n=i:self._select_tab_from_nav(n))
            btn.pack(side=tk.LEFT,padx=1,pady=1)
            self._tab_nav_buttons.append(btn)
        self._sync_tab_nav_selection()
        self.root.after_idle(self._ensure_selected_tab_visible)

    def _select_tab_from_nav(self,index):
        self.tabs.select(index)
        self._sync_tab_nav_selection()
        self.root.after_idle(self._ensure_selected_tab_visible)

    def _sync_tab_nav_selection(self):
        if not self._tab_nav_buttons:
            return
        try:
            selected=self.tabs.index(self.tabs.select())
        except Exception:
            selected=0
        for i,btn in enumerate(self._tab_nav_buttons):
            try:
                btn.state(['pressed'] if i==selected else ['!pressed'])
            except tk.TclError:
                pass
        self.root.after_idle(self._ensure_selected_tab_visible)

    def _ensure_selected_tab_visible(self):
        if self._tab_nav_canvas is None or not self._tab_nav_buttons:
            return
        try:
            selected=self.tabs.index(self.tabs.select())
            btn=self._tab_nav_buttons[selected]
            self._tab_nav_canvas.update_idletasks()
            total=max(1,self._tab_nav_inner.winfo_reqwidth())
            view=max(1,self._tab_nav_canvas.winfo_width())
            bx=btn.winfo_x(); bw=btn.winfo_width()
            left=self._tab_nav_canvas.canvasx(0); right=left+view
            if bx<left:
                self._tab_nav_canvas.xview_moveto(max(0.0,bx/total))
            elif bx+bw>right:
                self._tab_nav_canvas.xview_moveto(min(1.0,max(0.0,(bx+bw-view)/total)))
        except Exception:
            pass

    def _scroll_tab_strip(self,direction):
        if self._tab_nav_canvas is not None:
            self._tab_nav_canvas.xview_scroll(int(direction)*4,'units')

    def _on_tab_nav_mousewheel(self,event):
        if self._tab_nav_canvas is None:
            return 'break'
        delta=-1 if getattr(event,'delta',0)>0 else 1
        self._tab_nav_canvas.xview_scroll(delta*4,'units')
        return 'break'

    def _build_live_tab(self):
        tab=ttk.Frame(self.tabs)
        self.tabs.add(tab,text='Live STM data')

        outer=ttk.Panedwindow(tab,orient=tk.HORIZONTAL)
        outer.pack(fill=tk.BOTH,expand=True,padx=4,pady=4)

        # The live-control sidebar uses the same permanently visible toggle strip
        # as the main Controls sidebar. Stacking the Live button below Controls
        # avoids consuming a second vertical strip of graph width.
        left=ttk.Frame(outer,padding=(0,0,4,0))
        right=ttk.Frame(outer,padding=(4,0,0,0))
        outer.add(left,weight=0)
        outer.add(right,weight=1)
        self.live_pan=outer
        self.live_sidebar_frame=left
        self.live_sidebar_collapsed=False
        self.live_sidebar_toggle=ttk.Button(
            self.main_side_toggle_strip,text='◀',width=3,
            command=self._toggle_live_sidebar)
        self.live_sidebar_toggle.pack(side=tk.TOP,fill=tk.X,pady=(4,0))
        ttk.Label(self.main_side_toggle_strip,text='Live').pack(side=tk.TOP,pady=(3,0))

        control_scroll=ScrollableFrame(left)
        control_scroll.pack(fill=tk.BOTH,expand=True)
        controls=control_scroll.body

        conn_section=CollapsibleSection(controls,'Direct STM32 connection',expanded=True)
        conn_section.pack(fill=tk.X,pady=(0,4))
        conn=conn_section.body

        row1=ttk.Frame(conn); row1.pack(fill=tk.X,pady=2)
        ttk.Label(row1,text='COM port').pack(side=tk.LEFT,padx=(4,1))
        self.live_port_combo=ttk.Combobox(row1,textvariable=self.live_port,width=16)
        self.live_port_combo.pack(side=tk.LEFT,padx=(0,4))
        ttk.Button(row1,text='Refresh',command=self.live_refresh_ports).pack(side=tk.LEFT,padx=3)
        ttk.Label(row1,text='Baud').pack(side=tk.LEFT,padx=(10,1))
        ttk.Entry(row1,textvariable=self.live_baud,width=8).pack(side=tk.LEFT,padx=(0,4))

        row1b=ttk.Frame(conn); row1b.pack(fill=tk.X,pady=2)
        ttk.Label(row1b,text='Plot window s').pack(side=tk.LEFT,padx=(4,2))
        ttk.Entry(row1b,textvariable=self.live_view_seconds,width=7).pack(side=tk.LEFT,padx=(0,8))
        ttk.Label(row1b,text='Reported full scale').pack(side=tk.LEFT,padx=(4,2))
        ttk.Label(row1b,textvariable=self.live_stream_fs,width=9).pack(side=tk.LEFT,padx=(0,4))

        row2=ttk.Frame(conn); row2.pack(fill=tk.X,pady=2)
        self.btn_live_connect=ttk.Button(row2,text='Open ST-Link VCP',command=self.live_connect)
        self.btn_live_connect.pack(side=tk.LEFT,padx=3)
        self.btn_live_disconnect=ttk.Button(row2,text='Disconnect',command=self.live_disconnect)
        self.btn_live_disconnect.pack(side=tk.LEFT,padx=3)

        row3=ttk.Frame(conn); row3.pack(fill=tk.X,pady=2)
        self.btn_live_start=ttk.Button(row3,text='Start capture',command=self.live_start)
        self.btn_live_start.pack(side=tk.LEFT,padx=3)
        self.btn_live_stop=ttk.Button(row3,text='Stop capture',command=self.live_stop)
        self.btn_live_stop.pack(side=tk.LEFT,padx=3)
        ttk.Button(row3,text='Clear plot/capture',command=self.live_clear).pack(side=tk.LEFT,padx=3)

        row4=ttk.Frame(conn); row4.pack(fill=tk.X,pady=2)
        ttk.Button(row4,text='Save displayed data CSV',command=self.live_save_csv).pack(side=tk.LEFT,padx=3)
        ttk.Button(row4,text='Use capture as Reader dataset',command=self.live_use_as_reader).pack(side=tk.LEFT,padx=3)

        status_box=ttk.Frame(conn)
        status_box.pack(fill=tk.X,pady=(4,2))
        ttk.Label(status_box,text='Link').grid(row=0,column=0,sticky='w',padx=(4,2))
        ttk.Label(status_box,textvariable=self.live_link).grid(row=0,column=1,sticky='w',padx=(0,10))
        ttk.Label(status_box,text='Protocol').grid(row=0,column=2,sticky='w',padx=(4,2))
        ttk.Label(status_box,textvariable=self.live_fw_version).grid(row=0,column=3,sticky='w')
        ttk.Label(status_box,text='Capture').grid(row=1,column=0,sticky='w',padx=(4,2),pady=(2,0))
        ttk.Label(status_box,textvariable=self.live_capture,wraplength=240,justify=tk.LEFT).grid(row=1,column=1,columnspan=3,sticky='w',pady=(2,0))

        threshold_section=CollapsibleSection(controls,'Thresholds, logging and timeout',expanded=True)
        threshold_section.pack(fill=tk.X,pady=(0,4))
        threshold_body=threshold_section.body

        row5=ttk.Frame(threshold_body); row5.pack(fill=tk.X,pady=2)
        ttk.Label(row5,text='Vibration threshold').pack(side=tk.LEFT,padx=(4,2))
        threshold_modes=[
            'Gordon Office',
            'Fixed acceleration (g RMS)',
            'Fixed velocity (um/s RMS)',
            'Fixed displacement (um RMS)',
            'Off',
        ]
        self.live_threshold_mode_combo=ttk.Combobox(
            row5,textvariable=self.live_threshold_mode,values=threshold_modes,
            state='readonly',width=28)
        self.live_threshold_mode_combo.pack(side=tk.LEFT,padx=3)
        self.live_threshold_mode_combo.bind('<<ComboboxSelected>>',self._on_live_threshold_mode_changed)

        row6=ttk.Frame(threshold_body); row6.pack(fill=tk.X,pady=2)
        self.live_threshold_multiplier_label=ttk.Label(row6,text='Gordon multiplier')
        self.live_threshold_multiplier_label.pack(side=tk.LEFT,padx=(4,2))
        self.live_threshold_multiplier_entry=ttk.Entry(row6,textvariable=self.live_threshold_multiplier,width=7)
        self.live_threshold_multiplier_entry.pack(side=tk.LEFT,padx=(0,8))
        self.live_threshold_value_label=ttk.Label(row6,text='Fixed value')
        self.live_threshold_value_label.pack(side=tk.LEFT,padx=(4,2))
        self.live_threshold_value_entry=ttk.Entry(row6,textvariable=self.live_threshold_value,width=9)
        self.live_threshold_value_entry.pack(side=tk.LEFT,padx=(0,8))
        self.live_threshold_apply_button=ttk.Button(
            row6,text='Apply',command=self.live_apply_threshold)
        self.live_threshold_apply_button.pack(side=tk.LEFT,padx=6)
        self._update_live_threshold_control_states()

        ttk.Label(threshold_body,textvariable=self.live_threshold_status,wraplength=340,justify=tk.LEFT).pack(anchor=tk.W,padx=4,pady=(0,4))

        row7=ttk.Frame(threshold_body); row7.pack(fill=tk.X,pady=2)
        ttk.Label(row7,text='Display retention').pack(side=tk.LEFT,padx=(4,2))
        ttk.Entry(row7,textvariable=self.live_retention_seconds,width=8).pack(side=tk.LEFT,padx=2)
        ttk.Label(row7,text='sec').pack(side=tk.LEFT,padx=(1,12))
        self.btn_raw_log_start=ttk.Button(row7,text='Start raw log',command=self.live_start_raw_log)
        self.btn_raw_log_start.pack(side=tk.LEFT,padx=3)
        self.btn_raw_log_stop=ttk.Button(row7,text='Stop raw log',command=self.live_stop_raw_log,state=tk.DISABLED)
        self.btn_raw_log_stop.pack(side=tk.LEFT,padx=3)

        ttk.Label(threshold_body,textvariable=self.raw_log_status,wraplength=340,justify=tk.LEFT).pack(anchor=tk.W,padx=4,pady=(0,4))

        row8=ttk.Frame(threshold_body); row8.pack(fill=tk.X,pady=2)
        ttk.Checkbutton(row8,text='Enable capture timeout',variable=self.operator_timeout_enabled).pack(side=tk.LEFT,padx=(4,8))
        ttk.Label(row8,text='Duration').pack(side=tk.LEFT,padx=(2,2))
        ttk.Entry(row8,textvariable=self.operator_timeout_value,width=8).pack(side=tk.LEFT,padx=(0,3))
        ttk.Combobox(row8,textvariable=self.operator_timeout_units,values=['seconds','minutes'],state='readonly',width=9).pack(side=tk.LEFT,padx=(0,10))

        row9=ttk.Frame(threshold_body); row9.pack(fill=tk.X,pady=2)
        ttk.Label(row9,text='On timeout').pack(side=tk.LEFT,padx=(4,2))
        ttk.Combobox(row9,textvariable=self.operator_timeout_action,
                     values=['Ask','Continue same duration','Continue indefinitely','Stop'],
                     state='readonly',width=24).pack(side=tk.LEFT,padx=(0,10))

        ttk.Label(threshold_body,textvariable=self.live_diagnostics,justify=tk.LEFT,wraplength=340).pack(fill=tk.X,padx=4,pady=(2,0))

        correction_section=CollapsibleSection(controls,'Live correction, tilt and shock',expanded=True)
        correction_section.pack(fill=tk.X,pady=(0,4))
        motion=correction_section.body

        k_row=ttk.Frame(motion); k_row.pack(fill=tk.X,pady=(1,3))
        self.live_k_enable_check=ttk.Checkbutton(
            k_row,text='Apply live multi-frequency K correction',
            variable=self.live_k_correction_enabled,
            command=self._on_live_k_toggle)
        self.live_k_enable_check.pack(side=tk.LEFT,padx=(4,8))
        k_btns=ttk.Frame(motion)
        k_btns.pack(fill=tk.X,pady=(0,2))
        self.live_k_load_button=ttk.Button(
            k_btns,text='Load K-factor CSV(s)',command=self.load_live_k_correction)
        self.live_k_load_button.pack(side=tk.LEFT,padx=3)
        self.live_k_clear_button=ttk.Button(
            k_btns,text='Clear K factors',command=self.clear_live_k_correction)
        self.live_k_clear_button.pack(side=tk.LEFT,padx=3)
        ttk.Label(motion,textvariable=self.live_k_status,wraplength=340,justify=tk.LEFT).pack(anchor=tk.W,padx=4,pady=(0,4))

        tilt_row=ttk.Frame(motion); tilt_row.pack(fill=tk.X,pady=(1,3))
        ttk.Button(tilt_row,text='Zero tilt at current position',command=self.live_zero_tilt).pack(side=tk.LEFT,padx=3)
        ttk.Label(tilt_row,text='Averaging (samples)').pack(side=tk.LEFT,padx=(10,2))
        ttk.Entry(tilt_row,textvariable=self.tilt_average_samples,width=6).pack(side=tk.LEFT,padx=(0,8))
        ttk.Label(tilt_row,text='Max dynamic RMS (g)').pack(side=tk.LEFT,padx=(4,2))
        ttk.Entry(tilt_row,textvariable=self.live_tilt_max_dynamic_g,width=6).pack(side=tk.LEFT,padx=(0,4))
        ttk.Label(motion,textvariable=self.live_tilt_readout,wraplength=340,justify=tk.LEFT).pack(anchor=tk.W,padx=4,pady=(0,4))

        shock_row=ttk.Frame(motion); shock_row.pack(fill=tk.X,pady=(1,3))
        ttk.Checkbutton(shock_row,text='Enable shock detector',variable=self.live_shock_enabled,
                        command=self._on_live_shock_controls_changed).pack(side=tk.LEFT,padx=(4,8))
        ttk.Label(shock_row,text='Threshold (g)').pack(side=tk.LEFT,padx=(4,2))
        shock_entry=ttk.Entry(shock_row,textvariable=self.live_shock_threshold_g,width=7)
        shock_entry.pack(side=tk.LEFT,padx=(0,8))
        shock_entry.bind('<Return>',lambda e:self._on_live_shock_controls_changed())
        shock_entry.bind('<FocusOut>',lambda e:self._on_live_shock_controls_changed())
        ttk.Button(shock_row,text='Reset peak/events',command=self.live_reset_shock).pack(side=tk.LEFT,padx=3)
        ttk.Label(motion,text='Steady-state average: 50 samples (0.25 s @ 200 Hz)',wraplength=340,justify=tk.LEFT).pack(anchor=tk.W,padx=4)
        ttk.Label(motion,textvariable=self.live_shock_readout,wraplength=340,justify=tk.LEFT).pack(anchor=tk.W,padx=4,pady=(0,2))

        accel_section=CollapsibleSection(controls,'Accelerometer / processing configuration',expanded=False)
        accel_section.pack(fill=tk.X,pady=(0,4))
        accel_cfg=accel_section.body

        ac0=ttk.Frame(accel_cfg); ac0.pack(fill=tk.X,pady=2)
        ttk.Label(ac0,text='Processing method').pack(side=tk.LEFT,padx=(4,2))
        self.issue1_mode_button=ttk.Radiobutton(
            ac0,text='Issue 1 — Legacy FFT',variable=self.processing_method,
            value=ISSUE1_METHOD,command=self._processing_method_changed,
            style='Toolbutton')
        self.issue1_mode_button.pack(side=tk.LEFT,padx=(2,1))
        self.issue2_mode_button=ttk.Radiobutton(
            ac0,text='Issue 2 — PSD/Gordon',variable=self.processing_method,
            value=ISSUE2_METHOD,command=self._processing_method_changed,
            style='Toolbutton')
        self.issue2_mode_button.pack(side=tk.LEFT,padx=(1,8))
        ttk.Label(
            accel_cfg,textvariable=self.processing_method_details,
            wraplength=340,justify=tk.LEFT,
        ).pack(fill=tk.X,padx=4,pady=(0,3))

        ac1=ttk.Frame(accel_cfg); ac1.pack(fill=tk.X,pady=2)
        ttk.Label(ac1,text='Device').pack(side=tk.LEFT,padx=(4,2))
        ttk.Entry(ac1,textvariable=self.accel_device,width=12,state='readonly').pack(side=tk.LEFT,padx=(0,8))
        ttk.Label(ac1,text='Full scale').pack(side=tk.LEFT,padx=(4,2))
        ttk.Combobox(ac1,textvariable=self.accel_fs_g,values=[2,4,8,16],state='readonly',width=5).pack(side=tk.LEFT,padx=(0,4))
        ttk.Label(ac1,text='g').pack(side=tk.LEFT,padx=(0,8))
        ttk.Label(ac1,text='ODR').pack(side=tk.LEFT,padx=(4,2))
        self.accel_odr_combo=ttk.Combobox(
            ac1,textvariable=self.accel_odr_hz,values=[25,50,100,200,400,800],
            state='readonly',width=6)
        self.accel_odr_combo.pack(side=tk.LEFT,padx=(0,2))
        ttk.Label(ac1,text='Hz').pack(side=tk.LEFT,padx=(0,2))

        ac2=ttk.Frame(accel_cfg); ac2.pack(fill=tk.X,pady=2)
        ttk.Label(ac2,text='Operating mode').pack(side=tk.LEFT,padx=(4,2))
        ttk.Combobox(ac2,textvariable=self.accel_mode,values=['High Performance','Low Power'],state='readonly',width=17).pack(side=tk.LEFT,padx=(0,8))
        ttk.Checkbutton(ac2,text='Anti-alias bandwidth active',variable=self.accel_aa_enabled,state=tk.DISABLED).pack(side=tk.LEFT,padx=8)

        ac3=ttk.Frame(accel_cfg); ac3.pack(fill=tk.X,pady=2)
        ttk.Label(ac3,text='AA bandwidth').pack(side=tk.LEFT,padx=(4,2))
        ttk.Combobox(ac3,textvariable=self.accel_bw,values=['ODR/2','ODR/4','ODR/8','ODR/16'],state='readonly',width=8).pack(side=tk.LEFT,padx=(0,8))
        ttk.Label(ac3,text='FIFO watermark').pack(side=tk.LEFT,padx=(4,2))
        ttk.Spinbox(ac3,textvariable=self.accel_fifo_watermark,from_=1,to=32,width=6).pack(side=tk.LEFT,padx=(0,4))
        ttk.Label(ac3,text='samples').pack(side=tk.LEFT,padx=(0,8))

        ac4=ttk.Frame(accel_cfg); ac4.pack(fill=tk.X,pady=2)
        ttk.Checkbutton(ac4,text='Route FIFO threshold to INT1',variable=self.accel_int1_fifo_threshold).pack(side=tk.LEFT,padx=(4,8))
        ttk.Label(ac4,text='Default stream timeout').pack(side=tk.LEFT,padx=(4,2))
        ttk.Entry(ac4,textvariable=self.accel_stream_timeout_ms,width=10).pack(side=tk.LEFT,padx=(0,2))
        ttk.Label(ac4,text='ms').pack(side=tk.LEFT,padx=(0,8))

        ac5=ttk.Frame(accel_cfg); ac5.pack(fill=tk.X,pady=2)
        ttk.Label(ac5,text='Nominal sensitivity').pack(side=tk.LEFT,padx=(4,2))
        self.accel_sensitivity_label=ttk.Label(ac5,text='0.061 mg/LSB',width=13)
        self.accel_sensitivity_label.pack(side=tk.LEFT,padx=(0,10))
        ttk.Label(ac5,text='Data format').pack(side=tk.LEFT,padx=(4,2))
        ttk.Label(ac5,text='signed int16 X/Y/Z',width=18).pack(side=tk.LEFT,padx=(0,10))

        ac6=ttk.Frame(accel_cfg); ac6.pack(fill=tk.X,pady=2)
        ttk.Label(ac6,text='Application pre-filter').pack(side=tk.LEFT,padx=(4,2))
        self.accel_prefilter_combo=ttk.Combobox(
            ac6,textvariable=self.accel_prefilter,
            values=['None','Issue 1 FIR 3.5-90 Hz'],state='readonly',width=22,
        )
        self.accel_prefilter_combo.pack(side=tk.LEFT,padx=(0,10))
        ttk.Label(ac6,text='FFT N').pack(side=tk.LEFT,padx=(4,2))
        self.accel_fft_n_combo=ttk.Combobox(
            ac6,textvariable=self.accel_fft_n,values=[128,256,512,1024],
            state='readonly',width=7)
        self.accel_fft_n_combo.pack(side=tk.LEFT,padx=(0,8))
        ttk.Label(ac6,text='Window').pack(side=tk.LEFT,padx=(4,2))
        self.accel_window_combo=ttk.Combobox(
            ac6,textvariable=self.accel_fft_window,values=['Hann'],
            state='disabled',width=8)
        self.accel_window_combo.pack(side=tk.LEFT,padx=(0,8))

        ac7=ttk.Frame(accel_cfg); ac7.pack(fill=tk.X,pady=2)
        ttk.Label(ac7,text='Overlap %').pack(side=tk.LEFT,padx=(4,2))
        self.accel_overlap_entry=ttk.Entry(
            ac7,textvariable=self.accel_fft_overlap,width=7)
        self.accel_overlap_entry.pack(side=tk.LEFT,padx=(0,8))
        self.accel_ensemble_label=ttk.Label(ac7,text='PSD averages')
        self.accel_ensemble_label.pack(side=tk.LEFT,padx=(4,2))
        self.accel_averages_entry=ttk.Entry(
            ac7,textvariable=self.accel_psd_averages,width=6)
        self.accel_averages_entry.pack(side=tk.LEFT,padx=(0,8))

        ac8=ttk.Frame(accel_cfg); ac8.pack(fill=tk.X,pady=(3,1))
        ttk.Button(ac8,text='Apply processing settings',command=self.apply_accel_processing_settings).pack(side=tk.LEFT,padx=3)
        self.btn_program_accel=ttk.Button(ac8,text='Program accelerometer',command=self.program_accelerometer,state=tk.DISABLED)
        self.btn_program_accel.pack(side=tk.LEFT,padx=3)

        ac9=ttk.Frame(accel_cfg); ac9.pack(fill=tk.X,pady=(1,2))
        ttk.Button(ac9,text='Export requested config JSON',command=self.export_accelerometer_config).pack(side=tk.LEFT,padx=3)
        ttk.Label(ac9,text='Programming result:').pack(side=tk.LEFT,padx=(10,3))
        self.accel_program_badge=tk.Label(
            ac9,textvariable=self.accel_program_result,
            font=('Segoe UI',9,'bold'),relief=tk.GROOVE,
            padx=8,pady=2,bg='#6b7280',fg='white')
        self.accel_program_badge.pack(side=tk.LEFT,padx=(0,8))

        ttk.Label(accel_cfg,textvariable=self.accel_config_status,wraplength=340,justify=tk.LEFT).pack(fill=tk.X,padx=4,pady=(1,1))
        ttk.Label(accel_cfg,textvariable=self.accel_readback_status,wraplength=340,justify=tk.LEFT).pack(fill=tk.X,padx=4,pady=(1,2))

        for var in (self.accel_fs_g,self.accel_odr_hz,self.accel_mode,
                    self.accel_aa_enabled,self.accel_bw,self.accel_fifo_watermark,
                    self.accel_int1_fifo_threshold,self.accel_stream_timeout_ms):
            var.trace_add('write',self._accelerometer_controls_changed)
        self._update_accel_config_preview()

        ttk.Label(right,textvariable=self.live_stream_health,wraplength=980,justify=tk.LEFT,
                  font=('Segoe UI',10,'bold')).pack(fill=tk.X,padx=8,pady=(0,4))

        f=Figure(figsize=(10,6),dpi=100)
        canvas=FigureCanvasTkAgg(f,master=right)
        canvas.get_tk_widget().pack(fill=tk.BOTH,expand=True)
        tb=NavigationToolbar2Tk(canvas,right,pack_toolbar=False)
        tb.update()
        tb.pack(fill=tk.X)
        self.live_fig=f
        self.live_canvas=canvas
        self.live_refresh_ports()
        self._draw_live_empty()
    def _toggle_live_sidebar(self):
        """Collapse/restore the complete Live STM control sidebar horizontally."""
        try:
            if not self.live_sidebar_collapsed:
                self.live_pan.forget(self.live_sidebar_frame)
                self.live_sidebar_collapsed=True
                self.live_sidebar_toggle.configure(text='▶')
            else:
                # Restore the live controls to the left of the graph pane.
                self.live_pan.insert(0,self.live_sidebar_frame,weight=0)
                self.live_sidebar_collapsed=False
                self.live_sidebar_toggle.configure(text='◀')
        except tk.TclError:
            # Recover to an expanded usable layout if the pane state is ever
            # unexpectedly out of sync.
            try:
                panes=list(self.live_pan.panes())
                if str(self.live_sidebar_frame) not in panes:
                    self.live_pan.insert(0,self.live_sidebar_frame,weight=0)
            except Exception:
                pass
            self.live_sidebar_collapsed=False
            self.live_sidebar_toggle.configure(text='◀')

    def _build_help_tab(self):
        tab=ttk.Frame(self.tabs)
        self.tabs.add(tab,text='Help / info')

        outer=ttk.Frame(tab,padding=8)
        outer.pack(fill=tk.BOTH,expand=True)
        txt=tk.Text(outer,wrap=tk.WORD,height=30)
        scroll=ttk.Scrollbar(outer,orient=tk.VERTICAL,command=txt.yview)
        txt.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT,fill=tk.Y)
        txt.pack(side=tk.LEFT,fill=tk.BOTH,expand=True)

        help_text="""Live STM data help
==================

Live stream path
----------------
The live monitor reads the LIS2DUX12 acceleration stream from hw_test_vmm over the ST-Link Virtual COM port at 460800 baud. This path uses the prepared VMM sample-block stream and does not use ProtoComms, the System Diagnostics UART, SOM HTTP, JTAG or SWD.

The raw sample blocks contain signed int16 X/Y/Z counts together with metadata such as ODR, full-scale range and status. Converted acceleration is calculated as:
    acceleration_g = counts * fs_g / 32768

Accelerometer programming
-------------------------
Programming the accelerometer from this GUI requires firmware support for the CONFIGURE command and for configuration read-back. After a successful configuration, the firmware should acknowledge the request and return the effective HP/LP mode, ODR, full-scale, anti-alias bandwidth, FIFO and INT1 settings so the GUI can verify what was applied.

UI sections
-----------
The Direct STM32 connection section contains the ST-Link VCP connection controls, capture controls, threshold settings, retention settings and timeout handling.

The Accelerometer / processing configuration section contains the requested sensor setup and the processing defaults used by the live analysis views. Both sections can be collapsed or expanded using the arrow button to reduce clutter.

Shock detector
--------------
The offline Shock analysis and Shock translation tabs use dedicated baseline and reader shock captures for X, Y and Z. Shock datasets are isolated from the normal vibration Reader/Baseline datasets; loading shock files does not populate the PSD comparison, Gordon bands, Selected stage, vibration Transfer function or vibration Translation function tabs. Filenames should contain the word SHOCK plus the applied axis. Repeated threshold crossings from one impact/ring-down are merged into one physical event until the resultant has remained below threshold for the configurable Event release time. When both baseline and reader shock files exist for the selected applied axis, the plots are aligned to their respective peak shock so independently started captures can be compared.

Raw-count acquisition traceability

Before Start raw-count log begins recording, v24 asks for Reader ID, excited axis, requested vibration level, run number and optional notes. These fields are used to propose the CSV filename and are written to a companion *_settings.json file together with the accelerometer, capture and processing settings. Export derived correction also writes a companion *_settings.json traceability file recording the reader source, baseline source files and mappings, processing settings and all exported K coefficients.

The Live STM data summary includes a shock detector based on 1174-Y-056 Proposed Issue 2 section 6.2.2. Each X/Y/Z sample is compared with a 50-sample rolling steady-state average (0.25 s at 200 Hz), and the resultant magnitude is calculated as S = sqrt((X-Xavg)^2 + (Y-Yavg)^2 + (Z-Zavg)^2). The detector compares S with a configurable threshold in g. The Issue 2 draft defines the method and requires a configurable threshold but does not specify the numerical threshold value. The shock calculation uses raw converted XYZ acceleration and is not affected by live multi-frequency K correction.

Timeouts
--------
The operator capture timeout is separate from the low-level VMM START lease. The GUI can prompt, continue for the same duration, continue indefinitely, or stop capture when the operator timeout is reached. The low-level VMM START lease is renewed automatically in the background when capture is running.

File compatibility
------------------
The comparison workflow accepts both converted CSV files (time_s, X_g, Y_g, Z_g) and raw VMM count logs (for example x_counts, y_counts, z_counts with fs_g and timestamps). The GUI normalises both formats before comparison.

Live K correction
-----------------
A complete live correction contains 14 frequency-band K = Base / Reader coefficients for each axis (42 coefficients total). Load either one combined CSV or multiple axis-specific exported correction CSVs. The correction is multiplicative and is applied after Gordon-band RMS integration; it is not a DC offset and does not alter the raw time-domain acceleration or tilt calculation. When enabled, translated band values are used by the Live STM summary and live threshold decision. The Live K correction tab always shows raw versus translated values for review.

Baseline / reader correction workflow
-------------------------------------
Load one Reader CSV containing X, Y and Z acceleration channels. Use Load baseline CSV(s) to select up to three reference CSVs at once. Baseline filenames are inspected for X, Y or Z to identify the intentionally excited axis; if a filename is ambiguous the GUI asks for the axis.

The normal mapping is X correction ← X baseline, Y ← Y baseline and Z ← Z baseline. If one baseline is unavailable, use Correction baseline mapping to explicitly select another loaded axis as a surrogate. When a surrogate is used, its driven-axis acceleration channel is used as the reference profile for the missing target axis; cross-axis response from that file is not used by mistake.

Export derived correction creates a diagnostic 42-value correction from the currently loaded reader dataset. For the physical single-axis test-house workflow, use Store driven-axis K after each approved X-, Y- or Z-excited run, then Export final XYZ K to assemble the production candidate from the three separately driven-axis results with source traceability.

Selectable vibration methods
----------------------------
Issue 2 (default) retains the existing Welch PSD averaging and Gordon-band
integration path. Issue 1 is a separate legacy/reference path: raw XYZ passes
through a Hamming-designed 3.5-90 Hz FIR, then 19 overlapping 128-sample Hann
FFTs are averaged as Fourier amplitudes. It does not average PSDs or integrate
Gordon bands. The Gordon units selector can display those peak FFT-bin values
and their matching limits as equivalent RMS acceleration, velocity or
displacement; this is a unit conversion, not Issue 2 band integration. Use the
dedicated offline or live Issue 1 FFT ensemble tab. The Translation function
uses direct per-bin K = baseline ensemble amplitude / reader ensemble amplitude;
it requires time-domain baseline captures and does not use Issue 2 model fitting.

Recommended default Issue 2 live settings
-----------------------------------------
Device: LIS2DUX12
Full-scale: ±2 g
ODR: 200 Hz
Operating mode: High Performance
Anti-alias bandwidth: ODR/2
FIFO watermark: 32 samples
FFT length: 512
Window: Hann
Overlap: 50%
PSD averages: 4
"""
        txt.insert('1.0',help_text)
        txt.configure(state=tk.DISABLED)

    def _accelerometer_config_dict(self):
        """Snapshot all sensor and host-processing controls in exportable form."""
        fs_g=int(self.accel_fs_g.get())
        odr=int(self.accel_odr_hz.get())
        sensitivity_mg_lsb=fs_g/32768.0*1000.0
        bw_ratio={'ODR/2':2,'ODR/4':4,'ODR/8':8,'ODR/16':16}.get(self.accel_bw.get(),2)
        bw_hz=odr/float(bw_ratio)
        return {
            'device':self.accel_device.get(),
            'full_scale_g':fs_g,
            'odr_hz':odr,
            'operating_mode':self.accel_mode.get(),
            'anti_alias_enabled':bool(self.accel_aa_enabled.get()),
            'anti_alias_bandwidth':self.accel_bw.get(),
            'anti_alias_bandwidth_hz':bw_hz,
            'fifo_watermark':int(self.accel_fifo_watermark.get()),
            'int1_fifo_threshold':bool(self.accel_int1_fifo_threshold.get()),
            'stream_timeout_ms':int(self.accel_stream_timeout_ms.get()),
            'nominal_sensitivity_mg_per_lsb':sensitivity_mg_lsb,
            'data_format':'signed int16 X/Y/Z',
            'processing_method':self.processing_method.get(),
            'application_pre_filter':self.accel_prefilter.get(),
            'processing_rate_hz':odr,
            'fft_length':int(self.accel_fft_n.get()),
            'fft_window':self.accel_fft_window.get(),
            'fft_overlap_percent':float(self.accel_fft_overlap.get()),
            'psd_averages':int(self.accel_psd_averages.get()),
            'vibration_bands':'Gordon one-third-octave 4-80 Hz',
        }

    def _processing_method_changed(self,event=None):
        """Load the documented defaults for the selected software algorithm."""
        method=self.processing_method.get()
        if method==ISSUE1_METHOD:
            self.accel_odr_hz.set(int(ISSUE1_SAMPLE_RATE_HZ))
            self.accel_fft_n.set(ISSUE1_FFT_N)
            self.accel_fft_window.set('Hann')
            self.accel_fft_overlap.set(ISSUE1_OVERLAP*100.0)
            self.accel_psd_averages.set(ISSUE1_ENSEMBLE_COUNT)
            self.accel_prefilter.set('Issue 1 FIR 3.5-90 Hz')
            self.accel_ensemble_label.configure(text='FFTs in ensemble')
            self.processing_method_details.set(
                'Issue 1 legacy/reference: 3.5–90 Hz Hamming FIR, 128-sample '
                'Hann FFTs, 64-sample hop, 19-window amplitude ensemble, '
                '1.5625 Hz resolution. Press Apply processing settings to activate.')
        else:
            self.accel_odr_hz.set(200)
            self.accel_fft_n.set(512)
            self.accel_fft_window.set('Hann')
            self.accel_fft_overlap.set(50.0)
            self.accel_psd_averages.set(4)
            self.accel_prefilter.set('None')
            self.accel_ensemble_label.configure(text='PSD averages')
            self.processing_method_details.set(
                'Issue 2: Welch PSD averaging and Gordon one-third-octave '
                'integration. Press Apply processing settings to activate.')
        self._sync_processing_method_controls()
        self._update_processing_status()

    def _display_units_changed(self,event=None):
        """Refresh offline now and queue live plots after a unit-only change."""
        if self.reader.loaded:
            try:
                self.plot_bands()
                if any(p is not None and p.loaded for p in self.baselines.values()):
                    self.plot_translation()
            except Exception as exc:
                self.status.set(f'Could not refresh selected display units: {exc}')
        self.live_last_view_draw=0.0
        self.live_redraw_pending=True

    def _sync_processing_method_controls(self):
        """Grey controls that do not participate in the selected method."""
        issue1=self.processing_method.get()==ISSUE1_METHOD

        # Issue 1 is the fixed document algorithm. Issue 2 retains the existing
        # editable FFT/PSD controls. Hann and the mode-specific pre-filter are
        # informative fixed values in both modes.
        self.accel_odr_combo.configure(state='disabled' if issue1 else 'readonly')
        self.accel_fft_n_combo.configure(state='disabled' if issue1 else 'readonly')
        self.accel_overlap_entry.configure(state='disabled' if issue1 else 'normal')
        self.accel_averages_entry.configure(state='disabled' if issue1 else 'normal')
        self.accel_window_combo.configure(state='disabled')
        self.accel_prefilter_combo.configure(state='disabled')
        self.stage_block_combo.configure(state='disabled' if issue1 else 'readonly')
        self.stage_block_label.configure(state='disabled' if issue1 else 'normal')
        self.stage_explanation.configure(
            text=('Issue 1 displays all 19 FFT amplitudes and their ensemble.'
                  if issue1 else
                  'Blocks are the overlapping segments used in the PSD average.'))

        issue2_state='disabled' if issue1 else 'normal'
        issue2_combo_state='disabled' if issue1 else 'readonly'
        # Both algorithms can present their native result in equivalent RMS
        # acceleration, velocity or displacement units.
        self.units_combo.configure(state='readonly')
        self.auto_crop_check.configure(state=issue2_state)
        self.btn_export_correction.configure(state=issue2_state)
        self.live_threshold_mode_combo.configure(state=issue2_combo_state)
        self.live_threshold_apply_button.configure(state=issue2_state)
        self.live_k_enable_check.configure(state=issue2_state)
        self.live_k_load_button.configure(state=issue2_state)
        self.live_k_clear_button.configure(state=issue2_state)
        self._update_live_threshold_control_states()

    def _sync_applied_mode_tabs(self):
        """Enable analysis tabs belonging to the applied processing method."""
        issue1=self._issue1_active()
        issue1_tabs={'Issue 1 FFT ensemble','Live Issue 1 ensemble'}
        issue2_tabs={
            'PSD comparison','Transfer function',
            'Live FFT + PSD','Live K correction',
        }
        for tab_id in self.tabs.tabs():
            text=self.tabs.tab(tab_id,'text')
            if text in issue1_tabs:
                self.tabs.tab(tab_id,state='normal' if issue1 else 'disabled')
            elif text in issue2_tabs:
                reference_available=text=='PSD comparison' and (isinstance(self.reader,BaselineSpectrum) or any(isinstance(b,BaselineSpectrum) and b.loaded for b in self.baselines.values()))
                self.tabs.tab(tab_id,state='disabled' if issue1 and not reference_available else 'normal')
        selected=self.tabs.select()
        if selected and str(self.tabs.tab(selected,'state'))=='disabled':
            target='Calculation stages'
            replacement=next(
                (tab_id for tab_id in self.tabs.tabs()
                 if self.tabs.tab(tab_id,'text')==target),None)
            if replacement:
                self.tabs.select(replacement)

    def _issue1_active(self):
        return self.applied_processing_method==ISSUE1_METHOD

    def _requested_accelerometer_wire_config(self):
        """Build and validate the firmware-facing subset of the controls."""
        cfg=self._accelerometer_config_dict()
        requested=AccelerometerConfig(
            odr_hz=int(cfg['odr_hz']),
            fs_g=int(cfg['full_scale_g']),
            high_performance=cfg['operating_mode']=='High Performance',
            bandwidth_divisor=int(cfg['anti_alias_bandwidth'].split('/')[-1]),
            fifo_watermark=int(cfg['fifo_watermark']),
            int1_fifo_threshold=bool(cfg['int1_fifo_threshold']),
            stream_timeout_ms=int(cfg['stream_timeout_ms']),
        )
        requested.validate()
        return requested

    def _begin_accelerometer_config_attempt(self,requested,message='CONFIGURE sent'):
        """Track an asynchronous CONFIG request until ACK plus read-back arrive."""
        self.accel_config_attempt+=1
        attempt=self.accel_config_attempt
        self.accel_pending_config=requested
        self.accel_config_ack_ok=False
        self.accel_effective_config=None
        self._set_accel_program_result('pending')
        self.accel_config_status.set(f'{message}; waiting for ACK and register read-back...')
        self.accel_readback_status.set('Effective configuration: waiting for firmware CONFIG packet')
        self.root.after(5000,lambda:self._accelerometer_config_timeout(attempt))

    def _update_accel_config_preview(self):
        try:
            cfg=self._accelerometer_config_dict()
            self.accel_sensitivity_label.configure(
                text=f"{cfg['nominal_sensitivity_mg_per_lsb']:.3f} mg/LSB")
            requested=(
                f"Requested: ±{cfg['full_scale_g']} g, {cfg['odr_hz']} Hz, "
                f"{cfg['operating_mode']}, AA "
                f"{'ON' if cfg['anti_alias_enabled'] else 'OFF'} "
                f"{cfg['anti_alias_bandwidth']} "
                f"(~{cfg['anti_alias_bandwidth_hz']:g} Hz), "
                f"FIFO {cfg['fifo_watermark']}, INT1 "
                f"{'ON' if cfg['int1_fifo_threshold'] else 'OFF'}")
            self.accel_config_status.set(requested)
        except Exception as exc:
            self.accel_config_status.set(f'Configuration error: {exc}')

    def _set_accel_program_result(self,state,text=None):
        """Update the prominent sensor-programming result badge."""
        colours={
            'idle':('#6b7280','white'),
            'changed':('#2563eb','white'),
            'pending':('#d97706','white'),
            'success':('#15803d','white'),
            'failure':('#b91c1c','white'),
        }
        labels={
            'idle':'NOT PROGRAMMED',
            'changed':'CHANGED - NOT APPLIED',
            'pending':'SETTING...',
            'success':'SUCCESS - VERIFIED',
            'failure':'FAILED',
        }
        self.accel_program_result.set(text or labels[state])
        if hasattr(self,'accel_program_badge'):
            bg,fg=colours[state]
            self.accel_program_badge.configure(bg=bg,fg=fg)
        if hasattr(self,'btn_program_accel'):
            if state=='pending':
                self.btn_program_accel.configure(state=tk.DISABLED)
            elif self.live_thread and self.live_thread.is_alive():
                self.btn_program_accel.configure(state=tk.NORMAL)

    def _accelerometer_controls_changed(self,*_args):
        self._update_accel_config_preview()
        # A verified read-back only applies to the values that were sent. Once
        # a firmware-facing control changes, make it explicit that it must be
        # programmed again.
        self.accel_pending_config=None
        self.accel_config_ack_ok=False
        self._set_accel_program_result('changed')

    def _accelerometer_config_timeout(self,attempt):
        if attempt!=self.accel_config_attempt or self.accel_pending_config is None:
            return
        stage='register read-back' if self.accel_config_ack_ok else 'firmware acknowledgement'
        self.accel_config_status.set(f'CONFIGURE timed out waiting for {stage}.')
        self.accel_readback_status.set('Configuration was not verified; check the firmware and serial link.')
        self.accel_pending_config=None
        self.accel_config_ack_ok=False
        self._set_accel_program_result('failure','FAILED - TIMEOUT')

    def _update_processing_status(self):
        """Show the host settings most recently applied to live/offline analysis."""
        if self._issue1_active():
            hop=int(round(int(self.n.get())*(1-float(self.ov.get())/100.0)))
            detail=(f'{int(self.segs.get())} FFT amplitudes | {hop}-sample hop | '
                    f'FIR {ISSUE1_LOW_HZ:g}–{ISSUE1_HIGH_HZ:g} Hz')
        else:
            detail=f'{int(self.segs.get())} PSD averages'
        self.processing_status.set(
            f'Processing [{self.applied_processing_method}]: '
            f'{float(self.fs.get()):g} Hz | FFT {int(self.n.get())} | Hann | '
            f'{float(self.ov.get()):g}% overlap | {detail}'
            + (f' | Selected: {self.processing_method.get()} — press Apply'
               if self.processing_method.get()!=self.applied_processing_method else ''))

    def apply_accel_processing_settings(self):
        """Apply the host-side processing subset immediately."""
        try:
            if self.processing_method.get()==ISSUE1_METHOD:
                # Issue 1 is a deliberate fixed legacy/reference algorithm,
                # rather than a request to run Issue 2 with 19 PSD averages.
                self.accel_odr_hz.set(int(ISSUE1_SAMPLE_RATE_HZ))
                self.accel_fft_n.set(ISSUE1_FFT_N)
                self.accel_fft_window.set('Hann')
                self.accel_fft_overlap.set(ISSUE1_OVERLAP*100.0)
                self.accel_psd_averages.set(ISSUE1_ENSEMBLE_COUNT)
                self.accel_prefilter.set('Issue 1 FIR 3.5-90 Hz')
            cfg=self._accelerometer_config_dict()
            self.fs.set(float(cfg['processing_rate_hz']))
            self.n.set(int(cfg['fft_length']))
            self.ov.set(float(cfg['fft_overlap_percent']))
            self.segs.set(int(cfg['psd_averages']))
            self.applied_processing_method=cfg['processing_method']
            if self.live_k_correction_enabled.get() and self.live_k_method not in (None,self.applied_processing_method):
                self.live_k_correction_enabled.set(False)
                self.live_k_status.set(
                    f'Translation: loaded {self.live_k_method}, OFF — active processing is {self.applied_processing_method}')
            self._update_processing_status()
            self._sync_applied_mode_tabs()
            self.accel_config_status.set(
                'Host processing updated. Sensor register programming is separate.')
            if self.reader.loaded:
                self.refresh()
            self._request_live_redraw()
        except Exception as exc:
            messagebox.showerror('Accelerometer configuration',str(exc))

    def program_accelerometer(self):
        if self.live_client is None or not (self.live_thread and self.live_thread.is_alive()):
            messagebox.showerror('Accelerometer configuration','Open the ST-Link VCP and start capture first.')
            return
        try:
            requested=self._requested_accelerometer_wire_config()
            self.live_client.configure(requested)
            self._begin_accelerometer_config_attempt(requested)
        except Exception as exc:
            self.accel_pending_config=None
            self.accel_config_ack_ok=False
            self._set_accel_program_result('failure','FAILED - NOT SENT')
            messagebox.showerror('Accelerometer configuration',str(exc))

    def export_accelerometer_config(self):
        cfg=self._accelerometer_config_dict()
        path=filedialog.asksaveasfilename(
            defaultextension='.json',
            filetypes=[('JSON','*.json')],
            initialfile='1174_lis2dux12_requested_configuration.json')
        if not path:
            return
        import json
        Path(path).write_text(json.dumps(cfg,indent=2),encoding='utf-8')
        self.accel_config_status.set(f'Requested configuration exported: {Path(path).name}')

    def _build_live_analysis_tabs(self):
        descriptions={
            'Live raw': 'Waiting for live samples. This view shows decoded, scaled X/Y/Z data before mean removal or windowing.',
            'Live raw counts': 'Waiting for live samples. This view shows the exact signed int16 X/Y/Z counts supplied by SampleBlock.xyz.',
            'Live stages': 'Waiting for one complete FFT block to show each processing stage.',
            'Live FFT + PSD': 'Waiting for one complete FFT block to show spectral results.',
            'Live Issue 1 ensemble': 'Select and apply Issue 1 mode, then capture 1,280 samples for the 19-FFT amplitude ensemble.',
            'Live K correction': 'Load X/Y/Z band-specific K factors to compare raw and translated Gordon-band values.',
            'Live threshold': 'Waiting for averaged PSD data to compare all axes with the selected vibration threshold.',
        }
        for name,message in descriptions.items():
            tab=ttk.Frame(self.tabs)
            self.tabs.add(tab,text=name)
            if name in ('Live stages','Live FFT + PSD','Live Issue 1 ensemble'):
                controls=ttk.Frame(tab,padding=(5,3)); controls.pack(fill=tk.X)
                label=('Individual FFT axis' if name=='Live Issue 1 ensemble'
                       else ('Axis' if name=='Live stages' else 'Phase axis'))
                axis_var=(self.issue1_waterfall_axis if name=='Live Issue 1 ensemble' else self.axis)
                ttk.Label(controls,text=label).pack(side=tk.LEFT,padx=(2,2))
                cb=ttk.Combobox(controls,textvariable=axis_var,values=list('XYZ'),state='readonly',width=5)
                cb.pack(side=tk.LEFT,padx=(0,10))
                cb.bind('<<ComboboxSelected>>',lambda e:self._request_live_redraw())
                note=('19 individual filtered FFT amplitudes.' if name=='Live Issue 1 ensemble'
                      else ('Processing-stage axis only.' if name=='Live stages'
                            else 'Magnitude/PSD show X/Y/Z; selector controls the phase panel.'))
                ttk.Label(controls,text=note).pack(side=tk.LEFT,padx=4)
            f=Figure(figsize=(10,6),dpi=100)
            canvas=FigureCanvasTkAgg(f,master=tab)
            canvas.get_tk_widget().pack(fill=tk.BOTH,expand=True)
            tb=NavigationToolbar2Tk(canvas,tab,pack_toolbar=False)
            tb.update(); tb.pack(fill=tk.X)
            self.live_analysis_fig[name]=(f,canvas)
            self._draw_live_analysis_empty(name,message)

    def _request_live_redraw(self):
        """Mark the selected live view stale after a processing-control change."""
        self.live_last_view_draw=0.0
        self.live_redraw_pending=True
        if self.live_t:
            self._refresh_visible_live_view()

    def _draw_live_analysis_empty(self,name,message):
        f,c=self._clear_live_analysis_figure(name)
        ax=f.add_subplot(111); ax.axis('off')
        ax.text(.5,.55,name,ha='center',va='center',fontsize=16,fontweight='bold')
        ax.text(.5,.43,message,ha='center',va='center',fontsize=11,wrap=True)
        f.tight_layout(); c.draw_idle()

    def _clear_live_analysis_figure(self,name):
        f,c=self.live_analysis_fig[name]
        # Figure.clear() propagates a temporary zero x-limit through shared
        # logarithmic axes, producing repeated non-positive-xlim warnings.
        for existing in list(f.axes):
            existing.remove()
        return f,c

    def _draw_live_empty(self,message='Connect to the STM32 through the ST-Link Virtual COM port and start capture.'):
        f=self.live_fig; f.clear(); ax=f.add_subplot(111); ax.axis('off')
        ax.text(.5,.55,'Live STM Accelerometer',ha='center',va='center',fontsize=16,fontweight='bold')
        ax.text(.5,.43,message,ha='center',va='center',fontsize=11,wrap=True)
        f.tight_layout(); self.live_canvas.draw_idle()

    def live_refresh_ports(self):
        """Refresh the COM-port selector while preserving a valid selection."""
        try:
            ports=available_serial_ports()
            values=[]
            for dev,desc,hwid in ports:
                label=f'{dev} — {desc}' if desc else dev
                values.append(label)
            self.live_port_combo['values']=values
            # Preserve an explicitly typed COM port. Otherwise choose first
            # ST-Link-looking entry (available_serial_ports sorts those first).
            if values and (not self.live_port.get().strip() or self.live_port.get().strip()=='COM4'):
                self.live_port.set(values[0].split(' — ',1)[0])
            if ports:
                self.status.set('Serial ports: '+', '.join(p[0] for p in ports))
            else:
                self.status.set('No serial ports found. Connect the ST-Link pod and refresh.')
        except Exception as exc:
            self.status.set(f'Could not enumerate serial ports: {exc}')

    def _selected_live_port(self):
        text=self.live_port.get().strip()
        return text.split(' — ',1)[0].strip()

    def live_connect(self):
        """Open the selected VCP on a worker thread to keep Tk responsive."""
        if self.live_connect_thread and self.live_connect_thread.is_alive():
            return
        port=self._selected_live_port()
        if not port:
            messagebox.showerror('Live connection','Select or enter the ST-Link Virtual COM port.')
            return
        try:
            baud=int(self.live_baud.get())
        except Exception:
            messagebox.showerror('Live connection','Baud must be an integer (VMM default is 460800).')
            return

        if self.live_client is not None:
            self.live_disconnect()
        self.accel_effective_config=None
        self.accel_readback_status.set('Live read-back: waiting for firmware CONFIG packet')
        self.live_link.set('Opening...')
        self.live_fw_version.set('VMM v1')
        self.btn_live_connect.configure(state=tk.DISABLED)

        def work():
            try:
                client=VmmStreamClient(port,baud)
                self.live_queue.put(('connected',client))
            except Exception as exc:
                self.live_queue.put(('connect_error',str(exc)))

        self.live_connect_thread=threading.Thread(target=work,daemon=True)
        self.live_connect_thread.start()

    def live_disconnect(self):
        """Stop acquisition, close the VCP, and reset connection controls."""
        self.live_stop()
        th=self.live_thread
        if th and th.is_alive():
            th.join(timeout=1.0)
        client=self.live_client
        self.live_client=None
        if client is not None:
            try: client.close()
            except Exception: pass
        self.live_link.set('Disconnected')
        self.live_capture.set('Stopped')
        self.btn_program_accel.configure(state=tk.DISABLED)

    def _operator_timeout_seconds(self):
        value=float(self.operator_timeout_value.get())
        if not math.isfinite(value) or value<=0:
            raise ValueError('Capture timeout duration must be greater than zero.')
        return value*(60.0 if self.operator_timeout_units.get()=='minutes' else 1.0)

    def _reset_capture_session_state(self):
        """Initialise operator timeout, renewal, and recovery bookkeeping."""
        now=time.monotonic()
        self.capture_started_monotonic=now
        self.operator_period_started_monotonic=now
        self.current_capture_indefinite=(not bool(self.operator_timeout_enabled.get())
                                         or self.operator_timeout_action.get()=='Continue indefinitely')
        self.operator_timeout_prompt_open=False
        self.timeout_extensions=0; self.stream_renewals=0; self.recovery_attempts=0
        self.last_sample_block_monotonic=None; self.next_vmm_renewal_monotonic=None

    def live_start(self):
        """Validate settings and launch the live serial polling worker."""
        if self.live_client is None:
            messagebox.showerror('Live capture','Open the ST-Link Virtual COM port first.')
            return
        if self.live_thread and self.live_thread.is_alive(): return
        try:
            if self.operator_timeout_enabled.get():
                self._operator_timeout_seconds()
        except Exception as exc:
            messagebox.showerror('Live capture',str(exc)); return
        # Centre the tilt target on the first averaged reading of each capture.
        self.live_tilt_zero_roll=None; self.live_tilt_zero_pitch=None
        self.live_tilt_readout.set('Waiting for complete stable gravity window before auto-zero...')
        self.live_last_block_seq=None; self.live_first_timestamp_us=None
        self.live_stop_event.clear(); self.live_capture.set('Starting...')
        self._reset_capture_session_state()
        client=self.live_client
        try:
            requested=self._requested_accelerometer_wire_config()
            self._begin_accelerometer_config_attempt(
                requested,'Applying sensor configuration before capture')
        except Exception as exc:
            self.live_capture.set('Error')
            self._set_accel_program_result('failure','FAILED - INVALID SETTINGS')
            messagebox.showerror('Live capture',f'Invalid accelerometer configuration: {exc}')
            return
        # This finite firmware timeout is deliberately separate from the user's
        # operator timeout. Indefinite captures are implemented by transparent
        # renewal rather than relying on an undocumented zero/infinite value.
        vmm_timeout_ms=max(10_000,int(requested.stream_timeout_ms))
        renewal_interval=max(2.0,(vmm_timeout_ms/1000.0)*float(self.vmm_renew_fraction))
        stall_limit=max(2.0,4.0*max(1,int(requested.fifo_watermark))/max(1.0,float(requested.odr_hz)))

        def work():
            try:
                client.configure(requested)
                client.start(timeout_ms=vmm_timeout_ms)
                client.status()
                started=time.monotonic(); next_renew=started+renewal_interval
                self.live_queue.put(('capture_started',{'started':started,'next_renew':next_renew,'vmm_timeout_ms':vmm_timeout_ms}))
                last_raw_progress=0.0
                sample_blocks_at_start=client.stats.sample_blocks
                no_data_reported=False
                last_sample_at=started
                last_recovery_at=0.0
                consecutive_recovery_attempts=0
                while not self.live_stop_event.is_set():
                    got_sample=False
                    for packet in client.poll():
                        if isinstance(packet,SampleBlock) and packet.xyz:
                            got_sample=True; last_sample_at=time.monotonic(); consecutive_recovery_attempts=0
                            if self.raw_logger.active:
                                try:
                                    progress=self.raw_logger.write_block(packet)
                                    now=time.monotonic()
                                    if progress is not None and now-last_raw_progress>=1.0:
                                        self.live_queue.put(('raw_log_progress',progress)); last_raw_progress=now
                                except Exception as exc:
                                    self.raw_logger.stop(); self.live_queue.put(('raw_log_error',str(exc)))
                            fs_g=float(packet.fs_g or 2)
                            raw_counts=np.asarray(packet.xyz,dtype=np.int16)
                            arr=raw_counts.astype(float)*(fs_g/32768.0)
                            meta={
                                'seq':int(packet.seq),'timestamp_us':int(packet.timestamp_us),
                                'sample_period_us':int(packet.sample_period_us),'odr_hz':int(packet.odr_hz),
                                'fs_g':fs_g,'status':int(packet.status),
                                'frames_bad':int(client.stats.frames_bad),'last_error':str(client.stats.last_error),
                                'raw_counts':raw_counts,
                            }
                            self.live_queue.put(('sample_block',(arr,meta)))
                        elif isinstance(packet,AccelerometerConfig):
                            self.live_queue.put(('config_readback',packet))
                        elif isinstance(packet,tuple) and packet and packet[0]=='ack':
                            detail=packet[3] if len(packet)>3 else 0
                            if packet[1]==CMD_CONFIGURE:
                                self.live_queue.put(('config_ack',(packet[2],detail)))
                            else:
                                self.live_queue.put(('log',f'VMM ACK command={packet[1]} status={packet[2]} detail={detail}'))

                    now=time.monotonic()
                    if now>=next_renew and not self.live_stop_event.is_set():
                        client.renew(timeout_ms=vmm_timeout_ms)
                        next_renew=now+renewal_interval
                        self.live_queue.put(('stream_renewed',{'when':now,'next_renew':next_renew,'reason':'scheduled'}))

                    if (not no_data_reported and now-started>=3.0
                            and client.stats.sample_blocks==sample_blocks_at_start):
                        no_data_reported=True
                        self.live_queue.put(('no_data',(
                            client.stats.bytes_rx,client.stats.frames_ok,
                            client.stats.frames_bad,client.stats.last_error)))

                    # A true stream stall is independent of the operator timeout.
                    if now-last_sample_at>stall_limit and now-last_recovery_at>stall_limit:
                        consecutive_recovery_attempts+=1; last_recovery_at=now
                        if consecutive_recovery_attempts>3:
                            raise RuntimeError('Stream stalled and recovery failed after 3 START renewals.')
                        self.live_queue.put(('stream_recovery_attempt',consecutive_recovery_attempts))
                        try:
                            client.status()
                            client.renew(timeout_ms=vmm_timeout_ms)
                            next_renew=now+renewal_interval
                        except Exception as exc:
                            self.live_queue.put(('log',f'Stream recovery write failed: {exc}'))
                    time.sleep(.002)
                try: client.stop()
                except Exception: pass
                self.live_queue.put(('capture_stopped',None))
            except Exception as exc:
                self.live_queue.put(('capture_error',str(exc)))
        self.live_thread=threading.Thread(target=work,daemon=True); self.live_thread.start()

    def live_stop(self):
        """Signal the worker to finish; final UI cleanup arrives via its queue."""
        self.live_stop_event.set()
        if self.live_capture.get() not in ('Stopped','Error'): self.live_capture.set('Stopping...')

    @staticmethod
    def _safe_filename_token(value):
        text=str(value).strip()
        text=re.sub(r'[^A-Za-z0-9._-]+','_',text)
        text=re.sub(r'_+','_',text).strip('_.-')
        return text or 'NA'

    def _raw_log_metadata_dialog(self):
        """Collect the minimum test-house metadata before a raw-count log starts."""
        win=tk.Toplevel(self.root)
        win.title('Raw-count log details')
        win.transient(self.root)
        win.grab_set()
        win.resizable(False,False)
        body=ttk.Frame(win,padding=12); body.pack(fill=tk.BOTH,expand=True)

        defaults=self.last_raw_log_defaults
        reader=tk.StringVar(value=defaults.get('reader_id',''))
        axis=tk.StringVar(value=defaults.get('excited_axis','X'))
        level=tk.StringVar(value=defaults.get('vibration_level',''))
        run=tk.StringVar(value=defaults.get('run_number','1'))
        notes=tk.StringVar(value='')

        rows=[
            ('Reader ID',reader),
            ('Excited axis',axis),
            ('Requested vibration level',level),
            ('Run number',run),
            ('Notes (optional)',notes),
        ]
        for r,(label,var) in enumerate(rows):
            ttk.Label(body,text=label).grid(row=r,column=0,sticky='w',padx=(0,10),pady=4)
            if label=='Excited axis':
                w=ttk.Combobox(body,textvariable=var,values=list('XYZ'),state='readonly',width=28)
            else:
                w=ttk.Entry(body,textvariable=var,width=31)
            w.grid(row=r,column=1,sticky='ew',pady=4)
            if r==0: first=w
        ttk.Label(body,text='Example level: 0.5 x Gordon, Office, 1.5 x Gordon, 0.8 mm/s.',
                  wraplength=370,justify=tk.LEFT).grid(row=len(rows),column=0,columnspan=2,sticky='w',pady=(3,10))

        result={'value':None}
        def accept():
            rid=reader.get().strip(); lev=level.get().strip(); ax=axis.get().strip().upper()
            if not rid:
                messagebox.showwarning('Raw-count log','Enter a Reader ID.',parent=win); return
            if ax not in 'XYZ':
                messagebox.showwarning('Raw-count log','Select the excited X, Y or Z axis.',parent=win); return
            if not lev:
                messagebox.showwarning('Raw-count log','Enter the requested vibration level.',parent=win); return
            try:
                run_no=int(run.get().strip())
                if run_no<1: raise ValueError
            except Exception:
                messagebox.showwarning('Raw-count log','Run number must be a positive whole number.',parent=win); return
            result['value']={
                'reader_id':rid,
                'excited_axis':ax,
                'vibration_level':lev,
                'run_number':run_no,
                'notes':notes.get().strip(),
            }
            self.last_raw_log_defaults={
                'reader_id':rid,'excited_axis':ax,'vibration_level':lev,
                'run_number':str(run_no+1),'notes':''
            }
            win.destroy()
        def cancel(): win.destroy()
        buttons=ttk.Frame(body); buttons.grid(row=len(rows)+1,column=0,columnspan=2,sticky='e')
        ttk.Button(buttons,text='Cancel',command=cancel).pack(side=tk.RIGHT,padx=(6,0))
        ttk.Button(buttons,text='Continue',command=accept).pack(side=tk.RIGHT)
        win.protocol('WM_DELETE_WINDOW',cancel)
        try: first.focus_set()
        except Exception: pass
        self.root.wait_window(win)
        return result['value']

    def _settings_snapshot(self):
        """Return JSON-safe acquisition/processing settings for traceability."""
        effective=self.accel_effective_config
        if effective is not None:
            try:
                effective={k:getattr(effective,k) for k in vars(effective)}
            except Exception:
                effective=str(effective)
        return {
            'application_title':APP_TITLE,
            'capture_settings':{
                'port':self.live_port.get(),
                'baud':int(self.live_baud.get()),
                'requested_device':self.accel_device.get(),
                'requested_full_scale_g':int(self.accel_fs_g.get()),
                'requested_odr_hz':int(self.accel_odr_hz.get()),
                'requested_operating_mode':self.accel_mode.get(),
                'requested_antialias_enabled':bool(self.accel_aa_enabled.get()),
                'requested_antialias_bandwidth':self.accel_bw.get(),
                'fifo_watermark':int(self.accel_fifo_watermark.get()),
                'int1_fifo_threshold':bool(self.accel_int1_fifo_threshold.get()),
                'effective_configuration':effective,
            },
            'processing_settings':{
                'processing_method':self.applied_processing_method,
                'application_prefilter':('Issue 1 FIR 3.5-90 Hz'
                                         if self._issue1_active() else 'None'),
                'sample_rate_hz':float(self.fs.get()),
                'fft_length':int(self.n.get()),
                'fft_window':self.accel_fft_window.get(),
                'fft_overlap_percent':float(self.ov.get()),
                'psd_averages':int(self.segs.get()),
                'fft_ensemble_count':(int(self.segs.get()) if self._issue1_active() else None),
                'gordon_units':self.units.get(),
                'tilt_average_samples':int(self.tilt_average_samples.get()),
                'shock_enabled':bool(self.live_shock_enabled.get()),
                'shock_threshold_g':float(self.live_shock_threshold_g.get()),
                'shock_steady_state_samples':int(self.live_shock_window_samples),
                'live_k_correction_enabled':bool(self.live_k_correction_enabled.get()),
                'live_k_source_files':[str(Path(x).name) for x in self.live_k_source_files],
            },
        }

    def _write_raw_settings_log(self, csv_path, metadata):
        csv_path=Path(csv_path)
        sidecar=csv_path.with_name(csv_path.stem+'_settings.json')
        payload={
            'log_type':'1174 raw accelerometer acquisition settings',
            'capture_start_local':time.strftime('%Y-%m-%d %H:%M:%S'),
            'raw_csv':csv_path.name,
            'test':metadata,
        }
        payload.update(self._settings_snapshot())
        sidecar.write_text(json.dumps(payload,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
        return sidecar

    def live_start_raw_log(self):
        metadata=self._raw_log_metadata_dialog()
        if metadata is None: return
        stamp=time.strftime('%Y%m%d_%H%M%S')
        rid=self._safe_filename_token(metadata['reader_id'])
        lev=self._safe_filename_token(metadata['vibration_level'])
        initial=f"{rid}_{metadata['excited_axis']}_{lev}_Run{int(metadata['run_number']):02d}_{stamp}.csv"
        path=filedialog.asksaveasfilename(
            defaultextension='.csv',filetypes=[('Raw-count CSV','*.csv')],initialfile=initial)
        if not path: return
        try:
            self.raw_logger.start(path)
            self.raw_log_metadata=dict(metadata)
            self.raw_log_settings_path=self._write_raw_settings_log(path,metadata)
        except Exception as exc:
            try: self.raw_logger.stop()
            except Exception: pass
            messagebox.showerror('Raw-count log',str(exc)); return
        self.raw_log_status.set(
            f"Raw log: {metadata['reader_id']} / {metadata['excited_axis']} / {metadata['vibration_level']} / "
            f"Run {metadata['run_number']} → {Path(path).name}")
        self.btn_raw_log_start.configure(state=tk.DISABLED)
        self.btn_raw_log_stop.configure(state=tk.NORMAL)

    def live_stop_raw_log(self):
        blocks,samples,path=self.raw_logger.stop()
        name=path.name if path else 'file'
        sidecar=Path(self.raw_log_settings_path).name if self.raw_log_settings_path else 'no settings log'
        self.raw_log_status.set(f'Raw log: stopped — {samples:,} samples in {name}; settings {sidecar}')
        self.btn_raw_log_start.configure(state=tk.NORMAL)
        self.btn_raw_log_stop.configure(state=tk.DISABLED)

    def live_clear(self):
        self.live_t.clear(); self.live_x.clear(); self.live_y.clear(); self.live_z.clear()
        self.live_raw_x.clear(); self.live_raw_y.clear(); self.live_raw_z.clear()
        self.live_last_status=0
        self.live_stream_health.set('Stream health: waiting for data')
        self.live_packets=0; self.live_dropped_or_invalid=0
        self.live_last_block_seq=None; self.live_first_timestamp_us=None
        self.live_stream_time_offset=0.0; self.live_sequence_gaps=0; self.live_bad_frames=0
        self.live_total_samples=0
        self.live_stream_fs.set('Unknown')
        self._draw_live_empty('Capture cleared. Start capture to display live data.')
        self.live_last_view_draw=0.0
        self.live_last_selected_tab=None
        self.live_redraw_pending=False
        self.live_tilt_zero_roll=None; self.live_tilt_zero_pitch=None
        self.live_tilt_readout.set('Waiting for live samples')
        self.live_shock_peak_g=0.0; self.live_shock_events=0; self.live_shock_prev_exceeded=False; self.live_shock_last_processed_time=None
        self.live_shock_readout.set('Shock: waiting for 50 samples')
        for name in self.live_analysis_fig:
            self._draw_live_analysis_empty(name,'Capture cleared. Start capture to display live data.')

    def _append_live_samples(self,arr,meta=None):
        """Convert one raw sample block to g and append it to retained history."""
        meta=meta or {}
        period_us=int(meta.get('sample_period_us') or 0)
        odr_hz=float(meta.get('odr_hz') or 0)
        fs=(1e6/period_us) if period_us>0 else (odr_hz if odr_hz>0 else float(self.fs.get()))
        self.fs.set(fs)
        # Live processing follows the measured block timing, as before; keep
        # the read-only summary aligned with the value the calculations use.
        self._update_processing_status()
        n=len(arr)
        timestamp_us=int(meta.get('timestamp_us') or 0)
        if timestamp_us>0 and period_us>0:
            if self.live_first_timestamp_us is None:
                self.live_first_timestamp_us=timestamp_us
                self.live_stream_time_offset=(self.live_t[-1]+1.0/fs) if self.live_t else 0.0
            relative=(timestamp_us-self.live_first_timestamp_us)/1e6
            tt=(self.live_stream_time_offset+relative+np.arange(n,dtype=float)*(period_us/1e6)).tolist()
        else:
            start_time=(self.live_t[-1]+1.0/fs) if self.live_t else 0.0
            tt=(start_time+np.arange(n,dtype=float)/fs).tolist()
        self.live_t.extend(tt); self.live_x.extend(arr[:,0].tolist()); self.live_y.extend(arr[:,1].tolist()); self.live_z.extend(arr[:,2].tolist())
        raw_counts=meta.get('raw_counts')
        if raw_counts is not None:
            raw_counts=np.asarray(raw_counts,dtype=np.int16).reshape(-1,3)
            self.live_raw_x.extend(raw_counts[:,0].astype(int).tolist())
            self.live_raw_y.extend(raw_counts[:,1].astype(int).tolist())
            self.live_raw_z.extend(raw_counts[:,2].astype(int).tolist())
        self.live_total_samples+=n
        try: retention_s=max(1.0,float(self.live_retention_seconds.get()))
        except (ValueError,TypeError,tk.TclError): retention_s=120.0
        max_samples=max(1,int(round(retention_s*fs)))
        excess=len(self.live_t)-max_samples
        if excess>0:
            del self.live_t[:excess]; del self.live_x[:excess]; del self.live_y[:excess]; del self.live_z[:excess]
            if self.live_raw_x:
                del self.live_raw_x[:excess]; del self.live_raw_y[:excess]; del self.live_raw_z[:excess]
        self.live_packets+=1
        seq=meta.get('seq')
        if seq is not None:
            seq=int(seq)&0xFFFFFFFF
            if self.live_last_block_seq is not None:
                gap=(seq-((self.live_last_block_seq+1)&0xFFFFFFFF))&0xFFFFFFFF
                if 0<gap<0x80000000: self.live_sequence_gaps+=gap
            self.live_last_block_seq=seq
        self.live_bad_frames=int(meta.get('frames_bad') or self.live_bad_frames)
        self.live_dropped_or_invalid=self.live_sequence_gaps+self.live_bad_frames
        full_scale=float(meta.get('fs_g') or 2)
        status=int(meta.get('status') or 0)
        self.live_last_status=status
        self.live_stream_fs.set(f'±{full_scale:g} g')
        self.live_fw_version.set(f'VMM v1; {fs:g} Hz; ±{full_scale:g} g; status 0x{status:02X}')
        if self.accel_effective_config is None:
            try:
                req_fs=float(self.accel_fs_g.get()); req_odr=float(self.accel_odr_hz.get())
                fs_ok=abs(full_scale-req_fs)<1e-6
                odr_ok=abs(fs-req_odr)<max(0.5,0.01*req_odr)
                self.accel_readback_status.set(
                    f"VMM metadata: {fs:g} Hz, ±{full_scale:g} g — "
                    f"ODR {'MATCH' if odr_ok else 'DIFFERS'}, "
                    f"FS {'MATCH' if fs_ok else 'DIFFERS'}")
            except Exception:
                self.accel_readback_status.set(
                    f'VMM metadata: {fs:g} Hz, ±{full_scale:g} g')
        # Stream-health diagnostics and plots are intentionally refreshed by
        # the GUI scheduler, not once per incoming block.

    def _apply_accelerometer_readback(self,config):
        """Compare firmware-effective configuration with the pending request."""
        self.accel_effective_config=config
        mode='High Performance' if config.high_performance else 'Low Power'
        bw=f'ODR/{config.bandwidth_divisor}'
        pending=self.accel_pending_config
        try:
            requested=(
                {
                    'odr_hz':pending.odr_hz,
                    'full_scale_g':pending.fs_g,
                    'operating_mode':'High Performance' if pending.high_performance else 'Low Power',
                    'anti_alias_bandwidth':f'ODR/{pending.bandwidth_divisor}',
                    'fifo_watermark':pending.fifo_watermark,
                    'int1_fifo_threshold':pending.int1_fifo_threshold,
                    'stream_timeout_ms':pending.stream_timeout_ms,
                }
                if pending is not None else self._accelerometer_config_dict()
            )
            matches=(
                config.odr_hz==int(requested['odr_hz'])
                and config.fs_g==int(requested['full_scale_g'])
                and mode==requested['operating_mode']
                and bw==requested['anti_alias_bandwidth']
                and config.fifo_watermark==int(requested['fifo_watermark'])
                and config.int1_fifo_threshold==bool(requested['int1_fifo_threshold'])
                and config.stream_timeout_ms==int(requested['stream_timeout_ms'])
            )
        except Exception:
            matches=False
        if pending is not None and not self.accel_config_ack_ok:
            # STATUS can return the previously effective CONFIG before the
            # asynchronous CONFIGURE request is applied. Do not treat that
            # packet as proof (or as a mismatch); wait for CONFIGURE ACK and
            # the firmware's post-apply CONFIG packet.
            self.accel_readback_status.set(
                f'Pre-ACK read-back received: {mode}, {config.odr_hz} Hz, '
                f'Â±{config.fs_g} g, {bw}; waiting for confirmed read-back')
            self.accel_config_status.set(
                'Waiting for CONFIGURE acknowledgement before verifying registers.')
            self._set_accel_program_result('pending','WAITING FOR CONFIG ACK...')
            self.live_fw_version.set(
                f'VMM v1 config; {mode}; {config.odr_hz} Hz; Â±{config.fs_g} g; {bw}')
            return
        state='VERIFIED' if matches else 'DIFFERS FROM REQUEST'
        self.accel_readback_status.set(
            f'{state}: {mode}, {config.odr_hz} Hz, ±{config.fs_g} g, {bw}, '
            f'FIFO {config.fifo_watermark}, INT1 '
            f'{"ON" if config.int1_fifo_threshold else "OFF"}, '
            f'timeout {config.stream_timeout_ms} ms')
        self.accel_config_status.set(
            'Firmware register read-back verified.' if matches and pending is not None and self.accel_config_ack_ok
            else ('Firmware returned an effective configuration that differs from the requested values.'
                  if pending is not None and not matches
                  else 'Firmware configuration read-back received.'))
        if pending is not None:
            if matches and self.accel_config_ack_ok:
                self._set_accel_program_result('success')
            elif not matches:
                self._set_accel_program_result('failure','FAILED - READ-BACK MISMATCH')
            self.accel_pending_config=None
            self.accel_config_ack_ok=False
        self.live_fw_version.set(
            f'VMM v1 config; {mode}; {config.odr_hz} Hz; ±{config.fs_g} g; {bw}')

    def _format_duration(self,seconds):
        seconds=max(0,int(round(seconds)))
        h,rem=divmod(seconds,3600); m,sec=divmod(rem,60)
        return f'{h:02d}:{m:02d}:{sec:02d}' if h else f'{m:02d}:{sec:02d}'

    def _continue_timeout_period(self):
        self.operator_period_started_monotonic=time.monotonic()
        self.timeout_extensions+=1
        self.operator_timeout_prompt_open=False

    def _continue_capture_indefinitely(self):
        self.current_capture_indefinite=True
        self.operator_timeout_prompt_open=False

    def _stop_from_timeout(self):
        self.operator_timeout_prompt_open=False
        self.live_stop()

    def _show_operator_timeout_dialog(self):
        if self.operator_timeout_prompt_open or self.live_capture.get() not in ('Running','Running - no sample data'):
            return
        self.operator_timeout_prompt_open=True
        seconds=self._operator_timeout_seconds()
        if self.operator_timeout_units.get()=='minutes':
            amount=f'{self.operator_timeout_value.get():g} minute(s)'
        else:
            amount=f'{self.operator_timeout_value.get():g} second(s)'
        win=tk.Toplevel(self.root); win.title('Capture timeout reached'); win.transient(self.root); win.grab_set()
        ttk.Label(win,text=f'The configured {amount} capture period has completed.\n\nWhat would you like to do?',
                  justify=tk.LEFT,padding=14).pack(fill=tk.X)
        buttons=ttk.Frame(win,padding=(10,0,10,10)); buttons.pack(fill=tk.X)
        def finish(action):
            try: win.grab_release()
            except Exception: pass
            win.destroy(); action()
        ttk.Button(buttons,text=f'Continue another {amount}',command=lambda:finish(self._continue_timeout_period)).pack(side=tk.LEFT,padx=4)
        ttk.Button(buttons,text='Continue indefinitely',command=lambda:finish(self._continue_capture_indefinitely)).pack(side=tk.LEFT,padx=4)
        ttk.Button(buttons,text='Stop capture',command=lambda:finish(self._stop_from_timeout)).pack(side=tk.LEFT,padx=4)
        # Closing the dialog preserves data by extending one more period rather
        # than accidentally terminating a bench test.
        win.protocol('WM_DELETE_WINDOW',lambda:finish(self._continue_timeout_period))

    def _check_operator_timeout(self):
        """Handle user capture deadlines independently of the VMM lease timer."""
        if self.capture_started_monotonic is None or self.live_capture.get() not in ('Running','Running - no sample data'):
            return
        if not self.operator_timeout_enabled.get() or self.current_capture_indefinite:
            return
        try: period=self._operator_timeout_seconds()
        except Exception: return
        if self.operator_period_started_monotonic is None:
            self.operator_period_started_monotonic=time.monotonic()
        if time.monotonic()-self.operator_period_started_monotonic<period:
            return
        action=self.operator_timeout_action.get()
        if action=='Ask': self._show_operator_timeout_dialog()
        elif action=='Continue same duration': self._continue_timeout_period()
        elif action=='Continue indefinitely': self._continue_capture_indefinitely()
        elif action=='Stop': self._stop_from_timeout()

    def _update_live_diagnostics(self):
        now=time.monotonic()
        if self.capture_started_monotonic is None:
            self.live_diagnostics.set('Capture diagnostics: stopped'); return
        elapsed=now-self.capture_started_monotonic
        if not self.operator_timeout_enabled.get() or self.current_capture_indefinite:
            op='OFF - continued indefinitely' if self.current_capture_indefinite and self.operator_timeout_enabled.get() else 'OFF - indefinite capture'
        else:
            try:
                period=self._operator_timeout_seconds(); start=self.operator_period_started_monotonic or now
                op=self._format_duration(max(0.0,period-(now-start)))+' remaining'
            except Exception: op='invalid setting'
        renew='n/a'
        if self.next_vmm_renewal_monotonic is not None:
            renew=self._format_duration(max(0.0,self.next_vmm_renewal_monotonic-now))
        age='n/a' if self.last_sample_block_monotonic is None else f'{max(0.0,now-self.last_sample_block_monotonic):.2f} s'
        raw='Recording' if self.raw_logger.active else 'Off'
        self.live_diagnostics.set(
            f'Elapsed {self._format_duration(elapsed)} | Operator timeout {op} | VMM renewal {renew} | '
            f'Last block age {age} | Samples {self.live_total_samples:,} | Blocks {self.live_packets:,} | '
            f'Renewals {self.stream_renewals} | Extensions {self.timeout_extensions} | Recoveries {self.recovery_attempts} | '
            f'Seq gaps {self.live_sequence_gaps} | Bad frames {self.live_bad_frames} | Raw log {raw}')

    def _poll_live_queue(self):
        """Apply serial-worker events on Tk's main thread and schedule next poll."""
        redraw=False
        processed=0
        try:
            # Bound work per Tk callback so a backlog cannot monopolise the UI
            # thread. Remaining items are handled by the next callback.
            while processed<100:
                kind,payload=self.live_queue.get_nowait()
                processed+=1
                if kind=='connected':
                    self.live_client=payload
                    self.live_link.set('Connected'); self.btn_live_connect.configure(state=tk.NORMAL)
                    self.btn_program_accel.configure(state=tk.DISABLED)
                    self.live_fw_version.set('VMM v1')
                    self.status.set('ST-Link VCP opened. Start capture to send the VMM START command at 460800 baud.')
                elif kind=='connect_error':
                    self.live_link.set('Connection failed'); self.live_fw_version.set('VMM v1'); self.btn_live_connect.configure(state=tk.NORMAL)
                    self.btn_program_accel.configure(state=tk.DISABLED)
                    messagebox.showerror('Live connection',payload)
                elif kind=='capture_started':
                    self.live_capture.set('Running'); self.btn_program_accel.configure(state=tk.NORMAL)
                    if isinstance(payload,dict):
                        self.capture_started_monotonic=float(payload.get('started') or time.monotonic())
                        self.operator_period_started_monotonic=self.capture_started_monotonic
                        self.next_vmm_renewal_monotonic=float(payload.get('next_renew') or 0) or None
                elif kind=='capture_stopped':
                    self.live_capture.set('Stopped'); self.btn_program_accel.configure(state=tk.DISABLED)
                    self.next_vmm_renewal_monotonic=None; self.operator_timeout_prompt_open=False
                    self.capture_started_monotonic=None; self.operator_period_started_monotonic=None
                    if self.raw_logger.active:
                        blocks,samples,path=self.raw_logger.stop()
                        name=path.name if path else 'file'
                        self.raw_log_status.set(f'Raw log: stopped — {samples:,} samples in {name}')
                        self.btn_raw_log_start.configure(state=tk.NORMAL); self.btn_raw_log_stop.configure(state=tk.DISABLED)
                elif kind=='capture_error':
                    self.live_capture.set('Error'); self.live_link.set('Check connection')
                    self.capture_started_monotonic=None; self.operator_period_started_monotonic=None; self.next_vmm_renewal_monotonic=None
                    if self.raw_logger.active:
                        blocks,samples,path=self.raw_logger.stop()
                        name=path.name if path else 'file'
                        self.raw_log_status.set(f'Raw log: stopped after capture error — {samples:,} samples in {name}')
                        self.btn_raw_log_start.configure(state=tk.NORMAL); self.btn_raw_log_stop.configure(state=tk.DISABLED)
                    self.accel_pending_config=None; self.accel_config_ack_ok=False
                    self._set_accel_program_result('failure','FAILED - CAPTURE ERROR')
                    self.btn_program_accel.configure(state=tk.DISABLED)
                    messagebox.showerror('Live capture',payload)
                elif kind=='sample_block':
                    if self.live_capture.get()=='Running - no sample data':
                        self.live_capture.set('Running')
                    self.last_sample_block_monotonic=time.monotonic()
                    arr,meta=payload; self._append_live_samples(arr,meta); redraw=True
                elif kind=='stream_renewed':
                    self.stream_renewals+=1
                    self.next_vmm_renewal_monotonic=float(payload.get('next_renew') or 0) or None
                    self.status.set('VMM START lease renewed without clearing the capture.')
                elif kind=='stream_recovery_attempt':
                    self.recovery_attempts+=1
                    self.status.set(f'Stream stalled - attempting recovery ({payload}/3).')
                elif kind=='no_data':
                    bytes_rx,frames_ok,frames_bad,last_error=payload
                    self.live_capture.set('Running - no sample data')
                    self.status.set(
                        f'No accelerometer samples after 3 s: RX {bytes_rx} bytes, '
                        f'{frames_ok} valid frames, {frames_bad} bad frames'
                        + (f'; last error: {last_error}' if last_error else ''))
                elif kind=='raw_log_progress':
                    blocks,samples,path=payload
                    self.raw_log_status.set(f'Raw log: {samples:,} samples ({blocks:,} blocks) → {path.name}')
                elif kind=='raw_log_error':
                    self.raw_log_status.set('Raw log: error')
                    self.btn_raw_log_start.configure(state=tk.NORMAL)
                    self.btn_raw_log_stop.configure(state=tk.DISABLED)
                    messagebox.showerror('Raw-count log',payload)
                elif kind=='config_ack':
                    status,detail=payload
                    if status==0:
                        self.accel_config_ack_ok=True
                        self._set_accel_program_result('pending','ACK OK - VERIFYING...')
                        self.accel_config_status.set('Firmware accepted CONFIGURE; waiting for register read-back...')
                    else:
                        self.accel_pending_config=None
                        self.accel_config_ack_ok=False
                        self._set_accel_program_result('failure',f'FAILED - FW STATUS {status}')
                        self.accel_config_status.set(f'CONFIGURE failed: ACK status={status}, firmware detail={detail}')
                elif kind=='config_readback':
                    self._apply_accelerometer_readback(payload)
                elif kind=='log': self.status.set(payload)
        except queue.Empty:
            pass
        if redraw:
            self.live_redraw_pending=True
        self._check_operator_timeout()
        self._update_live_diagnostics()
        self._refresh_visible_live_view()
        self.root.after(25 if not self.live_queue.empty() else 100,self._poll_live_queue)

    def _refresh_visible_live_view(self):
        """Redraw only the selected live tab to limit UI processing load."""
        """Refresh only the visible live figure at a human-readable rate."""
        if not self.live_t:
            return
        try:
            selected=self.tabs.tab(self.tabs.select(),'text')
        except tk.TclError:
            return
        changed=selected!=self.live_last_selected_tab
        self.live_last_selected_tab=selected
        updaters={
            'Live STM data':self._update_live_plot,
            'Live raw':self._update_live_raw_plot,
            'Live raw counts':self._update_live_raw_counts_plot,
            'Live stages':self._update_live_stage_plot,
            'Live FFT + PSD':self._update_live_spectrum_plot,
            'Live Issue 1 ensemble':self._update_live_issue1_plot,
            'Live K correction':self._update_live_k_correction_plot,
            'Live threshold':self._update_live_threshold_plot,
        }
        updater=updaters.get(selected)
        if updater is None:
            return
        now=time.monotonic()
        # Tilt benefits from a faster visual update. Multi-axis FFT/PSD and
        # threshold figures are expensive and gain nothing from redrawing more
        # frequently than once per second at a 200 Hz sample rate.
        if selected in ('Live FFT + PSD','Live Issue 1 ensemble','Live K correction','Live threshold','Live stages','Live raw counts'):
            interval=1.0
        else:
            interval=0.5
        if not changed and (not self.live_redraw_pending or now-self.live_last_view_draw<interval):
            return
        if selected in ('Live STM data','Live raw counts'):
            self._update_stream_health()
        updater()
        self.live_last_view_draw=now
        self.live_redraw_pending=False

    def _live_welch(self,axis_values):
        """Compute the configured live Welch PSD when enough samples exist."""
        if self._issue1_active():
            return None
        n=int(self.n.get()); ov=float(self.ov.get())/100.; segs=int(self.segs.get()); hop=int(round(n*(1-ov))); need=n+(segs-1)*hop
        if len(axis_values)<need: return None
        data=np.asarray(axis_values[-need:],float); ps=[]
        for i in range(segs):
            s=i*hop; freq,p,*_=periodogram(data[s:s+n],float(self.fs.get())); ps.append(p)
        pstack=np.vstack(ps)
        return freq,pstack,np.mean(pstack,axis=0)

    def _live_welch_bands(self,axis_values):
        result=self._live_welch(axis_values)
        if result is None: return None
        freq,_,pavg=result
        tmp=Processor('Live'); return tmp.bands(freq,pavg)

    def _band_values_for_units(self,bands):
        mode=self.units.get()
        if mode=='RMS acceleration (g)': return np.asarray(bands.a_rms_g,float),'RMS g'
        if mode=='RMS displacement (µm)': return np.asarray(bands.x_rms_um,float),'RMS µm'
        return np.asarray(bands.v_rms_um_s,float),'RMS µm/s'

    def _live_k_complete(self):
        return self.live_k_method is not None and self.live_k_coordinates is not None and all(self.live_k_factors.get(axis) is not None for axis in 'XYZ')

    def _on_live_k_toggle(self):
        if self.live_k_correction_enabled.get() and not self._live_k_complete():
            self.live_k_correction_enabled.set(False)
            messagebox.showinfo('Live translation','Load a complete X/Y/Z translation set before enabling live translation.')
            return
        if self.live_k_correction_enabled.get() and self.live_k_method!=self.applied_processing_method:
            self.live_k_correction_enabled.set(False)
            messagebox.showerror('Live translation method mismatch',f'Loaded translation uses {self.live_k_method}; active processing uses {self.applied_processing_method}.')
            return
        state='ON' if self.live_k_correction_enabled.get() else 'OFF'
        if self._live_k_complete():
            self.live_k_status.set(f'Translation: {state}; {self.live_k_method}; {len(self.live_k_coordinates)} coefficients/axis')
        self.live_last_view_draw=0.0
        self.live_redraw_pending=True
        self._refresh_visible_live_view()

    def clear_live_k_correction(self):
        self.live_k_correction_enabled.set(False)
        self.live_k_factors={axis:None for axis in 'XYZ'}
        self.live_k_coordinates=None
        self.live_k_method=None
        self.live_k_source_files=[]
        self.live_k_status.set('Translation: not loaded')
        self.live_last_view_draw=0.0
        self.live_redraw_pending=True
        self._refresh_visible_live_view()

    def load_live_k_correction(self):
        """Load a complete Issue-1 or Issue-2 XYZ translation CSV set."""
        paths=filedialog.askopenfilenames(title='Load X/Y/Z translation CSV(s)',filetypes=[('CSV','*.csv'),('All files','*.*')])
        if not paths: return
        collected={axis:{} for axis in 'XYZ'}; source_files=[]; methods=set()
        try:
            for path in paths:
                df=pd.read_csv(path)
                cols={str(c).strip().lower():c for c in df.columns}
                method_col=cols.get('processing_method') or cols.get('method')
                if method_col is None:
                    file_methods={ISSUE2_METHOD}
                else:
                    file_methods={str(v).strip() for v in df[method_col].dropna().unique()}
                if len(file_methods)!=1: raise ValueError(f'{Path(path).name}: translation file contains mixed processing methods.')
                method=next(iter(file_methods)); methods.add(method)
                axis_col=cols.get('axis') or cols.get('driven_axis')
                coord_col=(cols.get('frequency_hz') if method==ISSUE1_METHOD else (cols.get('fc_hz') or cols.get('band_hz') or cols.get('frequency_hz')))
                k_col=cols.get('k_base_over_reader') or cols.get('k') or cols.get('k_factor') or cols.get('correction_factor')
                if coord_col is None or k_col is None:
                    raise ValueError(f'{Path(path).name}: expected coordinate and K_base_over_reader columns.')
                if axis_col is None:
                    name=Path(path).stem.upper(); hits=[]
                    for axis in 'XYZ':
                        padded=f'_{name}_'
                        if f'_{axis}_' in padded or name.startswith(axis+'_') or name.endswith('_'+axis): hits.append(axis)
                    if len(hits)!=1: raise ValueError(f'{Path(path).name}: axis/driven_axis column is required when filename is ambiguous.')
                    axes=np.full(len(df),hits[0],dtype=object)
                else:
                    axes=df[axis_col].astype(str).str.strip().str.upper().to_numpy()
                coords=pd.to_numeric(df[coord_col],errors='coerce').to_numpy(float)
                ks=pd.to_numeric(df[k_col],errors='coerce').to_numpy(float)
                for axis,coord,k in zip(axes,coords,ks):
                    if axis not in 'XYZ' or not np.isfinite(coord) or not np.isfinite(k) or k<=0:
                        raise ValueError(f'{Path(path).name}: invalid axis/coordinate/K row.')
                    collected[axis][round(float(coord),9)]=float(k)
                source_files.append(Path(path).name)
            if len(methods)!=1: raise ValueError('Cannot load a translation set containing both Issue 1 and Issue 2 data.')
            method=next(iter(methods))
            expected=(np.asarray(GORDON_FC,float) if method==ISSUE2_METHOD else np.fft.rfftfreq(ISSUE1_FFT_N,1.0/ISSUE1_SAMPLE_RATE_HZ))
            if method==ISSUE1_METHOD:
                expected=expected[(expected>=ISSUE1_LOW_HZ)&(expected<=ISSUE1_HIGH_HZ)]
            factors={}
            for axis in 'XYZ':
                vals=[]
                for coord in expected:
                    key=round(float(coord),9)
                    if key not in collected[axis]: raise ValueError(f'Incomplete {method} translation: missing {axis} at {coord:g} Hz.')
                    vals.append(collected[axis][key])
                factors[axis]=np.asarray(vals,float)
            self.live_k_method=method
            self.live_k_coordinates=np.asarray(expected,float)
            self.live_k_factors=factors
            self.live_k_source_files=source_files
            self.live_k_correction_enabled.set(False)
            self.live_k_status.set(f'Translation: loaded {method}, {len(expected)} coefficients/axis; OFF')
            self.status.set(f'Loaded complete XYZ {method} live translation set.')
            self.live_last_view_draw=0.0; self.live_redraw_pending=True
        except Exception as exc:
            messagebox.showerror('Load translation',str(exc))

    def _apply_live_k(self,axis,values,force=False):
        """Apply Issue-2 K after Gordon-band integration; raw samples stay unchanged."""
        arr=np.asarray(values,float)
        if not self._live_k_complete() or self.live_k_method!=ISSUE2_METHOD:
            return arr.copy()
        if not force and not self.live_k_correction_enabled.get():
            return arr.copy()
        return arr*np.asarray(self.live_k_factors[axis],float)

    def _apply_issue1_translation_result(self,result,force=False):
        """Return an Issue1Result with loaded K applied at Issue-1 FFT coordinates."""
        if not self._live_k_complete() or self.live_k_method!=ISSUE1_METHOD:
            return result
        if not force and not self.live_k_correction_enabled.get():
            return result
        ensemble=np.asarray(result.ensemble_amplitude_g,float).copy()
        coords=np.asarray(self.live_k_coordinates,float)
        for ai,axis in enumerate('XYZ'):
            for coord,k in zip(coords,self.live_k_factors[axis]):
                idx=int(np.argmin(np.abs(result.frequency_hz-float(coord))))
                if abs(float(result.frequency_hz[idx])-float(coord))<1e-6:
                    ensemble[ai,idx]*=float(k)
        exceeded=ensemble>result.threshold_amplitude_g[None,:]
        ratio=np.divide(ensemble,result.threshold_amplitude_g[None,:],out=np.zeros_like(ensemble),where=np.isfinite(result.threshold_amplitude_g[None,:])&(result.threshold_amplitude_g[None,:]>0))
        flat=int(np.argmax(ratio)); ai,bi=np.unravel_index(flat,ratio.shape)
        return replace(result,ensemble_amplitude_g=ensemble,exceeded=exceeded,worst_axis='XYZ'[ai],worst_frequency_hz=float(result.frequency_hz[bi]),worst_measured_g=float(ensemble[ai,bi]),worst_threshold_g=float(result.threshold_amplitude_g[bi]),worst_ratio=float(ratio[ai,bi]))

    def _live_threshold_limit(self):
        mode=self.live_threshold_mode.get()
        if mode=='Off':
            self.live_threshold_status.set('Threshold: disabled')
            return None
        try:
            if mode=='Gordon Office':
                multiplier=float(self.live_threshold_multiplier.get())
                if multiplier<=0: raise ValueError
                velocity=gordon_office_velocity_um_s(GORDON_FC)*multiplier
                acceleration=(velocity*1e-6)*(2*np.pi*GORDON_FC)/G0
                displacement=velocity/(2*np.pi*GORDON_FC)
                label=f'Gordon Office × {multiplier:g}'
            else:
                value=float(self.live_threshold_value.get())
                if value<=0: raise ValueError
                if mode=='Fixed acceleration (g RMS)':
                    acceleration=np.full_like(GORDON_FC,value)
                    velocity=acceleration*G0/(2*np.pi*GORDON_FC)*1e6
                    displacement=velocity/(2*np.pi*GORDON_FC)
                    label=f'Fixed {value:g} g RMS'
                elif mode=='Fixed velocity (um/s RMS)':
                    velocity=np.full_like(GORDON_FC,value)
                    acceleration=(velocity*1e-6)*(2*np.pi*GORDON_FC)/G0
                    displacement=velocity/(2*np.pi*GORDON_FC)
                    label=f'Fixed {value:g} µm/s RMS'
                elif mode=='Fixed displacement (um RMS)':
                    displacement=np.full_like(GORDON_FC,value)
                    velocity=displacement*(2*np.pi*GORDON_FC)
                    acceleration=(velocity*1e-6)*(2*np.pi*GORDON_FC)/G0
                    label=f'Fixed {value:g} µm RMS'
                else:
                    self.live_threshold_status.set('Threshold: unsupported method')
                    return None
        except (ValueError,TypeError,tk.TclError):
            self.live_threshold_status.set('Threshold: enter a value greater than zero')
            return None

        display_mode=self.units.get()
        if display_mode=='RMS acceleration (g)': limit=acceleration
        elif display_mode=='RMS displacement (µm)': limit=displacement
        else: limit=velocity
        return np.asarray(limit,float),label

    def _update_live_threshold_control_states(self):
        mode=self.live_threshold_mode.get()
        issue2_selected=self.processing_method.get()!=ISSUE1_METHOD
        gordon_enabled=issue2_selected and mode=='Gordon Office'
        fixed_enabled=issue2_selected and mode.startswith('Fixed ')
        self.live_threshold_multiplier_entry.configure(state='normal' if gordon_enabled else 'disabled')
        self.live_threshold_multiplier_label.configure(state='normal' if gordon_enabled else 'disabled')
        self.live_threshold_value_entry.configure(state='normal' if fixed_enabled else 'disabled')
        self.live_threshold_value_label.configure(state='normal' if fixed_enabled else 'disabled')
        fixed_labels={
            'Fixed acceleration (g RMS)':'Fixed value (g RMS)',
            'Fixed velocity (um/s RMS)':'Fixed value (µm/s RMS)',
            'Fixed displacement (um RMS)':'Fixed value (µm RMS)',
        }
        self.live_threshold_value_label.configure(text=fixed_labels.get(mode,'Fixed value'))

    def _on_live_threshold_mode_changed(self,event=None):
        self._update_live_threshold_control_states()
        self.live_apply_threshold()

    def live_apply_threshold(self):
        self._update_live_threshold_control_states()
        selected=self._live_threshold_limit()
        if self.live_threshold_mode.get()=='Off':
            if self.live_t: self._update_live_plot()
            return
        if selected is None: return
        if self.live_t:
            self.live_last_view_draw=0.0
            self.live_redraw_pending=True
            self._update_live_plot()
        else:
            self.live_threshold_status.set(f'Threshold configured: {selected[1]}; waiting for PSD data')

    def _update_live_plot(self):
        """Refresh the live summary: XYZ acceleration, XYZ Gordon bands and tilt target."""
        if not self.live_t:
            return
        f=self.live_fig
        f.clear()
        # Give the lower diagnostics substantially more room than the older
        # three-equal-column layout.  Gordon bands occupy the left two-thirds
        # of the lower area, while tilt and shock are stacked on the right.
        # This keeps the plots readable when the application is used on a
        # typical laptop / 1080p display.
        gs=f.add_gridspec(
            3,3,
            height_ratios=[1.15,0.88,0.88],
            width_ratios=[1.25,1.25,1.0],
            hspace=0.52,wspace=0.38,
        )
        ax_t=f.add_subplot(gs[0,:])
        ax_b=f.add_subplot(gs[1:,0:2])
        ax_tilt=f.add_subplot(gs[1,2])
        ax_shock=f.add_subplot(gs[2,2])

        fs=float(self.fs.get())
        view=max(2.,float(self.live_view_seconds.get()))
        nview=max(10,int(view*fs))
        t=np.asarray(self.live_t[-nview:])
        x=np.asarray(self.live_x[-nview:])
        y=np.asarray(self.live_y[-nview:])
        z=np.asarray(self.live_z[-nview:])
        ax_t.plot(t,x,label='X')
        ax_t.plot(t,y,label='Y')
        ax_t.plot(t,z,label='Z')
        ax_t.set(
            title=f'Live reader acceleration — last {min(view, len(t)/fs):.1f} s',
            xlabel='Capture time (s)',ylabel='Acceleration (g)')
        ax_t.grid(True)
        ax_t.legend(ncol=3)

        # Summary Gordon view: show all three reader axes together.
        axis_values={'X':self.live_x,'Y':self.live_y,'Z':self.live_z}
        band_results={axis:self._live_welch_bands(vals) for axis,vals in axis_values.items()}
        if any(v is None for v in band_results.values()):
            n=int(self.n.get())
            segs=int(self.segs.get())
            hop=int(round(n*(1-float(self.ov.get())/100.)))
            need=n+(segs-1)*hop
            available=min(len(self.live_x),len(self.live_y),len(self.live_z))
            ax_b.axis('off')
            if self._issue1_active():
                ax_b.text(.5,.5,'Issue 2 Gordon-band summary inactive\nUse Live Issue 1 ensemble',ha='center',va='center')
                self.live_threshold_status.set('Issue 2 threshold inactive in Issue 1 mode')
            else:
                ax_b.text(.5,.5,f'Waiting for PSD data\n{available} / {need} samples',ha='center',va='center')
                self.live_threshold_status.set(f'Threshold: waiting for PSD data ({available} / {need} samples)')
        else:
            plotted={}
            ylabel=None
            for axis in ('X','Y','Z'):
                raw_q,yl=self._band_values_for_units(band_results[axis])
                q=self._apply_live_k(axis,raw_q)
                plotted[axis]=q
                ylabel=yl
                suffix=' translated' if self.live_k_correction_enabled.get() and self._live_k_complete() else ''
                ax_b.semilogx(GORDON_FC,q,marker='o',label=f'{axis} axis{suffix}')

            threshold=self._live_threshold_limit()
            if threshold is not None:
                limit,threshold_label=threshold
                ax_b.semilogx(GORDON_FC,limit,marker='s',linewidth=1.6,label=threshold_label)
                worst_axis=None; worst_band=None; worst_ratio=-np.inf
                total_exceeded=0
                for axis,q in plotted.items():
                    exceeded=q>limit
                    total_exceeded+=int(np.sum(exceeded))
                    ratio=np.divide(q,limit,out=np.zeros_like(q),where=limit>0)
                    idx=int(np.argmax(ratio))
                    if ratio[idx]>worst_ratio:
                        worst_ratio=float(ratio[idx]); worst_axis=axis; worst_band=idx
                state='EXCEEDED' if total_exceeded else 'PASS'
                self.live_threshold_status.set(
                    f'{state}: {total_exceeded}/42 axis-bands; worst {worst_axis} '
                    f'{GORDON_FC[worst_band]:g} Hz = {worst_ratio:.2f}× limit')

            set_gordon_xaxis(ax_b,rotate=45)
            ax_b.set(title=('Live X / Y / Z Gordon bands - translated' if self.live_k_correction_enabled.get() and self._live_k_complete() else 'Live X / Y / Z Gordon bands - raw reader'),xlabel='Band centre (Hz)',ylabel=ylabel)
            ax_b.grid(True,which='both')
            ax_b.legend(fontsize=8)

        # Tilt and shock are raw time-domain safety/diagnostic calculations;
        # frequency-band K correction is deliberately not applied to either.
        self._draw_live_tilt_target(ax_tilt)
        self._draw_live_shock_detector(ax_shock)

        f.suptitle('Live vibration summary',y=0.995)
        f.subplots_adjust(top=0.94,bottom=0.09,left=0.08,right=0.98)
        self.live_canvas.draw_idle()

    def _on_live_shock_controls_changed(self):
        try:
            threshold=float(self.live_shock_threshold_g.get())
            if not np.isfinite(threshold) or threshold <= 0:
                raise ValueError
        except Exception:
            self.live_shock_readout.set('Shock: threshold must be > 0 g')
            return
        if self.live_t:
            self.live_last_view_draw=0.0
            self.live_redraw_pending=True
            self._update_live_plot()

    def live_reset_shock(self):
        self.live_shock_peak_g=0.0
        self.live_shock_events=0
        self.live_shock_prev_exceeded=False
        self.live_shock_last_processed_time=None
        self.live_shock_readout.set('Shock: peak/events reset')
        if self.live_t:
            self._update_live_plot()

    def _live_shock_series(self, nview=None):
        """Return moving-baseline resultant shock magnitude for retained samples."""
        """Return time and Issue-2 shock magnitude for the live XYZ stream.

        1174-Y-056 Proposed Issue 2 section 6.2.2 defines shock as the
        resultant of the instantaneous X/Y/Z changes from their averaged
        steady-state values. Figure 6 uses a 50-point average, which is
        0.25 s at the nominal 200 Hz ODR.

        This live implementation uses a causal 50-sample rolling average,
        because future samples are not available in real time. K correction
        is intentionally not applied: K is a frequency-band translation for
        vibration, while shock is defined directly from the time-domain XYZ.
        """
        n=min(len(self.live_t),len(self.live_x),len(self.live_y),len(self.live_z))
        w=int(self.live_shock_window_samples)
        if n < w:
            return None
        # Calculate enough history to make the displayed view continuous.
        if nview is None:
            fs=max(1.0,float(self.fs.get()))
            nview=max(w,int(max(2.0,float(self.live_view_seconds.get()))*fs))
        take=min(n,max(w,int(nview)+w-1))
        t=np.asarray(self.live_t[-take:],float)
        xyz=np.column_stack([
            np.asarray(self.live_x[-take:],float),
            np.asarray(self.live_y[-take:],float),
            np.asarray(self.live_z[-take:],float),
        ])
        kernel=np.ones(w,dtype=float)/float(w)
        avg=np.column_stack([np.convolve(xyz[:,i],kernel,mode='valid') for i in range(3)])
        inst=xyz[w-1:,:]
        delta=inst-avg
        s2=np.sum(delta*delta,axis=1)
        mag=np.sqrt(np.maximum(s2,0.0))
        return t[w-1:],mag,delta,avg

    def _draw_live_shock_detector(self,ax):
        ax.clear()
        ax.set_title('Live shock detector')
        if not bool(self.live_shock_enabled.get()):
            ax.axis('off')
            ax.text(.5,.55,'Shock detector OFF',ha='center',va='center',fontsize=13,fontweight='bold')
            ax.text(.5,.42,'Issue 2 resultant-magnitude calculation is disabled.',ha='center',va='center',wrap=True)
            self.live_shock_readout.set('Shock: detector off')
            return
        try:
            threshold=float(self.live_shock_threshold_g.get())
        except Exception:
            threshold=np.nan
        if not np.isfinite(threshold) or threshold <= 0:
            ax.axis('off')
            ax.text(.5,.5,'Set a shock threshold > 0 g',ha='center',va='center')
            self.live_shock_readout.set('Shock: invalid threshold')
            return
        series=self._live_shock_series()
        if series is None:
            available=min(len(self.live_x),len(self.live_y),len(self.live_z))
            ax.axis('off')
            ax.text(.5,.55,'Live shock detector',ha='center',va='center',fontsize=13,fontweight='bold')
            ax.text(.5,.42,f'Waiting for 50 samples ({available}/50)',ha='center',va='center')
            self.live_shock_readout.set(f'Shock: waiting for 50 samples ({available}/50)')
            return
        t,mag,delta,avg=series
        current=float(mag[-1]); window_peak=float(np.max(mag))

        # Process only shock samples not previously considered, so short events
        # are still detected even when the plot redraw rate is slower than 200 Hz.
        if self.live_shock_last_processed_time is None:
            new_mask=np.ones(len(t),dtype=bool)
        else:
            new_mask=t > float(self.live_shock_last_processed_time) + 1e-12
        if np.any(new_mask):
            new_mag=mag[new_mask]
            self.live_shock_peak_g=max(float(self.live_shock_peak_g),float(np.max(new_mag)))
            prev=bool(self.live_shock_prev_exceeded)
            for value in new_mag:
                over=bool(value > threshold)
                if over and not prev:
                    self.live_shock_events+=1
                prev=over
            self.live_shock_prev_exceeded=prev
            self.live_shock_last_processed_time=float(t[-1])
        else:
            self.live_shock_peak_g=max(float(self.live_shock_peak_g),window_peak)

        ax.plot(t,mag,label='Shock magnitude S')
        ax.axhline(threshold,linestyle='--',linewidth=1.5,label=f'Threshold {threshold:g} g')
        ax.set_xlabel('Capture time (s)')
        ax.set_ylabel('Shock magnitude (g)')
        ax.grid(True,alpha=.3)
        ax.legend(fontsize=8,loc='upper right')
        self.live_shock_readout.set(
            f'Shock: current {current:.3f} g; peak {self.live_shock_peak_g:.3f} g; '
            f'events {self.live_shock_events}; threshold {threshold:g} g')

    def _current_live_tilt(self):
        """Estimate roll/pitch only when the averaged vector resembles gravity."""
        """
        Estimate tilt from the low-frequency gravity vector.

        Tilt must not be calculated from a handful of raw samples because
        vibration is superposed on gravity. Require a complete averaging
        interval, then average X/Y/Z before calculating the angles.
        """
        if not self.live_x:
            return None

        fs=float(self.fs.get())
        navg=max(1,int(self.tilt_average_samples.get()))
        avg_s=navg/fs

        # Do not let the averaging window "grow in" from 1 sample. That caused
        # the first automatic zero reference to be based on an unstable partial
        # window and could create very large apparent relative tilt later.
        if min(len(self.live_x),len(self.live_y),len(self.live_z)) < navg:
            return {
                'ready':False,
                'needed':navg,
                'available':min(len(self.live_x),len(self.live_y),len(self.live_z)),
                'avg_s':avg_s,
            }

        x=np.asarray(self.live_x[-navg:],float)
        y=np.asarray(self.live_y[-navg:],float)
        z=np.asarray(self.live_z[-navg:],float)

        gx=float(np.mean(x)); gy=float(np.mean(y)); gz=float(np.mean(z))
        gmag=float(np.sqrt(gx*gx+gy*gy+gz*gz))

        # Dynamic content is estimated after removing the mean gravity vector.
        dx=x-gx; dy=y-gy; dz=z-gz
        dynamic_rms=float(np.sqrt(np.mean(dx*dx+dy*dy+dz*dz)))

        # Normalising is not essential for atan2, but makes the reported vector
        # explicitly represent gravity direction rather than scale error.
        if gmag > 1e-9:
            nx,ny,nz=gx/gmag,gy/gmag,gz/gmag
        else:
            nx=ny=nz=0.0

        roll=float(np.degrees(np.arctan2(ny,nz)))
        pitch=float(np.degrees(np.arctan2(-nx,np.sqrt(ny*ny+nz*nz))))

        min_g=float(self.live_tilt_min_gravity_g.get())
        max_g=float(self.live_tilt_max_gravity_g.get())
        max_dyn=float(self.live_tilt_max_dynamic_g.get())
        source_diag=self._raw_stream_diagnostics()
        lp_signature=bool(source_diag is not None and np.all(source_diag['multiple256']>0.95))
        source_fault=bool(self.live_last_status & 0x04) or lp_signature
        valid=(min_g <= gmag <= max_g) and (dynamic_rms <= max_dyn) and not source_fault

        return {
            'ready':True,
            'valid':valid,
            'roll':roll,
            'pitch':pitch,
            'gx':gx,'gy':gy,'gz':gz,
            'nx':nx,'ny':ny,'nz':nz,
            'gmag':gmag,
            'dynamic_rms':dynamic_rms,
            'source_fault':source_fault,
            'lp_signature':lp_signature,
            'navg':navg,
            'avg_s':avg_s,
        }

    def live_zero_tilt(self):
        current=self._current_live_tilt()
        if current is None:
            self.live_tilt_readout.set('Cannot zero: no live samples')
            return
        if not current.get('ready'):
            self.live_tilt_readout.set(
                f"Cannot zero yet: {current['available']}/{current['needed']} samples "
                f"for {current['avg_s']:.2f} s gravity average")
            return
        if not current.get('valid'):
            self.live_tilt_readout.set(
                f"Cannot zero: gravity estimate not stable "
                f"(|g|={current['gmag']:.3f}, dynamic RMS={current['dynamic_rms']:.3f} g)")
            return
        self.live_tilt_zero_roll=current['roll']
        self.live_tilt_zero_pitch=current['pitch']
        self.live_tilt_readout.set(
            f"Zero reference updated from {current['avg_s']:.2f} s stable gravity average")
        self._update_live_tilt_plot()

    @staticmethod
    def _angle_delta(angle,reference):
        return (float(angle)-float(reference)+180.0)%360.0-180.0

    def _draw_live_tilt_target(self,ax):
        """Draw the live two-axis tilt target onto an arbitrary axes."""
        current=self._current_live_tilt()
        ax.clear()
        if current is None:
            ax.axis('off')
            ax.text(.5,.5,'Waiting for live tilt data',ha='center',va='center')
            return

        if not current.get('ready'):
            self.live_tilt_readout.set(
                f"Establishing gravity estimate: {current['available']}/{current['needed']} "
                f"samples ({current['avg_s']:.2f} s required)")
            ax.axis('off')
            ax.text(.5,.55,'Live tilt',ha='center',va='center',fontsize=14,fontweight='bold')
            ax.text(.5,.43,
                    f"Waiting for a complete {current['avg_s']:.2f} s gravity-averaging window.",
                    ha='center',va='center',wrap=True)
            return

        roll=current['roll']; pitch=current['pitch']
        gx=current['gx']; gy=current['gy']; gz=current['gz']
        navg=current['navg']; gmag=current['gmag']
        dynamic_rms=current['dynamic_rms']; valid=current['valid']

        # Automatic zero only from a complete, stable gravity window.
        if self.live_tilt_zero_roll is None or self.live_tilt_zero_pitch is None:
            if valid:
                self.live_tilt_zero_roll=roll
                self.live_tilt_zero_pitch=pitch
            else:
                self.live_tilt_readout.set(
                    f"Tilt invalid/not zeroed: |g|={gmag:.3f} g, "
                    f"dynamic RMS={dynamic_rms:.3f} g, source fault={current.get('source_fault',False)}")
                ax.axis('off')
                ax.text(.5,.55,'Live tilt',ha='center',va='center',fontsize=14,fontweight='bold')
                ax.text(.5,.42,
                        'Gravity estimate is not stable enough to establish a zero reference.',
                        ha='center',va='center',wrap=True)
                return

        relative_roll=self._angle_delta(roll,self.live_tilt_zero_roll)
        relative_pitch=self._angle_delta(pitch,self.live_tilt_zero_pitch)
        magnitude=float(np.hypot(relative_pitch,relative_roll))

        plot_pitch=relative_pitch
        plot_roll=relative_roll
        if magnitude>10.0:
            scale=10.0/magnitude
            plot_pitch*=scale
            plot_roll*=scale

        if not valid:
            colour='tab:orange'
        else:
            colour='tab:green' if magnitude<=10.0 else 'tab:red'

        for radius in (2,4,6,8,10):
            ax.add_patch(Circle((0,0),radius,fill=False,
                                color='0.72' if radius<10 else '0.25',
                                linewidth=.7 if radius<10 else 1.5))
        ax.axhline(0,color='0.35',linewidth=.8)
        ax.axvline(0,color='0.35',linewidth=.8)
        ax.scatter([plot_pitch],[plot_roll],s=200,color=colour,edgecolor='black',zorder=5)
        ax.plot([0,plot_pitch],[0,plot_roll],color=colour,linewidth=1.2,zorder=4)
        ax.set_xlim(-10.5,10.5); ax.set_ylim(-10.5,10.5)
        ax.set_aspect('equal',adjustable='box')
        ax.set_xticks(np.arange(-10,11,2)); ax.set_yticks(np.arange(-10,11,2))
        ax.set_xlabel(
            f'Pitch {relative_pitch:+.2f}°    Roll {relative_roll:+.2f}°',
            labelpad=6)
        ax.set_ylabel('Roll (degrees)')
        ax.set_title('Live tilt target')
        ax.grid(True,alpha=.25)

        if not valid:
            state='INVALID / dynamic'
        else:
            state='within range' if magnitude<=10.0 else 'over range'

        # Keep the compact status string available to the control sidebar, but
        # do not draw the previous multi-line diagnostic box over the target.
        # The live summary only needs the current pitch/roll values below the
        # target; gravity-vector and dynamic-RMS diagnostics remain available
        # through the sidebar/readout when needed.
        readout=(f'Pitch {relative_pitch:+.2f} deg   Roll {relative_roll:+.2f} deg   {state}')
        self.live_tilt_readout.set(readout)

    def _update_live_tilt_plot(self):
        """Compatibility wrapper retained for older calls; the dedicated tab is no longer created."""
        return

    @staticmethod
    def _decode_vmm_status(status):
        """Decode status bits actually used by vmm_acquisition.c in the tagged source."""
        status=int(status)&0xFF
        flags=[]
        # vmm_types.h calls bit 0 overrun, but the current tagged acquisition
        # code does not presently set it. Preserve the intended meaning.
        if status & 0x01: flags.append('FIFO/overrun flag')
        if status & 0x02: flags.append('block timing gap')
        if status & 0x04: flags.append('raw sample hit int16 rail')
        unknown=status & ~0x07
        if unknown: flags.append(f'unknown bits 0x{unknown:02X}')
        return flags

    def _raw_stream_diagnostics(self):
        """Summarise VMM sequence, timing, and status health from raw metadata."""
        if not self.live_raw_x:
            return None
        fs=float(self.fs.get())
        nview=min(len(self.live_raw_x),max(32,int(round(max(2.0,float(self.live_view_seconds.get()))*fs))))
        raw=np.column_stack([
            np.asarray(self.live_raw_x[-nview:],dtype=np.int32),
            np.asarray(self.live_raw_y[-nview:],dtype=np.int32),
            np.asarray(self.live_raw_z[-nview:],dtype=np.int32),
        ])
        rail=np.sum((raw==32767)|(raw==-32768),axis=0)
        # In the current ST driver, low-power XL_ONLY_2X FIFO data are 8-bit
        # values left-shifted by 8, so virtually every value is a multiple of
        # 256. High-performance 16-bit data should not have this signature.
        multiple256=np.mean((raw % 256)==0,axis=0)
        return {
            'n':nview,
            'raw':raw,
            'mean':np.mean(raw,axis=0),
            'std':np.std(raw,axis=0),
            'min':np.min(raw,axis=0),
            'max':np.max(raw,axis=0),
            'rms':np.sqrt(np.mean(raw.astype(float)**2,axis=0)),
            'rail':rail,
            'multiple256':multiple256,
        }

    def _update_stream_health(self):
        d=self._raw_stream_diagnostics()
        if d is None:
            self.live_stream_health.set('Stream health: waiting for raw counts')
            return
        flags=self._decode_vmm_status(self.live_last_status)
        quantized=bool(np.all(d['multiple256']>0.95))
        rails=int(np.sum(d['rail']))
        messages=[]
        if flags: messages.append('status: '+', '.join(flags))
        if quantized:
            messages.append('8-bit-left-justified FIFO signature detected (>95% of counts are multiples of 256)')
        if rails:
            messages.append(f'{rails} int16 rail hits in current diagnostic window')
        if quantized:
            messages.append('tagged firmware source selects LIS2DUX12 low-power mode; VmmAcq_ReadFifoBlock transmits only xl[0] from the 2X 8-bit FIFO entry')
        if not messages: messages.append('no VMM status faults; no low-power 8-bit FIFO signature detected')
        prefix='STREAM NOT VALID FOR TILT/SPECTRAL ANALYSIS' if (quantized or (self.live_last_status & 0x04)) else 'Stream health'
        self.live_stream_health.set(prefix+': '+'; '.join(messages))

    def _update_live_raw_counts_plot(self):
        name='Live raw counts'
        d=self._raw_stream_diagnostics()
        if d is None:
            self._draw_live_analysis_empty(name,'Waiting for exact SampleBlock.xyz raw counts.')
            return
        f,c=self._clear_live_analysis_figure(name)
        gs=f.add_gridspec(2,2,height_ratios=[1.05,1.0])
        axstats=f.add_subplot(gs[0,0]); axhist=f.add_subplot(gs[0,1])
        axtrace=f.add_subplot(gs[1,:])
        raw=d['raw']; axes='XYZ'
        axstats.axis('off')
        lines=['Exact signed int16 counts from SampleBlock.xyz',
               f'Window: {d["n"]} samples    VMM status: 0x{self.live_last_status:02X}']
        status_flags=self._decode_vmm_status(self.live_last_status)
        lines.append('Status decode: '+(', '.join(status_flags) if status_flags else 'none'))
        lines.append('')
        lines.append('Axis      mean       std       min       max       RMS    rail hits   % multiple-of-256')
        for i,a in enumerate(axes):
            lines.append(f'{a:>2}  {d["mean"][i]:9.1f} {d["std"][i]:9.1f} {int(d["min"][i]):9d} {int(d["max"][i]):9d} '
                         f'{d["rms"][i]:9.1f} {int(d["rail"][i]):10d} {100*d["multiple256"][i]:16.1f}%')
        lines.append('')
        lines.append('Expected for a stationary level ±2 g sensor: one axis roughly ±16384 counts, other axes near 0; |G| ≈ 1 g.')
        lines.append('If nearly all counts are multiples of 256, the stream has the LIS2DUX12 low-power 8-bit-left-justified FIFO signature.')
        axstats.text(.01,.98,'\n'.join(lines),va='top',ha='left',family='monospace',fontsize=9)

        # Histogram of low byte: HP 16-bit data should use it; LP left-justified 8-bit makes it zero.
        lowbytes=(raw.astype(np.int32)&0xFF).ravel()
        axhist.hist(lowbytes,bins=np.arange(-.5,256.5,1),density=False)
        axhist.set_title('Low-byte distribution across X/Y/Z')
        axhist.set_xlabel('raw_count & 0xFF'); axhist.set_ylabel('occurrences'); axhist.grid(True,axis='y',alpha=.3)

        ntrace=min(len(raw),400)
        idx=np.arange(ntrace)
        for i,a in enumerate(axes): axtrace.plot(idx,raw[-ntrace:,i],label=a,linewidth=.8)
        axtrace.axhline(32767,color='0.4',linestyle='--',linewidth=.7)
        axtrace.axhline(-32768,color='0.4',linestyle='--',linewidth=.7)
        axtrace.set_title(f'Latest {ntrace} exact raw triples')
        axtrace.set_xlabel('sample'); axtrace.set_ylabel('signed int16 counts'); axtrace.grid(True); axtrace.legend()
        f.suptitle('VMM raw-count diagnostics — no g conversion')
        f.tight_layout(); c.draw_idle()

    def _update_live_raw_plot(self):
        if not self.live_t: return
        f,c=self._clear_live_analysis_figure('Live raw')
        axs=f.subplots(3,1,sharex=True)
        fs=float(self.fs.get()); view=max(2.,float(self.live_view_seconds.get())); nview=max(10,int(view*fs))
        t=np.asarray(self.live_t[-nview:])
        series=[np.asarray(self.live_x[-nview:]),np.asarray(self.live_y[-nview:]),np.asarray(self.live_z[-nview:])]
        colours=['tab:blue','tab:orange','tab:green']
        for ax,axis_name,values,colour in zip(axs,'XYZ',series,colours):
            ax.plot(t,values,color=colour,linewidth=.9,label=f'{axis_name} raw')
            ax.set_ylabel(f'{axis_name} (g)'); ax.grid(True); ax.legend(loc='upper right')
            if len(values):
                ax.text(.01,.04,f'latest {values[-1]:+.6f} g',transform=ax.transAxes,fontsize=8)
        axs[-1].set_xlabel('Capture time (s)')
        f.suptitle(
            f'Live decoded accelerometer samples (no filtering/windowing) - '
            f'last {min(view,len(t)/fs):.1f} s; block-reported full scale={self.live_stream_fs.get()}'
        )
        f.tight_layout(); c.draw_idle()

    def _draw_issue1_result(self,f,c,result,title,fs,selected_axis=None):
        """Render filtered data, individual FFTs, ensemble, and threshold result."""
        f.clear(); axs=f.subplots(2,2)
        first=int(result.window_starts[0])
        last=int(result.window_starts[-1])+ISSUE1_FFT_N
        filtered=result.filtered_xyz_g[first:last]
        time_axis=np.arange(len(filtered),dtype=float)/float(fs)
        colours={'X':'tab:blue','Y':'tab:orange','Z':'tab:green'}
        for axis_i,axis_name in enumerate('XYZ'):
            axs[0,0].plot(time_axis,filtered[:,axis_i],color=colours[axis_name],
                          linewidth=.8,label=axis_name)
        axs[0,0].set(title='Issue 1 FIR-filtered time history',xlabel='Ensemble time (s)',ylabel='Acceleration (g)')
        axs[0,0].legend(ncol=3)

        selected=selected_axis or self.issue1_waterfall_axis.get()
        selected_i='XYZ'.index(selected)
        active=(result.frequency_hz>=ISSUE1_LOW_HZ)&(result.frequency_hz<=ISSUE1_HIGH_HZ)
        for index,amplitude in enumerate(result.individual_amplitude_g[:,selected_i,:],start=1):
            axs[0,1].plot(result.frequency_hz[active],amplitude[active],alpha=.35,
                          linewidth=.75,label='Individual FFTs' if index==1 else None)
        axs[0,1].set(title=f'{selected}: 19 individual FFT amplitudes',xlabel='Frequency (Hz)',ylabel='Peak amplitude (g)')
        axs[0,1].legend(fontsize=8)

        for axis_i,axis_name in enumerate('XYZ'):
            measured=result.ensemble_amplitude_g[axis_i]
            axs[1,0].semilogy(result.frequency_hz[active],np.maximum(measured[active],1e-12),
                              color=colours[axis_name],label=f'{axis_name} ensemble')
            exceeded=result.exceeded[axis_i]&active
            if np.any(exceeded):
                axs[1,0].scatter(result.frequency_hz[exceeded],measured[exceeded],
                                 color='tab:red',edgecolor='black',s=35,zorder=5)
        axs[1,0].semilogy(
            result.frequency_hz[active],result.threshold_amplitude_g[active],
            color='black',linestyle='--',linewidth=1.5,label='Issue 1 derived threshold')
        axs[1,0].set(title='Ensemble-averaged Fourier amplitude',xlabel='Frequency (Hz)',ylabel='Peak amplitude (g)')
        axs[1,0].legend(fontsize=8)

        axs[1,1].axis('off')
        state='EXCEEDED' if np.any(result.exceeded) else 'PASS'
        axs[1,1].text(
            .04,.92,
            f'{state}\n\n'
            f'Worst axis: {result.worst_axis}\n'
            f'Worst frequency: {result.worst_frequency_hz:g} Hz\n'
            f'Measured FFT amplitude: {result.worst_measured_g:.6g} g peak\n'
            f'Threshold FFT amplitude: {result.worst_threshold_g:.6g} g peak\n'
            f'Measured / threshold: {result.worst_ratio:.3f}×\n\n'
            f'Frequency resolution: {float(fs)/ISSUE1_FFT_N:g} Hz\n'
            f'Windows: {len(result.window_starts)}; hop: '
            f'{int(result.window_starts[1]-result.window_starts[0])} samples',
            transform=axs[1,1].transAxes,va='top',fontsize=11,
        )
        for ax in (axs[0,0],axs[0,1],axs[1,0]):
            ax.grid(True,which='both')
            ax.set_xlim(ISSUE1_LOW_HZ,ISSUE1_HIGH_HZ) if ax is not axs[0,0] else None
        f.suptitle(title)
        f.tight_layout(); c.draw_idle()

    def _issue1_display_values(self,peak_acceleration_g,frequency_hz):
        """Convert Issue 1 peak FFT amplitudes to the selected RMS units."""
        frequency=np.asarray(frequency_hz,float)
        acceleration_rms_g=np.asarray(peak_acceleration_g,float)/np.sqrt(2.0)
        if self.units.get()=='RMS acceleration (g)':
            return acceleration_rms_g,'RMS g (per FFT bin)','g RMS'
        velocity_rms_um_s=np.divide(
            acceleration_rms_g*G0*1e6,2*np.pi*frequency,
            out=np.full_like(acceleration_rms_g,np.nan),where=frequency>0)
        if self.units.get()=='RMS displacement (µm)':
            displacement_rms_um=np.divide(
                velocity_rms_um_s,2*np.pi*frequency,
                out=np.full_like(velocity_rms_um_s,np.nan),where=frequency>0)
            return displacement_rms_um,'RMS µm (per FFT bin)','µm RMS'
        return velocity_rms_um_s,'RMS µm/s (per FFT bin)','µm/s RMS'

    def _draw_issue1_gordon_comparison(self,f,c,result,title):
        """Draw the legacy calculation using the Issue 2 comparison layout.

        The single combined axes is deliberately only a presentation change:
        Issue 1 still compares ensemble Fourier amplitudes at FFT-bin
        frequencies, rather than integrating a Welch PSD into Gordon bands.
        """
        f.clear(); ax=f.add_subplot(111)
        active=np.isfinite(result.threshold_amplitude_g)&(result.frequency_hz>0)
        frequency=np.asarray(result.frequency_hz[active],float)
        threshold,yl,summary_unit=self._issue1_display_values(
            result.threshold_amplitude_g[active],frequency)
        colours={'X':'tab:blue','Y':'tab:orange','Z':'tab:green'}
        exceeded_labelled=False
        displayed={}
        for axis_i,axis_name in enumerate('XYZ'):
            measured,_,_=self._issue1_display_values(
                result.ensemble_amplitude_g[axis_i,active],frequency)
            displayed[axis_name]=measured
            ax.semilogx(frequency,measured,
                        color=colours[axis_name],marker='o',linestyle='-',
                        label=f'Reader {axis_name}')
            exceeded=result.exceeded[axis_i]&active
            if np.any(exceeded):
                ax.scatter(result.frequency_hz[exceeded],
                           measured[result.exceeded[axis_i,active]],
                           color='tab:red',edgecolor='black',s=42,zorder=5,
                           label='Exceeded' if not exceeded_labelled else None)
                exceeded_labelled=True
        ax.semilogx(
            frequency,threshold,
            marker='s',color='0.25',linewidth=1.5,
            label='Gordon Office — Issue 1 Fourier limit')
        # Reuse the Gordon-centre tick layout from the Issue 2 comparison;
        # measured points remain at their actual Issue 1 FFT-bin frequencies.
        set_gordon_xaxis(ax,rotate=45)
        ax.set_xlabel('FFT frequency bin (Hz)')
        ax.set_ylabel(yl)
        state='EXCEEDED' if np.any(result.exceeded) else 'PASS'
        worst_index=int(np.argmin(np.abs(frequency-result.worst_frequency_hz)))
        worst_measured=float(displayed[result.worst_axis][worst_index])
        worst_threshold=float(threshold[worst_index])
        ax.set_title(
            f'{title} — {self.units.get()}\n'
            f'{state}: worst reader {result.worst_axis} at '
            f'{result.worst_frequency_hz:g} Hz; {worst_measured:.6g} / '
            f'{worst_threshold:.6g} {summary_unit} = {result.worst_ratio:.3f}×')
        ax.grid(True,which='both'); ax.legend(ncol=2,fontsize=8)
        f.tight_layout(); c.draw_idle()

    def _update_live_issue1_plot(self):
        name='Live Issue 1 ensemble'
        if not self._issue1_active():
            self._draw_live_analysis_empty(
                name,'Issue 1 is not active. Select it in Accelerometer / processing configuration and press Apply processing settings.')
            return
        required=issue1_required_samples()
        available=min(len(self.live_x),len(self.live_y),len(self.live_z))
        if available<required:
            self._draw_live_analysis_empty(name,f'Waiting for Issue 1 ensemble data: {available} / {required} samples.')
            return
        xyz=np.column_stack((self.live_x,self.live_y,self.live_z))
        start=len(xyz)-required
        try:
            result=issue1_ensemble_fft(xyz,float(self.fs.get()),start=start)
            result=self._apply_issue1_translation_result(result)
        except Exception as exc:
            self._draw_live_analysis_empty(name,f'Issue 1 processing unavailable: {exc}')
            return
        f,c=self._clear_live_analysis_figure(name)
        translated='translated environmental estimate' if self.live_k_correction_enabled.get() and self.live_k_method==ISSUE1_METHOD else 'raw reader response'
        self._draw_issue1_result(f,c,result,f'Live Issue 1 {translated}',float(self.fs.get()))

    def _update_live_stage_plot(self):
        if self._issue1_active():
            name='Live stages'; required=issue1_required_samples()
            available=min(len(self.live_x),len(self.live_y),len(self.live_z))
            if available<required:
                self._draw_live_analysis_empty(name,f'Waiting for Issue 1 calculation stages: {available} / {required} samples.')
                return
            xyz=np.column_stack((self.live_x,self.live_y,self.live_z))
            result=issue1_ensemble_fft(xyz,float(self.fs.get()),start=len(xyz)-required)
            result=self._apply_issue1_translation_result(result)
            state='EXCEEDED' if np.any(result.exceeded) else 'PASS'
            self.live_threshold_status.set(
                f'{state}: Issue 1 worst {result.worst_axis} '
                f'{result.worst_frequency_hz:g} Hz = {result.worst_ratio:.2f}× limit')
            f,c=self._clear_live_analysis_figure(name)
            self._draw_issue1_result(
                f,c,result,'Live Issue 1 calculation stages',float(self.fs.get()),
                selected_axis=self.axis.get())
            return
        name='Live stages'; n=int(self.n.get()); fs=float(self.fs.get())
        axis_name=self.axis.get(); values={'X':self.live_x,'Y':self.live_y,'Z':self.live_z}[axis_name]
        if len(values)<n:
            self._draw_live_analysis_empty(name,f'Waiting for one FFT block: {len(values)} / {n} samples.')
            return
        f,c=self._clear_live_analysis_figure(name); axs=f.subplots(3,2)
        raw=np.asarray(values[-n:],float)
        freq,psd,mean_removed,window,windowed,fft=periodogram(raw,fs)
        tx=np.arange(n,dtype=float)/fs
        axs[0,0].plot(tx,raw); axs[0,0].set_title('Raw decoded block')
        axs[0,1].plot(tx,mean_removed); axs[0,1].set_title('Mean removed')
        axs[1,0].plot(tx,window); axs[1,0].set_title('Hann window')
        axs[1,1].plot(tx,windowed); axs[1,1].set_title('Windowed samples')
        axs[2,0].plot(freq,np.abs(fft)); axs[2,0].set_title('FFT magnitude'); axs[2,0].set_xlabel('Frequency (Hz)')
        axs[2,1].semilogy(freq[1:],np.maximum(psd[1:],1e-18)); axs[2,1].set_title('Single-block PSD'); axs[2,1].set_xlabel('Frequency (Hz)'); axs[2,1].set_ylabel('g^2/Hz')
        for ax in axs.flat: ax.grid(True)
        f.suptitle(f'Live {axis_name}-axis processing stages - newest {n} samples ({n/fs:.2f} s)')
        f.tight_layout(); c.draw_idle()

    def _update_live_spectrum_plot(self):
        if self._issue1_active():
            self._draw_live_analysis_empty('Live FFT + PSD','Issue 2 FFT/PSD diagnostics are inactive. Use Live Issue 1 ensemble.')
            return
        name='Live FFT + PSD'; n=int(self.n.get()); fs=float(self.fs.get())
        if min(len(self.live_x),len(self.live_y),len(self.live_z))<n:
            have=min(len(self.live_x),len(self.live_y),len(self.live_z))
            self._draw_live_analysis_empty(name,f'Waiting for one FFT block: {have} / {n} samples.')
            return
        f,c=self._clear_live_analysis_figure(name); axs=f.subplots(2,2)
        selected=self.axis.get(); selected_fft=None
        for axis_name,values in zip('XYZ',[self.live_x,self.live_y,self.live_z]):
            freq,psd,_,_,_,fft=periodogram(np.asarray(values[-n:],float),fs)
            axs[0,0].plot(freq,np.abs(fft),label=axis_name)
            axs[1,0].semilogy(freq[1:],np.maximum(psd[1:],1e-18),label=axis_name)
            welch=self._live_welch(values)
            if welch is not None:
                wf,_,pavg=welch
                axs[1,1].semilogy(wf[1:],np.maximum(pavg[1:],1e-18),label=axis_name)
            if axis_name==selected:
                selected_fft=(freq,fft)
        if selected_fft is not None:
            sf,sfft=selected_fft
            phase=np.degrees(np.angle(sfft))
            meaningful=np.abs(sfft)>max(float(np.max(np.abs(sfft)))*1e-9,1e-15)
            axs[0,1].plot(sf[meaningful],phase[meaningful],'.',markersize=2)
        axs[0,0].set_title(f'FFT magnitude - newest {n}-sample block')
        axs[0,1].set_title(f'FFT phase - selected {selected} axis'); axs[0,1].set_ylabel('Phase (degrees)')
        axs[1,0].set_title('Single-block PSD')
        axs[1,1].set_title(f'Welch averaged PSD ({int(self.segs.get())} blocks, {float(self.ov.get()):g}% overlap)')
        for ax in axs.flat:
            ax.set_xlabel('Frequency (Hz)'); ax.grid(True,which='both')
        axs[0,0].set_ylabel('Magnitude'); axs[1,0].set_ylabel('g^2/Hz'); axs[1,1].set_ylabel('g^2/Hz')
        axs[0,0].legend(); axs[1,0].legend()
        if axs[1,1].lines:
            axs[1,1].legend()
        else:
            need=n+(int(self.segs.get())-1)*int(round(n*(1-float(self.ov.get())/100.)))
            axs[1,1].text(.5,.5,f'Waiting for averaged PSD\n{len(self.live_x)} / {need} samples',transform=axs[1,1].transAxes,ha='center',va='center')
        f.suptitle(f'Live full-spectrum diagnostics - nominal fs={fs:g} Hz, Nyquist={fs/2:g} Hz')
        f.tight_layout(); c.draw_idle()

    def _update_live_k_correction_plot(self):
        name='Live K correction'
        if self._issue1_active():
            if not self._live_k_complete() or self.live_k_method!=ISSUE1_METHOD:
                self._draw_live_analysis_empty(name,'Load a complete Issue 1 XYZ translation set to compare raw reader response with the translated environmental estimate.')
                return
            required=issue1_required_samples(); available=min(len(self.live_x),len(self.live_y),len(self.live_z))
            if available<required:
                self._draw_live_analysis_empty(name,f'Waiting for Issue 1 ensemble data: {available} / {required} samples.')
                return
            xyz=np.column_stack((self.live_x,self.live_y,self.live_z))
            raw=issue1_ensemble_fft(xyz,float(self.fs.get()),start=len(xyz)-required)
            translated=self._apply_issue1_translation_result(raw,force=True)
            active=(raw.frequency_hz>=ISSUE1_LOW_HZ)&(raw.frequency_hz<=ISSUE1_HIGH_HZ)
            f,c=self._clear_live_analysis_figure(name); axs=f.subplots(3,1,sharex=True)
            for ai,(axis,ax) in enumerate(zip('XYZ',axs)):
                raw_v,yl,_=self._issue1_display_values(raw.ensemble_amplitude_g[ai,active],raw.frequency_hz[active])
                tr_v,_,_=self._issue1_display_values(translated.ensemble_amplitude_g[ai,active],raw.frequency_hz[active])
                ax.semilogx(raw.frequency_hz[active],raw_v,marker='o',label=f'Raw reader {axis}')
                ax.semilogx(raw.frequency_hz[active],tr_v,marker='x',linestyle='--',label=f'Translated {axis}')
                ax.set_ylabel(f'{axis}: {yl}'); ax.grid(True,which='both'); ax.legend(fontsize=8)
            axs[-1].set_xlabel('Issue 1 FFT frequency (Hz)')
            f.suptitle('Live Issue 1 translation — raw reader response vs translated environmental estimate')
            f.tight_layout(); c.draw_idle()
            return
        if not self._live_k_complete():
            self._draw_live_analysis_empty(
                name,
                'No complete K-factor set loaded. Use Load K-factor CSV(s) in the Direct STM32 connection section.\n'
                'A complete set contains 14 Gordon-band K = Base / Reader coefficients for each of X, Y and Z.'
            )
            return
        results=[]
        for axis_name,values in zip('XYZ',[self.live_x,self.live_y,self.live_z]):
            bands=self._live_welch_bands(values)
            if bands is None:
                need=int(self.n.get())+(int(self.segs.get())-1)*int(round(int(self.n.get())*(1-float(self.ov.get())/100.)))
                self._draw_live_analysis_empty(name,f'Waiting for averaged PSD data: {len(values)} / {need} samples.')
                return
            raw,yl=self._band_values_for_units(bands)
            corrected=self._apply_live_k(axis_name,raw,force=True)
            results.append((axis_name,raw,corrected,yl))

        f,c=self._clear_live_analysis_figure(name)
        axs=f.subplots(3,1,sharex=True)
        for ax,(axis_name,raw,corrected,yl) in zip(axs,results):
            ax.semilogx(GORDON_FC,raw,marker='o',label=f'Raw reader {axis_name}')
            ax.semilogx(GORDON_FC,corrected,marker='x',linestyle='--',label=f'Translated {axis_name}')
            ax.set_ylabel(f'{axis_name}: {yl}')
            ax.grid(True,which='both')
            ax.legend(fontsize=8,loc='best')
            set_gordon_xaxis(ax,rotate=0)
        axs[-1].set_xlabel('One-third-octave band centre (Hz)')
        state='ON for live summary/threshold' if self.live_k_correction_enabled.get() else 'OFF for live summary/threshold'
        files=', '.join(self.live_k_source_files)
        f.suptitle(f'Live multi-frequency K correction - before / after ({state})\nSources: {files}')
        f.tight_layout(); c.draw_idle()

    def _update_live_threshold_plot(self):
        name='Live threshold'
        if self._issue1_active():
            required=issue1_required_samples()
            available=min(len(self.live_x),len(self.live_y),len(self.live_z))
            if available<required:
                self._draw_live_analysis_empty(name,f'Waiting for Issue 1 Gordon comparison: {available} / {required} samples.')
                return
            xyz=np.column_stack((self.live_x,self.live_y,self.live_z))
            result=issue1_ensemble_fft(xyz,float(self.fs.get()),start=len(xyz)-required)
            result=self._apply_issue1_translation_result(result)
            state='EXCEEDED' if np.any(result.exceeded) else 'PASS'
            self.live_threshold_status.set(
                f'{state}: Issue 1 worst {result.worst_axis} '
                f'{result.worst_frequency_hz:g} Hz = {result.worst_ratio:.2f}× limit')
            f,c=self._clear_live_analysis_figure(name)
            self._draw_issue1_gordon_comparison(
                f,c,result,'Live Issue 1 Fourier-amplitude comparison with Gordon Office')
            return
        threshold=self._live_threshold_limit()
        if threshold is None:
            message=('Threshold overlay is disabled.' if self.live_threshold_mode.get()=='Off'
                     else self.live_threshold_status.get())
            self._draw_live_analysis_empty(name,message)
            return
        limit,threshold_label=threshold
        results=[]
        for axis_name,values in zip('XYZ',[self.live_x,self.live_y,self.live_z]):
            bands=self._live_welch_bands(values)
            if bands is None:
                need=int(self.n.get())+(int(self.segs.get())-1)*int(round(int(self.n.get())*(1-float(self.ov.get())/100.)))
                self.live_threshold_status.set(f'Threshold: waiting for PSD data ({len(values)} / {need} samples)')
                self._draw_live_analysis_empty(name,f'Waiting for averaged PSD data: {len(values)} / {need} samples.')
                return
            values_for_units,yl=self._band_values_for_units(bands)
            values_for_units=self._apply_live_k(axis_name,values_for_units)
            results.append((axis_name,values_for_units,yl))

        # Put all three axes on one threshold graph.  This makes the live
        # decision much easier to read than three stacked plots and keeps the
        # axis colours consistent with the rest of the application.
        f,c=self._clear_live_analysis_figure(name)
        ax=f.add_subplot(111)
        colours={'X':'tab:blue','Y':'tab:orange','Z':'tab:green'}
        k_active=bool(self.live_k_correction_enabled.get() and self._live_k_complete())
        worst_ratio=-1.0; worst_axis=''; worst_band=0.0; total_exceeded=0
        ylabel=results[0][2] if results else self.units.get()

        # When live K correction is active, show the uncorrected reader result
        # as a light dashed reference and the corrected values as the solid
        # X/Y/Z traces used for the actual threshold decision.
        for axis_name,measured,yl in results:
            raw_bands=self._live_welch_bands({'X':self.live_x,'Y':self.live_y,'Z':self.live_z}[axis_name])
            raw_display,_=self._band_values_for_units(raw_bands)
            if k_active:
                ax.semilogx(
                    GORDON_FC,raw_display,color=colours[axis_name],linestyle='--',
                    linewidth=1.0,alpha=.35,label=f'{axis_name} raw')

            ratio=np.divide(measured,limit,out=np.zeros_like(measured),where=limit>0)
            exceeded=measured>limit; total_exceeded+=int(np.sum(exceeded))
            idx=int(np.argmax(ratio))
            if ratio[idx]>worst_ratio:
                worst_ratio=float(ratio[idx]); worst_axis=axis_name; worst_band=float(GORDON_FC[idx])
            label=f'{axis_name} translated' if k_active else f'{axis_name} raw'
            ax.semilogx(
                GORDON_FC,measured,color=colours[axis_name],marker='o',
                linewidth=1.8,label=label)
            if np.any(exceeded):
                ax.scatter(
                    GORDON_FC[exceeded],measured[exceeded],s=55,
                    color=colours[axis_name],edgecolor='black',zorder=5)

        ax.semilogx(
            GORDON_FC,limit,marker='s',color='0.20',linewidth=1.8,
            linestyle='-',label=threshold_label)
        set_gordon_xaxis(ax,rotate=45)
        ax.set_xlabel('One-third-octave band centre (Hz)')
        ax.set_ylabel(ylabel)
        ax.grid(True,which='both')
        ax.legend(fontsize=8,ncol=2,loc='best')

        state='EXCEEDED' if total_exceeded else 'PASS'
        correction_text='translated' if k_active else 'raw reader'
        self.live_threshold_status.set(
            f'{state}: {total_exceeded}/42 axis-bands; worst {worst_axis} {worst_band:g} Hz = {worst_ratio:.2f}× limit; '
            f'threshold decision uses {correction_text} data'
        )
        ax.set_title(
            f'Live X / Y / Z against {threshold_label} - {correction_text}\n'
            f'{state}: worst {worst_axis} {worst_band:g} Hz = {worst_ratio:.2f}× limit')
        f.tight_layout(); c.draw_idle()
        f.tight_layout(); c.draw_idle()

    def live_dataframe(self):
        """Return retained converted live samples in the standard CSV schema."""
        if not self.live_t: return pd.DataFrame(columns=['time_s','X_g','Y_g','Z_g'])
        return pd.DataFrame({'time_s':self.live_t,'X_g':self.live_x,'Y_g':self.live_y,'Z_g':self.live_z})

    def live_save_csv(self):
        if not self.live_t:
            messagebox.showinfo('Live capture','No live samples have been captured yet.'); return
        path=filedialog.asksaveasfilename(defaultextension='.csv',filetypes=[('CSV','*.csv')],initialfile='1174_live_accelerometer_capture.csv')
        if path:
            self.live_dataframe().to_csv(path,index=False); self.status.set(f'Saved live capture: {path}')

    def live_use_as_reader(self):
        if not self.live_t:
            messagebox.showinfo('Live capture','No live samples have been captured yet.'); return
        df=self.live_dataframe(); cols=Cols('time_s','X_g','Y_g','Z_g')
        self.reader=Processor('Reader')
        self.pair=PairAnalysis(self.base,self.reader)
        self.reader.set(df,cols,float(self.fs.get()),'Live STM capture',metadata={'source_format':'converted_g','estimated_fs_hz':float(self.fs.get())})
        self.reader_label.configure(text=f'Reader: Live STM capture ({len(df)} samples)')
        self.status.set('Live capture loaded as Reader dataset.')
        if self.base.loaded:
            try: self.refresh()
            except Exception: pass

    def _on_close(self):
        self.live_stop_event.set()
        try:
            self.raw_logger.stop()
        except Exception:
            pass
        client=self.live_client
        self.live_client=None
        if client is not None:
            try: client.close()
            except Exception: pass
        self.root.destroy()

    def guess_cols(self,df):
        return Cols(
            _column_lookup(df,['time_s','time','timestamp']),
            _column_lookup(df,['x_g','accel_x','acc_x']),
            _column_lookup(df,['y_g','accel_y','acc_y']),
            _column_lookup(df,['z_g','accel_z','acc_z']))
    def _baseline_summary_text(self):
        parts=[]
        for axis in 'XYZ':
            proc=self.baselines.get(axis)
            if proc is not None and proc.loaded:
                kind='PSD' if isinstance(proc,BaselineSpectrum) else 'time'
                parts.append(f"{axis}: {Path(proc.source).name} [{kind}] Source: {proc.metadata.get('selected_source','time-domain')}"
                             + (f"\nOrientation: {proc.metadata.get('orientation') or axis} -> {' / '.join(proc.metadata.get('axis_mapping',[axis]))}"
                                f"\nAvailable reference curves: {', '.join(k for k in getattr(proc,'traces',{}) if k=='Ref') or 'none'}"
                                f"\nFrequency range: {proc.freq_hz[0]:.3g} to {proc.freq_hz[-1]:.3g} Hz" if isinstance(proc,BaselineSpectrum) else ''))
            else:
                parts.append(f"{axis}: not loaded")
        reader_parts=[]
        for axis in 'XYZ':
            run=self.reader_runs.get(axis)
            if run is not None and run.loaded:
                reader_parts.append(f'{axis}-excited: {Path(run.source).name if run.source else run.label}')
        reader_text=', '.join(reader_parts) if reader_parts else 'Not loaded'
        return 'Baselines — ' + '\n'.join(parts) + f'\nReader runs: {reader_text}' + ('\nTranslation: Unavailable until reader data is loaded' if not reader_parts else '')

    def _copy_processor(self,source,label=None):
        """Clone processor state so an analysis cannot mutate displayed data."""
        p=Processor(label or source.label)
        p.set(source.original_df.copy(), source.cols, source.fs, source.source,
              metadata=dict(source.metadata), source_df=source.source_df.copy() if source.source_df is not None else None)
        return p

    def _mapped_baseline_processor(self,target_axis,source_axis=None):
        """Resolve a target axis to an independent time-domain baseline copy."""
        source_axis=(source_axis or self.baseline_use_axis[target_axis].get()).upper()
        src=self.baselines.get(source_axis)
        if src is None or not src.loaded:
            raise ValueError(f'No {source_axis}-axis baseline CSV is loaded for the {target_axis}-axis correction.')
        if isinstance(src,BaselineSpectrum):
            raise ValueError('PSD-only baseline: Gordon-band comparison, K derivation, translation and correction export are available. Time-domain overlay, synchronous phase, H1 and coherence require time-domain captures.')
        out=self._copy_processor(src,label=f'Baseline {target_axis} (using {source_axis})')
        # If another driven-axis baseline is used as a surrogate, use that
        # baseline's driven-axis acceleration as the reference profile for the
        # requested reader axis. This avoids accidentally using cross-axis
        # response from the surrogate file.
        if source_axis != target_axis:
            df=out.original_df.copy()
            src_col={'X':out.cols.x,'Y':out.cols.y,'Z':out.cols.z}[source_axis]
            dst_col={'X':out.cols.x,'Y':out.cols.y,'Z':out.cols.z}[target_axis]
            df[dst_col]=pd.to_numeric(df[src_col],errors='coerce').to_numpy(float)
            out.set(df,out.cols,out.fs,out.source,metadata=dict(out.metadata),source_df=out.source_df)
            out.metadata['surrogate_baseline_axis']=source_axis
        out.metadata['correction_target_axis']=target_axis
        return out

    def _activate_baseline_for_driven_axis(self):
        """Update the legacy active baseline from the current explicit mapping."""
        target=self.driven_axis.get().upper()
        source=self.baseline_use_axis[target].get().upper()
        mapped=self._mapped_baseline_processor(target,source)
        self.base.set(mapped.original_df.copy(),mapped.cols,mapped.fs,mapped.source,
                      metadata=dict(mapped.metadata),source_df=mapped.source_df)
        self.pair=PairAnalysis(self.base,self.reader)
        self.base_label.configure(text=self._baseline_summary_text()+f'\nActive {target} correction baseline: {source}')

    def _on_driven_axis_changed(self,event=None):
        try:
            self._activate_baseline_for_driven_axis()
            self.refresh()
        except Exception as exc:
            self.status.set(str(exc))
            self.refresh()

    def _on_baseline_mapping_changed(self,target_axis):
        if self.driven_axis.get().upper()==target_axis:
            self._on_driven_axis_changed()

    def load_baseline_csvs(self):
        """Load time-domain or PSD baselines and assign their excitation axes."""
        paths=filedialog.askopenfilenames(
            title='Select baseline/reference CSV or Excel files',
            filetypes=[('Baseline CSV/Excel','*.csv *.xlsx'),('All files','*.*')])
        if not paths:
            return
        loaded=[]; warnings_all=[]
        try:
            for path in paths:
                proc,warnings=read_baseline_file(path,float(self.fs.get()))
                orientation=baseline_orientation(path)
                axis=infer_excitation_axis_from_filename(path)
                if orientation is None and (axis is None or proc.metadata.get('selected_source')=='Ctl'):
                    response=simpledialog.askstring('Baseline orientation',
                        f'{Path(path).name}\nEnter Lateral (X/Y) or Vertical (Z)' +
                        (' (or X, Y, Z for a legacy CSV):' if proc.metadata.get('selected_source')!='Ctl' else ':'),parent=self.root)
                    if response is None: continue
                    choice=response.strip().upper()
                    if choice in ('LATERAL','LATERAL (X/Y)','X/Y'): orientation='Lateral'
                    elif choice in ('VERTICAL','VERTICAL (Z)'): orientation='Vertical'
                    elif choice in ('X','Y','Z') and proc.metadata.get('selected_source')!='Ctl': axis=choice
                    else: raise ValueError('Choose Lateral or Vertical for the test-house orientation.')
                if orientation=='Lateral':
                    # A filename containing an unambiguous X or Y token wins;
                    # otherwise use the operator's persistent lateral assignment.
                    if axis in ('X','Y'):
                        axes=axis
                    else:
                        choice=self.lateral_assignment.get()
                        axes={'X only':'X','Y only':'Y'}.get(choice,'XY')
                else:
                    axes='Z' if orientation=='Vertical' else axis
                if not axes:
                    raise ValueError(f'Could not determine axis assignment for {Path(path).name}.')
                proc.metadata.update({'orientation':orientation,'axis_mapping':list(axes),
                                      'shared_lateral_source':axes=='XY','excitation_axis':axes[0]})
                for axis in axes:
                    self.baselines[axis]=proc
                loaded.append(f'{"/".join(axes)}={Path(path).name}')
                warnings_all.extend([f'{Path(path).name}: {w}' for w in warnings])
            if not loaded:
                return
            # Preserve normal X→X/Y→Y/Z→Z mapping where the corresponding file
            # exists. Missing axes remain explicitly selectable via the mapping UI.
            for axis in 'XYZ':
                if self.baselines.get(axis) is not None:
                    self.baseline_use_axis[axis].set(axis)
            self.base_label.configure(text=self._baseline_summary_text())
            target=self.driven_axis.get().upper()
            mapped_axis=self.baseline_use_axis[target].get().upper()
            mapped_obj=self.baselines.get(mapped_axis)
            if mapped_obj is not None and mapped_obj.loaded and not isinstance(mapped_obj,BaselineSpectrum):
                try:
                    self._activate_baseline_for_driven_axis()
                except Exception as exc:
                    warnings_all.append(str(exc))
            msg='Loaded baselines: '+', '.join(loaded)
            self.status.set(msg)
            if warnings_all:
                messagebox.showwarning('Baseline CSVs loaded with warnings','\n'.join(warnings_all))
            self._sync_applied_mode_tabs()
            self.refresh()
            if any(isinstance(b,BaselineSpectrum) for b in self.baselines.values()):
                for tab_id in self.tabs.tabs():
                    if self.tabs.tab(tab_id,'text')=='PSD comparison': self.tabs.select(tab_id)
        except Exception as exc:
            messagebox.showerror('Load baseline error',str(exc))

    def _shock_filename_axis(self,path):
        """Return X/Y/Z for filenames containing SHOCK plus an axis token."""
        stem=Path(path).stem.upper()
        if 'SHOCK' not in stem:
            return None
        return infer_excitation_axis_from_filename(path)

    def load_shock_csvs(self,kind):
        """Load up to one X/Y/Z shock file for baseline or reader characterisation."""
        if kind not in ('baseline','reader'):
            raise ValueError('Shock file kind must be baseline or reader.')
        paths=filedialog.askopenfilenames(
            title=f'Select {kind} shock CSV(s) - filenames should contain SHOCK and X, Y or Z',
            filetypes=[('CSV','*.csv'),('All files','*.*')])
        if not paths:
            return
        target=self.shock_baselines if kind=='baseline' else self.shock_readers
        loaded=[]; warnings_all=[]
        try:
            for path in paths:
                axis=self._shock_filename_axis(path)
                if axis is None:
                    response=simpledialog.askstring(
                        'Shock axis',
                        f'Filename should contain SHOCK and the applied axis (X, Y or Z):\n{Path(path).name}\n\nEnter the applied shock axis:',
                        parent=self.root)
                    if response is None:
                        continue
                    axis=response.strip().upper()
                    if axis not in 'XYZ' or len(axis)!=1:
                        raise ValueError(f'Invalid shock axis entered for {Path(path).name}: {response!r}')
                    warnings_all.append(f'{Path(path).name}: axis was entered manually because filename inference failed.')
                source_df=pd.read_csv(path)
                fmt=detect_accelerometer_csv_format(source_df)
                if fmt=='baseline_psd':
                    raise ValueError(f'{Path(path).name}: shock analysis requires time-domain X/Y/Z data, not PSD data.')
                df,cols,fs,meta,warnings=normalize_accelerometer_dataframe(source_df,float(self.fs.get()))
                proc=Processor(f'{kind.title()} shock {axis}')
                meta=dict(meta); meta.update({'shock_axis':axis,'shock_file_kind':kind})
                proc.set(df,cols,fs,path,metadata=meta,source_df=source_df)
                target[axis]=proc
                loaded.append(f'{axis}={Path(path).name}')
                warnings_all.extend([f'{Path(path).name}: {w}' for w in warnings])
            if loaded:
                axes=[a for a in 'XYZ' if target.get(a) is not None and target[a].loaded]
                if self.offline_shock_axis.get() not in axes and axes:
                    self.offline_shock_axis.set(axes[0])
                self.status.set(f'Loaded {kind} shock files: '+', '.join(loaded))
                if warnings_all:
                    messagebox.showwarning('Shock CSVs loaded with warnings','\n'.join(warnings_all))
                self.plot_shock_analysis()
                self.plot_shock_translation()
        except Exception as exc:
            messagebox.showerror('Load shock CSV error',str(exc))

    def _shock_series_for_processor(self,proc,w):
        time_s=np.asarray(proc.time(),float)
        xyz=np.column_stack([np.asarray(proc.axis(a),float) for a in 'XYZ'])
        n=min(len(time_s),len(xyz)); time_s=time_s[:n]; xyz=xyz[:n,:]
        if n < w:
            raise ValueError(f'{proc.label} contains {n} samples; at least {w} are required.')
        # Causal baseline from the preceding w samples (including current sample),
        # matching the live/Issue-2 engineering implementation.
        cs=np.vstack([np.zeros((1,3)),np.cumsum(xyz,axis=0)])
        avg=(cs[w:]-cs[:-w])/float(w)
        inst=xyz[w-1:,:]
        delta=inst-avg
        t=time_s[w-1:]
        mag=np.sqrt(np.maximum(np.sum(delta*delta,axis=1),0.0))
        return t,delta,mag

    def _group_shock_events(self,t,mag,threshold,release_s):
        """Group threshold crossings, requiring a quiet release interval."""
        """Group threshold ringing into one physical event.

        Once S crosses threshold an event stays open until S has remained below
        threshold continuously for release_s. This prevents one impact/ring-down
        from being reported as several shocks merely because S crosses the
        threshold multiple times.
        """
        t=np.asarray(t,float); mag=np.asarray(mag,float)
        if len(t)==0:
            return []
        fs=1.0/max(float(np.median(np.diff(t))) if len(t)>1 else 1.0/DEFAULT_FS,1e-12)
        release_n=max(1,int(round(float(release_s)*fs)))
        events=[]; active=False; start=0; below=0
        for i,val in enumerate(mag):
            if not active:
                if val>=threshold:
                    active=True; start=i; below=0
                continue
            if val>=threshold:
                below=0
            else:
                below+=1
                if below>=release_n:
                    end=max(start,i-below)
                    g=np.arange(start,end+1,dtype=int)
                    peak=int(g[np.argmax(mag[g])])
                    events.append((start,end,peak))
                    active=False; below=0
        if active:
            end=len(mag)-1
            g=np.arange(start,end+1,dtype=int)
            peak=int(g[np.argmax(mag[g])])
            events.append((start,end,peak))
        return events

    def load_csv(self,which):
        if which=='base':
            self.load_baseline_csvs(); return
        path=filedialog.askopenfilename(title='Load reader CSV or Excel',
            filetypes=[('Reader CSV/Excel','*.csv *.xlsx'),('All files','*.*')])
        if not path:return
        try:
            proc,warnings=read_reader_file(path,float(self.fs.get()))
            if isinstance(proc,BaselineSpectrum):
                orientation=baseline_orientation(path)
                axis='Z' if orientation=='Vertical' else infer_excitation_axis_from_filename(path)
                if axis is None:
                    choices='X or Y' if orientation=='Lateral' else 'X, Y or Z'
                    response=simpledialog.askstring('Reader PSD axis',
                        f'{Path(path).name}\nWhich reader axis does this measured PSD represent? Enter {choices}:',parent=self.root)
                    if response is None: return
                    axis=response.strip().upper()
                    if axis not in (('X','Y') if orientation=='Lateral' else ('X','Y','Z')):
                        raise ValueError(f'Choose {choices} for the reader PSD axis.')
                proc.metadata.update({'excitation_axis':axis,'orientation':orientation})
                self.driven_axis.set(axis)
                description=f"PSD {axis}; source: {proc.metadata['selected_source']}; {proc.freq_hz[0]:.3g}?{proc.freq_hz[-1]:.3g} Hz"
            else:
                axis=infer_excitation_axis_from_filename(path) or self.driven_axis.get().upper()
                if axis not in 'XYZ': axis='X'
                proc.metadata['excitation_axis']=axis
                self.driven_axis.set(axis)
                description=f'{len(proc.df):,} samples; {proc.fs:.3f} Hz'
            self.reader_runs[axis]=proc
            self.reader=proc
            self.pair=PairAnalysis(self.base,self.reader)
            loaded=', '.join(a for a in 'XYZ' if self.reader_runs.get(a) is not None and self.reader_runs[a].loaded)
            self.reader_label.configure(text=f'Reader {axis}-excited: {Path(path).name} — {description}\nLoaded excitation slots: {loaded or "none"}')
            if warnings:
                messagebox.showwarning('Reader loaded with warnings','\n'.join(warnings))
            try:
                self._activate_baseline_for_driven_axis()
            except ValueError:
                pass
            self._sync_applied_mode_tabs()
            self.refresh()
        except Exception as e:messagebox.showerror('Load error',str(e))
    def load_demo_pair(self):
        base=Path(__file__).with_name('1174_dummy_office_level_lateral_axes.csv')
        reader=Path(__file__).with_name('1174_dummy_reader_pcb_mounting_effects.csv')
        if not(base.exists() and reader.exists()):
            self.status.set('Demo CSVs are not beside the application. Use Load baseline CSV(s) / Load reader CSV.')
            return
        try:
            bsrc=pd.read_csv(base)
            bdf,bcols,bfs,bmeta,_=normalize_accelerometer_dataframe(bsrc,float(self.fs.get()))
            # The demo is a generic paired dataset rather than three physical
            # baseline runs, so copy it to all three slots only for demo use.
            for axis in 'XYZ':
                proc=Processor(f'Baseline {axis}')
                meta=dict(bmeta); meta.update({'excitation_axis':axis,'demo_shared_baseline':True})
                proc.set(bdf.copy(),bcols,bfs,str(base),metadata=meta,source_df=bsrc)
                self.baselines[axis]=proc
                self.baseline_use_axis[axis].set(axis)
            rsrc=pd.read_csv(reader)
            rdf,rcols,rfs,rmeta,_=normalize_accelerometer_dataframe(rsrc,float(self.fs.get()))
            self.reader=Processor('Reader')
            self.reader.set(rdf,rcols,rfs,str(reader),metadata=rmeta,source_df=rsrc)
            self.reader_label.configure(text=f'Reader: {reader.name} ({len(rdf)} samples)')
            self._activate_baseline_for_driven_axis()
            self.refresh()
        except Exception as exc:
            self.status.set(f'Demo load failed: {exc}')

    def apply_autocrop(self):
        """
        Restore the original imported data, then optionally crop the analysis
        copies to the common paired interval and a whole number of complete
        PSD-analysis spans. Original imported DataFrames are never changed.
        """
        if isinstance(self.reader,BaselineSpectrum):
            raise ValueError('PSD-only reader cannot be cropped in time.')
        if not (self.base.loaded and self.reader.loaded):
            return None

        self.base.restore_original()
        self.reader.restore_original()

        n=int(self.n.get())
        ov=float(self.ov.get())/100.0
        segs=int(self.segs.get())
        hop=int(round(n*(1-ov)))
        span=n+(segs-1)*hop

        base_orig=len(self.base.df)
        reader_orig=len(self.reader.df)
        note='Auto-crop OFF'

        if self.auto_crop.get():
            # Prefer actual timestamps when both datasets contain usable time columns.
            used_time_overlap=False
            if self.base.cols.time and self.reader.cols.time:
                try:
                    tb=pd.to_numeric(self.base.df[self.base.cols.time],errors='coerce').to_numpy(float)
                    tr=pd.to_numeric(self.reader.df[self.reader.cols.time],errors='coerce').to_numpy(float)
                    mb=np.isfinite(tb); mr=np.isfinite(tr)
                    if np.sum(mb)>1 and np.sum(mr)>1:
                        t0=max(float(np.nanmin(tb[mb])),float(np.nanmin(tr[mr])))
                        t1=min(float(np.nanmax(tb[mb])),float(np.nanmax(tr[mr])))
                        if t1>t0:
                            self.base.df=self.base.df[(tb>=t0)&(tb<=t1)].reset_index(drop=True)
                            self.reader.df=self.reader.df[(tr>=t0)&(tr<=t1)].reset_index(drop=True)
                            used_time_overlap=True
                except Exception:
                    used_time_overlap=False

            # Ensure equal paired sample count after time-overlap selection, or
            # simply use the shortest dataset when timestamps are unavailable.
            common=min(len(self.base.df),len(self.reader.df))
            if common < span:
                raise ValueError(
                    f'Auto-crop found only {common} common samples, but one complete '
                    f'analysis span needs {span} samples.'
                )

            # Keep a whole number of complete analysis spans so no final partial
            # window silently enters model fitting/validation.
            usable=(common//span)*span
            self.base.df=self.base.df.iloc[:usable].reset_index(drop=True)
            self.reader.df=self.reader.df.iloc[:usable].reset_index(drop=True)
            note=('Auto-crop ON: common timestamp overlap + complete PSD spans'
                  if used_time_overlap else
                  'Auto-crop ON: common sample count + complete PSD spans')

        used=min(len(self.base.df),len(self.reader.df))
        duration=(used-1)/float(self.fs.get()) if used>1 else 0.0

        self.base_label.configure(
            text=f'Baseline: {Path(self.base.source).name if self.base.source else self.base.label} '
                 f'({base_orig} original → {len(self.base.df)} used samples)'
        )
        self.reader_label.configure(
            text=f'Reader: {Path(self.reader.source).name if self.reader.source else self.reader.label} '
                 f'({reader_orig} original → {len(self.reader.df)} used samples)'
        )
        return f'{note}; paired analysis duration ≈ {duration:.3f} s; {used} samples'

    def _paired_axis_context(self,target_axis):
        """Return independent mapped baseline/reader processors for one target axis.

        This avoids a global driven-axis state: each offline analysis view resolves the
        target axis through the explicit baseline mapping at the point of use.
        """
        if isinstance(self.reader,BaselineSpectrum):
            raise ValueError('PSD-only reader: time-domain pairing, phase, H1 and coherence are unavailable.')
        target_axis=target_axis.upper()
        source_axis=self.baseline_use_axis[target_axis].get().upper()
        base=self._mapped_baseline_processor(target_axis,source_axis)
        reader=self._copy_processor(self.reader,label='Reader')
        base.fs=reader.fs=float(self.fs.get())
        base.restore_original(); reader.restore_original()
        if self.auto_crop.get():
            n=int(self.n.get()); ov=float(self.ov.get())/100.0; segs=int(self.segs.get())
            hop=int(round(n*(1-ov))); span=n+(segs-1)*hop
            # Prefer timestamp overlap where both files provide useful time axes.
            try:
                tb=np.asarray(base.time(),float); tr=np.asarray(reader.time(),float)
                if len(tb) and len(tr) and np.all(np.isfinite(tb)) and np.all(np.isfinite(tr)):
                    lo=max(float(tb[0]),float(tr[0])); hi=min(float(tb[-1]),float(tr[-1]))
                    if hi>lo:
                        ib=np.flatnonzero((tb>=lo)&(tb<=hi)); ir=np.flatnonzero((tr>=lo)&(tr<=hi))
                        if len(ib) and len(ir):
                            base.df=base.df.iloc[ib].reset_index(drop=True)
                            reader.df=reader.df.iloc[ir].reset_index(drop=True)
            except Exception:
                pass
            common=min(len(base.df),len(reader.df))
            usable=(common//span)*span if common>=span else common
            if usable>0:
                base.df=base.df.iloc[:usable].reset_index(drop=True)
                reader.df=reader.df.iloc[:usable].reset_index(drop=True)
        return base,reader,PairAnalysis(base,reader),source_axis

    def params(self): return int(self.n.get()),float(self.ov.get())/100,int(self.segs.get()),int(self.start.get())
    def clear(self,name): f,c=self.fig[name]; f.clear(); return f,c
    def _show_issue2_inactive(self,name):
        """Keep Issue 2-only tabs from presenting calculations in Issue 1 mode."""
        if not self._issue1_active():
            return False
        f,c=self.clear(name); ax=f.add_subplot(111); ax.axis('off')
        ax.text(.5,.5,'Issue 2 view inactive while Issue 1 FFT ensemble is active.',
                ha='center',va='center',fontsize=12)
        c.draw_idle()
        return True
    def refresh(self):
        """Recompute every analysis view that has sufficient loaded inputs."""
        try:
            if not self.reader.loaded:
                self.plot_psd()
                self.base_label.configure(text=self._baseline_summary_text())
                return
            if isinstance(self.reader,BaselineSpectrum):
                self.plot_psd()
                for name in ('Raw comparison','Selected stage','Issue 1 FFT ensemble','Transfer function'):
                    if name not in self.fig: continue
                    f,c=self.clear(name); ax=f.add_subplot(111); ax.axis('off')
                    ax.text(.5,.5,'PSD-only reader: time-domain samples are unavailable.\nUse PSD comparison and Issue 2 Gordon bands / translation.',ha='center',va='center')
                    c.draw_idle()
                self.plot_bands(); self.plot_translation()
                self.base_label.configure(text=self._baseline_summary_text())
                self.status.set('Reader Ctl PSD loaded. Time-domain, Issue 1 FFT, H1 and coherence are unavailable.')
                return
            self.reader.fs=float(self.fs.get())
            errors=[]
            try:
                self.plot_shock_analysis()
            except Exception as exc:
                errors.append(f'Shock analysis: {exc}')
            try:
                self.plot_issue1_analysis()
            except Exception as exc:
                errors.append(f'Issue 1 analysis: {exc}')
            if self._issue1_active():
                for label,func in [
                    ('Calculation stages',self.plot_stage),
                    ('Gordon comparison',self.plot_bands)]:
                    try:
                        func()
                    except Exception as exc:
                        errors.append(f'{label}: {exc}')

            have_baseline=any(b is not None and b.loaded for b in self.baselines.values())
            if have_baseline:
                functions=[('Raw comparison',self.plot_raw)]
                if self._issue1_active():
                    for name in ('Transfer function',):
                        f,c=self.clear(name); ax=f.add_subplot(111); ax.axis('off')
                        ax.text(.5,.5,'Issue 2 view inactive while Issue 1 FFT ensemble is active.',
                                ha='center',va='center',fontsize=12)
                        c.draw_idle()
                    functions.append(('PSD comparison',self.plot_psd))
                    functions.append(('Translation function',self.plot_translation))
                else:
                    functions.extend([
                        ('Selected stage',self.plot_stage),
                        ('PSD comparison',self.plot_psd),('Gordon bands',self.plot_bands),
                        ('Transfer function',self.plot_transfer),
                        ('Translation function',self.plot_translation)])
                for label,func in functions:
                    try:
                        func()
                    except Exception as exc:
                        errors.append(f'{label}: {exc}')
                self.base_label.configure(text=self._baseline_summary_text())
            if errors:
                self.status.set('Processing completed with unavailable views: '+'; '.join(errors))
            elif self._issue1_active():
                self.status.set('Issue 1 legacy FFT-amplitude ensemble complete; Issue 2 PSD and transfer views are inactive.')
            elif have_baseline:
                self.status.set('Multi-axis processing complete — X/Y/Z views use the explicit correction baseline mapping.')
            else:
                self.status.set('Reader loaded — Shock analysis available. Load baseline CSV(s) for paired vibration views.')
        except Exception as e:
            self.status.set(str(e)); messagebox.showerror('Processing error',str(e))

    def _offline_issue1_result(self):
        """Calculate Issue 1 once from the loaded Reader and selected start."""
        if isinstance(self.reader,BaselineSpectrum):
            raise ValueError('Issue 1 FFT requires reader time samples; the loaded workbook contains only PSD.')
        if not self.reader.loaded:
            raise ValueError('Load a Reader CSV for offline Issue 1 analysis.')
        _,x,y,z=self.reader.xyz()
        xyz=np.column_stack((x,y,z))
        return issue1_ensemble_fft(
            xyz,float(self.fs.get()),start=int(self.start.get()))

    def plot_issue1_analysis(self):
        """Render the dedicated offline Issue 1 result for the Reader dataset."""
        f,c=self.clear('Issue 1 FFT ensemble')
        if not self._issue1_active():
            ax=f.add_subplot(111); ax.axis('off')
            ax.text(
                .5,.5,
                'Issue 1 is not active.\nSelect it in Accelerometer / processing '
                'configuration and press Apply processing settings.',
                ha='center',va='center',fontsize=12)
            c.draw_idle(); return
        if not self.reader.loaded:
            ax=f.add_subplot(111); ax.axis('off')
            ax.text(.5,.5,'Load a Reader CSV for offline Issue 1 analysis.',
                    ha='center',va='center')
            c.draw_idle(); return
        result=self._offline_issue1_result()
        source=Path(self.reader.source).name if self.reader.source else self.reader.label
        self._draw_issue1_result(
            f,c,result,f'Offline Issue 1 legacy FFT-amplitude ensemble — {source}',
            float(self.fs.get()))

    def plot_raw(self):
        f,c=self.clear('Raw comparison'); axs=f.subplots(3,1,sharex=True)
        for i,a in enumerate('XYZ'):
            try:
                base,reader,_,source=self._paired_axis_context(a)
                tb=base.time(); tr=reader.time(); xb=base.axis(a); xr=reader.axis(a); m=min(len(tb),len(tr),len(xb),len(xr))
                axs[i].plot(tb[:m],xb[:m],label=f'Baseline {a} (source {source})',alpha=.8)
                axs[i].plot(tr[:m],xr[:m],label=f'Reader {a}',alpha=.8)
                axs[i].set_ylabel(f'{a} (g)'); axs[i].grid(True); axs[i].legend(loc='upper right',fontsize=8)
            except Exception as exc:
                axs[i].axis('off'); axs[i].text(.5,.5,f'{a}: {exc}',ha='center',va='center',wrap=True)
        axs[-1].set_xlabel('Time (s)'); f.suptitle('X / Y / Z baseline-reference versus installed reader')
        f.tight_layout(); c.draw_idle()

    def plot_shock_analysis(self):
        """Offline measured-data shock analysis with X/Y/Z shock characterisation."""
        try:
            w=int(self.offline_shock_window.get())
            threshold=float(self.offline_shock_threshold_g.get())
            release_s=float(self.offline_shock_release_s.get())
        except Exception:
            raise ValueError('Shock steady-state window, threshold and event release must be numeric.')
        if w < 2:
            raise ValueError('Shock steady-state window must be at least 2 samples.')
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError('Shock threshold must be greater than 0 g.')
        if not np.isfinite(release_s) or release_s < 0:
            raise ValueError('Shock event release time must be zero or greater.')

        applied=self.offline_shock_axis.get().upper()
        reader=self.shock_readers.get(applied)
        baseline=self.shock_baselines.get(applied)
        # Shock files are deliberately isolated from normal vibration datasets.
        # This prevents transient shock captures from populating PSD/Gordon/FFT
        # vibration-analysis views.
        if reader is None or not reader.loaded:
            raise ValueError(f'Load a dedicated {applied}-axis shock Reader CSV first using Load shock reader CSV(s).')

        tr,dr,mr=self._shock_series_for_processor(reader,w)
        reader_events_idx=self._group_shock_events(tr,mr,threshold,release_s)
        reader_events=[]
        for start_i,end_i,peak_i in reader_events_idx:
            d=dr[peak_i]
            reader_events.append({
                'start':float(tr[start_i]),'end':float(tr[end_i]),'peak_time':float(tr[peak_i]),
                'peak':float(mr[peak_i]),'delta':d.copy(),
                'dominant':'XYZ'[int(np.argmax(np.abs(d)))],
            })

        baseline_series=None
        if baseline is not None and baseline.loaded:
            tb,db,mb=self._shock_series_for_processor(baseline,w)
            baseline_series=(tb,db,mb)

        f,c=self.clear('Shock analysis')
        axs=f.subplots(4,1,sharex=False,gridspec_kw={'height_ratios':[1,1,1,1.15]})
        labels=['ΔX shock (g)','ΔY shock (g)','ΔZ shock (g)']

        # When a baseline is present, align both records to their largest shock
        # peak. This makes independently started test-house captures directly
        # comparable without pretending their absolute timestamps are synced.
        reader_peak_idx=int(np.argmax(mr))
        tr_plot=tr-float(tr[reader_peak_idx]) if baseline_series is not None else tr
        if baseline_series is not None:
            tb,db,mb=baseline_series
            base_peak_idx=int(np.argmax(mb)); tb_plot=tb-float(tb[base_peak_idx])
        else:
            tb=db=mb=tb_plot=None

        for i,a in enumerate('XYZ'):
            if baseline_series is not None:
                axs[i].plot(tb_plot,db[:,i],linewidth=1.0,label=f'Baseline {a}',alpha=.8)
            axs[i].plot(tr_plot,dr[:,i],linewidth=1.0,label=f'Reader {a}',alpha=.9)
            axs[i].axhline(0,linewidth=.8)
            for ev in reader_events:
                lo=ev['start']-float(tr[reader_peak_idx]) if baseline_series is not None else ev['start']
                hi=ev['end']-float(tr[reader_peak_idx]) if baseline_series is not None else ev['end']
                axs[i].axvspan(lo,hi,alpha=.10)
            axs[i].set_ylabel(labels[i]); axs[i].grid(True); axs[i].legend(loc='upper right',fontsize=8)

        if baseline_series is not None:
            axs[3].plot(tb_plot,mb,label='Baseline resultant S',linewidth=1.0,alpha=.8)
        axs[3].plot(tr_plot,mr,label='Reader resultant S',linewidth=1.1)
        axs[3].axhline(threshold,linestyle='--',label=f'Detection threshold {threshold:g} g')
        for ev in reader_events:
            lo=ev['start']-float(tr[reader_peak_idx]) if baseline_series is not None else ev['start']
            hi=ev['end']-float(tr[reader_peak_idx]) if baseline_series is not None else ev['end']
            pk=ev['peak_time']-float(tr[reader_peak_idx]) if baseline_series is not None else ev['peak_time']
            axs[3].axvspan(lo,hi,alpha=.10); axs[3].plot(pk,ev['peak'],marker='o')
        axs[3].set_ylabel('Resultant S (g)')
        axs[3].set_xlabel('Time relative to peak shock (s)' if baseline_series is not None else 'Time (s)')
        axs[3].grid(True); axs[3].legend(loc='upper right',fontsize=8)

        peak_delta=dr[reader_peak_idx]
        dominant='XYZ'[int(np.argmax(np.abs(peak_delta)))]
        summary=(f'{applied}-applied shock: Reader peak S={mr[reader_peak_idx]:.3f} g; '
                 f'dominant measured axis {dominant}; physical events={len(reader_events)}; '
                 f'release={release_s:.3g} s')
        if baseline_series is not None:
            base_peak=float(np.max(mb)); ratio=(float(mr[reader_peak_idx])/base_peak if base_peak>1e-12 else float('nan'))
            summary += f'; Baseline peak={base_peak:.3f} g; Reader/Baseline peak={ratio:.3f}'

        f.suptitle('Measured-data shock analysis\n'+summary,fontsize=11)
        f.tight_layout(rect=(0,0,1,.94)); c.draw_idle()

    def _shock_summary_rows(self):
        try:
            w=int(self.offline_shock_window.get()); threshold=float(self.offline_shock_threshold_g.get()); release_s=float(self.offline_shock_release_s.get())
        except Exception:
            w=50; threshold=1.0; release_s=0.25
        rows=[]
        for applied in 'XYZ':
            b=self.shock_baselines.get(applied); r=self.shock_readers.get(applied)
            row={'applied_axis':applied,'baseline_file':'','reader_file':'','baseline_peak_s_g':np.nan,'reader_peak_s_g':np.nan,'Kshock_base_over_reader':np.nan,'reader_event_count':0,'dominant_reader_axis':'','threshold_g':threshold,'event_release_s':release_s,'steady_state_window_samples':w}
            if b is not None and b.loaded:
                row['baseline_file']=Path(b.source).name if getattr(b,'source',None) else b.label
                try:
                    _,db,mb=self._shock_series_for_processor(b,w); row['baseline_peak_s_g']=float(np.max(mb))
                except Exception: pass
            if r is not None and r.loaded:
                row['reader_file']=Path(r.source).name if getattr(r,'source',None) else r.label
                try:
                    tr,dr,mr=self._shock_series_for_processor(r,w); ri=int(np.argmax(mr))
                    row['reader_peak_s_g']=float(mr[ri]); row['dominant_reader_axis']='XYZ'[int(np.argmax(np.abs(dr[ri])))]
                    events=self._group_shock_events(tr,mr,threshold,release_s); row['reader_event_count']=len(events)
                    if events:
                        durations=[float(tr[e]-tr[s]) for s,e,_ in events]
                        row['max_event_duration_s']=max(durations)
                    else: row['max_event_duration_s']=0.0
                    row['reader_peak_dX_g']=float(dr[ri,0]); row['reader_peak_dY_g']=float(dr[ri,1]); row['reader_peak_dZ_g']=float(dr[ri,2])
                except Exception: pass
            if np.isfinite(row['baseline_peak_s_g']) and np.isfinite(row['reader_peak_s_g']) and row['reader_peak_s_g']>1e-12:
                row['Kshock_base_over_reader']=row['baseline_peak_s_g']/row['reader_peak_s_g']
            rows.append(row)
        return rows

    def refresh_shock_summary(self):
        tree=getattr(self,'shock_summary_tree',None)
        if tree is None: return
        for item in tree.get_children(): tree.delete(item)
        for row in self._shock_summary_rows():
            vals=(row['applied_axis'],row['baseline_file'] or 'Not loaded',row['reader_file'] or 'Not loaded',
                  f"{row['baseline_peak_s_g']:.3f}" if np.isfinite(row['baseline_peak_s_g']) else '—',
                  f"{row['reader_peak_s_g']:.3f}" if np.isfinite(row['reader_peak_s_g']) else '—',
                  f"{row['Kshock_base_over_reader']:.3f}" if np.isfinite(row['Kshock_base_over_reader']) else '—',
                  str(row['reader_event_count']) if row['reader_file'] else '—',row['dominant_reader_axis'] or '—')
            tree.insert('',tk.END,values=vals)

    def export_shock_results(self):
        rows=self._shock_summary_rows()
        if not any(r['reader_file'] for r in rows):
            messagebox.showinfo('Export shock results','Load at least one dedicated shock Reader CSV first.'); return
        path=filedialog.asksaveasfilename(defaultextension='.csv',filetypes=[('CSV','*.csv')],initialfile='1174_shock_characterisation_summary.csv')
        if not path: return
        pd.DataFrame(rows).to_csv(path,index=False)
        sidecar=Path(path).with_name(Path(path).stem+'_settings.json')
        sidecar.write_text(json.dumps({'log_type':'1174 shock characterisation summary','generated_local':time.strftime('%Y-%m-%d %H:%M:%S'),'application_title':APP_TITLE,'shock_processing':{'steady_state_window_samples':int(self.offline_shock_window.get()),'threshold_g':float(self.offline_shock_threshold_g.get()),'event_release_s':float(self.offline_shock_release_s.get()),'method':'raw XYZ deviation from local steady-state vector; dedicated shock path'},'rows':rows},indent=2,ensure_ascii=False,default=str),encoding='utf-8')
        self.status.set(f'Exported shock summary to {Path(path).name} with {sidecar.name}.')

    def plot_shock_translation(self):
        """Peak-response comparison for dedicated baseline/reader shock captures.

        These are independent transient records, so this is intentionally not an
        H1/FRF calculation. The view compares peak resultant shock and peak X/Y/Z
        components for each deliberately applied shock axis.
        """
        f,c=self.clear('Shock translation')
        axs=f.subplots(2,2)
        try:
            w=int(self.offline_shock_window.get())
            if w<2:
                raise ValueError('Steady-state window must be at least 2 samples.')
        except Exception as exc:
            for ax in axs.flat: ax.axis('off')
            axs[0,0].text(.5,.5,str(exc),ha='center',va='center',wrap=True)
            c.draw_idle(); return

        rows=[]
        for applied in 'XYZ':
            b=self.shock_baselines.get(applied); r=self.shock_readers.get(applied)
            if b is None or r is None or not b.loaded or not r.loaded:
                continue
            try:
                tb,db,mb=self._shock_series_for_processor(b,w)
                tr,dr,mr=self._shock_series_for_processor(r,w)
                bi=int(np.argmax(mb)); ri=int(np.argmax(mr))
                bcomp=np.max(np.abs(db),axis=0); rcomp=np.max(np.abs(dr),axis=0)
                bres=float(mb[bi]); rres=float(mr[ri])
                kres=bres/rres if rres>1e-12 else np.nan
                rows.append({'applied':applied,'baseline_peak':bres,'reader_peak':rres,'K':kres,
                             'baseline_components':bcomp,'reader_components':rcomp,
                             'tb':tb-float(tb[bi]),'mb':mb,'tr':tr-float(tr[ri]),'mr':mr})
            except Exception:
                continue

        if not rows:
            for ax in axs.flat: ax.axis('off')
            axs[0,0].text(.5,.5,'Load at least one matching dedicated SHOCK baseline/reader pair.\nShock files do not use the normal vibration datasets.',ha='center',va='center',wrap=True)
            f.suptitle('Shock translation — no matched shock pairs')
            f.tight_layout(); c.draw_idle(); return

        x=np.arange(len(rows)); labels=[q['applied'] for q in rows]; width=.34
        axs[0,0].bar(x-width/2,[q['baseline_peak'] for q in rows],width,label='Baseline')
        axs[0,0].bar(x+width/2,[q['reader_peak'] for q in rows],width,label='Reader')
        axs[0,0].set_xticks(x,labels); axs[0,0].set_xlabel('Applied shock axis'); axs[0,0].set_ylabel('Peak resultant S (g)')
        axs[0,0].set_title('Peak resultant shock'); axs[0,0].legend(fontsize=8)

        axs[0,1].bar(x,[q['K'] for q in rows])
        axs[0,1].axhline(1.0,color='0.5',linewidth=1)
        axs[0,1].set_xticks(x,labels); axs[0,1].set_xlabel('Applied shock axis'); axs[0,1].set_ylabel('Kshock = Baseline / Reader')
        axs[0,1].set_title('Peak-resultant translation ratio')

        xx=np.arange(3); compw=.22
        for j,q in enumerate(rows):
            ratios=np.divide(q['baseline_components'],q['reader_components'],out=np.full(3,np.nan),where=np.asarray(q['reader_components'])>1e-12)
            axs[1,0].bar(xx+(j-(len(rows)-1)/2)*compw,ratios,compw,label=f'{q["applied"]}-applied')
        axs[1,0].axhline(1.0,color='0.5',linewidth=1)
        axs[1,0].set_xticks(xx,list('XYZ')); axs[1,0].set_xlabel('Measured shock component'); axs[1,0].set_ylabel('Baseline peak / Reader peak')
        axs[1,0].set_title('X/Y/Z component peak ratios'); axs[1,0].legend(fontsize=8)

        selected=self.offline_shock_axis.get().upper()
        q=next((z for z in rows if z['applied']==selected),rows[0])
        axs[1,1].plot(q['tb'],q['mb'],label='Baseline resultant S')
        axs[1,1].plot(q['tr'],q['mr'],label='Reader resultant S')
        axs[1,1].axvline(0,color='0.5',linewidth=.8)
        axs[1,1].set_xlabel('Time relative to peak shock (s)'); axs[1,1].set_ylabel('Resultant S (g)')
        axs[1,1].set_title(f'{q["applied"]}-applied aligned shock'); axs[1,1].legend(fontsize=8)

        for ax in axs.flat: ax.grid(True,alpha=.35)
        note=' | '.join(f"{q['applied']}: Kshock={q['K']:.3f}" for q in rows if np.isfinite(q['K']))
        f.suptitle('Shock translation overview — peak-response comparison only\n'+note,fontsize=10)
        f.tight_layout(rect=(0,0,1,.94)); c.draw_idle()

    def plot_stage(self):
        if self._issue1_active() and not isinstance(self.reader,BaselineSpectrum):
            result=self._offline_issue1_result()
            f,c=self.clear('Selected stage')
            self._draw_issue1_result(
                f,c,result,'Issue 1 calculation stages',float(self.fs.get()),
                selected_axis=self.stage_axis.get())
            return
        if self._show_issue2_inactive('Selected stage'): return
        a=self.stage_axis.get()
        n,ov,segs,start=self.params()
        base,reader,_,source=self._paired_axis_context(a)
        block=max(1,min(int(self.stage_block.get()),segs))
        hop=int(round(n*(1-ov)))
        stage_start=start+(block-1)*hop

        sb=base.stage(a,stage_start,n)
        sr=reader.stage(a,stage_start,n)
        f,c=self.clear('Selected stage')
        axs=f.subplots(3,2)
        tx=np.arange(n)/base.fs

        for s,label,alpha in [(sb,'Baseline',.85),(sr,'Reader',.75)]:
            axs[0,0].plot(tx,s['raw'],label=label,alpha=alpha)
            axs[0,1].plot(tx,s['mean_removed'],label=label,alpha=alpha)
            axs[1,1].plot(tx,s['windowed'],label=label,alpha=alpha)
            axs[2,0].plot(s['freq'],s['fft_mag'],label=label,alpha=alpha)
            axs[2,1].semilogy(s['freq'][1:],np.maximum(s['psd'][1:],1e-18),label=label,alpha=alpha)

        axs[1,0].plot(tx,sb['window'])
        axs[0,0].set_title('Raw 512-sample block')
        axs[0,1].set_title('Mean removed')
        axs[1,0].set_title('Hann window')
        axs[1,1].set_title('Windowed')
        axs[2,0].set_title('FFT magnitude')
        axs[2,1].set_title('Single-block PSD (g²/Hz)')

        for ax in axs.flat:
            ax.grid(True)
            if ax is not axs[1,0]:
                ax.legend(fontsize=8)

        duration=n/base.fs
        f.suptitle(
            f'{a}-axis processing stages (baseline source {source}) — FFT block {block} of {segs}; '
            f'samples {stage_start}:{stage_start+n} ({duration:.2f} s)\n'
            f'PSD-average set begins at sample {start}; hop={hop} samples'
        )
        f.tight_layout()
        c.draw_idle()

    def _plot_reference_psd(self):
        spectra=list({id(obj):obj for obj in self.baselines.values()
                      if isinstance(obj,BaselineSpectrum) and obj.loaded}.values())
        plot_axes={}
        if isinstance(self.reader,BaselineSpectrum):
            target=self.reader.metadata['excitation_axis']
            source=self.baseline_use_axis[target].get()
            baseline=self.baselines.get(source)
            if isinstance(baseline,Processor) and baseline.loaded:
                spectrum=self._baseline_axis_spectrum(target,source)
                spectra.append(spectrum); plot_axes[id(spectrum)]=[target]
            elif not isinstance(baseline,BaselineSpectrum):
                spectra.append(None)
        if not spectra: return False
        f,c=self.clear('PSD comparison')
        labels={'Ctl':'Ctl = measured control accelerometer PSD',
                'Ref':'Ref = commanded/nominal target PSD'}
        for i,spec in enumerate(spectra):
            ax=f.add_subplot(len(spectra),1,i+1)
            traces={} if spec is None else getattr(spec,'traces',{'PSD':(spec.freq_hz,spec.psd_g2_per_hz)})
            for name,(freq,psd) in traces.items():
                if name not in ('Ctl','Ref','PSD'): continue
                ax.semilogy(freq,np.where(psd>0,psd,np.nan),
                            label=labels.get(name,name),linewidth=1.8,
                            linestyle='--' if name=='Ref' else '-')
            axes=plot_axes.get(id(spec),[a for a in 'XYZ' if self.baselines.get(self.baseline_use_axis[a].get()) is spec])
            if isinstance(self.reader,BaselineSpectrum):
                reader=self.reader; reader_axis=reader.metadata['excitation_axis']
                if spec is None or reader_axis in axes:
                    for name,(freq,psd) in getattr(reader,'traces',{'PSD':(reader.freq_hz,reader.psd_g2_per_hz)}).items():
                        if name not in ('Ctl','Ref','PSD'): continue
                        ax.semilogy(freq,np.where(psd>0,psd,np.nan),label=f'Reader {reader_axis} {name}',linestyle='--' if name=='Ref' else '-')
            elif self.reader.loaded:
                n,ov,segs,start=self.params()
                for a in axes:
                    try:
                        freq,_,psd,_=self.reader.welch(a,start,n,ov,segs)
                        ax.semilogy(freq[1:],np.maximum(psd[1:],1e-18),label=f'Reader {a}',linewidth=1)
                    except ValueError as exc:
                        ax.text(.01,.01,f'Reader {a}: {exc}',transform=ax.transAxes,fontsize=8)
            reader_only=spec is None
            if reader_only: spec=self.reader
            orientation=spec.metadata.get('orientation')
            suffix='Lateral (X/Y)' if orientation=='Lateral' else 'Vertical (Z)' if orientation=='Vertical' else '/'.join(axes)
            ax.set(title=f'Baseline/reference PSD – {suffix}\n{Path(spec.source).name}',
                   xlabel='Frequency (Hz)',ylabel='PSD (g²/Hz)',
                   xlim=(spec.freq_hz[0],spec.freq_hz[-1]))
            ax.grid(True,which='both'); ax.legend(fontsize=8,ncol=2)
        f.tight_layout(); c.draw_idle()
        return True

    def plot_psd(self):
        if self._plot_reference_psd(): return
        if self._show_issue2_inactive('PSD comparison'): return
        n,ov,segs,start=self.params(); f,c=self.clear('PSD comparison'); ax=f.add_subplot(111)
        colours={'X':'tab:blue','Y':'tab:orange','Z':'tab:green'}
        plotted=False
        for a in 'XYZ':
            try:
                base,reader,_,source=self._paired_axis_context(a)
                fb,_,pb,_=base.welch(a,start,n,ov,segs); fr,_,pr,_=reader.welch(a,start,n,ov,segs)
                ax.semilogy(fb[1:],np.maximum(pb[1:],1e-18),linestyle='--',color=colours[a],alpha=.75,label=f'Baseline {a} ({source})')
                ax.semilogy(fr[1:],np.maximum(pr[1:],1e-18),linestyle='-',color=colours[a],label=f'Reader {a}')
                plotted=True
            except Exception:
                continue
        ax.set_xlim(3,90); ax.set(title='X / Y / Z averaged PSD — dashed baseline, solid reader',xlabel='Frequency (Hz)',ylabel='g²/Hz'); ax.grid(True,which='both')
        if plotted: ax.legend(ncol=2,fontsize=8)
        f.tight_layout(); c.draw_idle()

    def plot_bands(self):
        if self._issue1_active() and not isinstance(self.reader,BaselineSpectrum):
            result=self._offline_issue1_result()
            f,c=self.clear('Gordon bands')
            self._draw_issue1_gordon_comparison(
                f,c,result,'Issue 1 Fourier-amplitude comparison with Gordon Office')
            return
        if not isinstance(self.reader,BaselineSpectrum) and self._show_issue2_inactive('Gordon bands'): return
        if not self.reader.loaded:return
        n,ov,segs,start=self.params(); f,c=self.clear('Gordon bands'); ax=f.add_subplot(111); mode=self.units.get(); limv=gordon_office_velocity_um_s(GORDON_FC)
        colours={'X':'tab:blue','Y':'tab:orange','Z':'tab:green'}; plotted=False
        worst_ratio=-np.inf; worst_axis=''; worst_index=0; worst_measured=np.nan
        if mode=='RMS acceleration (g)': lim=(limv*1e-6)*(2*np.pi*GORDON_FC)/G0; yl='RMS g'
        elif mode=='RMS displacement (µm)': lim=limv/(2*np.pi*GORDON_FC); yl='RMS µm'
        else: lim=limv; yl='RMS µm/s'
        for a in 'XYZ':
            source=self.baseline_use_axis[a].get().upper(); src=self.baselines.get(source)
            try:
                if isinstance(src,BaselineSpectrum) or isinstance(self.reader,BaselineSpectrum):
                    bb=self._baseline_axis_spectrum(a,source).bands(); rb,_=self._reader_axis_band_statistics(a)
                    if mode=='RMS acceleration (g)': b=bb.a_rms_g; r=rb.a_rms_g
                    elif mode=='RMS displacement (µm)': b=bb.x_rms_um; r=rb.x_rms_um
                    else: b=bb.v_rms_um_s; r=rb.v_rms_um_s
                else:
                    base,reader,pair,source=self._paired_axis_context(a); d=pair.paired_band_window(start,n,ov,segs); q=d[d.axis==a]
                    if mode=='RMS acceleration (g)': b=q.a_rms_g; r=q.reader_a_rms_g
                    elif mode=='RMS displacement (µm)': b=q.x_rms_um; r=q.reader_x_rms_um
                    else: b=q.v_rms_um_s; r=q.reader_v_rms_um_s
                ax.semilogx(GORDON_FC,b,marker='o',linestyle='--',color=colours[a],alpha=.75,label=f'Baseline {a} ({source})')
                ax.semilogx(GORDON_FC,r,marker='o',linestyle='-',color=colours[a],label=f'Reader {a}')
                reader_values=np.asarray(r,float)
                ratios=np.divide(reader_values,lim,out=np.zeros_like(reader_values),where=lim>0)
                index=int(np.argmax(ratios))
                if ratios[index]>worst_ratio:
                    worst_ratio=float(ratios[index]); worst_axis=a
                    worst_index=index; worst_measured=float(reader_values[index])
                plotted=True
            except Exception:
                continue
        ax.semilogx(GORDON_FC,lim,marker='s',color='0.25',linewidth=1.5,label='Gordon Office')
        set_gordon_xaxis(ax,rotate=45); ax.set_xlabel('1/3-octave band centre (Hz)'); ax.set_ylabel(yl)
        title=f'Issue 2 X / Y / Z Gordon-band RMS comparison — {mode}'
        if plotted:
            state='EXCEEDED' if worst_ratio>1.0 else 'PASS'
            title+=(f'\n{state}: worst reader {worst_axis} at {GORDON_FC[worst_index]:g} Hz; '
                   f'{worst_measured:.6g} / {float(lim[worst_index]):.6g} {yl} = '
                   f'{worst_ratio:.3f}×')
        ax.set_title(title); ax.grid(True,which='both')
        if plotted: ax.legend(ncol=2,fontsize=8)
        f.tight_layout(); c.draw_idle()

    def plot_transfer(self):
        if self._show_issue2_inactive('Transfer function'): return
        n,ov,segs,start=self.params()
        a=self.transfer_axis.get()
        source=self.baseline_use_axis[a].get()
        if isinstance(self.baselines.get(source),BaselineSpectrum) or isinstance(self.reader,BaselineSpectrum):
            f,c=self.clear('Transfer function'); ax=f.add_subplot(111); ax.axis('off')
            ax.text(.5,.5,'PSD-only data: H1, synchronous phase and coherence are unavailable.\nUse Gordon bands or Translation function for Ctl-based correction.',
                    ha='center',va='center',wrap=True)
            c.draw_idle(); return
        base,reader,pair,source=self._paired_axis_context(a)
        d=pair.paired_band_window(start,n,ov,segs)
        q=d[d.axis==a]
        freq,H,coh=pair.h1(a,n,ov)

        f,c=self.clear('Transfer function')
        axs=f.subplots(2,2)

        axs[0,0].semilogx(q.fc_hz,q.H_reader_over_base,marker='o',label=f'Driven {a}')
        axs[0,1].semilogx(q.fc_hz,q.K_base_over_reader,marker='o',label=f'Driven {a}')

        axs[0,0].axhline(1,color='0.5',linewidth=1)
        axs[0,0].set(title=f'{a}-axis band transfer H = Reader / Base',ylabel='Amplitude ratio')
        axs[0,1].axhline(1,color='0.5',linewidth=1)
        axs[0,1].set(title=f'{a}-axis correction K = Base / Reader',ylabel='Correction factor')
        set_gordon_xaxis(axs[0,0], rotate=45)
        set_gordon_xaxis(axs[0,1], rotate=45)

        m=(freq>=3.5)&(freq<=90)
        axs[1,0].plot(freq[m],np.abs(H[m]))
        axs[1,0].axhline(1,color='0.5',linewidth=1)
        axs[1,0].set(title=f'Driven {a}: H1 FRF magnitude',xlabel='Frequency (Hz)',ylabel='|H1|')
        axs[1,1].plot(freq[m],coh[m])
        axs[1,1].set_ylim(0,1.05)
        axs[1,1].set(title=f'Driven {a}: magnitude-squared coherence',xlabel='Frequency (Hz)',ylabel='Coherence')

        for ax in axs.flat:
            ax.grid(True,which='both')
        axs[0,0].legend(fontsize=8)
        axs[0,1].legend(fontsize=8)

        f.suptitle(
            f'Single-axis transfer characterisation — selected axis: {a}; baseline source {source}\n'
            'Non-driven reader axes are recorded but are not used to derive H or K.'
        )
        f.tight_layout()
        c.draw_idle()

    def _plot_issue1_translation(self):
        """Show driven-axis-only Issue 1 translations for every loaded reader slot."""
        f,c=self.clear('Translation function'); axs=f.subplots(2,2)
        colours={'X':'tab:blue','Y':'tab:orange','Z':'tab:green'}; notes=[]; yl='Peak g per FFT bin'
        for axis in 'XYZ':
            reader=self.reader_runs.get(axis)
            if reader is None:
                # Backward-compatible single active reader, but only when its
                # excitation metadata agrees with the requested axis.
                candidate=self.reader
                if getattr(candidate,'loaded',False) and str(candidate.metadata.get('excitation_axis',axis)).upper()==axis:
                    reader=candidate
            if reader is None or not reader.loaded or isinstance(reader,BaselineSpectrum):
                notes.append(f'{axis}: no {axis}-excited time-domain reader run'); continue
            try:
                result=self.derive_translation(axis,ISSUE1_METHOD)
                rf,reader_amp,_=self._issue1_axis_amplitudes_from_processor(reader,axis)
                idx=[int(np.argmin(np.abs(rf-fc))) for fc in result.coordinates_hz]
                raw_peak=reader_amp[idx]
                ref_peak=raw_peak*result.k_values
                raw_values,yl,_=self._issue1_display_values(raw_peak,result.coordinates_hz)
                ref_values,_,_=self._issue1_display_values(ref_peak,result.coordinates_hz)
                translated_values=raw_values*result.k_values
                axs[0,0].semilogx(result.coordinates_hz,result.k_values,marker='o',color=colours[axis],label=f'{axis} K')
                axs[0,1].semilogx(result.coordinates_hz,ref_values,marker='o',color=colours[axis],label=f'Reference {axis}')
                axs[1,0].semilogx(result.coordinates_hz,raw_values,marker='o',color=colours[axis],label=f'Raw {axis}-excited reader {axis}')
                axs[1,1].semilogx(result.coordinates_hz,translated_values,marker='o',color=colours[axis],label=f'Translated {axis}')
                notes.append(f'{axis}: {result.reference_domain} reference; median K {np.nanmedian(result.k_values):.3f}')
            except Exception as exc:
                notes.append(f'{axis}: unavailable ({exc})')
        axs[0,0].axhline(1.0,color='0.5',linewidth=1)
        axs[0,0].set_title('Issue 1 direct K = reference / driven-axis reader amplitude')
        axs[0,1].set_title('Reference Issue 1 amplitude')
        axs[1,0].set_title('Raw primary reader response')
        axs[1,1].set_title('Translated environmental estimate')
        for ax in axs.flat:
            ax.set_xlim(ISSUE1_LOW_HZ,ISSUE1_HIGH_HZ); ax.set_xlabel('Issue 1 FFT frequency (Hz)'); ax.grid(True,which='both')
            if ax.lines: ax.legend(fontsize=8)
        axs[0,0].set_ylabel('K factor')
        for ax in (axs[0,1],axs[1,0],axs[1,1]): ax.set_ylabel(yl)
        f.suptitle('Issue 1 X / Y / Z driven-axis translation overview\n'+' | '.join(notes),fontsize=10)
        f.tight_layout(); c.draw_idle()

    def plot_translation(self):
        if self._issue1_active() and not isinstance(self.reader,BaselineSpectrum):
            self._plot_issue1_translation()
            return
        if not isinstance(self.reader,BaselineSpectrum) and self._show_issue2_inactive('Translation function'): return
        f,c=self.clear('Translation function'); axs=f.subplots(2,2)
        colours={'X':'tab:blue','Y':'tab:orange','Z':'tab:green'}
        model_notes=[]; any_plot=False
        for a in 'XYZ':
            source=self.baseline_use_axis[a].get().upper()
            try:
                scores,rec,common,band,series,_=self._axis_correction_result(a,source)
                kval=np.array([band[float(fc)] for fc in GORDON_FC],float)
                axs[0,0].semilogx(GORDON_FC,kval,marker='o',color=colours[a],label=f'{a} K ({source})')
                # Build representative band values. PSD baselines expose explicit *_base/*_reader columns;
                # time-domain model series contains multiple windows, so use median by band.
                if 'a_rms_g_base' in series.columns:
                    q=series.copy()
                    base_v=q.v_rms_um_s_base.to_numpy(float) if 'v_rms_um_s_base' in q else q.a_rms_g_base.to_numpy(float)*G0/(2*np.pi*q.fc_hz.to_numpy(float))*1e6
                    reader_v=q.v_rms_um_s_reader.to_numpy(float) if 'v_rms_um_s_reader' in q else q.a_rms_g_reader.to_numpy(float)*G0/(2*np.pi*q.fc_hz.to_numpy(float))*1e6
                    fc=q.fc_hz.to_numpy(float)
                else:
                    q=series.groupby('fc_hz',as_index=False)[['v_rms_um_s','reader_v_rms_um_s']].median()
                    fc=q.fc_hz.to_numpy(float); base_v=q.v_rms_um_s.to_numpy(float); reader_v=q.reader_v_rms_um_s.to_numpy(float)
                k_for=np.array([band[float(x)] for x in fc],float); corrected=reader_v*k_for
                axs[0,1].semilogx(fc,base_v,marker='o',color=colours[a],label=f'Baseline {a}')
                axs[1,0].semilogx(fc,reader_v,marker='o',color=colours[a],label=f'Raw reader {a}')
                axs[1,1].semilogx(fc,corrected,marker='o',color=colours[a],label=f'Corrected {a}')
                model_notes.append(f'{a}: {rec}; single-K {common:.3f}; source {source}')
                any_plot=True
            except Exception as exc:
                model_notes.append(f'{a}: unavailable ({exc})')
        axs[0,0].axhline(1.0,color='0.5',linewidth=1); axs[0,0].set_title('Multi-frequency K = Base / Reader')
        axs[0,1].set_title('Baseline/reference band RMS velocity')
        axs[1,0].set_title('Raw reader band RMS velocity')
        axs[1,1].set_title('Reader after multi-frequency K correction')
        for ax in axs.flat:
            set_gordon_xaxis(ax,rotate=45); ax.set_xlabel('Band centre (Hz)'); ax.grid(True,which='both'); ax.legend(fontsize=8)
        axs[0,0].set_ylabel('K factor'); axs[0,1].set_ylabel('RMS velocity (µm/s)'); axs[1,0].set_ylabel('RMS velocity (µm/s)'); axs[1,1].set_ylabel('RMS velocity (µm/s)')
        f.suptitle('X / Y / Z translation overview\n'+' | '.join(model_notes),fontsize=10)
        f.tight_layout(); c.draw_idle()

    def plot_tilt(self):
        avg=max(1,int(self.tilt_average_samples.get()))/float(self.fs.get()); tb,xb,yb,zb,rb,pb,_=self.base.tilt(avg); tr,xr,yr,zr,rr,pr,_=self.reader.tilt(avg); m=min(len(tb),len(tr)); f,c=self.clear('Tilt comparison'); axs=f.subplots(2,2)
        # mean gravity vectors
        gb=np.array([np.mean(xb),np.mean(yb),np.mean(zb)]); gr=np.array([np.mean(xr),np.mean(yr),np.mean(zr)])
        axs[0,0].quiver([0,0],[0,0],[gb[0],gr[0]],[gb[2],gr[2]],angles='xy',scale_units='xy',scale=1); axs[0,0].set(title='Gravity vector: X-Z plane',xlabel='X (g)',ylabel='Z (g)'); axs[0,0].axis('equal')
        axs[0,1].quiver([0,0],[0,0],[gb[1],gr[1]],[gb[2],gr[2]],angles='xy',scale_units='xy',scale=1); axs[0,1].set(title='Gravity vector: Y-Z plane',xlabel='Y (g)',ylabel='Z (g)'); axs[0,1].axis('equal')
        br=float(np.mean(rb)); bp=float(np.mean(pb)); rr0=float(np.mean(rr)); rp0=float(np.mean(pr)); xpos=np.arange(2); w=.35; axs[1,0].bar(xpos-w/2,[br,bp],w,label='Baseline'); axs[1,0].bar(xpos+w/2,[rr0,rp0],w,label='Reader'); axs[1,0].set_xticks(xpos,['Roll-like','Pitch-like']); axs[1,0].set_ylabel('Angle (deg)'); axs[1,0].set_title(f'Mean tilt offsets: Δroll={rr0-br:+.3f}°, Δpitch={rp0-bp:+.3f}°'); axs[1,0].legend()
        axs[1,1].plot(tb[:m],rr[:m]-rb[:m],label='Reader - baseline roll'); axs[1,1].plot(tb[:m],pr[:m]-pb[:m],label='Reader - baseline pitch'); axs[1,1].set(title='Relative mounting tilt versus time',xlabel='Time (s)',ylabel='Angle difference (deg)'); axs[1,1].legend()
        for ax in axs.flat:ax.grid(True); f.suptitle('Static/mounting tilt comparison'); f.tight_layout(); c.draw_idle()
    def update_summary(self):
        n,ov,segs,start=self.params()
        a=self.driven_axis.get()
        scores,rec,common,band,series=self.pair.model_comparison(n,ov,segs,a)
        tilt_avg=max(1,int(self.tilt_average_samples.get()))/float(self.fs.get())
        _,xb,yb,zb,rb,pb,_=self.base.tilt(tilt_avg)
        _,xr,yr,zr,rr,pr,_=self.reader.tilt(tilt_avg)
        dr=float(np.mean(rr)-np.mean(rb))
        dp=float(np.mean(pr)-np.mean(pb))

        lines=[
            'SINGLE-AXIS TRANSLATION ASSESSMENT',
            '',
            f'Intentionally driven axis: {a}',
            'Transfer/correction calculations use only this axis.',
            'The other two measured axes are retained for cross-axis diagnostics only.',
            '',
            f'Recommended structure: {rec}',
            '',
            'Production form:',
            f'Base_est_{a}[band] = K_{a}[band] × Reader_RMS_{a}[band]',
            f'K_{a}[band] = Base_RMS_{a}[band] / Reader_RMS_{a}[band]',
            '',
            'Validation error (paired time windows):'
        ]
        for _,q in scores.iterrows():
            lines.append(
                f"{q['model']}: median {q.median_abs_error_pct:.2f}%, "
                f"95th {q.p95_abs_error_pct:.2f}%, RMSE {q.rmse_pct:.2f}%"
            )

        lines += [
            '',
            'Interpretation:',
            'H = Reader / Base on the driven axis.',
            'H > 1 means local amplification; H < 1 means attenuation/deadening.',
            'K = Base / Reader = 1/H.',
            '',
            f'Single-K candidate for {a}: {common:.4f}',
            '',
            'RELATIVE TILT',
            f'Δ roll-like angle: {dr:+.4f} deg',
            f'Δ pitch-like angle: {dp:+.4f} deg',
            '',
            'For production coefficients, combine controlled physical characterisation',
            'from repeated runs/readers for the same driven axis and vibration level/profile.'
        ]
        self.summary.configure(state=tk.NORMAL)
        self.summary.delete('1.0',tk.END)
        self.summary.insert('1.0','\n'.join(lines))
        self.summary.configure(state=tk.DISABLED)

    def _resolve_missing_baseline_mapping(self,target_axis):
        source=self.baseline_use_axis[target_axis].get().upper()
        if self.baselines.get(source) is not None and self.baselines[source].loaded:
            return source
        available=[a for a in 'XYZ' if self.baselines.get(a) is not None and self.baselines[a].loaded]
        if not available:
            raise ValueError('No baseline CSVs are loaded.')
        response=simpledialog.askstring(
            'Missing baseline',
            f'The {target_axis}-axis correction is mapped to baseline {source}, but that baseline is not loaded.\n\n'
            f'Available baseline axes: {", ".join(available)}\n\n'
            f'Enter which baseline axis to use as a surrogate for {target_axis}:',
            parent=self.root)
        if response is None:
            raise RuntimeError('Correction export cancelled by user.')
        chosen=response.strip().upper()
        if chosen not in available:
            raise ValueError(f'Baseline {chosen!r} is not available. Loaded axes: {", ".join(available)}')
        self.baseline_use_axis[target_axis].set(chosen)
        return chosen

    def _baseline_axis_spectrum(self,target_axis,source_axis):
        src=self.baselines.get(source_axis)
        if isinstance(src,BaselineSpectrum): return src
        base=self._mapped_baseline_processor(target_axis,source_axis)
        base.fs=float(self.fs.get())
        n,ov,segs,start=self.params()
        freq,_,psd,_=base.welch(target_axis,start,n,ov,segs)
        spectrum=BaselineSpectrum('Baseline PSD')
        spectrum.set(freq,psd,base.source,base.metadata)
        return spectrum

    def _reader_axis_band_statistics(self,target_axis):
        """Return reader Gordon RMS from its PSD or complete time-domain windows."""
        reader=self.reader_runs.get(target_axis) or self.reader
        excitation=str(getattr(reader,'metadata',{}).get('excitation_axis',target_axis)).upper()
        if excitation!=target_axis:
            raise ValueError(f'no {target_axis} reader data: the active reader run is {excitation}-excited and is cross-axis diagnostic only.')
        if isinstance(reader,BaselineSpectrum):
            axis=reader.metadata.get('excitation_axis')
            if target_axis!=axis:
                raise ValueError(f'Reader PSD represents {axis} only; no {target_axis} reader data is loaded.')
            bands=reader.bands()
            return bands,bands.copy()
        n,ov,segs,start=self.params()
        # Use all complete analysis spans from the reader. This is deliberately
        # independent of the frequency-domain baseline because there is no
        # synchronous time history to pair against.
        hop=int(round(n*(1-ov)))
        span=n+(segs-1)*hop
        total=len(reader.df)
        starts=list(range(0,total-span+1,span))
        if not starts:
            # Fall back to the selected start if the file only contains one span.
            starts=[start] if start>=0 and start+span<=total else []
        if not starts:
            raise ValueError(f'Reader file does not contain the {span} samples required for one complete PSD analysis span.')
        frames=[]
        for s in starts:
            f,_,p,_=reader.welch(target_axis,s,n,ov,segs)
            b=reader.bands(f,p).copy(); b['start_sample']=s; frames.append(b)
        allb=pd.concat(frames,ignore_index=True)
        med=allb.groupby('fc_hz',as_index=False)[['a_rms_g','v_rms_um_s','x_rms_um']].median()
        return med,allb

    def _axis_correction_from_psd_baseline(self,target_axis,source_axis,spectrum):
        """Derive band K values from an unpaired PSD reference and reader medians."""
        base_bands=spectrum.bands()
        reader_med,reader_windows=self._reader_axis_band_statistics(target_axis)
        q=base_bands.merge(reader_med,on='fc_hz',suffixes=('_base','_reader'))
        q['K_base_over_reader']=np.divide(
            q.a_rms_g_base.to_numpy(float),q.a_rms_g_reader.to_numpy(float),
            out=np.full(len(q),np.nan),where=q.a_rms_g_reader.to_numpy(float)>1e-12)
        valid=q.K_base_over_reader.replace([np.inf,-np.inf],np.nan).dropna()
        common=float(valid.median()) if len(valid) else float('nan')
        band={float(r.fc_hz):float(r.K_base_over_reader) for _,r in q.iterrows()}
        rec='Multi-frequency K (PSD baseline)'
        return rec,common,band,q,reader_windows

    def _axis_correction_result(self,target_axis,source_axis):
        """Select the PSD or paired-time-history correction workflow for one axis."""
        reader=self.reader_runs.get(target_axis) or self.reader
        excitation=str(getattr(reader,'metadata',{}).get('excitation_axis',target_axis)).upper()
        if excitation!=target_axis:
            raise ValueError(f'{target_axis} correction requires the {target_axis}-excited reader run, not {excitation}-excited cross-axis data.')
        src=self.baselines.get(source_axis)
        if isinstance(src,BaselineSpectrum) or isinstance(reader,BaselineSpectrum):
            src=self._baseline_axis_spectrum(target_axis,source_axis)
            rec,common,band,series,reader_windows=self._axis_correction_from_psd_baseline(target_axis,source_axis,src)
            return None,rec,common,band,series,src
        local_base=self._mapped_baseline_processor(target_axis,source_axis)
        # Work on independent Processor copies so export cannot alter the
        # currently displayed/cropped datasets.
        local_reader=self._copy_processor(reader,label='Reader')
        local_base.fs=local_reader.fs=float(self.fs.get())
        pair=PairAnalysis(local_base,local_reader)
        n,ov,segs,start=self.params()
        scores,rec,common,band,series=pair.model_comparison(n,ov,segs,target_axis)
        return scores,rec,common,band,series,local_base

    def _issue1_axis_amplitudes_from_processor(self,proc,axis):
        """Median Issue-1 ensemble amplitude for one primary axis across complete spans."""
        if isinstance(proc,BaselineSpectrum):
            raise ValueError('Time-domain processor required.')
        required=issue1_required_samples()
        total=len(proc.df)
        starts=list(range(0,total-required+1,required))
        if not starts:
            start=int(self.start.get())
            starts=[start] if start>=0 and start+required<=total else []
        if not starts:
            raise ValueError(f'{proc.label}: Issue 1 needs at least {required} samples for one 19-FFT ensemble.')
        xyz=np.column_stack([proc.axis(a) for a in 'XYZ'])
        values=[]; frequency=None
        for start in starts:
            result=issue1_ensemble_fft(xyz,float(proc.fs),start=start)
            frequency=result.frequency_hz
            values.append(result.ensemble_amplitude_g['XYZ'.index(axis)])
        return np.asarray(frequency,float),np.nanmedian(np.vstack(values),axis=0),starts

    def _derive_issue1_translation(self,target_axis,source_axis):
        reader=self.reader_runs.get(target_axis) or self.reader
        if reader is None or not reader.loaded or isinstance(reader,BaselineSpectrum):
            raise ValueError(f'Issue 1 requires a time-domain {target_axis}-excited reader capture.')
        excitation=str(reader.metadata.get('excitation_axis',target_axis)).upper()
        if excitation!=target_axis:
            raise ValueError(f'{target_axis} translation requires a {target_axis}-excited reader run; {excitation}-excited data are cross-axis diagnostic only.')
        rf,reader_amp,reader_starts=self._issue1_axis_amplitudes_from_processor(reader,target_axis)
        reference=self.baselines.get(source_axis)
        if reference is None or not reference.loaded:
            raise ValueError(f'No {source_axis} reference is loaded for {target_axis}.')
        if isinstance(reference,BaselineSpectrum):
            bf,base_amp=issue1_amplitude_from_psd(
                reference.freq_hz,reference.psd_g2_per_hz,float(reader.fs),ISSUE1_FFT_N,True)
            if not np.allclose(bf,rf):
                base_amp=np.interp(rf,bf,base_amp,left=np.nan,right=np.nan)
            reference_domain='PSD'
        else:
            mapped=self._mapped_baseline_processor(target_axis,source_axis)
            mapped.fs=float(reader.fs)
            _,base_amp,_=self._issue1_axis_amplitudes_from_processor(mapped,target_axis)
            reference_domain='time'
        active=(rf>=ISSUE1_LOW_HZ)&(rf<=ISSUE1_HIGH_HZ)&np.isfinite(base_amp)&np.isfinite(reader_amp)&(reader_amp>1e-12)
        coords=rf[active]
        k=base_amp[active]/reader_amp[active]
        if coords.size==0 or not np.all(np.isfinite(k)):
            raise ValueError('Issue 1 translation contains no valid frequency coordinates.')
        settings={
            'sample_rate_hz':float(reader.fs),'fft_length':ISSUE1_FFT_N,
            'fft_overlap_percent':ISSUE1_OVERLAP*100.0,'fft_ensemble_count':ISSUE1_ENSEMBLE_COUNT,
            'fir_hz':[ISSUE1_LOW_HZ,ISSUE1_HIGH_HZ],'fft_window':'Hann','fir_window':'Hamming',
        }
        return TranslationResult(
            ISSUE1_METHOD,target_axis,coords,k,reference.source,reader.source,
            reference_domain,source_axis,settings,
            {'reader_ensemble_starts':reader_starts,'reference_selected_source':reference.metadata.get('selected_source','time-domain')}
        )

    def _derive_issue2_translation(self,target_axis,source_axis):
        reader=self.reader_runs.get(target_axis) or self.reader
        if reader is None or not reader.loaded:
            raise ValueError(f'No {target_axis}-excited reader run is loaded.')
        scores,rec,common,band,series,local_base=self._axis_correction_result(target_axis,source_axis)
        reference=self.baselines[source_axis]
        coords=np.asarray(GORDON_FC,float)
        k=np.asarray([band[float(fc)] for fc in coords],float)
        settings={
            'sample_rate_hz':float(self.fs.get()),'fft_length':int(self.n.get()),'fft_window':'Hann',
            'fft_overlap_percent':float(self.ov.get()),'psd_averages':int(self.segs.get()),
            'gordon_band_centres_hz':[float(v) for v in GORDON_FC],
        }
        return TranslationResult(
            ISSUE2_METHOD,target_axis,coords,k,reference.source,reader.source,
            'PSD' if isinstance(reference,BaselineSpectrum) else 'time',source_axis,settings,
            {'recommended_model':rec,'single_K_candidate':float(common),'reference_selected_source':reference.metadata.get('selected_source','time-domain')}
        )

    def derive_translation(self,target_axis=None,method=None):
        """Authoritative translation path used by store/export for both methods."""
        axis=(target_axis or self.driven_axis.get()).upper()
        if axis not in 'XYZ': raise ValueError('Driven axis must be X, Y or Z.')
        source=self._resolve_missing_baseline_mapping(axis)
        selected=method or self.applied_processing_method
        if selected==ISSUE1_METHOD:
            return self._derive_issue1_translation(axis,source)
        if selected==ISSUE2_METHOD:
            return self._derive_issue2_translation(axis,source)
        raise ValueError(f'Unsupported processing method: {selected}')

    def _write_correction_export_log(self, csv_path, rows, mappings):
        """Write a sidecar JSON that records exactly how exported K factors were generated."""
        csv_path=Path(csv_path)
        sidecar=csv_path.with_name(csv_path.stem+'_settings.json')
        baseline_files={axis:(Path(obj.source).name if obj is not None and getattr(obj,'source','') else None)
                        for axis,obj in self.baselines.items()}
        payload={
            'log_type':'1174 vibration correction generation settings',
            'generated_local':time.strftime('%Y-%m-%d %H:%M:%S'),
            'application_title':APP_TITLE,
            'correction_csv':csv_path.name,
            'reader_source_file':Path(self.reader.source).name if self.reader.source else self.reader.label,
            'reader_details':dict(self.reader.metadata),
            'baseline_source_files':baseline_files,
            'baseline_mapping':dict(mappings),
            'baseline_details':{a:dict(obj.metadata) for a,obj in self.baselines.items() if obj is not None},
            'processing_settings':{
                'processing_method':ISSUE2_METHOD,
                'sample_rate_hz':float(self.fs.get()),
                'fft_length':int(self.n.get()),
                'fft_window':'Hann',
                'fft_overlap_percent':float(self.ov.get()),
                'psd_averages':int(self.segs.get()),
                'psd_set_start_sample':int(self.start.get()),
                'auto_crop_common_duration':bool(self.auto_crop.get()),
                'gordon_band_centres_hz':[float(v) for v in GORDON_FC],
            },
            'coefficients':rows,
        }
        sidecar.write_text(json.dumps(payload,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
        return sidecar

    def _update_stored_correction_status(self):
        parts=[]
        for axis in 'XYZ':
            q=self.stored_correction_axes.get(axis)
            parts.append(f"{axis} {'stored' if q else '—'}")
        self.stored_correction_status.set('Final XYZ correction: '+' | '.join(parts))

    def store_current_axis_correction(self):
        """Derive and retain the selected driven-axis translation for the active method."""
        axis=self.driven_axis.get().upper()
        try:
            result=self.derive_translation(axis,self.applied_processing_method)
            rows=result.rows()
            self.stored_correction_axes[axis]={
                'method':result.method,
                'result':result,
                'rows':rows,
                'reader_source_file':Path(result.reader_source).name if result.reader_source else '',
                'baseline_source_axis':result.reference_axis,
                'stored_local':time.strftime('%Y-%m-%d %H:%M:%S'),
            }
            self._update_stored_correction_status()
            self.status.set(f'Stored {axis}-axis {result.method} translation from the {axis}-excited reader run.')
        except Exception as exc:
            messagebox.showerror('Store translation',str(exc))

    def export_stored_xyz_correction(self):
        """Export one complete XYZ translation set; mixed processing methods are rejected."""
        missing=[a for a in 'XYZ' if not self.stored_correction_axes.get(a)]
        if missing:
            messagebox.showinfo('Export final XYZ translation','Store the driven-axis translation for '+', '.join(missing)+' before final export.'); return
        methods={self.stored_correction_axes[a].get('method') for a in 'XYZ'}
        if len(methods)!=1:
            messagebox.showerror('Export final XYZ translation','The stored X/Y/Z results use different processing methods. Re-derive all three using either Issue 1 or Issue 2.'); return
        method=next(iter(methods))
        path=filedialog.asksaveasfilename(defaultextension='.csv',filetypes=[('CSV','*.csv')],initialfile=f'1174_final_XYZ_{"issue1" if method==ISSUE1_METHOD else "issue2"}_translation.csv')
        if not path: return
        rows=[]
        for axis in 'XYZ': rows.extend(self.stored_correction_axes[axis]['rows'])
        pd.DataFrame(rows).to_csv(path,index=False)
        sidecar=Path(path).with_name(Path(path).stem+'_settings.json')
        axis_sources={}
        for axis in 'XYZ':
            result=self.stored_correction_axes[axis]['result']
            axis_sources[axis]={
                'method':result.method,'reference_source':result.reference_source,
                'reader_source':result.reader_source,'reference_domain':result.reference_domain,
                'reference_axis':result.reference_axis,'settings':result.settings,'metadata':result.metadata,
            }
        payload={'log_type':'1174 final XYZ vibration translation set','generated_local':time.strftime('%Y-%m-%d %H:%M:%S'),'application_title':APP_TITLE,'processing_method':method,'axis_sources':axis_sources,'coefficients':rows}
        sidecar.write_text(json.dumps(payload,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
        self.status.set(f'Exported complete XYZ {method} translation set to {Path(path).name}.')
        messagebox.showinfo('Final XYZ translation exported',f'Saved {len(rows)} coefficients to:\n{Path(path).name}\n\nTraceability:\n{sidecar.name}')

    def export_correction(self):
        """Export translations derivable from the currently loaded excitation slots."""
        available=[a for a in 'XYZ' if self.reader_runs.get(a) is not None and self.reader_runs[a].loaded]
        if not available and self.reader.loaded:
            available=[self.driven_axis.get().upper()]
        if not available:
            messagebox.showinfo('Export translation','Load at least one reader characterisation run first.'); return
        try:
            results=[]
            for axis in available:
                results.append(self.derive_translation(axis,self.applied_processing_method))
            path=filedialog.asksaveasfilename(
                defaultextension='.csv',filetypes=[('CSV','*.csv')],
                initialfile=f'1174_{"".join(r.driven_axis for r in results)}_{"issue1" if self.applied_processing_method==ISSUE1_METHOD else "issue2"}_translation.csv')
            if not path: return
            rows=[]
            for result in results: rows.extend(result.rows())
            pd.DataFrame(rows).to_csv(path,index=False)
            sidecar=Path(path).with_name(Path(path).stem+'_settings.json')
            payload={
                'log_type':'1174 vibration translation export','generated_local':time.strftime('%Y-%m-%d %H:%M:%S'),
                'application_title':APP_TITLE,'processing_method':self.applied_processing_method,
                'reader_details':dict(self.reader.metadata) if getattr(self.reader,'metadata',None) is not None else {},
                'axis_sources':{r.driven_axis:{'reference_source':r.reference_source,'reader_source':r.reader_source,'reference_domain':r.reference_domain,'reference_axis':r.reference_axis,'settings':r.settings,'metadata':r.metadata} for r in results},
                'coefficients':rows,
            }
            sidecar.write_text(json.dumps(payload,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
            self.status.set(f'Exported {len(rows)} {self.applied_processing_method} translation coefficients to {Path(path).name}.')
            messagebox.showinfo('Translation exported',f'Saved {len(rows)} coefficients to:\n{Path(path).name}\n\nTraceability:\n{sidecar.name}')
        except Exception as exc:
            messagebox.showerror('Export translation',str(exc))

    def save_plot(self):
        idx=self.tabs.index(self.tabs.select()); name=self.tabs.tab(idx,'text'); f,_=self.fig[name]; path=filedialog.asksaveasfilename(defaultextension='.png',filetypes=[('PNG','*.png'),('PDF','*.pdf')]);
        if path:f.savefig(path,dpi=200,bbox_inches='tight')

def main():
    root=tk.Tk(); App(root); root.mainloop()
if __name__=='__main__': main()
