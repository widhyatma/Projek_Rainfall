# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend - allows running as script
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import warnings
warnings.filterwarnings('ignore')

# --- Style: Standard Light Theme ---
plt.rcParams.update({
    'figure.facecolor':  'white',
    'axes.facecolor':    '#F8F9FA',
    'axes.edgecolor':    '#CCCCCC',
    'axes.labelcolor':   '#333333',
    'xtick.color':       '#555555',
    'ytick.color':       '#555555',
    'grid.color':        '#DDDDDD',
    'grid.alpha':        0.8,
    'text.color':        '#222222',
    'font.family':       'DejaVu Sans',
    'font.size':         10,
    'axes.titlesize':    12,
    'axes.titleweight':  'bold',
    'legend.facecolor':  'white',
    'legend.edgecolor':  '#CCCCCC',
    'legend.fontsize':   9,
    'savefig.facecolor': 'white',
    'savefig.dpi':       150,
    'savefig.bbox':      'tight',
})

# --- Color Palette (standard, accessible) ---
COLOR_ORIGINAL = '#2196F3'    # Blue - original sensor data
COLOR_LINEAR   = '#4CAF50'    # Green - linear interpolated
COLOR_LSTM     = '#F44336'    # Red - LSTM imputed
COLOR_TEMP     = '#E65100'    # Dark orange - temperature
COLOR_HUM      = '#1565C0'    # Dark blue - humidity
COLOR_PRESS    = '#6A1B9A'    # Purple - pressure
COLOR_DEW      = '#00838F'    # Teal - dew point
COLOR_ACCENT   = '#333333'    # Dark for titles

OUTPUTS_DIR = r'D:\Github\Projek_Rainfall\Model_Imputasi\outputs\plots'

print("[OK] Libraries imported. Standard light theme activated.")

DATA_PATH = r'D:\Github\Projek_Rainfall\Analisis_Meteorologi\cache_data\id-05_imputed.csv'

df = pd.read_csv(DATA_PATH)
df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert('Asia/Jakarta').dt.tz_localize(None)
df = df.set_index('datetime').sort_index()

# Rename dew ? dewpoint for consistency
if 'dew' in df.columns and 'dewpoint' not in df.columns:
    df = df.rename(columns={'dew': 'dewpoint'})

# ??? Overview ????????????????????????????????????????????????
print(f"?  Periode  : {df.index.min()} -- {df.index.max()}")
print(f"?  Total    : {len(df):,} baris data (resolusi 1 menit)")
print(f"\n{'='*55}")
print("data_source breakdown:")
print(df['data_source'].value_counts().to_string())
print(f"{'='*55}")
print("\n?  Statistik Deskriptif:")
print(df[['temperature', 'humidity', 'pressure', 'dewpoint']].describe().round(3).to_string())


src_counts = df['data_source'].value_counts()
labels_map = {
    'original':            'Observasi Asli',
    'linear_interpolated': 'Interpolasi Linear',
    'lstm_imputed':        'Imputasi LSTM BiLSTM',
}
labels  = [labels_map.get(k, k) for k in src_counts.index]
colors  = [COLOR_ORIGINAL, COLOR_LINEAR, COLOR_LSTM][:len(src_counts)]
explode = [0.05] * len(src_counts)

fig, ax = plt.subplots(figsize=(7, 5))
fig.patch.set_facecolor('#0F1117')
wedges, texts, autotexts = ax.pie(
    src_counts.values, labels=labels, colors=colors,
    autopct='%1.1f%%', startangle=140, explode=explode,
    pctdistance=0.82, wedgeprops=dict(width=0.55, edgecolor='#0F1117', linewidth=2)
)
for at in autotexts:
    at.set_fontsize(10); at.set_color('#0F1117'); at.set_fontweight('bold')
for t in texts:
    t.set_color('#C8D3F5'); t.set_fontsize(10)

ax.set_title('Komposisi Sumber Data -- Stasiun ID-05', pad=18, color='#FFCB6B', fontsize=13, fontweight='bold')
centre_circle = plt.Circle((0,0), 0.45, fc='#0F1117')
ax.add_artist(centre_circle)
ax.text(0, 0, f'{len(df):,}\nSampel', ha='center', va='center',
        fontsize=12, color='#C8D3F5', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUTS_DIR}/id05_data_source_donut.png')
plt.show()
print(f"\n  Original: {src_counts.get('original', 0):,} ({src_counts.get('original', 0)/len(df)*100:.1f}%)")
print(f"  Linear  : {src_counts.get('linear_interpolated', 0):,} ({src_counts.get('linear_interpolated', 0)/len(df)*100:.2f}%)")
print(f"  LSTM    : {src_counts.get('lstm_imputed', 0):,} ({src_counts.get('lstm_imputed', 0)/len(df)*100:.2f}%)")


vars_info = [
    ('temperature', 'Suhu (degC)',           COLOR_TEMP,  (20, 36)),
    ('humidity',    'Kelembapan (%)',       COLOR_HUM,   (40, 105)),
    ('pressure',    'Tekanan Udara (hPa)', COLOR_PRESS, (1000, 1020)),
    ('dewpoint',    'Titik Embun (degC)',    COLOR_DEW,   (15, 32)),
]

src_colors = {
    'original':            COLOR_ORIGINAL,
    'linear_interpolated': COLOR_LINEAR,
    'lstm_imputed':        COLOR_LSTM,
}

# Daily resample for readability
df_h = df.resample('1h').mean(numeric_only=True)
src_h = df['data_source'].resample('1h').agg(lambda x: x.mode().iloc[0] if len(x) > 0 else 'original')

fig = plt.figure(figsize=(18, 14))
gs = gridspec.GridSpec(4, 1, figure=fig, hspace=0.08)

for idx, (var, ylabel, color, ylim) in enumerate(vars_info):
    ax = fig.add_subplot(gs[idx])

    # Plot each source segment with distinct color
    for src, sc in src_colors.items():
        mask = src_h == src
        ax.plot(df_h.index[mask], df_h[var][mask],
                '.', color=sc, alpha=0.6, markersize=3, rasterized=True)

    # Overlay smooth line
    ax.plot(df_h.index, df_h[var].interpolate(), color=color,
            linewidth=1.1, alpha=0.85, rasterized=True)

    ax.set_ylabel(ylabel, fontsize=10, labelpad=8)
    ax.set_ylim(ylim)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, axis='both', which='major', alpha=0.4)
    ax.grid(True, axis='y',    which='minor', alpha=0.15)
    ax.tick_params(labelbottom=(idx == 3))

    # Shade LSTM imputed windows
    lstm_mask = src_h == 'lstm_imputed'
    starts = df_h.index[lstm_mask & (~lstm_mask.shift(1, fill_value=False))]
    ends   = df_h.index[lstm_mask & (~lstm_mask.shift(-1, fill_value=False))]
    for s, e in zip(starts, ends):
        ax.axvspan(s, e, color=COLOR_LSTM, alpha=0.12)

# Legend (once)
patches = [
    mpatches.Patch(color=COLOR_ORIGINAL, label='Observasi Asli'),
    mpatches.Patch(color=COLOR_LINEAR,   label='Interpolasi Linear'),
    mpatches.Patch(color=COLOR_LSTM,     label='Imputasi LSTM'),
]
axes = [fig.add_subplot(gs[i]) for i in range(4)]
fig.legend(handles=patches, loc='upper right', ncol=3,
           framealpha=0.9, fontsize=10, bbox_to_anchor=(0.99, 0.98))

# X-axis formatter on bottom panel
ax_bottom = fig.axes[-1]
ax_bottom.xaxis.set_major_formatter(mdates.DateFormatter('%d %b\n%H:%M'))
ax_bottom.xaxis.set_major_locator(mdates.DayLocator(interval=3))
ax_bottom.set_xlabel('Tanggal / Waktu (WIB)', fontsize=11)

fig.suptitle('Time-Series Meteorologi -- Stasiun ID-05\n(30 Hari Terakhir | Resolusi 1-Jam)',
             fontsize=15, fontweight='bold', color=COLOR_ACCENT, y=1.005)
plt.savefig(f'{OUTPUTS_DIR}/id05_timeseries_overview.png')
plt.show()


df_orig  = df[df['data_source'] == 'original'].copy()
df_lstm  = df[df['data_source'] == 'lstm_imputed'].copy()
df_lin   = df[df['data_source'] == 'linear_interpolated'].copy()

hourly_all  = df.groupby(df.index.hour)
hourly_orig = df_orig.groupby(df_orig.index.hour) if len(df_orig) > 0 else None
hourly_lstm = df_lstm.groupby(df_lstm.index.hour) if len(df_lstm) > 0 else None

hours = range(24)

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.patch.set_facecolor('#0F1117')

panels = [
    (axes[0, 0], 'temperature', 'Suhu (degC)',           COLOR_TEMP),
    (axes[0, 1], 'humidity',    'Kelembapan (%)',       COLOR_HUM),
    (axes[1, 0], 'pressure',    'Tekanan Udara (hPa)', COLOR_PRESS),
    (axes[1, 1], 'dewpoint',    'Titik Embun (degC)',    COLOR_DEW),
]

for ax, var, label, color in panels:
    mean_all  = [hourly_all[var].mean().get(h, np.nan) for h in hours]
    std_all   = [hourly_all[var].std().get(h, np.nan) for h in hours]

    ax.fill_between(hours,
                    np.array(mean_all) - np.array(std_all),
                    np.array(mean_all) + np.array(std_all),
                    alpha=0.2, color=color, label='?1 Std Dev')
    ax.plot(hours, mean_all, color=color, linewidth=2.5, label='Rata-rata (Semua)')

    if hourly_lstm is not None and len(df_lstm) > 0:
        mean_lstm = [hourly_lstm[var].mean().get(h, np.nan) for h in hours]
        ax.plot(hours, mean_lstm, '--', color=COLOR_LSTM, linewidth=1.5,
                alpha=0.85, label='Rata-rata (LSTM)')

    ax.set_xlabel('Jam (WIB)', fontsize=10)
    ax.set_ylabel(label, fontsize=10)
    ax.set_title(f'Pola Diurnal -- {label}', fontsize=11)
    ax.set_xlim(0, 23)
    ax.set_xticks(range(0, 24, 3))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.grid(True, which='major', alpha=0.4)
    ax.grid(True, which='minor', alpha=0.15)
    ax.legend(fontsize=9)

fig.suptitle('Pola Diurnal Meteorologi -- Stasiun ID-05',
             fontsize=14, fontweight='bold', color=COLOR_ACCENT)
plt.tight_layout()
plt.savefig(f'{OUTPUTS_DIR}/id05_diurnal_cycle.png')
plt.show()


from scipy.stats import gaussian_kde

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.patch.set_facecolor('#0F1117')

var_info2 = [
    (axes[0, 0], 'temperature', 'Suhu (degC)'),
    (axes[0, 1], 'humidity',    'Kelembapan (%)'),
    (axes[1, 0], 'pressure',    'Tekanan Udara (hPa)'),
    (axes[1, 1], 'dewpoint',    'Titik Embun (degC)'),
]

src_groups = {
    'Observasi Asli':        (df[df['data_source']=='original'],             COLOR_ORIGINAL),
    'Interpolasi Linear':    (df[df['data_source']=='linear_interpolated'],  COLOR_LINEAR),
    'Imputasi LSTM':         (df[df['data_source']=='lstm_imputed'],          COLOR_LSTM),
}

for ax, var, xlabel in var_info2:
    for label, (subset, color) in src_groups.items():
        vals = subset[var].dropna()
        if len(vals) < 5:
            continue
        ax.hist(vals, bins=60, density=True, alpha=0.25, color=color)
        kde = gaussian_kde(vals, bw_method='scott')
        x   = np.linspace(vals.min(), vals.max(), 300)
        ax.plot(x, kde(x), color=color, linewidth=2, label=f'{label} (n={len(vals):,})')

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel('Densitas', fontsize=10)
    ax.set_title(f'Distribusi -- {xlabel}', fontsize=11)
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=9)

fig.suptitle('Perbandingan Distribusi Data: Original vs Imputasi -- Stasiun ID-05',
             fontsize=13, fontweight='bold', color=COLOR_ACCENT)
plt.tight_layout()
plt.savefig(f'{OUTPUTS_DIR}/id05_distribution_comparison.png')
plt.show()


fig, axes = plt.subplots(2, 2, figsize=(18, 10))
fig.patch.set_facecolor('#0F1117')

pivot_vars = [
    (axes[0, 0], 'temperature', 'Suhu (degC)',           'YlOrRd'),
    (axes[0, 1], 'humidity',    'Kelembapan (%)',       'RdYlBu_r'),
    (axes[1, 0], 'pressure',    'Tekanan Udara (hPa)', 'viridis'),
    (axes[1, 1], 'dewpoint',    'Titik Embun (degC)',    'cool'),
]

for ax, var, label, cmap in pivot_vars:
    pivot = df.pivot_table(index=df.index.hour, columns=df.index.date, values=var, aggfunc='mean')
    im = ax.imshow(pivot.values, aspect='auto', cmap=cmap,
                   interpolation='nearest', vmin=pivot.values[~np.isnan(pivot.values)].min(),
                   vmax=pivot.values[~np.isnan(pivot.values)].max())
    ax.set_yticks(range(0, 24, 3))
    ax.set_yticklabels([f'{h:02d}:00' for h in range(0, 24, 3)], fontsize=8)

    n_days = pivot.shape[1]
    xtick_step = max(1, n_days // 10)
    ax.set_xticks(range(0, n_days, xtick_step))
    col_dates = [str(d) for d in pivot.columns]
    ax.set_xticklabels(col_dates[::xtick_step], rotation=35, ha='right', fontsize=8)

    ax.set_ylabel('Jam (WIB)', fontsize=9)
    ax.set_xlabel('Tanggal', fontsize=9)
    ax.set_title(f'Heatmap -- {label}', fontsize=11)
    plt.colorbar(im, ax=ax, pad=0.02, shrink=0.9, label=label)

fig.suptitle('Heatmap Variabilitas Harian Meteorologi -- Stasiun ID-05',
             fontsize=13, fontweight='bold', color=COLOR_ACCENT)
plt.tight_layout()
plt.savefig(f'{OUTPUTS_DIR}/id05_heatmap_daily.png')
plt.show()


lstm_mask = df['data_source'] == 'lstm_imputed'
changes   = lstm_mask.astype(int).diff().fillna(0)
start_idx = df.index[changes == 1].tolist()
end_idx   = df.index[changes == -1].tolist()

# Handle edge cases
if lstm_mask.iloc[0]:   start_idx.insert(0, df.index[0])
if lstm_mask.iloc[-1]:  end_idx.append(df.index[-1])

n_gaps = min(len(start_idx), 9)  # Show max 9 gaps
print(f"  Ditemukan {len(start_idx)} segmen imputasi LSTM. Menampilkan {n_gaps} pertama...")

if n_gaps == 0:
    print("  Tidak ada segmen LSTM ditemukan -- semua data mungkin sudah original.")
else:
    ncols = 3
    nrows = (n_gaps + ncols - 1) // ncols
    CONTEXT = pd.Timedelta('2H')

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4 * nrows))
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    fig.patch.set_facecolor('#0F1117')

    for i in range(n_gaps):
        ax = axes[i]
        s, e = start_idx[i], end_idx[i] if i < len(end_idx) else df.index[-1]
        ws   = max(df.index[0],  s - CONTEXT)
        we   = min(df.index[-1], e + CONTEXT)
        seg  = df.loc[ws:we]

        for src, sc in src_colors.items():
            m = seg['data_source'] == src
            ax.plot(seg.index[m], seg['temperature'][m], '.', color=sc, markersize=4, alpha=0.8)

        gap_start, gap_end = seg.index[seg['data_source']=='lstm_imputed'][[0, -1]]
        ax.axvspan(gap_start, gap_end, color=COLOR_LSTM, alpha=0.15, zorder=0)

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m %H:%M'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=7)
        ax.set_title(f'Gap #{i+1}: {s.strftime("%d %b %H:%M")} - {e.strftime("%H:%M")}',
                     fontsize=9, pad=4)
        ax.set_ylabel('Suhu (degC)', fontsize=8)
        ax.grid(True, alpha=0.3)
        gap_len = (e - s).seconds // 60
        ax.text(0.98, 0.97, f'  {gap_len} mnt', transform=ax.transAxes, ha='right',
                va='top', fontsize=8, color=COLOR_LSTM)

    # Hide unused axes
    for j in range(n_gaps, len(axes)):
        axes[j].set_visible(False)

    legend_patches = [
        mpatches.Patch(color=COLOR_ORIGINAL, label='Original'),
        mpatches.Patch(color=COLOR_LINEAR,   label='Lin. Interp.'),
        mpatches.Patch(color=COLOR_LSTM,     label='LSTM Imputed'),
    ]
    fig.legend(handles=legend_patches, loc='lower right', ncol=3, fontsize=9)
    fig.suptitle('Zoom-In Segmen Imputasi LSTM -- Suhu Stasiun ID-05',
                 fontsize=13, fontweight='bold', color=COLOR_ACCENT)
    plt.tight_layout()
    plt.savefig(f'{OUTPUTS_DIR}/id05_lstm_gaps_zoomin.png')
    plt.show()


import matplotlib.colors as mcolors

TARGET_VARS = ['temperature', 'humidity', 'pressure', 'dewpoint']
corr = df[TARGET_VARS].corr()

fig, ax = plt.subplots(figsize=(7, 6))
fig.patch.set_facecolor('#0F1117')

cmap = plt.cm.coolwarm
im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect='equal')

labels_var = ['Suhu', 'Kelembapan', 'Tekanan', 'Titik Embun']
ax.set_xticks(range(len(labels_var))); ax.set_xticklabels(labels_var, rotation=30, ha='right')
ax.set_yticks(range(len(labels_var))); ax.set_yticklabels(labels_var)

for i in range(len(labels_var)):
    for j in range(len(labels_var)):
        v = corr.values[i, j]
        tc = 'white' if abs(v) > 0.5 else '#C8D3F5'
        ax.text(j, i, f'{v:.2f}', ha='center', va='center', color=tc, fontsize=12, fontweight='bold')

plt.colorbar(im, ax=ax, shrink=0.85, label='Koefisien Korelasi')
ax.set_title('Matriks Korelasi Variabel Meteorologi -- ID-05',
             fontsize=12, fontweight='bold', color=COLOR_ACCENT, pad=12)
plt.tight_layout()
plt.savefig(f'{OUTPUTS_DIR}/id05_correlation_matrix.png')
plt.show()


ROLL = 360  # 6 jam @ 1-menit resolusi

fig, axes = plt.subplots(4, 1, figsize=(18, 14), sharex=True)
fig.patch.set_facecolor('#0F1117')

panels_roll = [
    (axes[0], 'temperature', 'Suhu (degC)',           COLOR_TEMP),
    (axes[1], 'humidity',    'Kelembapan (%)',       COLOR_HUM),
    (axes[2], 'pressure',    'Tekanan Udara (hPa)', COLOR_PRESS),
    (axes[3], 'dewpoint',    'Titik Embun (degC)',    COLOR_DEW),
]

for ax, var, ylabel, color in panels_roll:
    raw   = df[var]
    rmean = raw.rolling(ROLL, center=True, min_periods=60).mean()
    rstd  = raw.rolling(ROLL, center=True, min_periods=60).std()

    ax.plot(df.index, raw,   color=color,   linewidth=0.5, alpha=0.35, rasterized=True)
    ax.plot(df.index, rmean, color=color,   linewidth=2.0, alpha=0.95, label='Rolling Mean 6-Jam')
    ax.fill_between(df.index, rmean - rstd, rmean + rstd,
                    color=color, alpha=0.18, label='?1 Std Dev')

    # Highlight LSTM windows
    for i in range(len(start_idx)):
        s = start_idx[i]
        e = end_idx[i] if i < len(end_idx) else df.index[-1]
        ax.axvspan(s, e, color=COLOR_LSTM, alpha=0.2, zorder=0)

    ax.set_ylabel(ylabel, fontsize=10)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.35)

axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%d %b\n%H:%M'))
axes[-1].xaxis.set_major_locator(mdates.DayLocator(interval=3))
axes[-1].set_xlabel('Tanggal / Waktu (WIB)', fontsize=11)

fig.suptitle('Rolling Mean & Std Dev (6 Jam) -- Stasiun ID-05',
             fontsize=14, fontweight='bold', color=COLOR_ACCENT)
plt.tight_layout()
plt.savefig(f'{OUTPUTS_DIR}/id05_rolling_stats.png')
plt.show()


TARGET_VARS = ['temperature', 'humidity', 'pressure', 'dewpoint']
SRC_LABELS  = {
    'original':            'Observasi Asli',
    'linear_interpolated': 'Interpolasi Linear',
    'lstm_imputed':        'Imputasi LSTM',
}

rows = []
for src, label in SRC_LABELS.items():
    subset = df[df['data_source'] == src]
    if len(subset) == 0:
        continue
    for var in TARGET_VARS:
        vals = subset[var].dropna()
        if len(vals) == 0:
            continue
        rows.append({
            'Sumber':    label,
            'Variabel':  var,
            'Count':     len(vals),
            'Mean':      round(vals.mean(), 4),
            'Std':       round(vals.std(), 4),
            'Min':       round(vals.min(), 4),
            'Max':       round(vals.max(), 4),
            'Median':    round(vals.median(), 4),
        })

summary_df = pd.DataFrame(rows)
print("=" * 80)
print("RINGKASAN STATISTIK PER SUMBER DATA -- STASIUN ID-05")
print("=" * 80)
print(summary_df.to_string(index=False))

