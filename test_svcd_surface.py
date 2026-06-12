# -*- coding: utf-8 -*-
"""SVCD Surface 生成 — linear vs quadratic 平滑度对比"""
import sys, os, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROJECT_ROOT)

from shared.paths import get_project_root, data_path
import numpy as np, pandas as pd, netCDF4 as nc
from SVCD import SVCD
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = str(get_project_root())
OUTPUT = os.path.join(ROOT, 'test_result', 'SVCD_surface')
os.makedirs(OUTPUT, exist_ok=True)

for day_str in ['2020-01-01', '2020-07-01']:
    print(f'\n=== {day_str} ===')
    monitor_df = pd.read_csv(data_path('test_data/raw/Monitor/2020_DailyPM2.5Monitor.csv'))
    day_df = monitor_df[monitor_df['Date']==day_str].dropna(subset=['Lat','Lon','Conc'])
    ds = nc.Dataset(data_path('test_data/raw/CMAQ/2020_PM25.nc'), 'r')
    lon=ds.variables['lon'][:]; lat=ds.variables['lat'][:]
    from datetime import datetime
    didx = (datetime.strptime(day_str,'%Y-%m-%d')-datetime(2020,1,1)).days
    cg=ds.variables['pred_PM25'][didx]; ds.close()

    cmaqs,glons,glats=[],[],[]
    for _,r in day_df.iterrows():
        d=np.sqrt((lon-r['Lon'])**2+(lat-r['Lat'])**2); idx=np.argmin(d)
        ny,nx=lon.shape; rr,cc=idx//nx,idx%nx
        cmaqs.append(cg[rr,cc]); glons.append(lon[rr,cc]); glats.append(lat[rr,cc])
    day_df['CMAQ']=cmaqs; day_df['CMAQ_Lon']=glons; day_df['CMAQ_Lat']=glats
    train=day_df.dropna(subset=['Lon','Lat','CMAQ','Conc'])
    Xt=train[['CMAQ_Lon','CMAQ_Lat']].values; yt=train['Conc'].values; mt=train['CMAQ'].values

    gf=np.column_stack([lon.flatten(),lat.flatten()]); cf=cg.flatten()
    n_total=len(gf); bs=5000

    surfaces = {}
    for basis in ['linear','quadratic']:
        m=SVCD(basis=basis); m.fit(Xt,yt,mt)
        yp=np.empty(n_total)
        for s in range(0,n_total,bs): yp[s:min(s+bs,n_total)]=m.predict(gf[s:min(s+bs,n_total)],cf[s:min(s+bs,n_total)])
        surfaces[basis]=yp.reshape(cg.shape)

    diff = surfaces['quadratic'] - surfaces['linear']

    # Plot
    fig,axes=plt.subplots(2,2,figsize=(18,14))
    vmin=min(cg.min(), surfaces['linear'].min(), surfaces['quadratic'].min())
    vmax=max(cg.max(), surfaces['linear'].max(), surfaces['quadratic'].max())

    for ax,(title,data,cmap) in zip(axes.flat, [
        ('CMAQ 原始', cg, 'Spectral_r'),
        ('SVCD linear', surfaces['linear'], 'Spectral_r'),
        ('SVCD quadratic', surfaces['quadratic'], 'Spectral_r'),
        (f'quadratic - linear\n[{diff.min():.1f}, {diff.max():.1f}]', diff, 'RdBu_r'),
    ]):
        if ' - ' in title:
            vabs=max(abs(diff.min()),abs(diff.max()))
            im=ax.pcolormesh(lon,lat,data,cmap=cmap,vmin=-vabs,vmax=vabs,shading='auto',rasterized=True)
        else:
            im=ax.pcolormesh(lon,lat,data,cmap=cmap,vmin=vmin,vmax=vmax,shading='auto',rasterized=True)
        plt.colorbar(im,ax=ax,shrink=0.78,label='$\mu$g/m$^3$')
        ax.scatter(day_df['Lon'],day_df['Lat'],c='k',s=1.5,alpha=0.3,edgecolors='none')
        ax.set_title(title,fontsize=13)

    fig.suptitle(f'SVCD Linear vs Quadratic — {day_str} ({len(day_df)} sites)',fontsize=15,fontweight='bold')
    plt.tight_layout()
    fp=os.path.join(OUTPUT,f'SVCD_linear_vs_quadratic_{day_str}.png')
    plt.savefig(fp,dpi=150,bbox_inches='tight'); plt.close()
    print(f'  saved: {fp}')

print(f'\nDone -> {OUTPUT}')
