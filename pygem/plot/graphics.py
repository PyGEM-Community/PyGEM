"""
Python Glacier Evolution Model (PyGEM)

copyright © 2025 Brandon Tober <btober@cmu.edu>, David Rounce <drounce@cmu.edu>

Distributed under the MIT license

Graphics module with various plotting tools
"""

import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import torch
from scipy.stats import binned_statistic, linregress

from pygem.utils.stats import effective_n


def plot_modeloutput_section(
    model=None,
    ax=None,
    title='',
    lnlabel=None,
    legendon=True,
    lgdkwargs={'loc': 'upper right', 'fancybox': False, 'borderaxespad': 0, 'handlelength': 1},
    **kwargs,
):
    """Plots the result of the model output along the flowline.
    A paired down version of OGGMs graphics.plot_modeloutput_section()

    Parameters
    ----------
    model: obj
        either a FlowlineModel or a list of model flowlines.
    fig
    title
    """

    try:
        fls = model.fls
    except AttributeError:
        fls = model

    if ax is None:
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_axes([0.07, 0.08, 0.7, 0.84])
    else:
        fig = plt.gcf()
    # get n lines plotted on figure
    nlines = len(plt.gca().get_lines())

    height = np.array([])
    bed = np.array([])
    for cls in fls:
        height = np.concatenate((height, cls.surface_h))
        bed = np.concatenate((bed, cls.bed_h))
    ylim = [bed.min(), height.max()]

    # plot Centerlines
    cls = fls[-1]
    x = np.arange(cls.nx) * cls.dx * cls.map_dx

    if nlines == 0:
        if getattr(model, 'do_calving', False):
            ax.hlines(model.water_level, x[0], x[-1], linestyles=':', label='Water level', color='C0')
        # Plot the bed
        ax.plot(x, cls.bed_h, color='k', linewidth=2.5, label='Bed (Parab.)')

    # Plot glacier
    t1 = cls.thick[:-2]
    t2 = cls.thick[1:-1]
    t3 = cls.thick[2:]
    pnan = ((t1 == 0) & (t2 == 0)) & ((t2 == 0) & (t3 == 0))
    cls.surface_h[np.where(pnan)[0] + 1] = np.nan

    if 'srfcolor' in kwargs.keys():
        srfcolor = kwargs['srfcolor']
    else:
        srfcolor = '#003399'

    if 'srfls' in kwargs.keys():
        srfls = kwargs['srfls']
    else:
        srfls = '-'

    ax.plot(x, cls.surface_h, color=srfcolor, linewidth=2, ls=srfls, label=lnlabel)

    # Plot tributaries
    for i, inflow in zip(cls.inflow_indices, cls.inflows):
        if inflow.thick[-1] > 0:
            ax.plot(
                x[i],
                cls.surface_h[i],
                's',
                markerfacecolor='#993399',
                markeredgecolor='k',
                label='Tributary (active)',
            )
        else:
            ax.plot(
                x[i],
                cls.surface_h[i],
                's',
                markerfacecolor='w',
                markeredgecolor='k',
                label='Tributary (inactive)',
            )

    ax.set_ylim(ylim)
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xlabel('Distance along flowline (m)')
    ax.set_ylabel('Altitude (m)')
    if legendon:
        ax.legend(**lgdkwargs)
    # Title
    ax.set_title(title, loc='left')


def plot_mcmc_chain(
    m_primes, m_chain, pred_primes, pred_chain, obs, ar, title, ms=1, fontsize=8, show=False, fpath=None
):
    # Plot the trace of the parameters
    nparams = m_primes.shape[1]
    npreds = len(pred_chain.keys())
    N = nparams + npreds + 1
    fig, axes = plt.subplots(N, 1, figsize=(6, N * 1), sharex=True)
    # convert torch objects to numpy
    m_chain = m_chain.detach().numpy()
    m_primes = m_primes.detach().numpy()

    # get n_eff
    neff = [effective_n(arr) for arr in m_chain.T]
    # instantiate list to hold legend objs
    legs = []

    # axes[0] will always be tbias
    axes[0].plot(
        [],
        [],
        label=f'median={np.median(m_chain[:, 0]):.3f}\niqr={np.subtract(*np.percentile(m_chain[:, 0], [75, 25])):.3f}',
    )
    l0 = axes[0].legend(loc='upper right', handlelength=0, borderaxespad=0, fontsize=fontsize)
    legs.append(l0)
    axes[0].plot(m_primes[:, 0], '.', ms=ms, label='proposed', c='tab:blue')
    axes[0].plot(m_chain[:, 0], '.', ms=ms, label='accepted', c='tab:orange')
    hands, ls = axes[0].get_legend_handles_labels()

    # axes[0].add_artist(leg)
    axes[0].set_ylabel(r'$T_{bias}$', fontsize=fontsize)

    # axes[1] will always be kp
    axes[1].plot(m_primes[:, 1], '.', ms=ms, c='tab:blue')
    axes[1].plot(m_chain[:, 1], '.', ms=ms, c='tab:orange')
    axes[1].plot(
        [],
        [],
        label=f'median={np.median(m_chain[:, 1]):.3f}\niqr={np.subtract(*np.percentile(m_chain[:, 1], [75, 25])):.3f}',
    )
    l1 = axes[1].legend(loc='upper right', handlelength=0, borderaxespad=0, fontsize=fontsize)
    legs.append(l1)
    axes[1].set_ylabel(r'$K_p$', fontsize=fontsize)

    # axes[2] will always be ddfsnow
    axes[2].plot(m_primes[:, 2], '.', ms=ms, c='tab:blue')
    axes[2].plot(m_chain[:, 2], '.', ms=ms, c='tab:orange')
    axes[2].plot(
        [],
        [],
        label=f'median={np.median(m_chain[:, 2]):.3f}\niqr={np.subtract(*np.percentile(m_chain[:, 2], [75, 25])):.3f}',
    )
    l2 = axes[2].legend(loc='upper right', handlelength=0, borderaxespad=0, fontsize=fontsize)
    legs.append(l2)
    axes[2].set_ylabel(r'$fsnow$', fontsize=fontsize)

    if nparams > 3:
        # axes[3] will be rho_ablation if more than 3 model params
        m_chain[:, 3] = m_chain[:, 3]
        m_primes[:, 3] = m_primes[:, 3]
        axes[3].plot(m_primes[:, 3], '.', ms=ms, c='tab:blue')
        axes[3].plot(m_chain[:, 3], '.', ms=ms, c='tab:orange')
        axes[3].plot(
            [],
            [],
            label=f'median={np.median(m_chain[:, 3]):.3f}\niqr={np.subtract(*np.percentile(m_chain[:, 3], [75, 25])):.3f}',
        )
        l3 = axes[3].legend(loc='upper right', handlelength=0, borderaxespad=0, fontsize=fontsize)
        legs.append(l3)
        axes[3].set_ylabel(r'$\rho_{abl}$', fontsize=fontsize)

        # axes[4] will be rho_accumulation if more than 3 model params
        m_chain[:, 4] = m_chain[:, 4]
        m_primes[:, 4] = m_primes[:, 4]
        axes[4].plot(m_primes[:, 4], '.', ms=ms, c='tab:blue')
        axes[4].plot(m_chain[:, 4], '.', ms=ms, c='tab:orange')
        axes[4].plot(
            [],
            [],
            label=f'median={np.median(m_chain[:, 4]):.3f}\niqr={np.subtract(*np.percentile(m_chain[:, 4], [75, 25])):.3f}',
        )
        l4 = axes[4].legend(loc='upper right', handlelength=0, borderaxespad=0, fontsize=fontsize)
        legs.append(l4)
        axes[4].set_ylabel(r'$\rho_{acc}$', fontsize=fontsize)

    # plot predictions
    if 'glacierwide_mb_mwea' in pred_primes.keys():
        mb_obs = obs['glacierwide_mb_mwea']
        axes[nparams].fill_between(
            np.arange(len(ar)),
            mb_obs[0] - (2 * mb_obs[1]),
            mb_obs[0] + (2 * mb_obs[1]),
            color='grey',
            alpha=0.3,
        )
        axes[nparams].fill_between(
            np.arange(len(ar)),
            mb_obs[0] - mb_obs[1],
            mb_obs[0] + mb_obs[1],
            color='grey',
            alpha=0.3,
        )

        mb_primes = torch.stack(pred_primes['glacierwide_mb_mwea']).numpy()
        mb_chain = torch.stack(pred_chain['glacierwide_mb_mwea']).numpy()
        axes[nparams].plot(mb_primes, '.', ms=ms, c='tab:blue')
        axes[nparams].plot(mb_chain, '.', ms=ms, c='tab:orange')
        axes[nparams].plot(
            [],
            [],
            label=f'median={np.median(mb_chain):.3f}\niqr={np.subtract(*np.percentile(mb_chain, [75, 25])):.3f}',
        )
        ln2 = axes[nparams].legend(loc='upper right', handlelength=0, borderaxespad=0, fontsize=fontsize)
        legs.append(ln2)
        axes[nparams].set_ylabel(r'$\dot{{b}}$', fontsize=fontsize)
        nparams += 1

    # plot MAE for all other prediction keys
    for key in pred_primes.keys():
        if key == 'glacierwide_mb_mwea':
            continue

        # stack predictions first (shape: n_steps x ... x ...) - may end up being 2d or 3d
        pred_primes = torch.stack(pred_primes[key]).numpy()
        pred_chain = torch.stack(pred_chain[key]).numpy()

        # flatten all axes except the first (n_steps) -> 2D array (n_steps, M)
        pred_primes_flat = pred_primes.reshape(pred_primes.shape[0], -1)
        pred_chain_flat = pred_chain.reshape(pred_chain.shape[0], -1)

        # make obs array broadcastable (flatten if needed)
        obs_vals_flat = np.ravel(np.array(obs[key][0]))

        # compute mean residual per step
        mean_resid_primes = np.nanmean(pred_primes_flat - obs_vals_flat, axis=1)
        mean_resid_chain = np.nanmean(pred_chain_flat - obs_vals_flat, axis=1)

        axes[nparams].plot(mean_resid_primes, '.', ms=ms, c='tab:blue')
        axes[nparams].plot(mean_resid_chain, '.', ms=ms, c='tab:orange')

        if key == 'elev_change_1d':
            axes[nparams].set_ylabel(r'$\overline{\hat{dh} - dh}$', fontsize=fontsize)
        else:
            axes[nparams].set_ylabel(r'$\overline{\mathrm{pred} - \mathrm{obs}}$', fontsize=fontsize)
        legs.append(None)
        nparams += 1

    # axes[-1] will always be acceptance rate
    axes[-1].plot(ar, 'tab:orange', lw=1)
    axes[-1].plot(
        np.convolve(ar, np.ones(100) / 100, mode='valid'),
        'k',
        label='moving avg.',
        lw=1,
    )
    ln1 = axes[-1].legend(loc='upper left', handlelength=0.5, borderaxespad=0, fontsize=fontsize)
    legs.append(ln1)
    axes[-1].set_ylabel(r'$AR$', fontsize=fontsize)

    for i, ax in enumerate(axes):
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.tick_params(axis='both', direction='inout')
        ax.yaxis.set_label_coords(-0.1, 0.5)
        if i > m_primes.shape[1] - 1:
            continue
        ax.plot([], [], label=f'n_eff={neff[i]}')
        hands, ls = ax.get_legend_handles_labels()
        if i == 0:
            ax.legend(
                handles=[hands[1], hands[2], hands[3]],
                labels=[ls[1], ls[2], ls[3]],
                loc='upper left',
                borderaxespad=0,
                handlelength=0,
                fontsize=fontsize,
            )
        else:
            ax.legend(
                handles=[hands[-1]],
                labels=[ls[-1]],
                loc='upper left',
                borderaxespad=0,
                handlelength=0,
                fontsize=fontsize,
            )
        if legs[i] is not None:
            ax.add_artist(legs[i])

    axes[0].set_xlim([0, m_chain.shape[0]])
    axes[0].set_title(title, fontsize=fontsize)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1, wspace=0)
    if fpath:
        fig.savefig(fpath, dpi=400)
    if show:
        plt.show(block=True)  # wait until the figure is closed
    plt.close(fig)


def plot_resid_histogram(obs, preds, title, fontsize=8, show=False, fpath=None):
    # Plot the trace of the parameters
    fig, axes = plt.subplots(1, 1, figsize=(3, 2))
    # subtract obs from preds to get residuals
    diffs = np.concatenate([pred.flatten() - obs[0].flatten().numpy() for pred in preds])
    # mask nans to avoid error in np.histogram()
    diffs = diffs[~np.isnan(diffs)]
    # Calculate histogram counts and bin edges
    counts, bin_edges = np.histogram(diffs, bins=20)
    pct = counts / counts.sum() * 100
    bin_width = bin_edges[1] - bin_edges[0]
    axes.bar(
        bin_edges[:-1],
        pct,
        width=bin_width,
        edgecolor='black',
        color='gray',
        align='edge',
    )
    axes.set_xlabel('residuals (pred - obs)', fontsize=fontsize)
    axes.set_ylabel('count (%)', fontsize=fontsize)
    axes.set_title(title, fontsize=fontsize)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1, wspace=0)
    if fpath:
        fig.savefig(fpath, dpi=400)
    if show:
        plt.show(block=True)  # wait until the figure is closed
    plt.close(fig)


def plot_mcmc_elev_change_1d(
    preds, fls, obs, ela, title, fontsize=8, rate=True, uniform_area=True, show=False, fpath=None
):
    bin_z = np.array(obs['bin_centers'])
    bin_edges = np.array(obs['bin_edges'])

    # get initial thickness and surface area
    initial_area = fls[0].widths_m * fls[0].dx_meter
    initial_thickness = getattr(fls[0], 'thick', None)
    initial_surface_h = getattr(fls[0], 'surface_h', None)
    # sort initial surface height
    sorting = np.argsort(initial_surface_h)
    initial_surface_h = initial_surface_h[sorting]
    initial_area = initial_area[sorting]
    initial_thickness = initial_thickness[sorting]
    # get first and last non-zero thickness indices
    first, last = np.nonzero(initial_thickness)[0][[0, -1]]
    # rebin surfce area
    initial_area = binned_statistic(x=initial_surface_h, values=initial_area, statistic=np.nanmean, bins=bin_edges)[0]
    # use reference dataset bin area if available
    if 'bin_area' in obs:
        initial_area = obs['bin_area']

    if uniform_area:
        xvals = np.nancumsum(initial_area) * 1e-6
    else:
        xvals = bin_z

    # get date time spans
    labels = []
    nyrs = []
    for start, end in obs['dates']:
        labels.append(f'{start[:-2].replace("-", "")}:{end[:-3].replace("-", "")}')
        start_dt = datetime.strptime(start, '%Y-%m-%d')
        end_dt = datetime.strptime(end, '%Y-%m-%d')
        nyrs.append((end_dt - start_dt).days / 365.25)
    if not rate:
        nyrs[:] = 1
        ylbl = 'Elevation change (m)'
    else:
        ylbl = r'Elevation change (m yr$^{-1}$)'

    # instantiate subplots
    fig, ax = plt.subplots(
        nrows=len(labels),
        ncols=1,
        figsize=(5, len(labels) * 2),
        gridspec_kw={'hspace': 0.075},
        sharex=True,
        sharey=rate,
    )

    # Transform functions
    def cum_area_to_elev(x):
        return np.interp(x, xvals, bin_z)

    def elev_to_cum_area(x):
        return np.interp(x, bin_z, xvals)

    if not isinstance(ax, np.ndarray):
        ax = [ax]

    # loop through date spans
    for t in range(len(labels)):
        ax[t].xaxis.set_label_position('top')
        ax[t].xaxis.tick_top()  # move ticks to top
        ax[t].tick_params(axis='x', which='both', top=False)
        ax[t].axhline(y=0, c='grey', lw=0.5)
        preds = np.stack(preds)

        ax[t].fill_between(
            xvals,
            (obs['dh'][:, t] - obs['dh_sigma'][:, t]) / nyrs[t],
            (obs['dh'][:, t] + obs['dh_sigma'][:, t]) / nyrs[t],
            color='k',
            alpha=0.125,
        )
        ax[t].plot(xvals, obs['dh'][:, t] / nyrs[t], 'k-', marker='.', label='Obs.')
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            ax[t].fill_between(
                xvals,
                np.nanpercentile(preds[:, :, t], 5, axis=0) / nyrs[t],
                np.nanpercentile(preds[:, :, t], 95, axis=0) / nyrs[t],
                color='r',
                alpha=0.25,
            )
            ax[t].plot(
                xvals,
                np.nanmedian(preds[:, :, t], axis=0) / nyrs[t],
                'r-',
                marker='.',
                label='Pred.',
            )

        # for r in stack:
        #     axb.plot(bin_z, r, 'r', alpha=.0125)

        # dummy label for timespan
        ax[t].text(
            0.99175,
            0.980,
            labels[t],
            transform=ax[t].transAxes,
            fontsize=8,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(
                facecolor='white',
                edgecolor='black',
                alpha=1,
                boxstyle='square,pad=0.25',
            ),
            zorder=10,
        )

        secaxx = ax[t].secondary_xaxis('bottom', functions=(cum_area_to_elev, elev_to_cum_area))

        if t != len(labels) - 1:
            secaxx.tick_params(axis='x', labelbottom=False)
        else:
            secaxx.set_xlabel('Elevation (m)')

        if t == 0:
            leg = ax[t].legend(
                handlelength=1,
                borderaxespad=0,
                fancybox=False,
                loc='lower right',
                edgecolor='k',
                framealpha=1,
            )
            for legobj in leg.legend_handles:
                legobj.set_linewidth(2.0)
        # Turn off cumulative area ticks and labels
        ax[t].tick_params(axis='x', which='both', top=False, labeltop=False)

    ax[0].set_xlim(list(map(elev_to_cum_area, (initial_surface_h[first], initial_surface_h[last]))))

    for a in ax:
        # plot ela
        a.axvline(x=elev_to_cum_area(ela), c='k', ls=':', lw=1)

    ax[-1].text(
        0.0125,
        0.5,
        ylbl,
        horizontalalignment='left',
        rotation=90,
        verticalalignment='center',
        transform=fig.transFigure,
    )

    ax[0].set_title(title, fontsize=fontsize)
    # Remove overlapping tick labels from secaxx
    fig.canvas.draw()  # Force rendering to get accurate bounding boxes
    labels = secaxx.get_xticklabels()
    renderer = fig.canvas.get_renderer()
    bboxes = [label.get_window_extent(renderer) for label in labels]
    # Only show labels spaced apart by at least `min_spacing` pixels
    min_spacing = 15  # adjust as needed
    last_right = -float('inf')
    for label, bbox in zip(labels, bboxes):
        if bbox.x0 > last_right + min_spacing:
            last_right = bbox.x1
        else:
            label.set_visible(False)
    # save
    if fpath:
        fig.savefig(fpath, dpi=400)
    if show:
        plt.show(block=True)  # wait until the figure is closed
    plt.close(fig)


def plot_mcmc_snowline_1d(
    preds, fls, obs, title, fontsize=8, show=False, fpath=None, **kwargs
):
    # get variables, if not default
    vn = kwargs.get('vn', 'z')
    vn_min = kwargs.get('vn_min', 'z_min')
    vn_max = kwargs.get('vn_max', 'z_max')
    units = kwargs.get('units', '[m a.s.l]')
    inverty = kwargs.get('inverty', False)

    snowline_z = np.array(obs[vn])
    snowline_min = np.array(obs[vn_min])
    snowline_max = np.array(obs[vn_max])
    snowline_date = np.array(obs['date'])
    
    # xvals = snowline_date
    xvals = [datetime.strptime(d, '%Y-%m-%d') for d in snowline_date]

    # instantiate subplots
    fig, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(8.5, 3.5),
        gridspec_kw={'hspace': 0.075},
        sharex=True,
        sharey=False,
    )

    if not isinstance(ax, np.ndarray):
        ax = [ax]

    # set xlabels
    ax[0].set_xlim([xvals[0], xvals[-1]])
    ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    preds = np.stack(preds)

    yerr_min = obs[vn] - obs[vn_min]
    yerr_max = obs[vn_max] - obs[vn]
    ax[0].scatter(xvals, obs[vn], s=6, c='k', alpha=0.8, zorder=10, label='Obs.')
    ax[0].errorbar(
        xvals,
        obs[vn],
        yerr=np.array([yerr_min, yerr_max]),
        fmt='none',
        ecolor='k',
        elinewidth=1,
        capsize=1,
        alpha=0.25,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        yerr_min = np.nanmedian(preds, axis=0) - np.nanpercentile(preds, 5, axis=0)
        yerr_max = np.nanpercentile(preds, 95, axis=0) - np.nanmedian(preds, axis=0)
        ax[0].scatter(xvals, np.nanmedian(preds, axis=0), s=6, c='r', alpha=0.8, zorder=10, label='Pred.')
        ax[0].errorbar(
            xvals,
            np.nanmedian(preds, axis=0),
            yerr=np.array([yerr_min, yerr_max]),
            fmt='none',
            ecolor='r',
            elinewidth=1,
            capsize=1,
            alpha=0.25,
        )

    leg = ax[0].legend(
        handlelength=1,
        borderaxespad=0,
        fancybox=False,
        loc='upper right',
        edgecolor='k',
        framealpha=1,
    )
    for legobj in leg.legend_handles:
        legobj.set_linewidth(2.0)

    ylbl = f'Snowline {units}'
    ax[-1].text(
        0.0125,
        0.5,
        ylbl,
        horizontalalignment='left',
        rotation=90,
        verticalalignment='center',
        transform=fig.transFigure,
    )
    ax[0].set_title(title, fontsize=fontsize)
    if inverty:
        ax[0].invert_yaxis()

    # save
    if fpath:
        fig.savefig(fpath, dpi=250)
    if show:
        plt.show(block=True)  # wait until the figure is closed
    plt.close(fig)


def plot_mcmc_snowline_1v1_1d(
    preds, fls, obs, title, fontsize=8, show=False, fpath=None, **kwargs,
):
    # get variables, if not default
    vn = kwargs.get('vn', 'z')
    vn_min = kwargs.get('vn_min', 'z_min')
    vn_max = kwargs.get('vn_max', 'z_max')
    units = kwargs.get('units', '[m a.s.l]')
    limit_buff = kwargs.get('limit_buff', 50)

    snowline_z = np.array(obs[vn])
    snowline_min = np.array(obs[vn_min])
    snowline_max = np.array(obs[vn_max])
    snowline_date = np.array(obs['date'])

    # instantiate subplots
    fig, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(6, 6),
    )

    if not isinstance(ax, np.ndarray):
        ax = [ax]

    preds = np.stack(preds)  

    # add scatter plots
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        preds_sl = np.nanmedian(preds, axis=0)

        # mask out nan
        mask = ~np.isnan(obs[vn]) & ~np.isnan(preds_sl)
        obs_sl_nonan = obs[vn][mask]
        preds_sl_nonan = preds_sl[mask]

        ax[0].scatter(obs_sl_nonan, preds_sl_nonan, c='k', s=12, alpha=0.5)

    lims = [max(0, min(np.nanmin(obs_sl_nonan)-limit_buff, np.nanmin(preds_sl_nonan)-limit_buff)), 
            max(np.nanmax(obs_sl_nonan)+limit_buff, np.nanmax(preds_sl_nonan)+limit_buff)]

    ax[0].plot(lims, lims, ls='--', c='k', lw=1) # add 1-to-1 line

    # set plot limits
    ax[0].set_title(title, fontsize=fontsize)
    ax[0].set_ylabel(f'Modeled snowline {units}', labelpad=22, size=10, va='center', ha='center')
    ax[0].set_xlabel(f'Observed snowline {units}', labelpad=22, size=10, va='center', ha='center')
    ax[0].set_xlim(lims)
    ax[0].set_ylim(lims)
    ax[0].set_box_aspect(1)

    # correlation info (pearson r)
    if len(preds_sl_nonan) > 1 and len(obs_sl_nonan) > 1:
        slope, intercept, r_value, p_value, std_err = linregress(obs_sl_nonan, preds_sl_nonan)
        obs_count = len(preds_sl_nonan)

        fs = 10
        ax[0].text(0.03, 0.99, f'$r^2 = {r_value**2:.2f}$ (n={obs_count})', 
                   transform=ax[0].transAxes, ha='left', va='top', fontsize=fs)
        ax[0].text(0.03, 0.93, f'bias = {fmt_val(np.mean(obs_sl_nonan-preds_sl_nonan))}', 
                   transform=ax[0].transAxes, ha='left', va='top', fontsize=fs)
        ax[0].text(0.03, 0.88, f'mae = {fmt_val(np.mean(np.abs(obs_sl_nonan-preds_sl_nonan)))}', 
                   transform=ax[0].transAxes, ha='left', va='top', fontsize=fs)

    # save
    fig.tight_layout()
    if fpath:
        fig.savefig(fpath, dpi=250)
    if show:
        plt.show(block=True)  # wait until the figure is closed
    plt.close(fig)

def fmt_val(val):
    abs_val = abs(val)
    if abs_val < 1:
        return f"{val:.2f}"
    elif abs_val < 10:
        return f"{val:.1f}"
    else:
        return f"{val:.0f}"
