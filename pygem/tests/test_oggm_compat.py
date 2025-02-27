import numpy as np

from pygem import oggm_compat

do_plot = False


def test_single_flowline_glacier_directory():
    rid = "RGI60-15.03473"
    gdir = oggm_compat.single_flowline_glacier_directory(rid)
    assert gdir.rgi_area_km2 == 61.054

    if do_plot:
        import matplotlib.pyplot as plt
        from oggm import graphics

        f, (ax1, ax2) = plt.subplots(1, 2)
        graphics.plot_googlemap(gdir, ax=ax1)
        graphics.plot_inversion(gdir, ax=ax2)
        plt.show()


def test_get_glacier_zwh():
    rid = "RGI60-15.03473"
    gdir = oggm_compat.single_flowline_glacier_directory(rid)
    df = oggm_compat.get_glacier_zwh(gdir)

    # Ref area km2
    ref_area = gdir.rgi_area_km2
    ref_area_m2 = ref_area * 1e6

    # Check that glacier area is conserved at 0.1%
    np.testing.assert_allclose((df.w * df.dx).sum(), ref_area_m2, rtol=0.001)

    # Check that volume is within VAS at 25%
    vas_vol = 0.034 * ref_area**1.375
    vas_vol_m3 = vas_vol * 1e9
    np.testing.assert_allclose(
        (df.w * df.dx * df.h).sum(), vas_vol_m3, rtol=0.25
    )


def test_random_mb_run():
    rid = "RGI60-15.03473"
    gdir = oggm_compat.single_flowline_glacier_directory(rid, prepro_border=80)

    # This initializes the mass balance model, but does not run it
    mbmod = oggm_compat.RandomLinearMassBalance(
        gdir, seed=1, sigma_ela=300, h_perc=55
    )
    # HERE CAN BE THE LOOP SUCH THAT EVERYTHING IS ALREADY LOADED
    for i in [1, 2, 3, 4]:
        # Change the model parameter
        mbmod.param1 = i
        # Run the mass balance model with fixed geometry
        ts_mb = mbmod.get_specific_mb(years=[2000, 2001, 2002])

    # Run the glacier flowline model with a mass balance model
    from oggm.core.flowline import robust_model_run

    flmodel = robust_model_run(gdir, mb_model=mbmod, ys=0, ye=700)

    # Check that "something" is computed
    import xarray as xr

    ds = xr.open_dataset(gdir.get_filepath("model_diagnostics"))
    assert ds.isel(time=-1).volume_m3 > 0

    if do_plot:
        import matplotlib.pyplot as plt
        from oggm import graphics

        graphics.plot_modeloutput_section(flmodel)
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
        (ds.volume_m3 * 1e-9).plot(ax=ax1)
        ax1.set_ylabel("Glacier volume (km$^{3}$)")
        (ds.area_m2 * 1e-6).plot(ax=ax2)
        ax2.set_ylabel("Glacier area (km$^{2}$)")
        (ds.length_m * 1e3).plot(ax=ax3)
        ax3.set_ylabel("Glacier length (km)")
        plt.tight_layout()
        plt.show()
