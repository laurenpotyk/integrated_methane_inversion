import os
import numpy as np
import xarray as xr
import pandas as pd
import datetime
from shapely.geometry import Polygon
from src.inversion_scripts.utils_gosat import (
    filter_gosat,
    get_strdate,
    check_is_OH_element,
    check_is_BC_element,
)

from src.inversion_scripts.operators.gosat_operator_utilities_fixed import (
    get_gc_lat_lon,
    read_all_geoschem,
    merge_pressure_grids,
    remap,
    remap_sensitivities,
    get_gridcell_list,
    nearest_loc,
    VerticalGrid,
)


def apply_gosat_operator(
    filename,
    n_elements,
    gc_startdate,
    gc_enddate,
    xlim,
    ylim,
    gc_cache,
    build_jacobian,
    period_i,
    config,
    use_water_obs=False,
):
    """
    Apply the GOSAT operator to map GEOS-Chem methane data to GOSAT observation space.

    Arguments:
        filename       [str]        : GOSAT netcdf data file to read
        n_elements     [int]        : Number of state vector elements
        gc_startdate   [datetime64] : First day of inversion period for GEOS-Chem and GOSAT
        gc_enddate     [datetime64] : Last day of inversion period for GEOS-Chem and GOSAT
        xlim           [float]      : Longitude bounds for simulation domain
        ylim           [float]      : Latitude bounds for simulation domain
        gc_cache       [str]        : Path to GEOS-Chem output data
        build_jacobian [bool]       : If True, map GEOS-Chem sensitivities to GOSAT observation space
        period_i       [int]        : Kalman filter period
        config         [dict]       : Configuration dictionary
        use_water_obs  [bool]       : If True, use observations over water

    Returns:
        output         [dict]       : Dictionary with the following fields:
                                       - obs_GC: GEOS-Chem and GOSAT methane data
                                       - GOSAT methane
                                       - GEOS-Chem methane
                                       - GOSAT lat, lon
                                       - GOSAT lat index, lon index
                                       If build_jacobian=True, also include:
                                       - K: Jacobian matrix
    """

    GOSAT = read_gosat(filename)
    sat_ind = filter_gosat(GOSAT, xlim, ylim, gc_startdate, gc_enddate, use_water_obs)

    n_obs = len(sat_ind[0])
    print(f"Number of observations is {n_obs}")

    if build_jacobian:
        jacobian_K = np.zeros([n_obs, n_elements], dtype=np.float32)
        jacobian_K.fill(np.nan)

    all_strdate = []
    date_after_inversion = str(gc_enddate + np.timedelta64(1, "D"))[:10].replace("-", "")
    time_threshold = f"{date_after_inversion}_00"

    for k in range(n_obs):
        iSat = sat_ind[0][k]
        jSat = sat_ind[1][k]
        time = pd.to_datetime(str(GOSAT["time"][iSat, jSat]))
        strdate = get_strdate(time, time_threshold)
        all_strdate.append(strdate)
    all_strdate = list(set(all_strdate))

    all_date_gc = read_all_geoschem(all_strdate, gc_cache, n_elements, config, build_jacobian)

    obs_GC = np.zeros([n_obs, 6], dtype=np.float32)
    obs_GC.fill(np.nan)

    for k in range(n_obs):
        iSat = sat_ind[0][k]
        jSat = sat_ind[1][k]

        p_sat = GOSAT["pressures"][iSat, jSat, :]
        dry_air_subcolumns = GOSAT["dry_air_subcolumns"][iSat, jSat, :]
        apriori = GOSAT["methane_profile_apriori"][iSat, jSat, :]
        avkern = GOSAT["averaging_kernel"][iSat, jSat, :]

        time = pd.to_datetime(str(GOSAT["time"][iSat, jSat]))
        strdate = get_strdate(time, time_threshold)
        GEOSCHEM = all_date_gc[strdate]
        dlon = np.median(np.diff(GEOSCHEM["lon"]))
        dlat = np.median(np.diff(GEOSCHEM["lat"]))

        lat_center = GOSAT["latitude"][iSat, jSat]
        lon_center = GOSAT["longitude"][iSat, jSat]

        iGC = nearest_loc(lon_center, GEOSCHEM["lon"], tolerance=max(dlon, 0.5))
        jGC = nearest_loc(lat_center, GEOSCHEM["lat"], tolerance=max(dlat, 0.5))

        if np.isnan(iGC) or np.isnan(jGC):
            continue

        iGC = int(iGC)
        jGC = int(jGC)

        p_gc = GEOSCHEM["PEDGE"][iGC, jGC, :]
        gc_CH4 = GEOSCHEM["CH4"][iGC, jGC, :]

        vg = VerticalGrid(
            model_conc_at_layers=gc_CH4[None, :, None],
            model_edges=p_gc[None, :],
            satellite_edges=p_sat[None, :],
            interpolate_to_centers_or_edges="edges",
            save_interpolation="false",
            save_dir=f"/tmp/gosat_vg_{period_i}_{k}",
            expand_model_edges=True,
        )
        sat_CH4 = vg.interpolate().squeeze()

        # print("sat_CH4 shape:", sat_CH4.shape)
        # print("apriori shape:", apriori.shape)
        # print("avkern shape:", avkern.shape)
        # print("weights shape:", dry_air_subcolumns.shape)

        virtual_gosat = np.nansum(
            dry_air_subcolumns * (apriori + avkern * (sat_CH4 - apriori))
        )

        obs_GC[k, 0] = GOSAT["methane"][iSat, jSat]
        obs_GC[k, 1] = virtual_gosat
        obs_GC[k, 2] = GOSAT["longitude"][iSat, jSat]
        obs_GC[k, 3] = GOSAT["latitude"][iSat, jSat]
        obs_GC[k, 4] = iSat
        obs_GC[k, 5] = jSat

        if build_jacobian:
            sensi_lonlat = GEOSCHEM["jacobian_ch4"][iGC, jGC, :, :]

            vg_sensi = VerticalGrid(
                model_conc_at_layers=sensi_lonlat,
                model_edges=p_gc,
                satellite_edges=p_sat,
                interpolate_to_centers_or_edges="edges",
                save_interpolation="false",
                save_dir=f"/tmp/gosat_vg_jac_{period_i}_{k}",
                expand_model_edges=True,
            )
            sat_deltaCH4 = vg_sensi.interpolate().squeeze()

            gosat_sensitivity = np.nansum(
                dry_air_subcolumns[:, None] * avkern[:, None] * sat_deltaCH4,
                axis=0,
            )
            jacobian_K[k, :] = gosat_sensitivity

    output = {"obs_GC": obs_GC}

    if build_jacobian:
        output["K"] = jacobian_K

    return output


def read_gosat(filename):
    """
    Read GOSAT data and save important variables to a dictionary.

    Arguments:
        filename [str]: GOSAT netcdf data file to read.

    Returns:
        dat [dict]: Dictionary of important variables from GOSAT:
                            - methane
                            - Latitude
                            - Longitude
                            - QA value
                            - UTC time
                            - Averaging kernel
                            - methane prior profile
                            - Pressure levels
                            - Dry air subcolumns
                            - Pressure edges (vertical pressure profile)
    """

    dat = {}

    try:
        with xr.open_dataset(filename) as gosat_data:
            dat["methane"] = np.expand_dims(gosat_data["xch4"].values, axis=0)
            dat["qa_value"] = np.expand_dims(gosat_data["xch4_quality_flag"].values, axis=0)
            dat["longitude"] = np.expand_dims(gosat_data["longitude"].values, axis=0)
            dat["latitude"] = np.expand_dims(gosat_data["latitude"].values, axis=0)
            dat["time"] = np.expand_dims(gosat_data["time"].values, axis=0)

            dat["averaging_kernel"] = np.expand_dims(
                gosat_data["xch4_averaging_kernel"].values, axis=0
            )

            dat["methane_profile_apriori"] = np.expand_dims(
                gosat_data["ch4_profile_apriori"].values, axis=0
            )
            dat["dry_air_subcolumns"] = np.expand_dims(
                gosat_data["pressure_weight"].values, axis=0
            )

            dat["pressures"] = np.expand_dims(
                gosat_data["pressure_levels"].values, axis=0
            )

    except Exception as e:
        print(f"Error opening {filename}: {e}")
        return None

    return dat
