import os
import numpy as np
import xarray as xr
import pandas as pd
import datetime
from shapely.geometry import Polygon
from src.inversion_scripts.utils import (
    filter_gosat,
    get_strdate,
    check_is_OH_element,
    check_is_BC_element,
)

from src.inversion_scripts.operators.operator_utilities import (
    get_gc_lat_lon,
    read_all_geoschem,
    merge_pressure_grids,
    remap,
    remap_sensitivities,
    get_gridcell_list,
    nearest_loc,
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

    # Read GOSAT data
    GOSAT = read_gosat(filename)
    sat_ind = filter_gosat(GOSAT, xlim, ylim, gc_startdate, gc_enddate, use_water_obs)

    # Number of GOSAT observations
    n_obs = len(sat_ind[0])
    print(f"Number of observations is {n_obs}")

    # Initialize Jacobian if needed
    if build_jacobian:
        jacobian_K = np.zeros([n_obs, n_elements], dtype=np.float32)
        jacobian_K.fill(np.nan)

    # Prepare for reading GEOS-Chem data
    all_strdate = []
    date_after_inversion = str(gc_enddate + np.timedelta64(1, "D"))[:10].replace("-", "")
    time_threshold = f"{date_after_inversion}_00"

    # Extract dates from GOSAT observations
    for k in range(n_obs):
        iSat = sat_ind[0][k]
        jSat = sat_ind[1][k]
        time = pd.to_datetime(str(GOSAT["time"][iSat, jSat]))
        strdate = get_strdate(time, time_threshold)
        all_strdate.append(strdate)
    all_strdate = list(set(all_strdate))

    # Read GEOS-Chem data for relevant dates
    all_date_gc = read_all_geoschem(all_strdate, gc_cache, n_elements, config, build_jacobian)

    # Initialize array to store GOSAT and GEOS-Chem data
    obs_GC = np.zeros([n_obs, 6], dtype=np.float32)
    obs_GC.fill(np.nan)

    # Process each GOSAT observation
    for k in range(n_obs):
        print(k)
        iSat = sat_ind[0][k]
        jSat = sat_ind[1][k]
        p_sat = GOSAT["pressures"][iSat, jSat, :]  # Pressure levels
        dry_air_subcolumns = GOSAT["dry_air_subcolumns"][iSat, jSat, :]  # mol m-2
        apriori = GOSAT["methane_profile_apriori"][iSat, jSat, :]  # Prior methane profile (mol m-2)
        avkern = GOSAT["averaging_kernel"][iSat, jSat, :]  # Averaging kernel
        time = pd.to_datetime(str(GOSAT["time"][iSat, jSat]))
        strdate = get_strdate(time, time_threshold)
        GEOSCHEM = all_date_gc[strdate]
        dlon = np.median(np.diff(GEOSCHEM["lon"]))  # GEOS-Chem lon resolution
        dlat = np.median(np.diff(GEOSCHEM["lat"]))  # GEOS-Chem lat resolution

        # Find the GEOS-Chem grid cell closest to the GOSAT pixel center
        lat_center = GOSAT["latitude"][iSat, jSat]
        lon_center = GOSAT["longitude"][iSat, jSat]

        iGC = nearest_loc(lon_center, GEOSCHEM["lon"], tolerance=max(dlon, 0.5))
        jGC = nearest_loc(lat_center, GEOSCHEM["lat"], tolerance=max(dlat, 0.5))

        # Skip if no valid GEOS-Chem grid cell was found
        if np.isnan(iGC) or np.isnan(jGC):
            continue

        # Retrieve GEOS-Chem data for the closest grid cell
        p_gc = GEOSCHEM["PEDGE"][iGC, jGC, :]  # GEOS-Chem pressure edges
        gc_CH4 = GEOSCHEM["CH4"][iGC, jGC, :]  # GEOS-Chem methane

        # Merge GEOS-Chem and GOSAT pressure grids
        merged = merge_pressure_grids(p_sat, p_gc)

        # Remap GEOS-Chem methane to GOSAT pressure levels
        print("GEOS-Chem methane (gc_CH4):", np.nansum(gc_CH4))  # Print GEOS-Chem methane
        print("Merged pressure grid (p_merge):", merged["p_merge"])  # Print the merged pressure grid
        print("Data type for remap:", merged["data_type"])  # Print the data type for remap

        # Remapping methane to GOSAT pressure levels
        sat_CH4 = remap(gc_CH4, merged["data_type"], merged["p_merge"], merged["edge_index"], merged["first_gc_edge"], GOSAT=True)

        # Check the remapped GEOS-Chem methane
        print("Remapped GEOS-Chem methane (sat_CH4):", np.nansum(sat_CH4))

        # Convert from ppb to mol m-2 using dry air subcolumns
        sat_CH4_molm2 = sat_CH4 * 1e-9 * dry_air_subcolumns
        print("GEOS-Chem methane in mol m-2 (sat_CH4_molm2):", sat_CH4_molm2)

        # Debugging the components of the virtual GOSAT observation
        print("Averaging kernel (avkern):", avkern)
        print("Prior methane profile (apriori):", apriori)
        print("Dry air subcolumns:", dry_air_subcolumns)

        # Compute virtual GOSAT observation
        # virtual_gosat = (
        #     sum(apriori + avkern * (sat_CH4_molm2 - apriori)) / sum(dry_air_subcolumns) * 1e9
        # )
        virtual_gosat = np.nansum(dry_air_subcolumns * (apriori + avkern * (sat_CH4 - apriori)))

        # Print the computed virtual GOSAT observation
        print("Virtual GOSAT observation (virtual_gosat):", virtual_gosat)


        # Save GOSAT and virtual GOSAT data
        obs_GC[k, 0] = GOSAT["methane"][iSat, jSat]  # Actual GOSAT CH4 column observation
        obs_GC[k, 1] = virtual_gosat  # Virtual GOSAT CH4 column observation
        obs_GC[k, 2] = GOSAT["longitude"][iSat, jSat]  # GOSAT longitude
        obs_GC[k, 3] = GOSAT["latitude"][iSat, jSat]  # GOSAT latitude
        obs_GC[k, 4] = iSat  # GOSAT index of longitude
        obs_GC[k, 5] = jSat  # GOSAT index of latitude

        # Build Jacobian if required
        if build_jacobian:
            sensi_lonlat = GEOSCHEM["jacobian_ch4"][iGC, jGC, :, :]  # GEOS-Chem sensitivities
            sat_deltaCH4 = remap_sensitivities(sensi_lonlat, merged["data_type"], merged["p_merge"], merged["edge_index"], merged["first_gc_edge"], GOSAT=True)
            avkern_tiled = np.transpose(np.tile(avkern, (n_elements, 1)))
            dry_air_subcolumns_tiled = np.transpose(np.tile(dry_air_subcolumns, (n_elements, 1)))

            # Calculate sensitivity
            # gosat_sensitivity = np.sum(
            #     avkern_tiled * sat_deltaCH4 * dry_air_subcolumns_tiled, 0
            # ) / sum(dry_air_subcolumns)  # unitless
            for n_elem in range(n_elements):
                # Calculate gosat_sensitivity for all elements at once
                gosat_sensitivity = np.nansum(dry_air_subcolumns[:, None] * avkern[:, None] * sat_deltaCH4, axis=0)

                # Store the result in the jacobian matrix for this observation
                jacobian_K[k, :] = gosat_sensitivity


    # Prepare output
    output = {"obs_GC": obs_GC}

    # Include Jacobian if required
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

    # Initialize dictionary for GOSAT data
    dat = {}

    try:
        # Open the GOSAT dataset once and extract important variables
        with xr.open_dataset(filename) as gosat_data:
            # Store methane (CH4), QA value, latitude, longitude, and time
            dat["methane"] = np.expand_dims(gosat_data["xch4"].values, axis=0)
            dat["qa_value"] = np.expand_dims(gosat_data["xch4_quality_flag"].values, axis=0)
            dat["longitude"] = np.expand_dims(gosat_data["longitude"].values, axis=0)
            dat["latitude"] = np.expand_dims(gosat_data["latitude"].values, axis=0)
            dat["time"] = np.expand_dims(gosat_data["time"].values, axis=0)

            # Store averaging kernel
            dat["averaging_kernel"] = np.expand_dims(gosat_data["xch4_averaging_kernel"].values, axis=0)

            # Store prior methane profile, pressure levels, and dry air subcolumns
            dat["methane_profile_apriori"] = np.expand_dims(gosat_data["ch4_profile_apriori"].values, axis=0)
            dat["dry_air_subcolumns"] = np.expand_dims(gosat_data["pressure_weight"].values, axis=0)

            # Store vertical pressure edges
            dat["pressures"] = np.expand_dims(gosat_data["pressure_levels"].values, axis=0)

    except Exception as e:
        print(f"Error opening {filename}: {e}")
        return None

    return dat

