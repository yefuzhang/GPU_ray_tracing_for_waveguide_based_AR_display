import numpy as np
import os
import time
from concurrent.futures import ThreadPoolExecutor
import time
from SRG_LUT_function import LUT_srg_rcwa_parallel_joblib, LUT_srg_rcwa_parallel_joblib1
from couplers_coor import couplers_coor_full_color

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # Safe on Windows

    # Units
    um = 1e-6
    nm = 1e-9
    deg = np.pi / 180

    print("Initializing ......")
    print("Import waveguide design and couplers' coordinates ......")
    num_FOV_x = 100
    num_FOV_y = 75

    IC, FC, FC_offset, OC, OC_offset, \
    eff_reg1, eff_reg2, eff_reg_FOV, eff_reg_FOV_range, \
    lut_TIR, lut_gap, \
    Lambda_ic, phi_ic, Lambda_fc, phi_fc, Lambda_oc, phi_oc, n_g, lmd, \
    th_in_ic, phi_in_ic, th_out_ic, phi_out_ic, th_out_fc, phi_out_fc, \
    th_out_ic2, phi_out_ic2, th_out_oc, phi_out_oc,\
    kx_ic, ky_ic, kx_fc, ky_fc, kx_oc, ky_oc  = couplers_coor_full_color(num_FOV_x=num_FOV_x, num_FOV_y=num_FOV_y)
    print("Complete!")

    
    th_out_fc_lut = th_out_fc[..., np.newaxis]
    phi_out_fc_lut = phi_out_fc[..., np.newaxis]
    th_out_oc_lut = th_out_oc[..., np.newaxis]
    phi_out_oc_lut = phi_out_oc[..., np.newaxis]

    ## set surface relief grating parameters
    # in-coupler
    front_ic = 0*deg # front angle (rad)
    back_ic = 0*deg # back angle (rad)
    thick_ic = 1*um # thickness (m)
    fill_factor_ic = 0.5 # fill factor
    n_material_ic = n_g # grating refarction index
    # folding-coupler (number of FC is 7)
    front_fc = np.array([0, 0, 0, 0, 0, 0, 0])*deg # front angle (rad)
    back_fc = np.array([0, 0, 0, 0, 0, 0, 0])*deg # back angle (rad)
    thick_fc = np.array([1, 1, 1, 1, 1, 1, 1])*um # thickness (m)
    fill_factor_fc = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]) # fill factor
    n_material_fc = n_g # grating refarction index
    # out-coupler (number of OC is 6)
    front_oc = np.array([0, 0, 0, 0, 0, 0])*deg # front angle (rad)
    back_oc = np.array([0, 0, 0, 0, 0, 0])*deg # back angle (rad)
    thick_oc = np.array([1, 1, 1, 1, 1, 1])*um # thickness (m)
    fill_factor_oc = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]) # fill factor
    n_material_oc = n_g # grating refarction index

    start_time = time.time()
    #########################
    # Generate in-coupler LUT
    #########################
    lut_ic1 = []
    lut_ic2 = []
    lut_ic3 = []
    for num in range(len(lmd)):
        # 1st interaction
        result_ic_te_1 = LUT_srg_rcwa_parallel_joblib(
            n_material_ic, 0, 0, front_ic, back_ic, thick_ic, Lambda_ic*nm, fill_factor_ic,
            lmd[num]*nm, th_in_ic[num, :, :], phi_in_ic[num, :, :]-phi_ic, 1, n_g, 1, 0)
        result_ic_tm_1 = LUT_srg_rcwa_parallel_joblib(
            n_material_ic, 0, 0, front_ic, back_ic, thick_ic, Lambda_ic*nm, fill_factor_ic,
            lmd[num]*nm, th_in_ic[num, :, :], phi_in_ic[num, :, :]-phi_ic, 1, n_g, 0, 1)
        # multi-interaction
        result_ic_te_2 = LUT_srg_rcwa_parallel_joblib(
            n_material_ic, 0, 0, front_ic, back_ic, thick_ic, Lambda_ic*nm, fill_factor_ic,
            lmd[num]*nm, th_out_ic[num, :, :], phi_out_ic[num, :, :]-phi_ic, n_g, 1, 1, 0)
        result_ic_tm_2 = LUT_srg_rcwa_parallel_joblib(
            n_material_ic, 0, 0, front_ic, back_ic, thick_ic, Lambda_ic*nm, fill_factor_ic,
            lmd[num]*nm, th_out_ic[num, :, :], phi_out_ic[num, :, :]-phi_ic, n_g, 1, 0, 1)
        # 2nd order consideration in-coupler
        result_ic_te_3 = LUT_srg_rcwa_parallel_joblib(
            n_material_ic, 0, 0, front_ic, back_ic, thick_ic, Lambda_ic*nm, fill_factor_ic,
            lmd[num]*nm, th_out_ic2[num, :, :], phi_out_ic2[num, :, :]-phi_ic, n_g, 1, 1, 0)
        result_ic_tm_3 = LUT_srg_rcwa_parallel_joblib(
            n_material_ic, 0, 0, front_ic, back_ic, thick_ic, Lambda_ic*nm, fill_factor_ic,
            lmd[num]*nm, th_out_ic2[num, :, :], phi_out_ic2[num, :, :]-phi_ic, n_g, 1, 0, 1)
        th_in_ic_lut = th_in_ic[num,:,:]
        th_in_ic_lut = th_in_ic_lut[..., np.newaxis]
        phi_in_ic_lut = phi_in_ic[num,:,:]
        phi_in_ic_lut = phi_in_ic_lut[..., np.newaxis]
        th_out_ic_lut = th_out_ic[num,:,:]
        th_out_ic_lut = th_out_ic_lut[..., np.newaxis]
        phi_out_ic_lut = phi_out_ic[num,:,:]
        phi_out_ic_lut = phi_out_ic_lut[..., np.newaxis]
        th_out_ic2_lut = th_out_ic2[num,:,:]
        th_out_ic2_lut = th_out_ic2_lut[..., np.newaxis]
        phi_out_ic2_lut = phi_out_ic2[num,:,:]
        phi_out_ic2_lut = phi_out_ic2_lut[..., np.newaxis]
        # in-coupler 1st interaction lut
        lut_ic1.append(np.concatenate((th_in_ic_lut, phi_in_ic_lut, result_ic_te_1, result_ic_tm_1), axis=2))
        # in-coupler multi-interaction lut
        lut_ic2.append(np.concatenate((th_out_ic_lut, phi_out_ic_lut, result_ic_te_2, result_ic_tm_2), axis=2))
        # in-coupler 2nd consideration lut
        lut_ic3.append(np.concatenate((th_out_ic2_lut, phi_out_ic2_lut, result_ic_te_3, result_ic_tm_3), axis=2))
    lut_ic1 = np.stack(lut_ic1, axis=0)
    lut_ic2 = np.stack(lut_ic2, axis=0)
    lut_ic3 = np.stack(lut_ic3, axis=0)
    print("LUT shape:", lut_ic1.shape)
    np.save('lut_ic1_fullColor.npy', lut_ic1)
    print("LUT shape:", lut_ic2.shape)
    np.save('lut_ic2_fullColor.npy', lut_ic2)
    print("LUT shape:", lut_ic3.shape)
    np.save('lut_ic3_fullColor.npy', lut_ic3)
    ##############################
    # Generate folding-coupler LUT
    ##############################
    ## note: Since FCs' structure are all the same in this demo, the code just generated one FC LUT and stacked 7 times to get the FC LUT.
    ##       For general situation, please use a for loop to generate 7 LUTs and then stack them into one LUT.
    lut_fc1 = []
    lut_fc2 = []
    for num in range(len(lmd)):
        # 1st interaction
        result_fc_te_1 = LUT_srg_rcwa_parallel_joblib1(
            n_material_fc, 0, 0, front_fc[0], back_fc[0], thick_fc[0], Lambda_fc*nm, fill_factor_fc[0],
            lmd[num]*nm, th_out_ic[num, :, :], phi_out_ic[num, :, :]-phi_fc, n_g, 1, 1, 0)
        result_fc_tm_1 = LUT_srg_rcwa_parallel_joblib1(
            n_material_fc, 0, 0, front_fc[0], back_fc[0], thick_fc[0], Lambda_fc*nm, fill_factor_fc[0],
            lmd[num]*nm, th_out_ic[num, :, :], phi_out_ic[num, :, :]-phi_fc, n_g, 1, 0, 1)
        # multi-interaction
        result_fc_te_2 = LUT_srg_rcwa_parallel_joblib1(
            n_material_fc, 0, 0, front_fc[0], back_fc[0], thick_fc[0], Lambda_fc*nm, fill_factor_fc[0],
            lmd[num]*nm, th_out_fc[num, :, :], phi_out_fc[num, :, :]-phi_fc, n_g, 1, 1, 0)
        result_fc_tm_2 = LUT_srg_rcwa_parallel_joblib1(
            n_material_fc, 0, 0, front_fc[0], back_fc[0], thick_fc[0], Lambda_fc*nm, fill_factor_fc[0],
            lmd[num]*nm, th_out_fc[num, :, :], phi_out_fc[num, :, :]-phi_fc, n_g, 1, 0, 1)
        th_out_ic_lut = th_out_ic[num,:,:]
        th_out_ic_lut = th_out_ic_lut[..., np.newaxis]
        phi_out_ic_lut = phi_out_ic[num,:,:]
        phi_out_ic_lut = phi_out_ic_lut[..., np.newaxis]
        th_out_fc_lut = th_out_fc[num,:,:]
        th_out_fc_lut = th_out_fc_lut[..., np.newaxis]
        phi_out_fc_lut = phi_out_fc[num,:,:]
        phi_out_fc_lut = phi_out_fc_lut[..., np.newaxis]
        # folding-coupler 1st interaction lut
        lut_fc1_lut = np.concatenate((th_out_ic_lut, phi_out_ic_lut, result_fc_te_1, result_fc_tm_1), axis=2)
        list_of_arrays = [lut_fc1_lut, lut_fc1_lut, lut_fc1_lut, lut_fc1_lut, lut_fc1_lut, lut_fc1_lut, lut_fc1_lut]
        lut_fc1.append(np.stack(list_of_arrays))
        # folding-coupler multi-interaction lut
        lut_fc2_lut = np.concatenate((th_out_fc_lut, phi_out_fc_lut, result_fc_te_2, result_fc_tm_2), axis=2)
        list_of_arrays = [lut_fc2_lut, lut_fc2_lut, lut_fc2_lut, lut_fc2_lut, lut_fc2_lut, lut_fc2_lut, lut_fc2_lut]
        lut_fc2.append(np.stack(list_of_arrays))
    lut_fc1 = np.stack(lut_fc1, axis=1)
    lut_fc2 = np.stack(lut_fc2, axis=1)
    print("LUT shape:", lut_fc1.shape)
    np.save('lut_fc1_fullColor.npy', lut_fc1)
    print("LUT shape:", lut_fc2.shape)
    np.save('lut_fc2_fullColor.npy', lut_fc2)
    ##########################
    # Generate out-coupler LUT
    ##########################
    ## note: Since OCs' structure are all the same in this demo, the code just generated one OC LUT and stacked 6 times to get the OC LUT.
    ##       For general situation, please use a for loop to generate 6 LUTs and then stack them into one LUT.
    lut_oc1 = []
    lut_oc2 = []
    for num in range(len(lmd)):
        # 1st interaction
        result_oc_te_1 = LUT_srg_rcwa_parallel_joblib(
            n_material_oc, 0, 0, front_oc[0], back_oc[0], thick_oc[0], Lambda_oc*nm, fill_factor_oc[0],
            lmd[num]*nm, th_out_fc[num, :, :], phi_out_fc[num, :, :]-phi_oc, n_g, 1, 1, 0)
        result_oc_tm_1 = LUT_srg_rcwa_parallel_joblib(
            n_material_oc, 0, 0, front_oc[0], back_oc[0], thick_oc[0], Lambda_oc*nm, fill_factor_oc[0],
            lmd[num]*nm, th_out_fc[num, :, :], phi_out_fc[num, :, :]-phi_oc, n_g, 1, 0, 1)
        result_oc_te_2 = LUT_srg_rcwa_parallel_joblib(
            n_material_oc, 0, 0, front_oc[0], back_oc[0], thick_oc[0], Lambda_oc*nm, fill_factor_oc[0],
            lmd[num]*nm, th_out_oc[num, :, :], phi_out_oc[num, :, :]-phi_oc, n_g, 1, 1, 0)
        result_oc_tm_2 = LUT_srg_rcwa_parallel_joblib(
            n_material_oc, 0, 0, front_oc[0], back_oc[0], thick_oc[0], Lambda_oc*nm, fill_factor_oc[0],
            lmd[num]*nm, th_out_oc[num, :, :], phi_out_oc[num, :, :]-phi_oc, n_g, 1, 0, 1)
        th_out_fc_lut = th_out_fc[num,:,:]
        th_out_fc_lut = th_out_fc_lut[..., np.newaxis]
        phi_out_fc_lut = phi_out_fc[num,:,:]
        phi_out_fc_lut = phi_out_fc_lut[..., np.newaxis]
        th_out_oc_lut = th_out_oc[num,:,:]
        th_out_oc_lut = th_out_oc_lut[..., np.newaxis]
        phi_out_oc_lut = phi_out_oc[num,:,:]
        phi_out_oc_lut = phi_out_oc_lut[..., np.newaxis]
        # out-coupler 1st interaction lut
        lut_oc1_lut = np.concatenate((th_out_fc_lut, phi_out_fc_lut, result_oc_te_1, result_oc_tm_1), axis=2)
        list_of_arrays = [lut_oc1_lut, lut_oc1_lut, lut_oc1_lut, lut_oc1_lut, lut_oc1_lut, lut_oc1_lut]
        lut_oc1.append(np.stack(list_of_arrays))
        # out-coupler 2nd consideration lut
        lut_oc2_lut = np.concatenate((th_out_oc_lut, phi_out_oc_lut, result_oc_te_2, result_oc_tm_2), axis=2)
        list_of_arrays = [lut_oc2_lut, lut_oc2_lut, lut_oc2_lut, lut_oc2_lut, lut_oc2_lut, lut_oc2_lut]
        lut_oc2.append(np.stack(list_of_arrays))
    lut_oc1 = np.stack(lut_oc1, axis=1)
    lut_oc2 = np.stack(lut_oc2, axis=1)
    print("LUT shape:", lut_oc1.shape)
    np.save('lut_oc1_fullColor.npy', lut_oc1)
    print("LUT shape:", lut_oc2.shape)
    np.save('lut_oc2_fullColor.npy', lut_oc2)

    end_time = time.time()

    print(f"Elapsed time of LUT generation: {end_time - start_time:.2f} seconds")