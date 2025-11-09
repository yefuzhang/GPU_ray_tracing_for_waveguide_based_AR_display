import numpy as np
import time
import math
from numba import cuda, int32, float32, uint32
from couplers_coor import couplers_coor
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.colors import LogNorm
from scipy.io import savemat
from numba.cuda.random import xoroshiro128p_uniform_float32


def generate_points_in_polygon(polygon_vertices, num_points):
    xmin, ymin = np.min(polygon_vertices, axis=0)
    xmax, ymax = np.max(polygon_vertices, axis=0)
    polygon_path = Path(polygon_vertices)
    points_inside = []
    while len(points_inside) < num_points:
        num_to_generate = (num_points - len(points_inside)) * 2
        random_points = np.random.uniform(low=[xmin, ymin], high=[xmax, ymax], size=(num_to_generate, 2))
        is_inside = polygon_path.contains_points(random_points)
        points_inside.extend(random_points[is_inside])

    return np.array(points_inside[:num_points])

@cuda.jit(device=True)
def get_uniform_random_number(rng_states, index):
    s = rng_states[index]
    if s == 0:
        s = uint32(0x6D2B79F5) ^ uint32(index + 1)
    s ^= (s << 13) & uint32(0xFFFFFFFF)
    s ^= (s >> 17)
    s ^= (s << 5)  & uint32(0xFFFFFFFF)
    rng_states[index] = s
    return (s & uint32(0xFFFFFFFF)) * (1.0 / 4294967296.0)

@cuda.jit(device=True)
def is_inside_polygon(x, y, poly, start, end):
    n_vertices = end - start
    inside = False
    j = n_vertices - 1
    for i in range(n_vertices):
        xi = poly[start + i, 0]
        yi = poly[start + i, 1]
        xj = poly[start + j, 0]
        yj = poly[start + j, 1]
        if ((yi > y) != (yj > y)) and \
           (x < (xj - xi) * (y - yi) / (yj - yi + 1e-20) + xi):
            inside = not inside
        j = i
    return inside

@cuda.jit(device=True)
def point_on_segment(px, py, poly, idx_a, idx_b, tol):
    x1 = poly[idx_a, 0]
    y1 = poly[idx_a, 1]
    x2 = poly[idx_b, 0]
    y2 = poly[idx_b, 1]
    if (px < min(x1, x2) - tol) or (px > max(x1, x2) + tol) or \
       (py < min(y1, y2) - tol) or (py > max(y1, y2) + tol):
        return False
    return abs((x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)) <= tol

@cuda.jit(device=True)
def is_inside_or_on_edge(px, py, poly, start, end):
    n_vertices = end - start
    j = n_vertices - 1
    for i in range(n_vertices):
        if point_on_segment(px, py, poly, start + j, start + i, 1e-12):
            return True
        j = i
    return is_inside_polygon(px, py, poly, start, end)

@cuda.jit(device=True)
def point_on_segment_4d(px, py, fov_array, m, n, idx_a, idx_b, tol):
    x1 = fov_array[m, n, idx_a, 0]
    y1 = fov_array[m, n, idx_a, 1]
    x2 = fov_array[m, n, idx_b, 0]
    y2 = fov_array[m, n, idx_b, 1]
    if (px < min(x1, x2) - tol) or (px > max(x1, x2) + tol) or \
       (py < min(y1, y2) - tol) or (py > max(y1, y2) + tol):
        return False
    return abs((x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)) <= tol

@cuda.jit(device=True)
def is_inside_polygon_4d(x, y, fov_array, m, n):
    n_vertices = fov_array.shape[2]
    inside = False
    j = n_vertices - 1
    for i in range(n_vertices):
        xi = fov_array[m, n, i, 0]
        yi = fov_array[m, n, i, 1]
        xj = fov_array[m, n, j, 0]
        yj = fov_array[m, n, j, 1]
        if ((yi > y) != (yj > y)) and \
           (x < (xj - xi) * (y - yi) / (yj - yi + 1e-20) + xi):
            inside = not inside 
        j = i
    return inside

@cuda.jit(device=True)
def is_inside_or_on_edge_4d(px, py, fov_array, m, n):
    n_vertices = fov_array.shape[2]
    j = n_vertices - 1
    for i in range(n_vertices):
        if point_on_segment_4d(px, py, fov_array, m, n, j, i, 1e-12):
            return True
        j = i
    return is_inside_polygon_4d(px, py, fov_array, m, n)

@cuda.jit(device=True)
def LUT_find_position(lut, theta, phi):
    J = lut.shape[1]
    K = lut.shape[2]
    best_d2 = 1e-5
    for j in range(J):
        for k in range(K):
            dt = theta - lut[0, j, k]
            dp = phi - lut[1, j, k]
            d2 = dt*dt + dp*dp
            if d2 < best_d2:
                return j, k, True
    return -1, -1, False

@cuda.jit(device=True)
def _wrap_minus_pi_to_pi(x):
    two_pi = 2.0 * math.pi
    x = x + math.pi
    x = x - two_pi * math.floor(x / two_pi)
    x = x - math.pi
    return x

@cuda.jit(device=True)
def E_field_cal(
    Ete_abs, Etm_abs, delta_tem,
    E_te_te, E_te_tm, E_tm_te, E_tm_tm):
    phase = complex(math.cos(delta_tem), math.sin(delta_tem))
    te_in = complex(Ete_abs, 0.0)
    tm_in = phase * Etm_abs
    a = E_te_te
    b = E_tm_te
    c = E_te_tm
    d = E_tm_tm
    Ete_out = a * te_in + b * tm_in
    Etm_out = c * te_in + d * tm_in 
    Ete_out_abs = math.hypot(Ete_out.real, Ete_out.imag)
    Etm_out_abs = math.hypot(Etm_out.real, Etm_out.imag)
    eps = 1e-20
    phi_te = math.atan2(Ete_out.imag, Ete_out.real) if Ete_out_abs >= eps else 0.0
    phi_tm = math.atan2(Etm_out.imag, Etm_out.real) if Etm_out_abs >= eps else 0.0
    delta_out = _wrap_minus_pi_to_pi(phi_tm - phi_te)

    return Ete_out_abs, Etm_out_abs, delta_out

@cuda.jit(device=True)
def add_to_EB_atomic_val(
    matrix_EB, m, n, x, y,
    xmin, xmax, ymin, ymax, value):
    Ny = matrix_EB.shape[2]
    Nx = matrix_EB.shape[3]
    dx = (xmax - xmin) / Nx
    dy = (ymax - ymin) / Ny
    ix = int(math.floor((x - xmin) / dx))
    iy = int(math.floor((y - ymin) / dy))
    cuda.atomic.add(matrix_EB, (n, m, iy, ix), value)
    return True

@cuda.jit
def zero_out_kernel(array):
    idx = cuda.grid(1)
    if idx < array.size:
        array.flat[idx] = 0

@cuda.jit
def reset_counter_kernel(counter):
    if cuda.threadIdx.x == 0 and cuda.blockIdx.x == 0:
        counter[0] = 0

@cuda.jit
def pack_active_to_front(src, dst, src_len, out_count):
    i = cuda.grid(1)
    if i >= src_len:
        return
    Ete = src[i, 8]
    Etm = src[i, 9]
    efficiency = Ete * Ete + Etm * Etm
    if src[i, 12] != 0 and efficiency > 0:
        pos = cuda.atomic.add(out_count, 0, 1)
        # write packed row at the front of dst
        for j in range(src.shape[1]):
            dst[pos, j] = src[i, j]

@cuda.jit
def process_rays_kernel(
    vectors, useful_count_in, d_total_ray_counter,
    MAX_STEPS, 
    IC, FC, FC_offset, OC, OC_offset,
    eff_reg1, eff_reg2, eff_reg_FOV, eff_reg_FOV_range,
    lut_ic1, lut_ic2, lut_fc1, lut_fc2, lut_oc,
    lut_TIR, lut_gap,
    matrix_EB):

    idx = cuda.grid(1)
    N = int(useful_count_in)
    if idx >= N:
        return
    if vectors[idx, 12] == 0:
        return
    x = vectors[idx, 0]
    y = vectors[idx, 1]
    gap_x = vectors[idx, 2]
    gap_y = vectors[idx, 3]
    theta = vectors[idx, 4]
    phi = vectors[idx, 5]
    m = int(vectors[idx, 6])
    n = int(vectors[idx, 7])
    Ete = vectors[idx, 8]
    Etm = vectors[idx, 9]
    delta_phase = vectors[idx, 10]
    region_state = vectors[idx, 11]
    flag = vectors[idx, 12]
    FOV_total = eff_reg_FOV.shape[0]

    if region_state == 0:
        theta = lut_ic2[m, n, 0]
        phi = lut_ic2[m, n, 1]
        Ete, Etm, delta_phase = E_field_cal(Ete, Etm, delta_phase,
                                            lut_ic1[m, n, 8], lut_ic1[m, n, 11],
                                            lut_ic1[m, n, 20], lut_ic1[m, n, 23])
        delta_phase += lut_TIR[m, n, 0]
        gap_x = lut_gap[m, n, 0]
        gap_y = lut_gap[m, n, 1]
        x += gap_x
        y += gap_y
        region_state = 1
    
    if region_state == 1:
        for _ in range(int(MAX_STEPS)):
            if not is_inside_or_on_edge(x, y, IC, 0, IC.shape[0]):
                for i in range(len(FC_offset)-1): 
                    start_FC = FC_offset[i]
                    end_FC = FC_offset[i+1]
                    if is_inside_or_on_edge(x, y, FC, start_FC, end_FC):
                        vectors[idx, 8], vectors[idx, 9], vectors[idx, 10] = E_field_cal(
                            Ete, Etm, delta_phase,
                            lut_fc1[i, m, n, 3], lut_fc1[i, m, n, 6], 
                            lut_fc1[i, m, n, 15], lut_fc1[i, m, n, 18])
                        vectors[idx, 10] += lut_TIR[m, n, 0] 
                        vectors[idx, 0] = x + gap_x 
                        vectors[idx, 1] = y + gap_y
                        vectors[idx, 2] = gap_x 
                        vectors[idx, 3] = gap_y
                        vectors[idx, 4] = theta.real 
                        vectors[idx, 5] = phi.real
                        vectors[idx, 6] = m 
                        vectors[idx, 7] = n
                        vectors[idx, 11] = 2 
                        vectors[idx, 12] = 1 
                        new_ray_index = cuda.atomic.add(d_total_ray_counter, 0, 1)
                        vectors[new_ray_index, 8], vectors[new_ray_index, 9], vectors[new_ray_index, 10] = E_field_cal(
                            Ete, Etm, delta_phase,
                            lut_fc1[i, m, n, 4], lut_fc1[i, m, n, 7],  
                            lut_fc1[i, m, n, 16], lut_fc1[i, m, n, 19]) 
                        vectors[new_ray_index, 10] += lut_TIR[m, n, 1] 
                        vectors[new_ray_index, 0] = x + lut_gap[m, n, 2] 
                        vectors[new_ray_index, 1] = y + lut_gap[m, n, 3]
                        vectors[new_ray_index, 2] = lut_gap[m, n, 2] 
                        vectors[new_ray_index, 3] = lut_gap[m, n, 3]
                        vectors[new_ray_index, 4] = lut_fc2[i, m, n, 0].real 
                        vectors[new_ray_index, 5] = lut_fc2[i, m, n, 1].real
                        vectors[new_ray_index, 6] = m 
                        vectors[new_ray_index, 7] = n
                        vectors[new_ray_index, 11] = 3 
                        vectors[new_ray_index, 12] = 1 
                        return
                delta_phase += 2*lut_TIR[m, n, 0] 
                x += gap_x
                y += gap_y
            else: 
                Ete, Etm, delta_phase = E_field_cal(
                    Ete, Etm, delta_phase,
                    lut_ic2[m, n, 3], lut_ic2[m, n, 6],
                    lut_ic2[m, n, 15], lut_ic2[m, n, 18]) 
                delta_phase += lut_TIR[m, n, 0]
                x += gap_x
                y += gap_y
        if region_state == 1:
            vectors[idx, 12] = 0
            return

    if region_state == 2 or region_state == 3:
        if not is_inside_or_on_edge(x, y, eff_reg1, 0, eff_reg1.shape[0]):
            vectors[idx, 12] = 0
            return

        for _ in range(int(MAX_STEPS)):
            for i in range(len(FC_offset)-1):
                start_FC = FC_offset[i]
                end_FC = FC_offset[i+1]
                if is_inside_or_on_edge(x, y, FC, start_FC, end_FC):
                    if region_state == 2: 
                        vectors[idx, 8], vectors[idx, 9], vectors[idx, 10] = E_field_cal(
                            Ete, Etm, delta_phase,
                            lut_fc1[i, m, n, 3], lut_fc1[i, m, n, 6],
                            lut_fc1[i, m, n, 15], lut_fc1[i, m, n, 18]) 
                        vectors[idx, 10] += lut_TIR[m, n, 0] 
                        vectors[idx, 0] = x + gap_x 
                        vectors[idx, 1] = y + gap_y
                        vectors[idx, 2] = gap_x 
                        vectors[idx, 3] = gap_y
                        vectors[idx, 4] = theta.real 
                        vectors[idx, 5] = phi.real
                        vectors[idx, 6] = m 
                        vectors[idx, 7] = n
                        vectors[idx, 11] = region_state 
                        vectors[idx, 12] = 1 
                        new_ray_index = cuda.atomic.add(d_total_ray_counter, 0, 1)
                        vectors[new_ray_index, 8], vectors[new_ray_index, 9], vectors[new_ray_index, 10] = E_field_cal(
                            Ete, Etm, delta_phase,
                            lut_fc1[i, m, n, 4], lut_fc1[i, m, n, 7], 
                            lut_fc1[i, m, n, 16], lut_fc1[i, m, n, 19])
                        vectors[new_ray_index, 10] += lut_TIR[m, n, 1] 
                        vectors[new_ray_index, 0] = x + lut_gap[m, n, 2] 
                        vectors[new_ray_index, 1] = y + lut_gap[m, n, 3]
                        vectors[new_ray_index, 2] = lut_gap[m, n, 2] 
                        vectors[new_ray_index, 3] = lut_gap[m, n, 3]
                        vectors[new_ray_index, 4] = lut_fc2[i, m, n, 0].real 
                        vectors[new_ray_index, 5] = lut_fc2[i, m, n, 1].real
                        vectors[new_ray_index, 6] = m
                        vectors[new_ray_index, 7] = n
                        vectors[new_ray_index, 11] = 3 
                        vectors[new_ray_index, 12] = 1 
                        return
                    
                    elif region_state == 3: 
                        vectors[idx, 8], vectors[idx, 9], vectors[idx, 10] = E_field_cal(
                            Ete, Etm, delta_phase,
                            lut_fc2[i, m, n, 3], lut_fc2[i, m, n, 6],  
                            lut_fc2[i, m, n, 15], lut_fc2[i, m, n, 18]) 
                        vectors[idx, 10] += lut_TIR[m, n, 1] 
                        vectors[idx, 0] = x + gap_x 
                        vectors[idx, 1] = y + gap_y
                        vectors[idx, 2] = gap_x 
                        vectors[idx, 3] = gap_y
                        vectors[idx, 4] = theta.real 
                        vectors[idx, 5] = phi.real
                        vectors[idx, 6] = m 
                        vectors[idx, 7] = n
                        vectors[idx, 11] = region_state 
                        vectors[idx, 12] = 1 
                        new_ray_index = cuda.atomic.add(d_total_ray_counter, 0, 1)
                        vectors[new_ray_index, 8], vectors[new_ray_index, 9], vectors[new_ray_index, 10] = E_field_cal(
                            Ete, Etm, delta_phase,
                            lut_fc2[i, m, n, 2], lut_fc2[i, m, n, 5], 
                            lut_fc2[i, m, n, 14], lut_fc2[i, m, n, 17]) 
                        vectors[new_ray_index, 10] += lut_TIR[m, n, 0] 
                        vectors[new_ray_index, 0] = x + lut_gap[m, n, 0] 
                        vectors[new_ray_index, 1] = y + lut_gap[m, n, 1]
                        vectors[new_ray_index, 2] = lut_gap[m, n, 0] 
                        vectors[new_ray_index, 3] = lut_gap[m, n, 1]
                        vectors[new_ray_index, 4] = lut_fc1[i, m, n, 0].real 
                        vectors[new_ray_index, 5] = lut_fc1[i, m, n, 1].real
                        vectors[new_ray_index, 6] = m 
                        vectors[new_ray_index, 7] = n
                        vectors[new_ray_index, 11] = 2 
                        vectors[new_ray_index, 12] = 1 
                        return
            if not is_inside_or_on_edge(x, y, eff_reg2, 0, eff_reg2.shape[0]):
                if region_state == 3:
                    region_state = 4
                    break
                else:
                    vectors[idx, 12] = 0
                    return
            else:
                delta_phase += 2*lut_TIR[m, n, 0] 
                x += gap_x
                y += gap_y
    if region_state == 4:  
        for _ in range(int(MAX_STEPS)):
            if not is_inside_or_on_edge(x, y, eff_reg1, 0, eff_reg1.shape[0]):
                vectors[idx, 12] = 0
                return

            hit = False
            for i in range(len(OC_offset) - 1):
                start_OC = OC_offset[i]
                end_OC = OC_offset[i+1]
                if is_inside_or_on_edge(x, y, OC, start_OC, end_OC):
                    hit = True
                    if is_inside_or_on_edge_4d(x, y, eff_reg_FOV, m, n):
                        Ete_out, Etm_out, _ = E_field_cal(
                            Ete, Etm, delta_phase,
                            lut_oc[i, m, n, 10], lut_oc[i, m, n, 13],
                            lut_oc[i, m, n, 22], lut_oc[i, m, n, 25]) 
                        efficiency = Ete_out * Ete_out + Etm_out * Etm_out
                        if efficiency > 0:
                            xmin = eff_reg_FOV_range[m, n, 0]
                            xmax = eff_reg_FOV_range[m, n, 1]
                            ymin = eff_reg_FOV_range[m, n, 2]
                            ymax = eff_reg_FOV_range[m, n, 3]
                            add_to_EB_atomic_val(matrix_EB, m, n, x, y, xmin, xmax, ymin, ymax, efficiency)
                    Ete, Etm, delta_phase = E_field_cal(
                        Ete, Etm, delta_phase,
                        lut_oc[i, m, n, 3],  lut_oc[i, m, n, 6],
                        lut_oc[i, m, n, 15], lut_oc[i, m, n, 18]) 
                    delta_phase += lut_TIR[m, n, 1]
                    x += gap_x
                    y += gap_y
                    efficiency = Ete * Ete + Etm * Etm
                    if efficiency < 0:
                        vectors[idx, 12] = 0
                        return
                    break
            if not hit:
                delta_phase += 2 * lut_TIR[m, n, 1]
                x += gap_x
                y += gap_y

@cuda.jit
def process_rays_kernel_pro(
    x_v, y_v, gap_x_v, gap_y_v, pol_v, azi_v, m_v, n_v, te_v, tm_v, delta_phase_v,
    rng_states,
    IC, FC, FC_offset, OC, OC_offset, n_g,
    eff_reg1, eff_reg2, eff_reg_FOV, eff_reg_FOV_range,
    lut_ic1, lut_ic2, lut_ic3, lut_fc1, lut_fc2, lut_oc1, lut_oc2,
    lut_TIR, lut_gap,
    matrix_EB):
    idx = cuda.grid(1)
    N = x_v.shape[0]
    if idx >= N:
        return
    x = x_v[idx]
    y = y_v[idx]
    gap_x = gap_x_v[idx]
    gap_y = gap_y_v[idx]
    theta = pol_v[idx]
    phi = azi_v[idx]
    m = int(m_v[idx]) 
    n = int(n_v[idx])
    Ete = te_v[idx]
    Etm = tm_v[idx]
    delta_phase = delta_phase_v[idx]
    ener = 1
    threshold = 1e-15
    Ete1, Etm1, delta_phase1 = E_field_cal(
        Ete, Etm, delta_phase,
        lut_ic1[m, n, 13], lut_ic1[m, n, 18], 
        lut_ic1[m, n, 33], lut_ic1[m, n, 38]) 
    Ete2, Etm2, delta_phase2 = E_field_cal(
        Ete, Etm, delta_phase,
        lut_ic1[m, n, 15], lut_ic1[m, n, 20], 
        lut_ic1[m, n, 35], lut_ic1[m, n, 40]) 
    efficiency1 = (Ete1 * Ete1 + Etm1 * Etm1) * math.cos(lut_ic2[m, n, 0].real) / math.cos(lut_ic1[m, n, 0].real) * n_g
    efficiency2 = (Ete2 * Ete2 + Etm2 * Etm2) * math.cos(lut_ic3[m, n, 0].real) / math.cos(lut_ic1[m, n, 0].real) * n_g
    rand_val = get_uniform_random_number(rng_states, idx)
    if rand_val <= efficiency1:
        theta = lut_ic2[m, n, 0]
        phi = lut_ic2[m, n, 1]
        norm = math.sqrt(Ete1*Ete1 + Etm1*Etm1)
        Ete = Ete1/norm
        Etm = Etm1/norm
        delta_phase = delta_phase1 + lut_TIR[m, n, 0] 
        gap_x = lut_gap[m, n, 0]
        gap_y = lut_gap[m, n, 1]
        x += gap_x
        y += gap_y
        ener *= efficiency1
        if not is_inside_or_on_edge(x, y, IC, 0, IC.shape[0]):
            region_state = 2
        else:
            region_state = 0
    elif rand_val <= efficiency1+efficiency2:
        theta = lut_ic3[m, n, 0]
        phi = lut_ic3[m, n, 1]
        norm = math.sqrt(Ete2*Ete2 + Etm2*Etm2)
        Ete = Ete2/norm
        Etm = Etm2/norm
        delta_phase = delta_phase2 + lut_TIR[m, n, 2]
        gap_x = lut_gap[m, n, 4]
        gap_y = lut_gap[m, n, 5]
        x += gap_x
        y += gap_y
        ener *= efficiency2
        if not is_inside_or_on_edge(x, y, IC, 0, IC.shape[0]): 
            return
        else:
            region_state = 1
    else:
        return
    for _ in range(1e5):
        if not is_inside_or_on_edge(x, y, eff_reg1, 0, eff_reg1.shape[0]):
            return
        if region_state == 0:
            Ete1, Etm1, delta_phase1 = E_field_cal(
                Ete, Etm, delta_phase,
                lut_ic2[m, n, 4], lut_ic2[m, n, 9], 
                lut_ic2[m, n, 24], lut_ic2[m, n, 29]) 
            Ete2, Etm2, delta_phase2 = E_field_cal(
                Ete, Etm, delta_phase,
                lut_ic2[m, n, 6], lut_ic2[m, n, 11], 
                lut_ic2[m, n, 26], lut_ic2[m, n, 31]) 
            efficiency1 = (Ete1 * Ete1 + Etm1 * Etm1) * math.cos(lut_ic2[m, n, 0].real) / math.cos(theta.real) 
            efficiency2 = (Ete2 * Ete2 + Etm2 * Etm2) * math.cos(lut_ic3[m, n, 0].real) / math.cos(theta.real) 
            rand_val = get_uniform_random_number(rng_states, idx)
            if rand_val <= efficiency1:
                theta = lut_ic2[m, n, 0]
                phi = lut_ic2[m, n, 1]
                norm = math.sqrt(Ete1*Ete1 + Etm1*Etm1)
                Ete = Ete1/norm
                Etm = Etm1/norm
                delta_phase = delta_phase1 + lut_TIR[m, n, 0] 
                gap_x = lut_gap[m, n, 0]
                gap_y = lut_gap[m, n, 1]
                x += gap_x
                y += gap_y
                ener *= efficiency1
                if not is_inside_or_on_edge(x, y, IC, 0, IC.shape[0]): 
                    region_state = 2
                else:
                    region_state = 0
            elif rand_val <= efficiency1+efficiency2:
                theta = lut_ic3[m, n, 0]
                phi = lut_ic3[m, n, 1]
                norm = math.sqrt(Ete2*Ete2 + Etm2*Etm2)
                Ete = Ete2/norm
                Etm = Etm2/norm
                delta_phase = delta_phase2 + lut_TIR[m, n, 2] 
                gap_x = lut_gap[m, n, 4] 
                gap_y = lut_gap[m, n, 5]
                x += gap_x
                y += gap_y
                ener *= efficiency2
                if not is_inside_or_on_edge(x, y, IC, 0, IC.shape[0]): 
                    return
                else:
                    region_state = 1
            else:
                return
        
        elif region_state == 1:
            Ete1, Etm1, delta_phase1 = E_field_cal(
                Ete, Etm, delta_phase,
                lut_ic3[m, n, 2], lut_ic3[m, n, 22], 
                lut_ic3[m, n, 7], lut_ic3[m, n, 27]) 
            Ete2, Etm2, delta_phase2 = E_field_cal(
                Ete, Etm, delta_phase,
                lut_ic3[m, n, 4], lut_ic3[m, n, 9], 
                lut_ic3[m, n, 24], lut_ic3[m, n, 29]) 
            efficiency1 = (Ete1 * Ete1 + Etm1 * Etm1) * math.cos(lut_ic2[m, n, 0].real) / math.cos(theta.real)
            efficiency2 = (Ete2 * Ete2 + Etm2 * Etm2) * math.cos(lut_ic3[m, n, 0].real) / math.cos(theta.real)
            rand_val = get_uniform_random_number(rng_states, idx)
            if rand_val <= efficiency1:
                theta = lut_ic2[m, n, 0]
                phi = lut_ic2[m, n, 1]
                norm = math.sqrt(Ete1*Ete1 + Etm1*Etm1)
                Ete = Ete1/norm
                Etm = Etm1/norm
                delta_phase = delta_phase1 + lut_TIR[m, n, 0]
                gap_x = lut_gap[m, n, 0]
                gap_y = lut_gap[m, n, 1]
                x += gap_x
                y += gap_y
                ener *= efficiency1
                if not is_inside_or_on_edge(x, y, IC, 0, IC.shape[0]):
                    region_state = 2
                else:
                    region_state = 0
            elif rand_val <= efficiency1+efficiency2:
                theta = lut_ic3[m, n, 0]
                phi = lut_ic3[m, n, 1]
                norm = math.sqrt(Ete2*Ete2 + Etm2*Etm2)
                Ete = Ete2/norm
                Etm = Etm2/norm
                delta_phase = delta_phase2 + lut_TIR[m, n, 2] 
                gap_x = lut_gap[m, n, 4] 
                gap_y = lut_gap[m, n, 5]
                x += gap_x
                y += gap_y
                ener *= efficiency2
                if not is_inside_or_on_edge(x, y, IC, 0, IC.shape[0]): 
                    return
                else:
                    region_state = 1
            else:
                return
        elif region_state == 2: 
            hit = False
            for i in range(len(FC_offset)-1):
                start_FC = FC_offset[i]
                end_FC = FC_offset[i+1]
                if is_inside_or_on_edge(x, y, FC, start_FC, end_FC):
                    hit = True
                    Ete1, Etm1, delta_phase1 = E_field_cal(
                        Ete, Etm, delta_phase,
                        lut_fc1[i, m, n, 3], lut_fc1[i, m, n, 6],   
                        lut_fc1[i, m, n, 15], lut_fc1[i, m, n, 18]) 
                    Ete2, Etm2, delta_phase2 = E_field_cal(
                        Ete, Etm, delta_phase,
                        lut_fc1[i, m, n, 2], lut_fc1[i, m, n, 5],
                        lut_fc1[i, m, n, 14], lut_fc1[i, m, n, 17]) 
                    efficiency1 = (Ete1 * Ete1 + Etm1 * Etm1) * math.cos(lut_fc1[i, m, n, 0].real) / math.cos(theta.real)
                    efficiency2 = (Ete2 * Ete2 + Etm2 * Etm2) * math.cos(lut_fc2[i, m, n, 0].real) / math.cos(theta.real)
                    ener1 = ener * efficiency1
                    ener2 = ener * efficiency2
                    rand_val = get_uniform_random_number(rng_states, idx)
                    if rand_val <= efficiency1 and ener1 > threshold:
                        theta = lut_fc1[i, m, n, 0]
                        phi = lut_fc1[i, m, n, 1]
                        norm = math.sqrt(Ete1*Ete1 + Etm1*Etm1)
                        Ete = Ete1/norm
                        Etm = Etm1/norm
                        delta_phase = delta_phase1 + lut_TIR[m, n, 0] 
                        gap_x = lut_gap[m, n, 0]
                        gap_y = lut_gap[m, n, 1]
                        x += gap_x
                        y += gap_y
                        ener = ener1 * 1.0
                        region_state = 2
                    elif rand_val <= efficiency1+efficiency2 and ener2 > threshold:
                        theta = lut_fc2[i, m, n, 0]
                        phi = lut_fc2[i, m, n, 1]
                        norm = math.sqrt(Ete2*Ete2 + Etm2*Etm2)
                        Ete = Ete2/norm
                        Etm = Etm2/norm
                        delta_phase = delta_phase2 + lut_TIR[m, n, 1] 
                        gap_x = lut_gap[m, n, 2]
                        gap_y = lut_gap[m, n, 3]
                        x += gap_x
                        y += gap_y
                        ener = ener2 * 1.0
                        region_state = 3
                    else:
                        return
                    break
            if not hit:
                x += gap_x
                y += gap_y
                delta_phase += 2*lut_TIR[m, n, 0]
        elif region_state == 3:
            hit = False
            for i in range(len(FC_offset)-1):
                start_FC = FC_offset[i]
                end_FC = FC_offset[i+1]
                if is_inside_or_on_edge(x, y, FC, start_FC, end_FC):
                    hit = True
                    Ete1, Etm1, delta_phase1 = E_field_cal(
                        Ete, Etm, delta_phase,
                        lut_fc2[i, m, n, 4], lut_fc2[i, m, n, 7], 
                        lut_fc2[i, m, n, 16], lut_fc2[i, m, n, 19]) 
                    Ete2, Etm2, delta_phase2 = E_field_cal(
                        Ete, Etm, delta_phase,
                        lut_fc2[i, m, n, 3], lut_fc2[i, m, n, 6], 
                        lut_fc2[i, m, n, 15], lut_fc2[i, m, n, 18]) 
                    efficiency1 = (Ete1 * Ete1 + Etm1 * Etm1) * math.cos(lut_fc1[i, m, n, 0].real) / math.cos(theta.real)
                    efficiency2 = (Ete2 * Ete2 + Etm2 * Etm2) * math.cos(lut_fc2[i, m, n, 0].real) / math.cos(theta.real)
                    ener1 = ener * efficiency1
                    ener2 = ener * efficiency2
                    rand_val = get_uniform_random_number(rng_states, idx)
                    if rand_val <= efficiency1 and ener1 > threshold:
                        theta = lut_fc1[i, m, n, 0]
                        phi = lut_fc1[i, m, n, 1]
                        norm = math.sqrt(Ete1*Ete1 + Etm1*Etm1)
                        Ete = Ete1/norm
                        Etm = Etm1/norm
                        delta_phase = delta_phase1 + lut_TIR[m, n, 0] 
                        gap_x = lut_gap[m, n, 0] 
                        gap_y = lut_gap[m, n, 1]
                        x += gap_x
                        y += gap_y
                        ener = ener1 * 1.0
                        region_state = 2
                    elif rand_val <= efficiency1+efficiency2 and ener2 > threshold:
                        theta = lut_fc2[i, m, n, 0]
                        phi = lut_fc2[i, m, n, 1]
                        norm = math.sqrt(Ete2*Ete2 + Etm2*Etm2)
                        Ete = Ete2/norm
                        Etm = Etm2/norm
                        delta_phase = delta_phase2 + lut_TIR[m, n, 1] 
                        gap_x = lut_gap[m, n, 2] 
                        gap_y = lut_gap[m, n, 3]
                        x += gap_x
                        y += gap_y
                        ener = ener2 * 1.0
                        region_state = 3
                    else:
                        return
                    break
            if not hit:
                if not is_inside_or_on_edge(x, y, eff_reg2, 0, eff_reg2.shape[0]):
                    region_state = 4
                else:
                    x += gap_x
                    y += gap_y
                    delta_phase += 2*lut_TIR[m, n, 1]
        elif region_state == 4: 
            hit = False
            for i in range(len(OC_offset)-1):
                start_OC = OC_offset[i]
                end_OC = OC_offset[i+1]
                if is_inside_or_on_edge(x, y, OC, start_OC, end_OC):
                    hit = True
                    Ete1, Etm1, delta_phase1 = E_field_cal(
                        Ete, Etm, delta_phase,
                        lut_oc1[i, m, n, 4],  lut_oc1[i, m, n, 9],
                        lut_oc1[i, m, n, 24], lut_oc1[i, m, n, 29]) 
                    Ete2, Etm2, delta_phase2 = E_field_cal(
                        Ete, Etm, delta_phase,
                        lut_oc1[i, m, n, 2], lut_oc1[i, m, n, 7],
                        lut_oc1[i, m, n, 22], lut_oc1[i, m, n, 27]) 
                    Ete3, Etm3, delta_phase3 = E_field_cal(
                        Ete, Etm, delta_phase,
                        lut_oc1[i, m, n, 13], lut_oc1[i, m, n, 18], 
                        lut_oc1[i, m, n, 33], lut_oc1[i, m, n, 38]) 
                    efficiency1 = (Ete1 * Ete1 + Etm1 * Etm1) * math.cos(lut_oc1[i, m, n, 0].real) / math.cos(theta.real)
                    efficiency2 = (Ete2 * Ete2 + Etm2 * Etm2) * math.cos(lut_oc2[i, m, n, 0].real) / math.cos(theta.real)
                    efficiency3 = (Ete3 * Ete3 + Etm3 * Etm3) * math.cos(lut_ic1[m, n, 0].real) / math.cos(theta.real) / n_g
                    ener1 = ener * efficiency1
                    ener2 = ener * efficiency2
                    ener3 = ener * efficiency3
                    rand_val = get_uniform_random_number(rng_states, idx)
                    if rand_val <= efficiency1 and ener1 > threshold:
                        theta = lut_oc1[i, m, n, 0]
                        phi = lut_oc1[i, m, n, 1]
                        norm = math.sqrt(Ete1*Ete1 + Etm1*Etm1)
                        Ete = Ete1/norm
                        Etm = Etm1/norm
                        delta_phase = delta_phase1 + lut_TIR[m, n, 1] 
                        gap_x = lut_gap[m, n, 2] 
                        gap_y = lut_gap[m, n, 3]
                        x += gap_x
                        y += gap_y
                        ener = ener1 * 1.0
                        region_state = 4
                    elif rand_val <= efficiency1+efficiency2 and ener2 > threshold:
                        theta = lut_oc2[i, m, n, 0]
                        phi = lut_oc2[i, m, n, 2]
                        norm = math.sqrt(Ete2*Ete2 + Etm2*Etm2)
                        Ete = Ete2/norm
                        Etm = Etm2/norm
                        delta_phase = delta_phase2 + lut_TIR[m, n, 3] 
                        gap_x = lut_gap[m, n, 6]
                        gap_y = lut_gap[m, n, 7]
                        x += gap_x
                        y += gap_y
                        ener = ener2 * 1.0
                        region_state = 5
                    elif rand_val <= efficiency1+efficiency2+efficiency3 and ener3 > threshold:
                        if is_inside_or_on_edge_4d(x, y, eff_reg_FOV, m, n):
                            xmin = eff_reg_FOV_range[m, n, 0]
                            xmax = eff_reg_FOV_range[m, n, 1]
                            ymin = eff_reg_FOV_range[m, n, 2]
                            ymax = eff_reg_FOV_range[m, n, 3]
                            add_to_EB_atomic_val(matrix_EB, m, n, x, y, xmin, xmax, ymin, ymax, 1.0)
                            return
                        else:
                            return
                    else:
                        return
                    break
            if not hit:
                x += gap_x
                y += gap_y
                delta_phase += 2*lut_TIR[m, n, 1]

        elif region_state == 5:
            hit = False
            for i in range(len(OC_offset)-1):
                start_OC = OC_offset[i]
                end_OC = OC_offset[i+1]
                if is_inside_or_on_edge(x, y, OC, start_OC, end_OC):
                    hit = True
                    Ete1, Etm1, delta_phase1 = E_field_cal(
                        Ete, Etm, delta_phase,
                        lut_oc2[i, m, n, 6], lut_oc2[i, m, n, 11],
                        lut_oc2[i, m, n, 26], lut_oc2[i, m, n, 31])
                    Ete2, Etm2, delta_phase2 = E_field_cal(
                        Ete, Etm, delta_phase,
                        lut_oc2[i, m, n, 4],  lut_oc2[i, m, n, 9],
                        lut_oc2[i, m, n, 24], lut_oc2[i, m, n, 29])
                    Ete3, Etm3, delta_phase3 = E_field_cal(
                        Ete, Etm, delta_phase,
                        lut_oc2[i, m, n, 15], lut_oc2[i, m, n, 20],
                        lut_oc2[i, m, n, 35], lut_oc2[i, m, n, 40])
                    efficiency1 = (Ete1 * Ete1 + Etm1 * Etm1) * math.cos(lut_oc1[i, m, n, 0].real) / math.cos(theta.real)
                    efficiency2 = (Ete2 * Ete2 + Etm2 * Etm2) * math.cos(lut_oc2[i, m, n, 0].real) / math.cos(theta.real)
                    efficiency3 = (Ete3 * Ete3 + Etm3 * Etm3) * math.cos(lut_ic1[m, n, 0].real) / math.cos(theta.real) / n_g
                    ener1 = ener * efficiency1
                    ener2 = ener * efficiency2
                    ener3 = ener * efficiency3
                    rand_val = get_uniform_random_number(rng_states, idx)
                    if rand_val <= efficiency1 and ener1 > threshold:
                        theta = lut_oc1[i, m, n, 0]
                        phi = lut_oc1[i, m, n, 1]
                        norm = math.sqrt(Ete1*Ete1 + Etm1*Etm1)
                        Ete = Ete1/norm
                        Etm = Etm1/norm
                        delta_phase = delta_phase1 + lut_TIR[m, n, 1]
                        gap_x = lut_gap[m, n, 2]
                        gap_y = lut_gap[m, n, 3]
                        x += gap_x
                        y += gap_y
                        ener = ener1 * 1.0
                        region_state = 4
                    elif rand_val <= efficiency1+efficiency2 and ener2 > threshold:
                        theta = lut_oc2[i, m, n, 0]
                        phi = lut_oc2[i, m, n, 2]
                        norm = math.sqrt(Ete2*Ete2 + Etm2*Etm2)
                        Ete = Ete2/norm
                        Etm = Etm2/norm
                        delta_phase = delta_phase2 + lut_TIR[m, n, 3]
                        gap_x = lut_gap[m, n, 6]
                        gap_y = lut_gap[m, n, 7]
                        x += gap_x
                        y += gap_y
                        ener = ener2 * 1.0
                        region_state = 5
                    elif rand_val <= efficiency1+efficiency2+efficiency3 and ener3 > threshold:
                        if is_inside_or_on_edge_4d(x, y, eff_reg_FOV, m, n):
                            xmin = eff_reg_FOV_range[m, n, 0]
                            xmax = eff_reg_FOV_range[m, n, 1]
                            ymin = eff_reg_FOV_range[m, n, 2]
                            ymax = eff_reg_FOV_range[m, n, 3]
                            add_to_EB_atomic_val(matrix_EB, m, n, x, y, xmin, xmax, ymin, ymax, 1.0)
                            return
                        else:
                            return
                    else:
                        return
                    break
            if not hit:
                return

@cuda.jit
def process_rays_kernel_pro_fullColor(
    x_v, y_v, gap_x_v, gap_y_v, pol_v, azi_v, m_v, n_v, lmd_num, te_v, tm_v, delta_phase_v,
    rng_states,
    IC, FC, FC_offset, OC, OC_offset, n_g,
    eff_reg1, eff_reg2, eff_reg_FOV, eff_reg_FOV_range,
    lut_ic1, lut_ic2, lut_ic3, lut_fc1, lut_fc2, lut_oc1, lut_oc2,
    lut_TIR, lut_gap,
    matrix_EB):
    idx = cuda.grid(1)
    N = x_v.shape[0]
    if idx >= N:
        return
    x = x_v[idx]
    y = y_v[idx]
    gap_x = gap_x_v[idx] 
    gap_y = gap_y_v[idx]
    theta = pol_v[idx] 
    phi = azi_v[idx]
    m = int(m_v[idx])
    n = int(n_v[idx])
    lmd_num = int(lmd_num[idx])
    Ete = te_v[idx] 
    Etm = tm_v[idx]
    delta_phase = delta_phase_v[idx]
    ener = 1
    threshold = 0
    Ete1, Etm1, delta_phase1 = E_field_cal(
        Ete, Etm, delta_phase,
        lut_ic1[lmd_num, m, n, 13], lut_ic1[lmd_num, m, n, 18], 
        lut_ic1[lmd_num, m, n, 33], lut_ic1[lmd_num, m, n, 38])
    Ete2, Etm2, delta_phase2 = E_field_cal(
        Ete, Etm, delta_phase,
        lut_ic1[lmd_num, m, n, 15], lut_ic1[lmd_num, m, n, 20], 
        lut_ic1[lmd_num, m, n, 35], lut_ic1[lmd_num, m, n, 40]) 
    efficiency1 = (Ete1 * Ete1 + Etm1 * Etm1) * math.cos(lut_ic2[lmd_num, m, n, 0].real) / math.cos(lut_ic1[lmd_num, m, n, 0].real) * n_g
    efficiency2 = (Ete2 * Ete2 + Etm2 * Etm2) * math.cos(lut_ic3[lmd_num, m, n, 0].real) / math.cos(lut_ic1[lmd_num, m, n, 0].real) * n_g
    rand_val = get_uniform_random_number(rng_states, idx)
    if rand_val <= efficiency1:
        theta = lut_ic2[lmd_num, m, n, 0]
        phi = lut_ic2[lmd_num, m, n, 1]
        norm = math.sqrt(Ete1*Ete1 + Etm1*Etm1)
        Ete = Ete1/norm
        Etm = Etm1/norm
        delta_phase = delta_phase1 + lut_TIR[lmd_num, m, n, 0] 
        gap_x = lut_gap[lmd_num, m, n, 0] 
        gap_y = lut_gap[lmd_num, m, n, 1]
        x += gap_x
        y += gap_y
        ener *= efficiency1
        if not is_inside_or_on_edge(x, y, IC, 0, IC.shape[0]): 
            region_state = 2
        else:
            region_state = 0
    elif rand_val <= efficiency1+efficiency2:
        theta = lut_ic3[lmd_num, m, n, 0]
        phi = lut_ic3[lmd_num, m, n, 1]
        norm = math.sqrt(Ete2*Ete2 + Etm2*Etm2)
        Ete = Ete2/norm
        Etm = Etm2/norm
        delta_phase = delta_phase2 + lut_TIR[lmd_num, m, n, 2] 
        gap_x = lut_gap[lmd_num, m, n, 4]
        gap_y = lut_gap[lmd_num, m, n, 5]
        x += gap_x
        y += gap_y
        ener *= efficiency2
        if not is_inside_or_on_edge(x, y, IC, 0, IC.shape[0]):
            return
        else:
            region_state = 1
    else:
        return
    for _ in range(1e5):
        if not is_inside_or_on_edge(x, y, eff_reg1, 0, eff_reg1.shape[0]):
            return
        if region_state == 0:
            Ete1, Etm1, delta_phase1 = E_field_cal(
                Ete, Etm, delta_phase,
                lut_ic2[lmd_num, m, n, 4], lut_ic2[lmd_num, m, n, 9], 
                lut_ic2[lmd_num, m, n, 24], lut_ic2[lmd_num, m, n, 29]) 
            Ete2, Etm2, delta_phase2 = E_field_cal(
                Ete, Etm, delta_phase,
                lut_ic2[lmd_num, m, n, 6], lut_ic2[lmd_num, m, n, 11], 
                lut_ic2[lmd_num, m, n, 26], lut_ic2[lmd_num, m, n, 31]) 
            efficiency1 = (Ete1 * Ete1 + Etm1 * Etm1) * math.cos(lut_ic2[lmd_num, m, n, 0].real) / math.cos(theta.real)
            efficiency2 = (Ete2 * Ete2 + Etm2 * Etm2) * math.cos(lut_ic3[lmd_num, m, n, 0].real) / math.cos(theta.real)
            rand_val = get_uniform_random_number(rng_states, idx)
            if rand_val <= efficiency1:
                theta = lut_ic2[lmd_num, m, n, 0]
                phi = lut_ic2[lmd_num, m, n, 1]
                norm = math.sqrt(Ete1*Ete1 + Etm1*Etm1)
                Ete = Ete1/norm
                Etm = Etm1/norm
                delta_phase = delta_phase1 + lut_TIR[lmd_num, m, n, 0] 
                gap_x = lut_gap[lmd_num, m, n, 0] 
                gap_y = lut_gap[lmd_num, m, n, 1]
                x += gap_x
                y += gap_y
                ener *= efficiency1
                if not is_inside_or_on_edge(x, y, IC, 0, IC.shape[0]): 
                    region_state = 2
                else:
                    region_state = 0
            elif rand_val <= efficiency1+efficiency2:
                theta = lut_ic3[lmd_num, m, n, 0]
                phi = lut_ic3[lmd_num, m, n, 1]
                norm = math.sqrt(Ete2*Ete2 + Etm2*Etm2)
                Ete = Ete2/norm
                Etm = Etm2/norm
                delta_phase = delta_phase2 + lut_TIR[lmd_num, m, n, 2] 
                gap_x = lut_gap[lmd_num, m, n, 4] 
                gap_y = lut_gap[lmd_num, m, n, 5]
                x += gap_x
                y += gap_y
                ener *= efficiency2
                if not is_inside_or_on_edge(x, y, IC, 0, IC.shape[0]): 
                    return
                else:
                    region_state = 1
            else:
                return
        elif region_state == 1:
            Ete1, Etm1, delta_phase1 = E_field_cal(
                Ete, Etm, delta_phase,
                lut_ic3[lmd_num, m, n, 2], lut_ic3[lmd_num, m, n, 22], 
                lut_ic3[lmd_num, m, n, 7], lut_ic3[lmd_num, m, n, 27]) 
            Ete2, Etm2, delta_phase2 = E_field_cal(
                Ete, Etm, delta_phase,
                lut_ic3[lmd_num, m, n, 4], lut_ic3[lmd_num, m, n, 9],
                lut_ic3[lmd_num, m, n, 24], lut_ic3[lmd_num, m, n, 29]) 
            efficiency1 = (Ete1 * Ete1 + Etm1 * Etm1) * math.cos(lut_ic2[lmd_num, m, n, 0].real) / math.cos(theta.real)
            efficiency2 = (Ete2 * Ete2 + Etm2 * Etm2) * math.cos(lut_ic3[lmd_num, m, n, 0].real) / math.cos(theta.real)
            rand_val = get_uniform_random_number(rng_states, idx)
            if rand_val <= efficiency1:
                theta = lut_ic2[lmd_num, m, n, 0]
                phi = lut_ic2[lmd_num, m, n, 1]
                norm = math.sqrt(Ete1*Ete1 + Etm1*Etm1)
                Ete = Ete1/norm
                Etm = Etm1/norm
                delta_phase = delta_phase1 + lut_TIR[lmd_num, m, n, 0] 
                gap_x = lut_gap[lmd_num, m, n, 0]
                gap_y = lut_gap[lmd_num, m, n, 1]
                x += gap_x
                y += gap_y
                ener *= efficiency1
                if not is_inside_or_on_edge(x, y, IC, 0, IC.shape[0]): 
                    region_state = 2
                else:
                    region_state = 0
            elif rand_val <= efficiency1+efficiency2:
                theta = lut_ic3[lmd_num, m, n, 0]
                phi = lut_ic3[lmd_num, m, n, 1]
                norm = math.sqrt(Ete2*Ete2 + Etm2*Etm2)
                Ete = Ete2/norm
                Etm = Etm2/norm
                delta_phase = delta_phase2 + lut_TIR[lmd_num, m, n, 2] 
                gap_x = lut_gap[lmd_num, m, n, 4] 
                gap_y = lut_gap[lmd_num, m, n, 5]
                x += gap_x
                y += gap_y
                ener *= efficiency2
                if not is_inside_or_on_edge(x, y, IC, 0, IC.shape[0]): 
                    return
                else:
                    region_state = 1
            else:
                return
        elif region_state == 2: 
            hit = False
            for i in range(len(FC_offset)-1):
                start_FC = FC_offset[i]
                end_FC = FC_offset[i+1]
                if is_inside_or_on_edge(x, y, FC, start_FC, end_FC):
                    hit = True
                    Ete1, Etm1, delta_phase1 = E_field_cal(
                        Ete, Etm, delta_phase,
                        lut_fc1[i, lmd_num, m, n, 3], lut_fc1[i, lmd_num, m, n, 6],
                        lut_fc1[i, lmd_num, m, n, 15], lut_fc1[i, lmd_num, m, n, 18])
                    Ete2, Etm2, delta_phase2 = E_field_cal(
                        Ete, Etm, delta_phase,
                        lut_fc1[i, lmd_num, m, n, 2], lut_fc1[i, lmd_num, m, n, 5],   
                        lut_fc1[i, lmd_num, m, n, 14], lut_fc1[i, lmd_num, m, n, 17]) 
                    efficiency1 = (Ete1 * Ete1 + Etm1 * Etm1) * math.cos(lut_fc1[i, lmd_num, m, n, 0].real) / math.cos(theta.real)
                    efficiency2 = (Ete2 * Ete2 + Etm2 * Etm2) * math.cos(lut_fc2[i, lmd_num, m, n, 0].real) / math.cos(theta.real)
                    ener1 = ener * efficiency1
                    ener2 = ener * efficiency2
                    rand_val = get_uniform_random_number(rng_states, idx)
                    if rand_val <= efficiency1 and ener1 > threshold:
                        theta = lut_fc1[i, lmd_num, m, n, 0]
                        phi = lut_fc1[i, lmd_num, m, n, 1]
                        norm = math.sqrt(Ete1*Ete1 + Etm1*Etm1)
                        Ete = Ete1/norm
                        Etm = Etm1/norm
                        delta_phase = delta_phase1 + lut_TIR[lmd_num, m, n, 0] 
                        gap_x = lut_gap[lmd_num, m, n, 0] 
                        gap_y = lut_gap[lmd_num, m, n, 1]
                        x += gap_x
                        y += gap_y
                        ener = ener1 * 1.0
                        region_state = 2
                    elif rand_val <= efficiency1+efficiency2 and ener2 > threshold:
                        theta = lut_fc2[i, lmd_num, m, n, 0]
                        phi = lut_fc2[i, lmd_num, m, n, 1]
                        norm = math.sqrt(Ete2*Ete2 + Etm2*Etm2)
                        Ete = Ete2/norm
                        Etm = Etm2/norm
                        delta_phase = delta_phase2 + lut_TIR[lmd_num, m, n, 1] 
                        gap_x = lut_gap[lmd_num, m, n, 2] 
                        gap_y = lut_gap[lmd_num, m, n, 3]
                        x += gap_x
                        y += gap_y
                        ener = ener2 * 1.0
                        region_state = 3
                    else:
                        return
                    break
            if not hit:
                x += gap_x
                y += gap_y
                delta_phase += 2*lut_TIR[lmd_num, m, n, 0]
        elif region_state == 3:
            hit = False
            for i in range(len(FC_offset)-1):
                start_FC = FC_offset[i]
                end_FC = FC_offset[i+1]
                if is_inside_or_on_edge(x, y, FC, start_FC, end_FC):
                    hit = True
                    Ete1, Etm1, delta_phase1 = E_field_cal(
                        Ete, Etm, delta_phase,
                        lut_fc2[i, lmd_num, m, n, 4], lut_fc2[i, lmd_num, m, n, 7],
                        lut_fc2[i, lmd_num, m, n, 16], lut_fc2[i, lmd_num, m, n, 19]) 
                    Ete2, Etm2, delta_phase2 = E_field_cal(
                        Ete, Etm, delta_phase,
                        lut_fc2[i, lmd_num, m, n, 3], lut_fc2[i, lmd_num, m, n, 6],
                        lut_fc2[i, lmd_num, m, n, 15], lut_fc2[i, lmd_num, m, n, 18]) 
                    efficiency1 = (Ete1 * Ete1 + Etm1 * Etm1) * math.cos(lut_fc1[i, lmd_num, m, n, 0].real) / math.cos(theta.real)
                    efficiency2 = (Ete2 * Ete2 + Etm2 * Etm2) * math.cos(lut_fc2[i, lmd_num, m, n, 0].real) / math.cos(theta.real)
                    ener1 = ener * efficiency1
                    ener2 = ener * efficiency2
                    rand_val = get_uniform_random_number(rng_states, idx)
                    if rand_val <= efficiency1 and ener1 > threshold:
                        theta = lut_fc1[i, lmd_num, m, n, 0]
                        phi = lut_fc1[i, lmd_num, m, n, 1]
                        norm = math.sqrt(Ete1*Ete1 + Etm1*Etm1)
                        Ete = Ete1/norm
                        Etm = Etm1/norm
                        delta_phase = delta_phase1 + lut_TIR[lmd_num, m, n, 0] 
                        gap_x = lut_gap[lmd_num, m, n, 0] 
                        gap_y = lut_gap[lmd_num, m, n, 1]
                        x += gap_x
                        y += gap_y
                        ener = ener1 * 1.0
                        region_state = 2
                    elif rand_val <= efficiency1+efficiency2 and ener2 > threshold:
                        theta = lut_fc2[i, lmd_num, m, n, 0]
                        phi = lut_fc2[i, lmd_num, m, n, 1]
                        norm = math.sqrt(Ete2*Ete2 + Etm2*Etm2)
                        Ete = Ete2/norm
                        Etm = Etm2/norm
                        delta_phase = delta_phase2 + lut_TIR[lmd_num, m, n, 1] 
                        gap_x = lut_gap[lmd_num, m, n, 2] 
                        gap_y = lut_gap[lmd_num, m, n, 3]
                        x += gap_x
                        y += gap_y
                        ener = ener2 * 1.0
                        region_state = 3
                    else:
                        return
                    break
            if not hit:
                if not is_inside_or_on_edge(x, y, eff_reg2, 0, eff_reg2.shape[0]):
                    region_state = 4
                else:
                    x += gap_x
                    y += gap_y
                    delta_phase += 2*lut_TIR[lmd_num, m, n, 1]

        elif region_state == 4: 
            hit = False
            for i in range(len(OC_offset)-1):
                start_OC = OC_offset[i]
                end_OC = OC_offset[i+1]
                if is_inside_or_on_edge(x, y, OC, start_OC, end_OC):
                    hit = True
                    Ete1, Etm1, delta_phase1 = E_field_cal(
                        Ete, Etm, delta_phase,
                        lut_oc1[i, lmd_num, m, n, 4],  lut_oc1[i, lmd_num, m, n, 9],
                        lut_oc1[i, lmd_num, m, n, 24], lut_oc1[i, lmd_num, m, n, 29]) 
                    Ete2, Etm2, delta_phase2 = E_field_cal(
                        Ete, Etm, delta_phase,
                        lut_oc1[i, lmd_num, m, n, 2], lut_oc1[i, lmd_num, m, n, 7],
                        lut_oc1[i, lmd_num, m, n, 22], lut_oc1[i, lmd_num, m, n, 27])
                    Ete3, Etm3, delta_phase3 = E_field_cal(
                        Ete, Etm, delta_phase,
                        lut_oc1[i, lmd_num, m, n, 13], lut_oc1[i, lmd_num, m, n, 18],
                        lut_oc1[i, lmd_num, m, n, 33], lut_oc1[i, lmd_num, m, n, 38])
                    efficiency1 = (Ete1 * Ete1 + Etm1 * Etm1) * math.cos(lut_oc1[i, lmd_num, m, n, 0].real) / math.cos(theta.real)
                    efficiency2 = (Ete2 * Ete2 + Etm2 * Etm2) * math.cos(lut_oc2[i, lmd_num, m, n, 0].real) / math.cos(theta.real)
                    efficiency3 = (Ete3 * Ete3 + Etm3 * Etm3) * math.cos(lut_ic1[lmd_num, m, n, 0].real) / math.cos(theta.real) / n_g
                    ener1 = ener * efficiency1
                    ener2 = ener * efficiency2
                    ener3 = ener * efficiency3
                    rand_val = get_uniform_random_number(rng_states, idx)
                    if rand_val <= efficiency1 and ener1 > threshold:
                        theta = lut_oc1[i, lmd_num, m, n, 0]
                        phi = lut_oc1[i, lmd_num, m, n, 1]
                        norm = math.sqrt(Ete1*Ete1 + Etm1*Etm1)
                        Ete = Ete1/norm
                        Etm = Etm1/norm
                        delta_phase = delta_phase1 + lut_TIR[lmd_num, m, n, 1] 
                        gap_x = lut_gap[lmd_num, m, n, 2] 
                        gap_y = lut_gap[lmd_num, m, n, 3]
                        x += gap_x
                        y += gap_y
                        ener = ener1 * 1.0
                        region_state = 4
                    elif rand_val <= efficiency1+efficiency2 and ener2 > threshold:
                        theta = lut_oc2[i, lmd_num, m, n, 0]
                        phi = lut_oc2[i, lmd_num, m, n, 2]
                        norm = math.sqrt(Ete2*Ete2 + Etm2*Etm2)
                        Ete = Ete2/norm
                        Etm = Etm2/norm
                        delta_phase = delta_phase2 + lut_TIR[lmd_num, m, n, 3] 
                        gap_x = lut_gap[lmd_num, m, n, 6] 
                        gap_y = lut_gap[lmd_num, m, n, 7]
                        x += gap_x
                        y += gap_y
                        ener = ener2 * 1.0
                        region_state = 5
                    elif rand_val <= efficiency1+efficiency2+efficiency3 and ener3 > threshold:
                        if is_inside_or_on_edge_4d(x, y, eff_reg_FOV, m, n):
                            xmin = eff_reg_FOV_range[m, n, 0]
                            xmax = eff_reg_FOV_range[m, n, 1]
                            ymin = eff_reg_FOV_range[m, n, 2]
                            ymax = eff_reg_FOV_range[m, n, 3]
                            add_to_EB_atomic_val(matrix_EB[lmd_num,:,:,:,:], m, n, x, y, xmin, xmax, ymin, ymax, 1.0)
                            return
                        else:
                            return
                    else:
                        return
                    break
            if not hit:
                x += gap_x
                y += gap_y
                delta_phase += 2*lut_TIR[lmd_num, m, n, 1]
        elif region_state == 5:
            hit = False
            for i in range(len(OC_offset)-1):
                start_OC = OC_offset[i]
                end_OC = OC_offset[i+1]
                if is_inside_or_on_edge(x, y, OC, start_OC, end_OC):
                    hit = True
                    Ete1, Etm1, delta_phase1 = E_field_cal(
                        Ete, Etm, delta_phase,
                        lut_oc2[i, lmd_num, m, n, 6], lut_oc2[i, lmd_num, m, n, 11],
                        lut_oc2[i, lmd_num, m, n, 26], lut_oc2[i, lmd_num, m, n, 31])
                    Ete2, Etm2, delta_phase2 = E_field_cal(
                        Ete, Etm, delta_phase,
                        lut_oc2[i, lmd_num, m, n, 4],  lut_oc2[i, lmd_num, m, n, 9],
                        lut_oc2[i, lmd_num, m, n, 24], lut_oc2[i, lmd_num, m, n, 29])
                    Ete3, Etm3, delta_phase3 = E_field_cal(
                        Ete, Etm, delta_phase,
                        lut_oc2[i, lmd_num, m, n, 15], lut_oc2[i, lmd_num, m, n, 20],
                        lut_oc2[i, lmd_num, m, n, 35], lut_oc2[i, lmd_num, m, n, 40])
                    efficiency1 = (Ete1 * Ete1 + Etm1 * Etm1) * math.cos(lut_oc1[i, lmd_num, m, n, 0].real) / math.cos(theta.real)
                    efficiency2 = (Ete2 * Ete2 + Etm2 * Etm2) * math.cos(lut_oc2[i, lmd_num, m, n, 0].real) / math.cos(theta.real)
                    efficiency3 = (Ete3 * Ete3 + Etm3 * Etm3) * math.cos(lut_ic1[lmd_num, m, n, 0].real) / math.cos(theta.real) / n_g
                    ener1 = ener * efficiency1
                    ener2 = ener * efficiency2
                    ener3 = ener * efficiency3
                    rand_val = get_uniform_random_number(rng_states, idx)
                    if rand_val <= efficiency1 and ener1 > threshold:
                        theta = lut_oc1[i, lmd_num, m, n, 0]
                        phi = lut_oc1[i, lmd_num, m, n, 1]
                        norm = math.sqrt(Ete1*Ete1 + Etm1*Etm1)
                        Ete = Ete1/norm
                        Etm = Etm1/norm
                        delta_phase = delta_phase1 + lut_TIR[lmd_num, m, n, 1] 
                        gap_x = lut_gap[lmd_num, m, n, 2]
                        gap_y = lut_gap[lmd_num, m, n, 3]
                        x += gap_x
                        y += gap_y
                        ener = ener1 * 1.0
                        region_state = 4
                    elif rand_val <= efficiency1+efficiency2 and ener2 > threshold:
                        theta = lut_oc2[i, lmd_num, m, n, 0]
                        phi = lut_oc2[i, lmd_num, m, n, 2]
                        norm = math.sqrt(Ete2*Ete2 + Etm2*Etm2)
                        Ete = Ete2/norm
                        Etm = Etm2/norm
                        delta_phase = delta_phase2 + lut_TIR[lmd_num, m, n, 3]
                        gap_x = lut_gap[lmd_num, m, n, 6] 
                        gap_y = lut_gap[lmd_num, m, n, 7]
                        x += gap_x
                        y += gap_y
                        ener = ener2 * 1.0
                        region_state = 5
                    elif rand_val <= efficiency1+efficiency2+efficiency3 and ener3 > threshold:
                        if is_inside_or_on_edge_4d(x, y, eff_reg_FOV, m, n):
                            xmin = eff_reg_FOV_range[m, n, 0]
                            xmax = eff_reg_FOV_range[m, n, 1]
                            ymin = eff_reg_FOV_range[m, n, 2]
                            ymax = eff_reg_FOV_range[m, n, 3]
                            add_to_EB_atomic_val(matrix_EB[lmd_num,:,:,:,:], m, n, x, y, xmin, xmax, ymin, ymax, 1.0)
                            return
                        else:
                            return
                    else:
                        return
                    break
            if not hit:
                return