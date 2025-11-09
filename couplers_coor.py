import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, MultiPolygon, LineString, GeometryCollection
from shapely.validation import make_valid
from shapely.ops import unary_union, polygonize
from shapely import affinity

deg = np.pi / 180

def filter_to_polygons(geom):
    if geom.is_empty:
        return Polygon()

    if isinstance(geom, (Polygon, MultiPolygon)):
        return geom

    if isinstance(geom, GeometryCollection):
        polys = [g for g in geom.geoms if isinstance(g, (Polygon, MultiPolygon))]
        if not polys:
            return Polygon()
        elif len(polys) == 1:
            return polys[0]
        else:
            return MultiPolygon(polys)

    # All other cases (LineString, Point, etc.)
    return Polygon()

def plot_polygons(geom, title="Polygon(s)"):
    fig, ax = plt.subplots()
    if geom.geom_type == 'Polygon':
        x, y = geom.exterior.xy
        ax.plot(x, y, color='blue')
    elif geom.geom_type == 'MultiPolygon':
        for poly in geom.geoms:
            x, y = poly.exterior.xy
            ax.plot(x, y, color='green')
    else:
        print(f"Unsupported geometry type: {geom.geom_type}")
    ax.set_title(title)
    ax.set_aspect('equal')
    plt.show()

def plot_filled_polygons(geom, title="Filled Polygon(s)", facecolor='skyblue', edgecolor='black'):
    fig, ax = plt.subplots()
    patches = []

    def add_polygon(p):
        # Exterior ring
        patches.append(MplPolygon(list(p.exterior.coords), closed=True))
        # Interior rings (holes)
        for interior in p.interiors:
            patches.append(MplPolygon(list(interior.coords), closed=True, fill=True))

    if isinstance(geom, Polygon):
        add_polygon(geom)
    elif isinstance(geom, MultiPolygon):
        for poly in geom.geoms:
            add_polygon(poly)
    else:
        print("Unsupported geometry type:", geom.geom_type)
        return

    p_collection = PatchCollection(patches, facecolor=facecolor, edgecolor=edgecolor, linewidth=1)
    ax.add_collection(p_collection)
    ax.autoscale()
    ax.set_aspect('equal')
    ax.set_title(title)
    plt.show()

def overlap_FOV(polygon1, polygon2):
    if not polygon1.is_valid:
        polygon1 = make_valid(polygon1)
        polygon1 = filter_to_polygons(polygon1)
    if not polygon2.is_valid:
        polygon2 = make_valid(polygon2)
        polygon2 = filter_to_polygons(polygon2)

    # Compute and sanitize the overlap region
    raw_overlap = polygon1.intersection(polygon2)
    overlap_region = filter_to_polygons(raw_overlap)

    # Compute and sanitize the modified polygon
    if overlap_region.is_empty:
        modified_polygon = polygon2
    else:
        raw_modified = polygon2.difference(overlap_region)
        modified_polygon = filter_to_polygons(raw_modified)

    return overlap_region, modified_polygon

def polygon_to_xy(polygon, xs_list, ys_list):
    if polygon.is_empty:
        return

    if isinstance(polygon, Polygon):
        x, y = polygon.exterior.xy
        xs_list.append(x)  # x is already an array
        ys_list.append(y)

    elif isinstance(polygon, MultiPolygon):
        for poly in polygon.geoms:
            if not poly.is_empty:
                x, y = poly.exterior.xy
                xs_list.append(x)
                ys_list.append(y)

def count_polygons(geometry):
    if geometry.is_empty:
        return 0
    elif isinstance(geometry, Polygon):
        return 1
    elif isinstance(geometry, MultiPolygon):
        return len(geometry.geoms)
    else:
        raise TypeError("Input is not a Polygon or MultiPolygon.")

def couplers_coor_full_color(num_FOV_x = 120, num_FOV_y = 80):
    """## Design goal of waveguide"""
    # Field of View (FoV)
    AR = 4 / 3  # aspect ratio
    FoV_x = 18 * deg  # AR=4/3; FoV_x=40*deg;
    FoV_y = FoV_x / AR
    FoV_X = np.linspace(-FoV_x / 2, FoV_x / 2, 50)  # coordinates for each FoV point
    FoV_Y = np.linspace(-FoV_y / 2, FoV_y / 2, 50)

    # Wavelength
    lmd = np.array([465, 532, 630])
    k0 = 2 * np.pi / lmd

    # Substrate parameters
    n_g = 1.9  # index of glass
    n_air = 1
    x = 60
    y = 50  # glass size
    t = 0.7  # thickness of the waveguide substrate

    # number of folding and out couplers
    num_FC = 7
    num_OC = 6

    # Input pupil size (circular) in mm
    r = 2  # radius

    # Input pupil position
    x_ic0 = -28  # coordinates of center of input coupler/pupil
    y_ic0 = 15
    n = 100  # number of points
    t_ic = np.linspace(0, 2 * np.pi, n)
    X_ic = x_ic0 + r * np.sin(t_ic)
    Y_ic = y_ic0 + r * np.cos(t_ic)

    # Eyebox size (mm)
    x_eb = 12
    y_eb = 8
    er = -20  # same side or different side for LE and HE have same folding

    # Eyebox position (mm)
    x_eb0 = 0  # x_eb0 = 13
    y_eb0 = 15
    X_eb = np.array([-x_eb / 2, -x_eb / 2, x_eb / 2, x_eb / 2]) + x_eb0
    Y_eb = np.array([-y_eb / 2, y_eb / 2, y_eb / 2, -y_eb / 2]) + y_eb0

    # Out-coupler size (mm)
    x_oc = np.tan(FoV_x / 2) * abs(er) * 2 + x_eb
    y_oc = np.tan(FoV_y / 2) * abs(er) * 2 + y_eb

    # Out-coupler position
    X_oc = np.array([-x_oc / 2, -x_oc / 2, x_oc / 2, x_oc / 2]) + x_eb0
    Y_oc = np.array([-y_oc / 2, y_oc / 2, y_oc / 2, -y_oc / 2]) + y_eb0

    # Maximum and minimum TIR angle for design
    theta_min = np.arcsin(n_air / n_g)  # minimum TIR angle
    theta_max = np.arctan(r / t)        # maximum TIR angle if pupil is continuous at largest TIR angle

    """## Checking periods of input and output coupler for designed FoV"""

    # Horizontal period and k_g direction of input coupler
    Lambda_ic = 388
    phi_ic = -38 * deg

    # Horizontal period and k_g direction of output coupler
    Lambda_oc = 388
    phi_oc = -142 * deg

    # Incoupler k vector
    kg_ic = 2 * np.pi / Lambda_ic
    kgx_ic = kg_ic * np.cos(phi_ic)
    kgy_ic = kg_ic * np.sin(phi_ic)

    # Reverse direction of outcoupler k vector
    kg_oc = 2 * np.pi / Lambda_oc
    kgx_oc = kg_oc * np.cos(phi_oc + 180 * deg)
    kgy_oc = kg_oc * np.sin(phi_oc + 180 * deg)

    """## Calculating period and shape of folding couplers"""

    # k-vector and horizontal period of folding coupler
    kgx_fc = kgx_oc - kgx_ic
    kgy_fc = kgy_oc - kgy_ic
    Lambda_fc = 2 * np.pi / np.sqrt(kgx_fc**2 + kgy_fc**2)
    # print(Lambda_fc)
    phi_fc = np.arctan2(kgy_fc, kgx_fc)  # use arctan2 for stability

    kk = 0

    # Preallocate result arrays
    kx0 = np.zeros((len(lmd), len(FoV_X)*len(FoV_Y)))
    ky0 = np.zeros((len(lmd), len(FoV_X)*len(FoV_Y)))
    kx_ic = np.zeros_like(kx0)
    ky_ic = np.zeros_like(ky0)
    kx_fc = np.zeros_like(kx0)
    ky_fc = np.zeros_like(ky0)
    x_f = []
    y_f = []

    
    for ii in range(len(FoV_X)):
        for jj in range(len(FoV_Y)):
            for num_lmd in range(len(lmd)):
                # k-vector in air
                th_inc = np.arctan(np.sqrt(np.tan(FoV_X[ii])**2 + np.tan(FoV_Y[jj])**2))
                phi_inc = np.arctan2(np.tan(FoV_Y[jj]), np.tan(FoV_X[ii]))
                kx0[num_lmd,kk] = n_air * k0[num_lmd] * np.sin(th_inc) * np.cos(phi_inc)
                ky0[num_lmd,kk] = n_air * k0[num_lmd] * np.sin(th_inc) * np.sin(phi_inc)

                # k-vector after incoupler
                kx_ic[num_lmd,kk] = kx0[num_lmd,kk] + kgx_ic
                ky_ic[num_lmd,kk] = ky0[num_lmd,kk] + kgy_ic
                kz_ic = np.sqrt(k0[num_lmd]**2 * n_g**2 - kx_ic[num_lmd,kk]**2 - ky_ic[num_lmd,kk]**2)

                # tangent lines of input pupil for different FoV after input coupler
                k1 = ky_ic[num_lmd,kk] / kx_ic[num_lmd,kk]
                b11 = y_ic0 - k1 * x_ic0 + r * np.sqrt(1 + k1**2)
                b12 = y_ic0 - k1 * x_ic0 - r * np.sqrt(1 + k1**2)

                # k-vector after folding
                kx_fc[num_lmd,kk] = kx_ic[num_lmd,kk] + kgx_fc
                ky_fc[num_lmd,kk] = ky_ic[num_lmd,kk] + kgy_fc
                kz_fc = np.sqrt(k0[num_lmd]**2 * n_g**2 - kx_fc[num_lmd,kk]**2 - ky_fc[num_lmd,kk]**2)

                # position of each edge
                dx = er * np.tan(th_inc) * np.cos(phi_inc)
                dy = er * np.tan(th_inc) * np.sin(phi_inc)

                x_ed_left_t = x_eb0 - x_eb / 2 + dx
                y_ed_left_t = y_eb0 + y_eb / 2 + dy
                x_ed_right_b = x_eb0 + x_eb / 2 + dx
                y_ed_right_b = y_eb0 - y_eb / 2 + dy
                x_ed_left_b = x_eb0 - x_eb / 2 + dx
                y_ed_left_b = y_eb0 - y_eb / 2 + dy
                x_ed_right_t = x_eb0 + x_eb / 2 + dx
                y_ed_right_t = y_eb0 + y_eb / 2 + dy

                # tangent lines of output coupler for different FoV after folding coupler
                k2 = ky_fc[num_lmd,kk] / kx_fc[num_lmd,kk]
                if k2 <= 0:
                    b21 = y_ed_left_b - k2 * x_ed_left_b
                    b22 = y_ed_right_t - k2 * x_ed_right_t
                else:
                    b21 = y_ed_left_t - k2 * x_ed_left_t
                    b22 = y_ed_right_b - k2 * x_ed_right_b

                # intersection points of two lines (x = (b2 - b1)/(k1 - k2))
                for b1 in [b11, b12]:
                    for b2 in [b22, b21]:
                        x_inter = (b2 - b1) / (k1 - k2)
                        y_inter = k1 * x_inter + b1
                        x_f.append(x_inter)
                        y_f.append(y_inter)
            kk += 1

    """## Generate 9 FoV for optimization"""

    # Generate 9 FoVs
    FoV_X_9c = np.array([
        -FoV_x / 2, np.finfo(float).eps, FoV_x / 2,
        -FoV_x / 2, np.finfo(float).eps, FoV_x / 2,
        FoV_x / 2, np.finfo(float).eps, -FoV_x / 2
    ])
    FoV_Y_9c = np.array([
        FoV_y / 2, FoV_y / 2, FoV_y / 2,
        np.finfo(float).eps, np.finfo(float).eps, np.finfo(float).eps,
        -FoV_y / 2, -FoV_y / 2, -FoV_y / 2
    ])

    FoV_x_600c = np.linspace(-FoV_x / 2, FoV_x / 2, num_FOV_x)
    FoV_y_600c = np.linspace(-FoV_y / 2, FoV_y / 2, num_FOV_y)
    FoV_X_600c, FoV_Y_600c = np.meshgrid(FoV_x_600c, FoV_y_600c, indexing='ij')

    # Compute convex hull of the folded coupling region
    x_coor_all = []
    y_coor_all = []
    x_f = np.array(x_f)
    y_f = np.array(y_f)
    points_fc = np.vstack((x_f, y_f)).T
    hull_fc = ConvexHull(points_fc)
    bd = hull_fc.vertices
    x_coor_all.extend(x_f[bd])
    y_coor_all.extend(y_f[bd])

    # Rotate the folding region
    points = np.vstack((x_f, y_f))[:, bd]
    angle = np.pi / 2 + phi_ic
    rotation_2d = np.array([[np.cos(angle), np.sin(angle)],
                [-np.sin(angle), np.cos(angle)]])
    rotated_points = rotation_2d @ points

    # Define slicing region
    start_line = np.max(rotated_points[1, :])
    end_line = np.min(rotated_points[1, :])
    slice_width = (start_line - end_line)/(num_FC+ 0.001)
    num_slices = int(np.ceil((start_line - end_line) / slice_width))
    end_width = (start_line - end_line) % slice_width
    if end_width < slice_width / 4:
        num_slices -= 1

    # Preallocate arrays for FoV shape
    x_fc_FOV = np.zeros((len(FoV_Y_9c)*len(lmd), 4))
    y_fc_FOV = np.zeros((len(FoV_Y_9c)*len(lmd), 4))
    kx0_9c = np.zeros(len(FoV_Y_9c)*len(lmd))
    ky0_9c = np.zeros(len(FoV_Y_9c)*len(lmd))

    # Compute shape for each 9 FoV region
    for ii in range(len(FoV_Y_9c)):
        for num_lmd in range(len(lmd)):
            th_inc = np.arctan(np.sqrt(np.tan(FoV_X_9c[ii])**2 + np.tan(FoV_Y_9c[ii])**2))
            phi_inc = np.arctan2(np.tan(FoV_Y_9c[ii]), np.tan(FoV_X_9c[ii]))

            kx0_9c[ii] = n_air * k0[num_lmd] * np.sin(th_inc) * np.cos(phi_inc)
            ky0_9c[ii] = n_air * k0[num_lmd] * np.sin(th_inc) * np.sin(phi_inc)

            kx_ic_ = kx0_9c[ii] + kgx_ic
            ky_ic_ = ky0_9c[ii] + kgy_ic
            kz_ic = np.sqrt(k0[num_lmd]**2 * n_g**2 - kx_ic_**2 - ky_ic_**2)

            k1 = ky_ic_ / kx_ic_
            b11 = y_ic0 - k1 * x_ic0 + r * np.sqrt(1 + k1**2)
            b12 = y_ic0 - k1 * x_ic0 - r * np.sqrt(1 + k1**2)

            kx_fc_ = kx_ic_ + kgx_fc
            ky_fc_ = ky_ic_ + kgy_fc
            kz_fc = np.sqrt(k0[num_lmd]**2 * n_g**2 - kx_fc_**2 - ky_fc_**2)

            dx = er * np.tan(th_inc) * np.cos(phi_inc)
            dy = er * np.tan(th_inc) * np.sin(phi_inc)

            x_ed_l_t = x_eb0 - x_eb / 2 + dx
            y_ed_l_t = y_eb0 + y_eb / 2 + dy
            x_ed_r_b = x_eb0 + x_eb / 2 + dx
            y_ed_r_b = y_eb0 - y_eb / 2 + dy
            x_ed_l_b = x_eb0 - x_eb / 2 + dx
            y_ed_l_b = y_eb0 - y_eb / 2 + dy
            x_ed_r_t = x_eb0 + x_eb / 2 + dx
            y_ed_r_t = y_eb0 + y_eb / 2 + dy

            k2 = ky_fc_ / kx_fc_
            if k2 <= 0:
                b21 = y_ed_l_b - k2 * x_ed_l_b
                b22 = y_ed_r_t - k2 * x_ed_r_t
            else:
                b21 = y_ed_l_t - k2 * x_ed_l_t
                b22 = y_ed_r_b - k2 * x_ed_r_b

            x_fc_FOV[len(lmd)*ii+num_lmd, 0] = (b22 - b11) / (k1 - k2)
            x_fc_FOV[len(lmd)*ii+num_lmd, 1] = (b21 - b11) / (k1 - k2)
            x_fc_FOV[len(lmd)*ii+num_lmd, 2] = (b21 - b12) / (k1 - k2)
            x_fc_FOV[len(lmd)*ii+num_lmd, 3] = (b22 - b12) / (k1 - k2)

            y_fc_FOV[len(lmd)*ii+num_lmd, 0] = k1 * x_fc_FOV[len(lmd)*ii+num_lmd, 0] + b11
            y_fc_FOV[len(lmd)*ii+num_lmd, 1] = k1 * x_fc_FOV[len(lmd)*ii+num_lmd, 1] + b11
            y_fc_FOV[len(lmd)*ii+num_lmd, 2] = k1 * x_fc_FOV[len(lmd)*ii+num_lmd, 2] + b12
            y_fc_FOV[len(lmd)*ii+num_lmd, 3] = k1 * x_fc_FOV[len(lmd)*ii+num_lmd, 3] + b12

    # Store valid slice polygons
    points_x_FOV_fc = []
    points_y_FOV_fc = []
    effective_x_FOV_fc = []
    effective_y_FOV_fc = []
    for i in range(len(FoV_Y_9c)*len(lmd)):
        x_coor = np.hstack((x_fc_FOV[i, :],X_ic))
        y_coor = np.hstack((y_fc_FOV[i, :],Y_ic))
        points_eff = np.vstack((x_coor, y_coor)).T
        hull_eff = ConvexHull(points_eff)
        bd = hull_eff.vertices
        x_coor_all.extend(x_coor[bd].flatten())
        y_coor_all.extend(y_coor[bd].flatten())
        poly_FOV_fc = Polygon(np.column_stack((x_fc_FOV[i, :], y_fc_FOV[i, :])))
        poly_FOV_fc = make_valid(poly_FOV_fc)
        polygon_to_xy(poly_FOV_fc,effective_x_FOV_fc,effective_y_FOV_fc)

    x_coor_eff_reg_2 = np.array(x_coor_all)
    y_coor_eff_reg_2 = np.array(y_coor_all)
    bd = ConvexHull(np.column_stack((x_coor_eff_reg_2, y_coor_eff_reg_2))).vertices
    eff_reg2_x = x_coor_eff_reg_2[bd]
    eff_reg2_y = y_coor_eff_reg_2[bd]
    coords = np.column_stack((eff_reg2_x, eff_reg2_y))
    line = LineString(coords)
    simplified_line = line.simplify(tolerance=1e-3)
    eff_reg2_x, eff_reg2_y = simplified_line.xy

    # plt.fill(eff_reg2_x, eff_reg2_y, 'r', alpha=0.3, edgecolor='black')

    for i in range(1, num_slices + 1):
        # Current slice y-bounds in rotated coordinates
        col_start = start_line - (i - 1) * slice_width
        col_end = start_line - i * slice_width

        points_x = rotated_points[0, :]
        points_y = rotated_points[1, :]

        # Polygon of the current folded region
        poly_fc = Polygon(np.column_stack((points_x, points_y)))

        # Clip polygon between the top and bottom lines
        if i == num_slices:
            band = Polygon([
                (-x, col_start), (x, col_start),
                (x, end_line), (-x, end_line)
            ])
        else:
            band = Polygon([
                (-x, col_start), (x, col_start),
                (x, col_end), (-x, col_end)
            ])

        clipped = poly_fc.intersection(band)
        if clipped.is_empty:
            continue

        # Convert back to original coordinates
        if clipped.geom_type == 'Polygon':
            coords = np.array(clipped.exterior.coords).T
        elif clipped.geom_type == 'MultiPolygon':
            coords = np.array(list(clipped.geoms[0].exterior.coords)).T
        else:
            continue

        # Rotate back
        inv_rotation = np.array([[np.cos(angle), -np.sin(angle)],
                    [np.sin(angle),  np.cos(angle)]])
        restored = inv_rotation @ coords
        px_restored, py_restored = restored[0, :], restored[1, :]
        p_restored_poly = Polygon(np.column_stack((px_restored, py_restored)))
        p_restored_poly = make_valid(p_restored_poly)

        points_x_FOV_fc.append(px_restored)
        points_y_FOV_fc.append(py_restored)

    # === Rotate out-coupler polygon ===
    points_oc = np.vstack((X_oc, Y_oc)).T
    hull_oc = ConvexHull(points_oc)
    bd = hull_oc.vertices
    points = np.vstack((X_oc[bd], Y_oc[bd]))

    # Rotate the out-coupler region
    angle_oc = 3 * np.pi / 2 + phi_oc
    rotation_2d_oc = np.array([
        [np.cos(angle_oc), np.sin(angle_oc)],
        [-np.sin(angle_oc), np.cos(angle_oc)]
    ])
    rotated_oc = rotation_2d_oc @ points

    # Define slicing parameters
    start_line = np.max(rotated_oc[1, :])
    end_line = np.min(rotated_oc[1, :])
    slice_width = (start_line - end_line)/(num_OC + 0.001)
    num_slices = int(np.ceil((start_line - end_line) / slice_width))
    end_width = (start_line - end_line) % slice_width
    if end_width < slice_width / 4:
        num_slices -= 1

    # Effective out-coupler region polygons
    x_oc_FOV = np.zeros((len(FoV_Y_9c), 4))
    y_oc_FOV = np.zeros((len(FoV_Y_9c), 4))
    for j in range(len(FoV_Y_9c)):
        th_inc = np.arctan(np.sqrt(np.tan(FoV_X_9c[j])**2 + np.tan(FoV_Y_9c[j])**2))
        phi_inc = np.arctan2(np.tan(FoV_Y_9c[j]), np.tan(FoV_X_9c[j]))

        dx = er * np.tan(th_inc) * np.cos(phi_inc)
        dy = er * np.tan(th_inc) * np.sin(phi_inc)

        x_oc_FOV[j, :] = np.array([
            x_eb0 - x_eb/2 + dx,
            x_eb0 - x_eb/2 + dx,
            x_eb0 + x_eb/2 + dx,
            x_eb0 + x_eb/2 + dx
        ])

        y_oc_FOV[j, :] = np.array([
            y_eb0 + y_eb/2 + dy,
            y_eb0 - y_eb/2 + dy,
            y_eb0 - y_eb/2 + dy,
            y_eb0 + y_eb/2 + dy
        ])

    # Effective out-coupler region polygons
    x_oc_FOV_600c = np.zeros((len(FoV_x_600c), len(FoV_y_600c), 4))
    y_oc_FOV_600c = np.zeros((len(FoV_x_600c), len(FoV_y_600c), 4))
    x_oc_FOV_600c_max_min = np.zeros((len(FoV_x_600c), len(FoV_y_600c), 2))
    y_oc_FOV_600c_max_min = np.zeros((len(FoV_x_600c), len(FoV_y_600c), 2))
    for i in range(len(FoV_x_600c)):
        for j in range(len(FoV_y_600c)):
            th_inc = np.arctan(np.sqrt(np.tan(FoV_X_600c[i, j])**2 + np.tan(FoV_Y_600c[i, j])**2))
            phi_inc = np.arctan2(np.tan(FoV_Y_600c[i, j]), np.tan(FoV_X_600c[i, j]))

            dx = er * np.tan(th_inc) * np.cos(phi_inc)
            dy = er * np.tan(th_inc) * np.sin(phi_inc)

            x_oc_FOV_600c[i, j, :] = np.array([
                x_eb0 - x_eb/2 + dx,
                x_eb0 - x_eb/2 + dx,
                x_eb0 + x_eb/2 + dx,
                x_eb0 + x_eb/2 + dx
            ])

            y_oc_FOV_600c[i, j, :] = np.array([
                y_eb0 + y_eb/2 + dy,
                y_eb0 - y_eb/2 + dy,
                y_eb0 - y_eb/2 + dy,
                y_eb0 + y_eb/2 + dy
            ])

            x_oc_FOV_600c_max_min[i, j, 0] = x_eb0 - x_eb/2 + dx
            x_oc_FOV_600c_max_min[i, j, 1] = x_eb0 + x_eb/2 + dx
            
            y_oc_FOV_600c_max_min[i, j, 0] = y_eb0 - y_eb/2 + dy
            y_oc_FOV_600c_max_min[i, j, 1] = y_eb0 + y_eb/2 + dy

    # Highlight combined region
    points_x_FOV_oc = []
    points_y_FOV_oc = []
    
    for i in range(len(FoV_Y_9c)):
        for num_lmd in range(len(lmd)):
            effective_x = np.concatenate([x_oc_FOV[i, :], x_fc_FOV[i*len(lmd)+num_lmd, :]])
            effective_y = np.concatenate([y_oc_FOV[i, :], y_fc_FOV[i*len(lmd)+num_lmd, :]])
            bd = ConvexHull(np.column_stack((effective_x, effective_y))).vertices
            x_coor_all.extend(effective_x[bd].flatten())
            y_coor_all.extend(effective_y[bd].flatten())

    x_coor_all = np.array(x_coor_all)
    y_coor_all = np.array(y_coor_all)
    bd = ConvexHull(np.column_stack((x_coor_all, y_coor_all))).vertices
    effective_x = x_coor_all[bd]
    effective_y = y_coor_all[bd]
    coords = np.column_stack((effective_x, effective_y))
    line = LineString(coords)
    simplified_line = line.simplify(tolerance=1e-3)
    effective_x, effective_y = simplified_line.xy
    # plt.fill(effective_x, effective_y, 'r', alpha=0.3, edgecolor='black')

    # === Slice and intersect out-coupler ===
    for i in range(1, num_slices + 1):
        col_start = start_line - (i - 1) * slice_width
        col_end = start_line - i * slice_width

        px = rotated_oc[0, :]
        py = rotated_oc[1, :]

        # Clip polygon between the top and bottom lines
        if i == num_slices:
            slice_band = Polygon([
                (-x, col_start), (x, col_start),
                (x, end_line), (-x, end_line)
            ])
        else:
            slice_band = Polygon([
                (-x, col_start), (x, col_start),
                (x, col_end), (-x, col_end)
            ])

        poly_oc = Polygon(np.column_stack((px, py)))
        clipped = poly_oc.intersection(slice_band)
        if clipped.is_empty:
            continue

        # Handle multi-polygon case
        if clipped.geom_type == 'Polygon':
            coords = np.array(clipped.exterior.coords).T
        elif clipped.geom_type == 'MultiPolygon':
            coords = np.array(list(clipped.geoms[0].exterior.coords)).T
        else:
            continue

        # Rotate back
        inv_rotation_oc = np.array([
            [np.cos(angle_oc), -np.sin(angle_oc)],
            [np.sin(angle_oc),  np.cos(angle_oc)]
        ])
        restored = inv_rotation_oc @ coords
        px_restored, py_restored = restored[0, :], restored[1, :]
        p_restored_poly = Polygon(np.column_stack((px_restored, py_restored)))

        points_x_FOV_oc.append(px_restored)
        points_y_FOV_oc.append(py_restored)

    # f_proj = 10 # define the projection lens focal length
    # x_dp = f_proj * np.tan(FoV_x/2)
    # y_dp = f_proj * np.tan(FoV_y/2)
    # X_dp = np.linspace(-x_dp, x_dp, 30)
    # Y_dp = np.linspace(-y_dp, y_dp, 20)

    # FoV_X_9c = np.arctan(X_dp / f_proj)  # coordinates for each FoV point
    # FoV_Y_9c = np.arctan(Y_dp / f_proj)
    # FoV_X_9c_grid, FoV_Y_9c_grid = np.meshgrid(FoV_X_9c, FoV_Y_9c, indexing='ij')
    # FoV_X_9c = FoV_X_9c_grid.flatten()
    # FoV_Y_9c = FoV_Y_9c_grid.flatten()

    th_in_ic = np.zeros((len(lmd), len(FoV_x_600c), len(FoV_y_600c)))
    phi_in_ic = np.zeros((len(lmd), len(FoV_x_600c), len(FoV_y_600c)))
    th_out_ic = np.zeros((len(lmd), len(FoV_x_600c), len(FoV_y_600c)))
    phi_out_ic = np.zeros((len(lmd), len(FoV_x_600c), len(FoV_y_600c)))
    th_out_ic2 = np.zeros((len(lmd), len(FoV_x_600c), len(FoV_y_600c)))
    phi_out_ic2 = np.zeros((len(lmd), len(FoV_x_600c), len(FoV_y_600c)))
    th_out_fc = np.zeros((len(lmd), len(FoV_x_600c), len(FoV_y_600c)))
    phi_out_fc = np.zeros((len(lmd), len(FoV_x_600c), len(FoV_y_600c)))
    th_out_oc = np.zeros((len(lmd), len(FoV_x_600c), len(FoV_y_600c)))
    phi_out_oc = np.zeros((len(lmd), len(FoV_x_600c), len(FoV_y_600c)))
    th_out_oc_glow = np.zeros((len(lmd), len(FoV_x_600c), len(FoV_y_600c)))
    lut_gap = np.zeros((len(lmd), len(FoV_x_600c), len(FoV_y_600c), 8))
    lut_TIR = np.zeros((len(lmd), len(FoV_x_600c), len(FoV_y_600c), 4))
    lut_Fresnel = np.zeros((len(FoV_x_600c), len(FoV_y_600c), 4))

    for num_lmd in range(len(lmd)):
        for i in range(len(FoV_x_600c)):
            for j in range(len(FoV_y_600c)):
                th_in_ic[num_lmd, i, j] = np.arctan(np.sqrt(np.tan(FoV_X_600c[i, j])**2 + np.tan(FoV_Y_600c[i, j])**2))
                phi_in_ic[num_lmd, i, j] = np.arctan2(np.tan(FoV_Y_600c[i, j]), np.tan(FoV_X_600c[i, j]))

                # k-vector in air
                kx = n_air * k0[num_lmd] * np.sin(th_in_ic[num_lmd, i, j]) * np.cos(phi_in_ic[num_lmd, i, j])
                ky = n_air * k0[num_lmd] * np.sin(th_in_ic[num_lmd, i, j]) * np.sin(phi_in_ic[num_lmd, i, j])

                # center k-vector block polar angle in glass & te tm reflection
                th_glass = np.arcsin(np.sin(th_in_ic[num_lmd, i, j])/n_g)
                th_out_oc_glow[num_lmd, i, j] = th_glass
                r_TE = (n_g*np.cos(th_glass)-np.cos(th_in_ic[num_lmd, i, j]))/(n_g*np.cos(th_glass)+np.cos(th_in_ic[num_lmd, i, j]))
                r_TM = (np.cos(th_glass)-n_g*np.cos(th_in_ic[num_lmd, i, j]))/(np.cos(th_glass)+n_g*np.cos(th_in_ic[num_lmd, i, j]))
                lut_Fresnel[i, j, 0] = r_TE
                lut_Fresnel[i, j, 1] = r_TM
                lut_Fresnel[i, j, 2] = 2 * t  * np.tan(th_glass) * np.cos(phi_in_ic[num_lmd, i, j])
                lut_Fresnel[i, j, 3] = 2 * t  * np.tan(th_glass) * np.cos(phi_in_ic[num_lmd, i, j])

                # after input coupler (for opposite FC)
                kxg_ic = kx - kgx_ic
                kyg_ic = ky - kgy_ic
                kzg_ic = np.sqrt(k0[num_lmd]**2 * n_g**2 - kxg_ic**2 - kyg_ic**2)
                th_out_ic2[num_lmd, i, j] = np.arctan(np.sqrt((kxg_ic**2 + kyg_ic**2) / kzg_ic**2))
                phi_out_ic2[num_lmd, i, j] = np.arctan2(kyg_ic, kxg_ic)

                lut_gap[num_lmd, i, j, 4] = 2 * t * np.tan(th_out_ic2[num_lmd, i, j]) * np.cos(phi_out_ic2[num_lmd, i, j])
                lut_gap[num_lmd, i, j, 5] = 2 * t * np.tan(th_out_ic2[num_lmd, i, j]) * np.sin(phi_out_ic2[num_lmd, i, j])

                # after input coupler
                kxg_ic = kx + kgx_ic
                kyg_ic = ky + kgy_ic
                kzg_ic = np.sqrt(k0[num_lmd]**2 * n_g**2 - kxg_ic**2 - kyg_ic**2)
                th_out_ic[num_lmd, i, j] = np.arctan(np.sqrt((kxg_ic**2 + kyg_ic**2) / kzg_ic**2))
                phi_out_ic[num_lmd, i, j] = np.arctan2(kyg_ic, kxg_ic)

                lut_gap[num_lmd, i, j, 0] = 2 * t * np.tan(th_out_ic[num_lmd, i, j]) * np.cos(phi_out_ic[num_lmd, i, j])
                lut_gap[num_lmd, i, j, 1] = 2 * t * np.tan(th_out_ic[num_lmd, i, j]) * np.sin(phi_out_ic[num_lmd, i, j])

                # after folding-coupler
                kxg_fc = kxg_ic + kgx_fc
                kyg_fc = kyg_ic + kgy_fc
                kzg_fc = np.sqrt(k0[num_lmd]**2 * n_g**2 - kxg_fc**2 - kyg_fc**2)
                th_out_fc[num_lmd, i, j] = np.arctan(np.sqrt((kxg_fc**2 + kyg_fc**2) / kzg_fc**2))
                phi_out_fc[num_lmd, i, j] = np.arctan2(kyg_fc, kxg_fc)

                # after out-coupler
                kxg_oc = kxg_fc - 2*kgx_oc
                kyg_oc = kyg_fc - 2*kgy_oc
                kzg_oc = np.sqrt(k0[num_lmd]**2 * n_g**2 - kxg_oc**2 - kyg_oc**2)
                th_out_oc[num_lmd, i, j] = np.arctan(np.sqrt((kxg_oc**2 + kyg_oc**2) / kzg_oc**2))
                phi_out_oc[num_lmd, i, j] = np.arctan2(kyg_oc, kxg_oc)

                lut_gap[num_lmd, i, j, 2] = 2 * t * np.tan(th_out_fc[num_lmd, i, j]) * np.cos(phi_out_fc[num_lmd, i, j])
                lut_gap[num_lmd, i, j, 3] = 2 * t * np.tan(th_out_fc[num_lmd, i, j]) * np.sin(phi_out_fc[num_lmd, i, j])

                lut_gap[num_lmd, i, j, 6] = 2 * t * np.tan(th_out_oc[num_lmd, i, j]) * np.cos(phi_out_oc[num_lmd, i, j])
                lut_gap[num_lmd, i, j, 7] = 2 * t * np.tan(th_out_oc[num_lmd, i, j]) * np.sin(phi_out_oc[num_lmd, i, j])

                term = n_g**2 * np.sin(th_out_ic[num_lmd, i, j])**2 - 1
                sqrt_term = np.sqrt(term)
                delta_s = 2 * np.arctan(sqrt_term / (n_g * np.cos(th_out_ic[num_lmd, i, j])))
                delta_p = 2 * np.arctan(n_g * sqrt_term / np.cos(th_out_ic[num_lmd, i, j]))
                lut_TIR[num_lmd, i, j, 0] = delta_s - delta_p

                term = n_g**2 * np.sin(th_out_fc[num_lmd, i, j])**2 - 1
                sqrt_term = np.sqrt(term)
                delta_s = 2 * np.arctan(sqrt_term / (n_g * np.cos(th_out_fc[num_lmd, i, j])))
                delta_p = 2 * np.arctan(n_g * sqrt_term / np.cos(th_out_fc[num_lmd, i, j]))
                lut_TIR[num_lmd, i, j, 1] = delta_s - delta_p

                term = n_g**2 * np.sin(th_out_ic2[num_lmd, i, j])**2 - 1
                sqrt_term = np.sqrt(term)
                delta_s = 2 * np.arctan(sqrt_term / (n_g * np.cos(th_out_ic2[num_lmd, i, j])))
                delta_p = 2 * np.arctan(n_g * sqrt_term / np.cos(th_out_ic2[num_lmd, i, j]))
                lut_TIR[num_lmd, i, j, 2] = delta_s - delta_p

                term = n_g**2 * np.sin(th_out_oc[num_lmd, i, j])**2 - 1
                sqrt_term = np.sqrt(term)
                delta_s = 2 * np.arctan(sqrt_term / (n_g * np.cos(th_out_oc[num_lmd, i, j])))
                delta_p = 2 * np.arctan(n_g * sqrt_term / np.cos(th_out_oc[num_lmd, i, j]))
                lut_TIR[num_lmd, i, j, 3] = delta_s - delta_p
    
    # generate IC coordinates
    IC = np.stack((X_ic, Y_ic), axis=1)

    # generate FC coordinates and offset
    FC_x = np.concatenate(points_x_FOV_fc)
    FC_y = np.concatenate(points_y_FOV_fc)
    FC = np.stack((FC_x, FC_y), axis=1)
    lengths = [len(arr) for arr in points_x_FOV_fc]
    FC_offset = np.cumsum([0] + lengths)

    # generate OC coordinates and offset
    OC_x = np.concatenate(points_x_FOV_oc)
    OC_y = np.concatenate(points_y_FOV_oc)
    OC = np.stack((OC_x, OC_y), axis=1)
    lengths = [len(arr) for arr in points_x_FOV_oc]
    OC_offset = np.cumsum([0] + lengths)

    # generate the whole effective region
    eff_reg1 = np.stack((effective_x, effective_y), axis=1)

    # generate effective region for IC and FC
    eff_reg2 = np.stack((eff_reg2_x, eff_reg2_y), axis=1)

    # generate effective out-coupled region and the range (min-max) for 600 FoVS
    eff_reg_FOV = np.stack((x_oc_FOV_600c, y_oc_FOV_600c), axis=-1)
    eff_reg_FOV_range = np.concatenate((x_oc_FOV_600c_max_min, y_oc_FOV_600c_max_min), axis=2)

    return (IC, # coordinates of in-coupler (mm)
            FC, FC_offset, # coordinates of folding-couplers (mm)
            OC, OC_offset, # coordinates of out-couplers (mm)
            eff_reg1, # effective regions for the whole system (mm)
            eff_reg2, # effective regions for IC&FC (mm)
            eff_reg_FOV, eff_reg_FOV_range, # effective out-coupled regions (mm)
            lut_TIR, lut_gap, lut_Fresnel, # Look-up table for TIR phase shift (rad) and propagation direction (mm)
            Lambda_ic, phi_ic, Lambda_fc, phi_fc, Lambda_oc, phi_oc, n_g, lmd, # grating orientation (rad) and period (nm)
            th_in_ic, phi_in_ic, th_out_ic, phi_out_ic, th_out_fc, phi_out_fc, # corresponding polar and azimuth angles (rad) for 600 FoVs (global coordinate)
            th_out_ic2, phi_out_ic2, th_out_oc, phi_out_oc, th_out_oc_glow, # 2nd order corresponding polar and azimuth angles (rad)
            kx0, ky0, kx_ic, ky_ic, kx_fc, ky_fc) # k-vectors