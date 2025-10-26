import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler, TensorDataset, DataLoader
from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils.ReadCapi import ReadCapi2
from utils.ReadSC import ReadSC2
from utils.Align import AlignData
from utils.MatchIOFiles import matching_io
from utils.represent_data import DataRepresentation
import plotly.graph_objects as go

#from run_CorrNN import train_weighted_corr_NN
#from setuprun import setup_and_run



def main():
    '''
    path = "/home/jayantdubey/Desktop/TrainData"
    testpath = "/home/jayantdubey/Desktop/TestData"

    io_pair = np.array(matching_io(path + "/sc", path + "/capi"))


    # --- load each trial, compute its own H, and pack (x,y,Hrep) ---
    for i in range(len(io_pair)):
    #for i in range(2):
        CAPIreader = ReadCapi2(io_pair[i, 0])
        SCreader   = ReadSC2(io_pair[i, 1])

        Synced = AlignData(CAPIreader, SCreader)
        Synced.check_timezone()
        Synced.interpret2(delete_missing=True)

        sc = Synced.adjusted_sc_clean    # [N, 11]
        capi = Synced.adjusted_capi_clean  # [N, 7]

        D = DataRepresentation(sc_array=sc, capi_array=capi)'''

    data = torch.load(f"TacGat/datasets/Val/0.pt")
    sc = data["sc"]
    capi = data["capi"]
    Dv = DataRepresentation(sc_array=sc, capi_array=capi)

    traj = Dv.target

    plot_traj(traj)

import plotly.graph_objects as go

def plot_traj(traj):
    fig = go.Figure()

    # traj is assumed to be shape [N, 3] in XYZ order
    fig.add_trace(go.Scatter3d(
        x=traj[:, 0],
        y=traj[:, 1],
        z=traj[:, 2],
        mode='lines',
        line=dict(width=4, color='green'),
        name='Trajectory'
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z"
        ),
        width=700,
        height=500,
        showlegend=False
    )

    fig.show()

   


def weighted_output(input_data, target_data, V):
    # Unit vector for reference axis 
    ref_axis = torch.tensor(V).float()
    ref_axis = ref_axis / ref_axis.norm()

    #  Compute angular deviation from ref_axis
    cos_vals = torch.clamp((target_data @ ref_axis), -1.0, 1.0)
    angles = torch.acos(cos_vals)    # in radians, shape [N]

    bins = torch.tensor([0.0, 0.02, 0.05, 0.1, 0.2, np.pi/2])
    bin_ids = torch.bucketize(angles, bins)

    counts = torch.bincount(bin_ids, minlength=len(bins)+1).float()
    counts[counts == 0] = 1.0

    w = 1.0 / torch.sqrt(counts[bin_ids]) 
    w = w / w.mean()



    #  Flatten BEFORE dataset creation
    if input_data.ndim == 3:
        N, win, feat = input_data.shape
        input_data = input_data.reshape(N, win * feat)   # [N, win*feat]
    else:
        N, win, feat = input_data.shape[0], 1, input_data.shape[1]

    print(f"Input shape after flatten: {input_data.shape}")   # Expect [N, win*feat]
    print(f"Target shape: {target_data.shape}")               # Expect [N, 3] or [N,3]

    # Debug window length vs features:
    if input_data.ndim == 2:
        print(f"N = {input_data.shape[0]}, Features = {input_data.shape[1]}")
    else:
        print("⚠ Still 3D! Something went wrong.")

    # build dataset/dataloader
    dataset = TensorDataset(input_data, target_data, w)

    sampler = WeightedRandomSampler(weights=w.cpu().numpy(),
                                    num_samples=len(w),
                                    replacement=True)
    loader = DataLoader(dataset, batch_size=64, sampler=sampler)

    return loader



def quiver_plot(ax, F, G,length=0.15, color='red', normalize=True):

    if normalize:
        p = ax.quiver(
        F[:, 0], F[:, 1], F[:, 2],
        G[:, 0], G[:, 1], G[:, 2],
        length=length, color=color, normalize=True
        )

    else:
        p = ax.quiver(
        F[:, 0], F[:, 1], F[:, 2],
        G[:, 0], G[:, 1], G[:, 2],
        color=color, normalize=False, 
        arrow_length_ratio=0.015
        )

    return p


def zero_integral_sections(data, tol=1e-2, min_len=100, M=3, max_tol=1.0, smooth_win=1):
    """
    Split into ranges whose *local* integral returns near zero.
    Resets the running integral at each detected boundary.
    Doubles tol until >= M segments found (or tol hits max_tol).
    """
    
    Ndim = data.ndim

    if Ndim < 2:
        x = np.asarray(data, dtype=float)
        if smooth_win > 1:
            w = np.ones(int(smooth_win)) / int(smooth_win)
            x = np.convolve(x, w, mode='same')
        x -= np.mean(x) # substracts data with the mean of the entire matrix

        n = len(x)  
        if n == 0:
            return np.empty((0, 2), dtype=int)

        def segment_with_tol(t):
            segs = []
            start = 0
            run = 0.0
            left_band = False  # require |run|>t before accepting next near-zero
            for i, v in enumerate(x):
                run += v
                if not left_band and abs(run) > t:
                    left_band = True
                if left_band and abs(run) <= t and (i - start) >= min_len:
                    segs.append([start, i])
                    start = i + 1
                    run = 0.0
                    left_band = False
            # tail segment if any length remains
            if start < n:
                if segs and (n - 1 - segs[-1][1]) < min_len:
                    # merge short tail into last segment
                    segs[-1][1] = n - 1
                else:
                    segs.append([start, n - 1])
            return np.array(segs, dtype=int)

        segs = segment_with_tol(tol)
        while len(segs) < M and tol < max_tol:
            tol *= 2
            segs = segment_with_tol(tol)

        return segs




def fisher_transform(H_set, start, end):
    '''
    Input [N, H.shape[0], H.shape[1]]: Set of cosine similarity transform matrices associated with N trials
    '''

    print(H_set.shape)

    standard_dev = 1 / np.sqrt(end - start - 3)
    distributions = np.zeros((H_set.shape[0],H_set.shape[1],H_set.shape[2]))
    for idx, corr_mat in enumerate(H_set):
        
        theta_y = corr_mat[:, 0]
        theta_z = corr_mat[:, 1]
        theta_x = corr_mat[:, 2]
        mag = corr_mat[:, 3]

        
        z_y = 0.5*np.log( (1+theta_y) / (1-theta_y))
        z_z = 0.5*np.log( (1+theta_z) / (1-theta_z))
        z_x = 0.5*np.log( (1+theta_x) / (1-theta_x))
        z_m = 0.5*np.log( (1+mag) / (1-mag))

        distributions[idx, :, :] = np.stack([z_y, z_z, z_x, z_m], axis=1)    
    
    return distributions, standard_dev



def plot_prediction(target_data, pred):

    # Extract predicted and true quaternions/translations
    q_pred = pred[:, :4]
    t_pred = pred[:, 4:]

    q_true = target_data[:, :4]
    t_true = target_data[:, 4:]

    # Convert to Euler
    euler_pred = R.from_quat(q_pred).as_euler('yzx', degrees=True)
    euler_true = R.from_quat(q_true).as_euler('yzx', degrees=True)

    # Plot Euler angles
    labels = ['θx', 'θy', 'θz']
    for i in range(3):
        plt.figure()
        plt.plot(euler_true[7500:15000, i], '--', label=f"true {labels[i]}", alpha=0.6)
        plt.plot(euler_pred[7500:15000, i], label=f"pred {labels[i]}")
        plt.title(f"Euler angle {labels[i]}")
        plt.legend()

    # Plot translations
    labels_t = ['tx', 'ty', 'tz']
    for i in range(3):
        plt.figure()
        plt.plot(t_true[7500:15000, i], '--', label=f"true {labels_t[i]}", alpha=0.6)
        plt.plot(t_pred[7500:15000, i], label=f"pred {labels_t[i]}")
        plt.title(f"Translation {labels_t[i]}")
        plt.legend()

    plt.show()

    

def plot_subset_traj(ax, traj, M_list):

    ax.scatter(0,0,0, color='red')  # #origin of coordinate frame
    #ax.scatter(x1[0], x1[1], x1[2], color = 'purple')   # Point A
    #ax.plot([x1[0], x2[0]], [x1[1], x2[1]], [x1[2], x2[2]], color='black', label='Line (x1-x2)')    # vector d_
    ax.plot([-1.0*traj[:, 6], M_list[:, 0]], [-1.0*traj[:, 5], M_list[:, 1]], [-1.0*traj[:, 4], M_list[:, 2]], color='green', label='Plane') # M
    plt.show()

def viz_pred(pred_transform, target_data):

        # Extract predicted and true quaternions/translations
    q_pred = pred_transform[:, :4]
    t_pred = pred_transform[:, 4:]

    q_true = target_data[:, :4]
    t_true = target_data[:, 4:]

    # Convert to Euler
    euler_pred = R.from_quat(q_pred).as_euler('yzx', degrees=True)
    euler_true = R.from_quat(q_true).as_euler('yzx', degrees=True)

    # Print MSE and MAE for translation:
    p = (t_true - t_pred)**2
    ps = np.sum(p)
    MSE = np.mean(ps)
    MAE = np.mean(np.sum(np.abs(t_true - t_pred)))

    print(f"MSE: {MSE} mm")
    print(f"MAE: {MAE} mm")

    # Print MSE and MAE for translation:
    # Print MSE and MAE for translation:
    p = (euler_true - euler_pred)**2
    ps = np.sum(p)
    MSE = np.mean(ps)
    MAE = np.mean(np.sum(np.abs(t_true - t_pred)))

    print(f"MSE: {MSE} mm")
    print(f"MAE: {MAE} mm")


    # Plot Euler angles
    labels = ['θx', 'θy', 'θz']
    for i in range(3):
        plt.figure()
        plt.plot(euler_true[7500:15000, i], '--', label=f"true {labels[i]}", alpha=0.6)
        plt.plot(euler_pred[7500:15000, i], label=f"pred {labels[i]}")
        plt.title(f"Euler angle {labels[i]}")
        plt.legend()

    # Plot translations
    labels_t = ['tx', 'ty', 'tz']
    for i in range(3):
        plt.figure()
        plt.plot(t_true[7500:15000, i], '--', label=f"true {labels_t[i]}", alpha=0.6)
        plt.plot(t_pred[7500:15000, i], label=f"pred {labels_t[i]}")
        plt.title(f"Translation {labels_t[i]}")
        plt.legend()

    plt.show()


def vizall(input_data, target_data,start, end):

    #start, end = 20300, 21000

    impulse, D_list_0, delta_pressure, waveforms_whole = input_data
    x1, x2, traj_whole, M_list, Disp_list, delta_pose, net_disp, d_ = target_data

    delta_pose = np.array(delta_pose)
    delta_pressure = np.array(delta_pressure)

    V_ = traj_whole #- M_list # All trajectory vectors defined w.r.t to the "origin" - a loop closure
    delta_V = V_[1:] - V_[:-1]  # The direction vectors change from v_t - v_t-1    

    print(f"Area dv_x/dt = {np.trapz([delta_V[:,0]])}")
    print(f"Area dv_y/dt = {np.trapz([delta_V[:,1]])}")
    print(f"Area dv_y/dt = {np.trapz([delta_V[:,2]])}")

    plt.figure()
    plt.plot(delta_V[:,0])
    plt.plot(delta_V[:,1])
    plt.plot(delta_V[:,2])
    plt.show()

    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x1[0], x1[1], x1[2], color = 'purple')   # Point A
    ax.scatter(x2[0], x2[1], x2[2], color = 'blue') # Point B
    #ax.plot([x1[0], x2[0]], [x1[1], x2[1]], [x1[2], x2[2]], color='black', label='Line (x1-x2)')    # vector d_

    ax.plot([traj_whole[:, 0], M_list[:, 0]], [traj_whole[:, 1], M_list[:, 1]], [traj_whole[:, 2], M_list[:, 2]], color='green', label='Plane') # M
    

    #planes, coll_points = normals_from_d(traj_whole, M_list)  # Computes normals 
    #n_, x0_ = planes[:,0,:], planes[:,1,:]

    #quiver_plot(ax, M_list, V_, color='black', normalize=False)    # V_ 
    #quiver_plot(ax, traj_whole, n_,length=0.15, color='red', normalize=True)  # Normals w.r.t AB
    #quiver_plot(ax, traj_whole[1:], delta_V, length=0.15, color='orange', normalize=False)

    ax.set_title("Relative Pose Displacement")
    
    plt.figure()
    plt.plot(np.linspace(0,Disp_list.shape[0],Disp_list.shape[0]),Disp_list)
    plt.title("Pose displacement")

    plt.figure()
    plt.plot(np.linspace(0,delta_pose.shape[0],delta_pose.shape[0]),delta_pose)
    plt.title("Pose velocity")

    print(f"Pressure Impulse: {impulse}")
    print(f"Pose Net Displacement: {net_disp}")

    plt.figure()
    plt.title("Pressure velocity")
    plt.plot(np.linspace(0,delta_pressure.shape[0],delta_pressure.shape[0]),delta_pressure)

    plt.figure()
    plt.title("Relative Pressure Displacement")
    plt.plot(np.linspace(0,D_list_0.shape[0],D_list_0.shape[0]),D_list_0)

    plt.show()

main()













'''
Archived training function

 # ---- Load trained model ----
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    # Initialize model with same input shape as used in training
    
    model = ResNetSpline(dummy_input).to(device)
    model.load_state_dict(checkpoint)
    model.eval()

    device = next(model.parameters()).device

    # ---- Inference loop ----
    results = []
    mse = torch.nn.MSELoss()

    with torch.no_grad():
        for block in ranges_pose:

            A_ = traj_whole[block[0]]
            B_ = traj_whole[block[1]]
            D_ = B_ - A_

            print(f"Testing block: {block[0]}-{block[1]}")

            impulse, D_list_0, delta_pressure, waveforms = scviz(sc_array=waveforms_whole,start=block[0],end=block[1])
            x1, x2, traj, M_list, Disp_list, delta_pose, net_disp, d_ = capiviz(capi_array=traj_whole, start=block[0], end=block[1],cutoff=True)
            

            # Get raw position and velocity vectors 
            pos_np = traj - M_list
            vel_np = pos_np[1:] - pos_np[:-1]  # The direction vectors change from v_t - v_t-1
            pos_np = pos_np[1:, :]    # trim to keep consistent with the N-1 readings of vel  

            pos = torch.as_tensor(pos_np, dtype=torch.float32)
            vel = torch.as_tensor(vel_np, dtype=torch.float32)
            pressure_readings = torch.as_tensor(waveforms, dtype=torch.float32)


            # Normalize same as training
            pressure_readings = (pressure_readings - pressure_readings.mean(dim=0, keepdim=True)) / (pressure_readings.std(dim=0, keepdim=True) + 1e-8)
            pos_target = pos - pos.mean(dim=0, keepdim=True)
            vel_target = vel - vel.mean(dim=0, keepdim=True)

            pressure_readings = pressure_readings.to(device)
            pos_target = pos_target.to(device)
            vel_target = vel_target.to(device)

            # Forward pass
            grad_3d, plane_9d = model(pressure_readings)

            # Integrate gradients to get predicted trajectory (for optional visualization)
            pos_pred = torch.cumsum(grad_3d, dim=0)
            pos_pred -= pos_pred.mean(dim=0, keepdim=True)


            # Compute block metrics (optional)
            loss_pos = mse(pos_pred, pos_target)
            loss_vel = mse(grad_3d, vel_target)
            print(f"pos_loss: {loss_pos.item():.3f}, vel_loss: {loss_vel.item():.3f}")

            results.append({
                "vel_pred": grad_3d.detach().cpu(),
                "pos_pred": pos_pred.detach().cpu(),
                "pos_target": pos_target.detach().cpu(),
                "vel_target": vel_target.detach().cpu(),
            })

            grad_3d = grad_3d.detach().cpu().numpy()
            pos_pred = pos_pred.detach().cpu().numpy()

            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            quiver_plot(ax, pos_pred, grad_3d, length=0.15, color='orange', normalize=False)
            quiver_plot(ax, pos_np, vel_np, length=0.15, color='blue', normalize=False)
            plt.show()
        



    return results
--------------------------------------------------------------------------------    

plane stuff use in main loop:

    #n_avg, k_avg = compute_avg_plane(ax, traj_data=traj_whole)
    #compute_parabolic_surface(ax, traj_whole)
    #plane_coeffs = (n_avg[0], n_avg[1], n_avg[2], k_avg)  # from your average-plane computation

    #compute_parabolic_surface(ax, traj_whole, plane_coeffs)

--------------------------------------------------------------------------------    
zero integral loops in main loop:

    ranges_pose = zero_integral_sections(delta_pose, tol=1e-2, min_len=50)
    print(f"Number of ranges: {ranges_pose.shape}")

    plt.figure()
    plt.plot(delta_pose)
    plt.title("Pose velocity")
    # alternate colors
    colors = ['#cce5ff', "#7475b9"]  # light blue / light green
    for idx, (start, end) in enumerate(ranges_pose):
        plt.axvspan(start, end, color=colors[idx % 2], alpha=0.3)
    
    plt.figure()
    plt.plot(delta_mag)
    plt.title("pressure velocity")

    # Compare how well the closed loops of the pressure and pose aligns
    
    pose_close_loops = find_closed_loops(delta_avg_pose)
    pressure_close_loops = find_closed_loops(delta_mag_pressure)

    for idx, entry in enumerate(pressure_close_loops):
        print(f"sensor {idx}: {entry.shape}")
    
    print(f"pose: {pose_close_loops.shape}")
    
    plt.show()

------------------------------------------------------------------------------------
Archived find surface to point neural network

def train_loop_trajectory(model_class, input_data, target_data, plane_coeffs=None, ranges_pose=None):
    """
    Trains the FindSurface model on trajectory data.
    If plane_coeffs are provided, the model will be tethered to that plane
    using hyper_plane_loss.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    dummy_input = input_data[:10, :]    # just to get shape
    model = model_class(dummy_input).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # Prepare dataset and dataloader
    Data = TrajectoryDataset(input_data, target_data, ranges_pose)
    dataloader = DataLoader(Data, batch_size=1, shuffle=True)

    num_epochs = 100
    for epoch in range(num_epochs):
        total_loss = 0.0
        total_samples = 0

        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            # Forward pass
            coeffs = model(x.squeeze(0))

            # Select loss function
            if plane_coeffs is not None:
                loss = model.hyper_plane_loss(x.squeeze(0), coeffs, y.squeeze(0), plane_coeffs)
            else:
                loss = model.hyper_loss(x.squeeze(0), coeffs, y.squeeze(0))

            # Backward and optimize
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.shape[0]
            total_samples += x.shape[0]

        mean_loss = total_loss / total_samples
        print(f"Epoch {epoch+1:03d} | Mean loss: {mean_loss:.6f}")

    print("Training complete.")
    return model

from paraNN import FindSurface

def compute_parabolic_surface(ax, traj_data, plane_coeffs):
    """
    Fits a hyperbolic surface tethered to an average plane and plots it.
    plane_coeffs = (A0, B0, C0, K0) from compute_avg_plane()
    """

    # Split data
    X_M, Y_M, Z_M = traj_data[:, 0], traj_data[:, 1], traj_data[:, 2]
    Z = traj_data[:, 2]
    input_data = traj_data[:, :2]

    # --- Train model in tethered mode ---
    print("\n[Training tethered hyperbolic surface model...]")
    A, B, C, K = train_loop(
        input_data,
        Z,
        plane_coeffs=plane_coeffs,
        mode="tethered"
    )

    print(f"\nFinal coefficients:\nA={A:.4f}, B={B:.4f}, C={C:.4f}, K={K:.4f}")

    # --- Create grid for plotting ---
    x = np.linspace(min(X_M), max(X_M), 50)
    y = np.linspace(min(Y_M), max(Y_M), 50)
    X, Y = np.meshgrid(x, y)

    # --- Average plane base ---
    A0, B0, C0, K0 = plane_coeffs
    Z_plane = -(A0 * X + B0 * Y + K0) / C0

    # --- Tethered hyperbolic surface ---
    Z_surface = Z_plane + C * ((X ** 2) / (A ** 2) - (Y ** 2) / (B ** 2))

    # --- Plot ---
    ax.plot_surface(X, Y, Z_surface, alpha=0.6, color='orange', label='Hyper Surface')
    ax.plot_surface(X, Y, Z_plane, alpha=0.2, color='gray', label='Avg Plane')

    ax.set_title("Tethered Hyperbolic Surface")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    return A, B, C, K

# Find the plane that minimizes the distance to all points for numpy data
import random

def normals_from_d(points, M_list):

    
    This 


    planes = np.empty((points.shape[0],2,3))
    collinear_points = np.empty((points.shape[0],3,3))
    A_, B_ = points[0], points[-1]

    try:
        v = zip(points, M_list)
    
    except ValueError as e:
        print(f"{e}: Could not broadcast points and M_list")

    # Calculates the normal at x0_ w.r.t d_, a vector formed by 
    for idx, v_ in enumerate(v):
        
        try:
            #n_ = np.cross((v_[0] - A_), (v_[0] - v_[1]))
            n_ = np.cross((v_[0] - A_), (v_[0] - B_))

        except n_ == 0:
            collinear_points = np.array([v_[0], v_[1], A_])
            print(f"Failed collinear test at {v_[0], v_[1], A_}")
        

        planes[idx] = np.array([v_[0],n_])  # normal vector n_ at point x0_


    return planes, collinear_points

def get_spaced_sample(traj_data, k, min_distance):
    """
    Select k points from traj_data such that each pair is at least
    min_distance apart in Euclidean space.
    """
    n = traj_data.shape[0]
    if k > n:
        raise ValueError("Sample size k cannot be greater than trajectory length.")
    if k == 0:
        return []

    selected = []
    attempts = 0
    max_attempts = 10000  # safety break

    while len(selected) < k and attempts < max_attempts:
        idx = random.randint(0, n-1)
        candidate = traj_data[idx]
        if not selected:
            selected.append(idx)
        else:
            # compute distance to all already selected points
            dists = [np.linalg.norm(candidate - traj_data[i]) for i in selected]
            if all(d > min_distance for d in dists):
                selected.append(idx)
        attempts += 1

    if len(selected) < k:
        raise RuntimeError("Could not find enough spaced samples with given min_distance.")

    return selected

def compute_avg_plane(ax, traj_data, n_iter=25000):
    """
    Repeatedly fits planes to random triplets of points and
    returns the average normal (A,B,C) and average offset K.
    """
    normal_sum = np.zeros(3)
    k_sum = 0.0

    for i in range(n_iter):
        n_, k = find_plane(ax, traj_data)
        normal_sum += n_
        k_sum += k

    # Average
    n_avg = normal_sum / n_iter
    k_avg = k_sum / n_iter

    # Normalize so the normal vector has unit length
    #n_norm = np.linalg.norm(n_avg)
    #n_avg /= n_norm
    #k_avg /= n_norm

    print(f"Average plane coefficients: A={n_avg[0]:.4f}, B={n_avg[1]:.4f}, C={n_avg[2]:.4f}, K={k_avg:.4f}")

    # Optional plot
    X_M, Y_M, Z_M = traj_data[:, 0], traj_data[:, 1], traj_data[:, 2]
    x = np.linspace(min(X_M), max(X_M), 10)
    y = np.linspace(min(Y_M), max(Y_M), 10)
    X, Y = np.meshgrid(x, y)
    Z = (-n_[0]*X - n_[1]*Y - k_avg) / n_[2]
    ax.plot_surface(X, Y, Z, alpha=0.5, color='gray')

    return n_avg, k_avg

def find_plane(ax, traj_data):
    """
    Picks 3 random spaced points, computes the plane,
    and returns its coefficients.
    """
    points = get_spaced_sample(traj_data, 3, 6)
    plane_normal, plane_constant = compute_plane(ax, traj_data, points)
    
    return plane_normal, plane_constant
    
    #ax.scatter(traj_data[points[0]][0], traj_data[points[0]][1], traj_data[points[0]][2], color='red')  # origin of coordinate frame
    #ax.scatter(traj_data[points[1]][0], traj_data[points[1]][1], traj_data[points[1]][2], color='green')  # origin of coordinate frame
    #ax.scatter(traj_data[points[2]][0], traj_data[points[2]][1], traj_data[points[2]][2], color='blue')  # origin of coordinate frame
    
    #plt.show()

# Find the plane from 3 points, and returns the normal vector and k, 
# The commented out visual shows how the equation is made and plotted

def compute_plane(ax, traj_data, points):
    """
    Computes plane from 3 points in traj_data.
    Returns normal vector and constant K in Ax+By+Cz+K=0.
    """
    A, B, C = traj_data[points[0]], traj_data[points[1]], traj_data[points[2]]
    n_ = np.cross(A - B, A - C)
    n_ = n_ / np.linalg.norm(n_)  # normalize normal
    K = -np.dot(n_, C)

    return n_, K




# Used for pressure primarily, clean the pressure points with a moving average
# and remove points in a window that fall far out of the standard deviation 
def remove_outliers(pressure_data):
    pass


def find_closed_loops(data):

    if data.ndim > 1:
        ranges_pressure = []
        # Plots the individual closed loops for each pressure sensor
        for idx, sensor_delta in enumerate(data.T):
            ranges_pose = zero_integral_sections(sensor_delta, tol=1e-4, min_len=50)
            #print(f"Number of ranges: {ranges_pose.shape}")

            plt.figure()
            plt.plot(np.linspace(0,sensor_delta.shape[0],sensor_delta.shape[0]),sensor_delta)
            plt.title(f"Pressure velocity {idx}")

                # alternate colors
            colors = ['#cce5ff', "#7475b9"]  # light blue / light green
            for idx, (start, end) in enumerate(ranges_pose):
                plt.axvspan(start, end, color=colors[idx % 2], alpha=0.3)
        
            #plt.show()

            ranges_pressure.append(ranges_pose)

        return ranges_pressure
    
    else:
        # Plots the closed loops of the pose

        ranges_pose = zero_integral_sections(data, tol=1e-2, min_len=50)
        #print(f"Number of ranges: {ranges_pose.shape}")

        plt.figure()
        plt.plot(np.linspace(0,data.shape[0],data.shape[0]),data)
        plt.title("Pose velocity")

            # alternate colors
        colors = ['#cce5ff', "#7475b9"]  # light blue / light green
        for idx, (start, end) in enumerate(ranges_pose):
            plt.axvspan(start, end, color=colors[idx % 2], alpha=0.3)
    
        #plt.show()

        return ranges_pose

    
OLD CAPIVIZ
    
    def capiviz(self, cutoff=True):

        traj_whole = self.target_data[self.start:self.end,:]
        
        x1 = traj_whole[0, :]   # # Point A from least energy index
        traj_whole = traj_whole - x1    # Centers x1, and defines every vector w.r.t to x1
        
        x1 = np.zeros_like(traj_whole[0, :])  # define x1 in the local frame

        x2 = traj_whole[-1, :]

        M_list = np.zeros(traj_whole.shape)
        theta = np.zeros(traj_whole.shape)
        relative_mag = np.zeros(traj_whole.shape[0])
        
        for idx, x0 in enumerate(traj_whole):
            
            # Find parameter t along d_ that minimizes the distance between x0 and d_    
            #t = np.dot(x0, d_) / np.dot(d_, d_) + 1e-8

            if cutoff:
                if t > 1:
                    t = 1
                elif t < 0:
                    t = 0

            d = np.abs(np.cross(x0-x1,x0-x2))/np.abs(x2-x1)

            # Point along x1x2 and minimizes distance with point x0
            M = t*d_
            M_list[idx] = M
            dist = np.linalg.norm(np.cross(x0 - x1, x0 - x2))/ (np.linalg.norm(d)+ 1e-8)
            # -----------------------------------------------------------------------#
            # NOTE: I actually use these values
            # Distance from x0 to the line (perpendicular distance)
            
            mag = np.linalg.norm(x0)
            cosines = x0 / mag if mag > 1e-12 else np.zeros_like(x0)

            theta[idx] = np.arccos(cosines)    # theta_x, theta_y, theta_z in radians 
            relative_mag[idx] = mag  # magnitude of x0 

            #ax.plot([x0[0], M[0]], [x0[1], M[1]], [x0[2], M[2]], color='purple', label='Perpendicular')


        # Compute velocity
        delta = relative_mag[1:] - relative_mag[:-1] 
        
        slide_win = 10
        total_time_len = delta.shape[0]
        slide_stride = 1
        
        # moving average
        delta_avg = np.zeros(total_time_len)
        for i in range(total_time_len):
            start_i = max(0, i - slide_win)
            block = delta[start_i:i]  
            if len(block) > 0:
                delta_avg[i] = np.mean(block)
        
        
        # Compute net displacement by taking the integral of the velocity
        net_displacement = np.trapz(delta_avg)


        return x1, x2, traj_whole, M_list, relative_mag, delta_avg, net_displacement, theta
'''
