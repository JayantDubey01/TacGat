import torch
import numpy as np
import roma
from scipy.spatial.transform import Rotation as R
from scipy import linalg
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation 
import pyquaternion

class ReadCapi():

    def __init__(self, filename):
        self.CAPI_data = filename

    def is_zero_quat(self,quat):
        return np.allclose(quat,[0.0,0.0,0.0,0.0])

    # Transformation of tool2 to tool1, returned as Euler
    def ref_transform(self,tool2_quat, tool1_quat):

        if self.is_zero_quat(tool2_quat) or self.is_zero_quat(tool1_quat):
            return np.array([0.0,0.0,0.0])

        q1 = R.from_quat(tool1_quat,scalar_first=True)
        q2 = R.from_quat(tool2_quat,scalar_first=True)

        q2_inv = q2.inv()
        q2_from_q1 = q1 * q2_inv

        euler = q2_from_q1.as_euler('zyx',degrees=True) # Polaris Convention

        return euler


    # Transforms array of tool 2 data to tool 1's frame. 
    def convert_to_reference(self,tool2_quaternion, tool1_quaternion,tool2_translation, tool1_translation):
        # Euler angle representation of tool 2 w.r.t tool 1 array
        tool2_transformed_array = np.zeros((len(tool1_quaternion),6))

        for i in range(len(tool1_quaternion)): 
            
            # Find Transformation of Tool 2's Quaternion to Tool1 as reference frame. Converted into Euler Angles.
            tool2_euler_transformation = self.ref_transform(tool2_quaternion[i,:],tool1_quaternion[i,:])
            corrected_rotation = -1 * tool2_euler_transformation

            # Find Transformation of Tool 2's Translation to Tool1 as reference frame. 
            translation = np.subtract(tool2_translation[i,:],tool1_translation[i,:])
            transformation = np.concatenate((corrected_rotation,translation)) # concatenate to corrected_rotation to combine as [rx,rz,ry,x,y,z] vector

            # Update array with 6-dof transformation vector
            tool2_transformed_array[i] = transformation
        
        return tool2_transformed_array

    # Used to clean API-recorded data. Checks tool status column since the recorded value for missing tools is ~3e+12  
    def handle_bad_line_capi(self,bad_line):
        print("Bad CAPI line")

        status_line_tool1, status_line_tool2 = 6,20

        if bad_line[status_line_tool1] == 'Missing':
            s1,e1 = 6,12
            for i in range(s1, e1+1):
                bad_line[i] = 0
        
        if bad_line[status_line_tool2] == 'Missing':
            s2,e2 = 21,28
            for i in range(s2,e2+1):
                bad_line[i] = 0
        
        return bad_line

    # Used to clean CAPI-recorded data. Missing lines are filled with NaN values. 
    def handle_capi_lines(self,filename):
        capi_df = pd.read_csv(filename)

        nan_mask = capi_df.isna().to_numpy()
        MissingData = np.argwhere(nan_mask)

        # Need to make sure that if one tool has NaN, then the second tool also needs to have NaNs
        MissingData = np.unique(MissingData,axis=0)

        print(f"Number of missing data points {MissingData.shape[0]}")

        # For each row, if there exists a NaN, then make sure the rigid body columns of both Tool1 and Tool2 are filled with NaNs
        for idx,val in enumerate(MissingData):
            row,col = val   # unpack the 2D element since a missing point is described as row,val
            
            capi_df.iloc[row,6:10] = [0,0,0,0]   # Clean Quaternion
            capi_df.iloc[row,10:13] = [0,0,0]    # Clean Translation

            capi_df.iloc[row,21:25] = [0,0,0,0]   # Clean Quaternion
            capi_df.iloc[row,25:28] = [0,0,0]    # Clean Translation

        # Replace all NaNs with 0
        #capi_df.fillna(0,inplace=True)

        return capi_df


    # Reads CAPI-recorded file. Cleans it by replacing recorded NaN values with zero. Then performs a change in reference transformation on Tool2's Quaternion and Translation.
    # It transforms Tool2 from the camera reference frame to Tool1 as the reference frame. The Rotation is converted from a quaternion representation to Euler angles. 
    def readCapi(self):
        capi_df = self.handle_capi_lines(self.CAPI_data)

        capi_df.to_csv("CleanedCapi.csv",index=False)

        # The quaternions of each tool. Extract from specified columns from the file.
        tool1_quaternion_capi = capi_df.iloc[:, 6:10].to_numpy()
        tool2_quaternion_capi = capi_df.iloc[:, 21:25].to_numpy()

        # The translation of each tool. Extracted from specified columns from the file.
        tool1_translation = capi_df.iloc[:, 11:14].to_numpy()
        tool2_translation = capi_df.iloc[:,26:29].to_numpy()

        # Transform Tool2's quaternion to Tool1's reference frame. Converted to Euler Angles. Stored as an np array. 
        self.tool2_transformed_capi = self.convert_to_reference(tool2_quaternion_capi,tool1_quaternion_capi,tool2_translation,tool1_translation)
        self.CAPITime = capi_df.iloc[:,15].to_numpy()

class ReadCapi2():

    # Tool1 is the reference tool, and Tool2 are the sunglasses. 
    def __init__(self, filename):
        self.CAPI_data = filename
        self.raw_data = pd.read_csv(self.CAPI_data) 

        print(f"Reading file: {self.CAPI_data}")
        # Find column indices of quaternion and translation
        self.T1_Qidx = [self.raw_data.columns.get_loc("Q0"), self.raw_data.columns.get_loc("Qx"), self.raw_data.columns.get_loc("Qy"), self.raw_data.columns.get_loc("Qz")]
        self.T1_Tidx = [self.raw_data.columns.get_loc("Tx"), self.raw_data.columns.get_loc("Ty"), self.raw_data.columns.get_loc("Tz")]
        self.T2_Qidx = [self.raw_data.columns.get_loc("Q0.1"), self.raw_data.columns.get_loc("Qx.1"), self.raw_data.columns.get_loc("Qy.1"), self.raw_data.columns.get_loc("Qz.1")]
        self.T2_Tidx = [self.raw_data.columns.get_loc("Tx.1"), self.raw_data.columns.get_loc("Ty.1"), self.raw_data.columns.get_loc("Tz.1")]
        self.CAPITime = self.raw_data.loc[:,"Timestamp"].to_numpy()

        # Clean data - in scalar-fist form
        self.clean_data, self.MissingDataList = self.handle_capi_lines() 

        # Convert to numpy arrays, convert quaternions to scalara-last form
        # Rearranging the order of the quaternion to [Qx,Qy,Qz,W]
        T1Q = self.clean_data.iloc[:, self.T1_Qidx[0]:self.T1_Qidx[-1] + 1].to_numpy()
        T1Q = T1Q[:, [1,2,3,0]] 
        T1T = self.clean_data.iloc[:, self.T1_Tidx[0]:self.T1_Tidx[-1] + 1].to_numpy()

        T2Q = self.clean_data.iloc[:, self.T2_Qidx[0]:self.T2_Qidx[-1] + 1].to_numpy()
        T2Q = T2Q[:, [1,2,3,0]] 
        T2T = self.clean_data.iloc[:, self.T2_Tidx[0]:self.T2_Tidx[-1] + 1].to_numpy()
        #self.CAPITime = self.clean_data.loc[:,"Timestamp"].to_numpy()
        self.N = len(self.CAPITime)

        # Gather statistical properties of the dataset. Should be a useful feature. 
        self.avg_t1_q = self.quatWAvg(T1Q)
        #self.avg_t1_q = R.from_quat(self.avg_t1_q)
        self.avg_t1_t = np.mean(T1T, axis=0)
        #self.avg_unit_u = np.concatenate((self.avg_q, self.avg_t), axis=0)

        
        # Expresses T2's quaternion and translation vectors in the reference frame of T1. All operations and objects are soley quaternions.
        self.T2_Ref = self.convert_to_reference(self.avg_t1_q ,self.avg_t1_t ,T2Q,T2T)

        #T2Q_Ref = self.T2_Ref[:, 0:4].astype(float)   # Quaternion
        #T2T_Ref = self.T2_Ref[:, 4:7].astype(float)   # Translation

        # RE-EXPRESS Tool 2 into Tool 1's instrinsic coordinate frame
        T2Q_Ref, T2T_Ref = self.Realign2Bore()

        # TENSORIZE
        # Tool2 Tensor
        self.T2Q_Ref = torch.Tensor(T2Q_Ref) 
        #self.T2Q_Ref = self.T2Q_Ref / self.T2Q_Ref.norm(p=2, dim=-1, keepdim=True)

        self.T2T_Ref = torch.Tensor(T2T_Ref)

        # Tool2_Ref Quat and Trans vector
        self.T2 = torch.hstack((self.T2Q_Ref,self.T2T_Ref))

        # Visualize
        #self.visualize(T1Q,T1T,T2Q_Ref,T2T_Ref)
    
    def Realign2Bore(self):

        # Tool 2 pose w.r.t Tool 1
        R_T1_from_T2 = R.from_quat(self.T2_Ref[:, :4])
        t_T1_from_T2 = self.T2_Ref[:, 4:]

        # (1) Axes relabeling for Tool 2: intrinsic Z(-180) then X(+90)
        #A_T2 = R.from_euler('zx', [-180, +90], degrees=True)  # post-multiply  TEST 1  OKAY    
        #A_T2 = R.from_euler('xy', [+90,+180], degrees=True)  # post-multiply   TEST 2  
        
        A_T2 = R.from_euler('XY', [+90,+180], degrees=True)  # post-multiply      TEST 3   BEST SO FAR 90 DEGREE ROTATION ABOUT X THOUGH
        
        #A_T2 = R.from_euler('XY', [+90,+180], degrees=True)  # post-multiply      TEST 4   
        #A_T2 = R.from_euler('x', [+90], degrees=True)  # post-multiply      TEST 4   

        #A_T2 = R.from_euler('ZX', [0,+90], degrees=True)  # post-multiply      TEST 5



        # (2) 180° intrinsic flip about Tool 1 X
        A_T1 = R.from_euler('Y', 0, degrees=True)           # pre-multiply

        # New rotation: A_T1 * R * A_T2
        R_T1p_from_T2p = A_T1 * R_T1_from_T2 * A_T2

        # New translation: only affected by Tool-1's intrinsic flip
        t_T1p_from_T2p = A_T1.apply(t_T1_from_T2)

        # Output: Tool 2 pose in the final Tool 1 frame
        return R_T1p_from_T2p.as_quat(), t_T1p_from_T2p
    
    def quatWAvg(self, Q):
        '''
        Averaging Quaternions.

        Arguments:
            Q(ndarray): an Mx4 ndarray of quaternions.
            weights(list): an M elements list, a weight for each quaternion.
        '''

        # Form the symmetric accumulator matrix
        A = np.zeros((4, 4))
        M = Q.shape[0]
        wSum = 0

        for i in range(M):
            q = Q[i, :]
            w_i = 1.0
            A += w_i * (np.outer(q, q)) # rank 1 update
            wSum += w_i

        # scale
        A /= wSum

        # Get the eigenvector corresponding to largest eigen value
        return np.linalg.eigh(A)[1][:, -1]


    def is_unit_quat(self,quat):
        return np.allclose(quat,[0.0,0.0,0.0,1.0])

    # Transformation of tool2 quaternion to tool1 quaternion
    def ref_rotation(self,tool2_quat, tool1_quat):

        if self.is_unit_quat(tool2_quat) or self.is_unit_quat(tool1_quat):
            return np.array([0.0,0.0,0.0,1.0])

        q1 = R.from_quat(tool1_quat)
        q2 = R.from_quat(tool2_quat)

        q2_inv = q2.inv()
        q2_from_q1 = q1 * q2_inv

        return q2_from_q1.as_quat()
    
    def ref_translation(self, tool2_trans, tool1_quat, tool1_trans):
        #avg_t1_q = R.from_quat(self.avg_t1_q)

        tool1_quat = R.from_quat(tool1_quat)

        translation_world = np.subtract(tool1_trans,tool2_trans)
        translation_t1 = tool1_quat.inv().apply(translation_world)

        return translation_t1

    # Transforms array of Tool2 data to Tool1's frame. 
    def convert_to_reference(self, T1Q, T1T, T2Q, T2T):

        # Normalize all quaternions
        T1Q = R.from_quat(T1Q)
        T1Q = T1Q / T1Q.norm(p=2, dim=-1, keepdim=True)

        T2Q = R.from_quat(T2Q)
        T2Q = T2Q / T2Q.norm(p=2, dim=-1, keepdim=True)
        
        # Quaternion representation of Tool2 w.r.t tool 1 for whole collected dataset
        tool2_transformed_array = np.zeros((self.N,7),dtype=float)
        for i in range(self.N): 
            
            # Find Transformation of Tool2's Quaternion with Tool1 as reference frame
            rotation = self.ref_rotation(T2Q[i,:],T1Q[i,:])
            translation = self.ref_translation(T2T[i, :],T1Q[i,:], T1T[i,:])

            # Resulting Quaternion - Translation vector
            transformation = np.concatenate((rotation,translation)) # concatenate to corrected_rotation to combine as [Qx, Qy, Qz, W, x, y, z] vector


            # Store Quat-Trans vector
            tool2_transformed_array[i] = transformation
        
        return tool2_transformed_array

    '''
    CLEAN RECORDED DATA
    '''
    # Used to clean CAPI-recorded data. Missing lines are filled with NaN values. 
    def handle_capi_lines(self):
        
        # Produce a Boolean Mask for all NaN values, then create a list of indices of where the True values are. 
        capi_df = self.raw_data
        nan_mask = capi_df.isna().to_numpy()

        # Produces an Nx2 matrix of the missing data coordinates
        MissingData = np.argwhere(nan_mask)

        # The row indices are repeated in MissingData since there are NaN values in all the columns of the quaternion and translation vectors.    
        MissingData = np.unique(MissingData[:,0])
        MissingDataList = []

        # For each row, if there exists a NaN, then make sure the rigid body columns of both Tool1 and Tool2 are filled with 0s
        for row in MissingData:
            
            capi_df.iloc[row,self.T1_Qidx] = [1.0,0.0,0.0,0.0]   # Resets to Identity Quaternion ie no rotation
            capi_df.iloc[row,self.T1_Tidx] = [0.0,0.0,0.0]    # Zero Translation

            capi_df.iloc[row,self.T2_Qidx] = [1.0,0.0,0.0,0.0]   
            capi_df.iloc[row,self.T2_Tidx] = [0.0,0.0,0.0]

            MissingDataList.append(self.CAPITime[row])
        
        print(f"Number of Missing Data Points: {len(MissingDataList)}")

        return capi_df, MissingDataList
    
        # Used to clean API-recorded data. Checks tool status column since the recorded value for missing tools is ~3e+12  
    def _handle_bad_line_api(self,bad_line):
        print("Bad CAPI line")

        status_line_tool1, status_line_tool2 = 6,20

        if bad_line[status_line_tool1] == 'Missing':
            s1,e1 = 6,12
            for i in range(s1, e1+1):
                bad_line[i] = 0
        
        if bad_line[status_line_tool2] == 'Missing':
            s2,e2 = 21,28
            for i in range(s2,e2+1):
                bad_line[i] = 0
        
        return bad_line
    
    #@NOTE: Visualize quaternions in the transformed version
    def visualize(self, T1Q, T1T, T2Q_Ref, T2T_Ref):
        frame_num = len(self.T2Q_Ref)

        # Create figure and 3D axis
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        min = np.min(np.minimum(T1T,T2T_Ref))
        max = np.max(np.maximum(T1T, T2T_Ref))
        ax.set_xlim([min, max])
        ax.set_ylim([min, max])
        ax.set_zlim([min, max])
        
        # Initialize two line objects (for v1 and v2)
        line1, = ax.plot([], [], [], 'o-', lw=2, label="q1 (T1Q)", color='r')  # Red for v1
        line2, = ax.plot([], [], [], 'o-', lw=2, label="q2 (T2Q_Ref)", color='b')  # Blue for v2
        
        ax.legend()

        # Animation function
        def animate(i):
            
         
            q1 = T1Q[i]  # Quaternion v1
            q2 = T2Q_Ref[i]  # Quaternion v2
            
            # Convert quaternions to rotation objects
            r1 = R.from_quat(q1)
            r2 = R.from_quat(q2)

            t1 = T1T[i]
            t2 = T2T_Ref[i]

            # Rotate a unit vector (e.g., x-axis) by each quaternion
            v1_initial = np.array([1, 0, 0])  # Unit vector along x-axis
            v2_initial = np.array([1, 0, 0])  # Unit vector along x-axis
            
            v1_rotated = (r1.apply(v1_initial)*-1) + t1
            v2_rotated = r2.apply(v2_initial) + t2

            # Update the first vector (v1)
            line1.set_data([0, v1_rotated[0]], [0, v1_rotated[1]])
            line1.set_3d_properties([0, v1_rotated[2]])

            # Update the second vector (v2)
            line2.set_data([0, v2_rotated[0]], [0, v2_rotated[1]])
            line2.set_3d_properties([0, v2_rotated[2]])


            return line1, line2

        # Create the animation
        ani = FuncAnimation(fig, animate, frames=frame_num, interval=50, blit=True)

        # Display the animation
        plt.show()





