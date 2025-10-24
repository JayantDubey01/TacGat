import numpy as np
import matplotlib.pyplot as plt
import struct

class ReadSC():
    def __init__(self,filename):
        self.SC_data = filename
        self.read_file()
        self.extractCleanData()

    
    def read_file(self):
        with open(self.SC_data, 'rb') as f:
            # Reading the header as before
            self.v = struct.unpack('f', f.read(4))[0]
            self.Np = struct.unpack('i', f.read(4))[0]
            self.NTemp = struct.unpack('i', f.read(4))[0]
            self.Nothers = struct.unpack('i', f.read(4))[0]
            self.Pconv = struct.unpack('i', f.read(4))[0]
            self.Vpi = struct.unpack('i', f.read(4))[0]
            self.Rdown = struct.unpack('i', f.read(4))[0]
            self.calib_Woffs = np.frombuffer(f.read(4 * self.Np), dtype=np.float32)
            self.calib_Wrel = np.frombuffer(f.read(4 * self.Np), dtype=np.float32)
            self.calib_Wglob = struct.unpack('f', f.read(4))[0]
            self.hdr_length = f.tell()

            # Jump over the header
            f.seek(self.hdr_length, 0)

            # Determine number of entries (N)
            entry_length = 4 * (2 * self.Np + 1 * self.NTemp + 4) + 1 * 8
            f.seek(0, 2)  # Move to end of file
            file_length = f.tell()
            N = int((file_length - self.hdr_length) / entry_length)

            # Initialize arrays
            self.Vi = np.zeros((self.Np, N))
            self.Wght = np.zeros((self.Np, N))
            self.T_C = np.zeros((self.NTemp, N))
            self.Vcage = np.zeros(N)
            self.AmbNs = np.zeros(N)
            self.B0flag = np.zeros(N, dtype=np.int32)
            self.tstamp = np.zeros(N, dtype=np.int64)
            self.colcode = np.zeros(N, dtype=np.int32)

            # Read all entries
            f.seek(self.hdr_length, 0)
            for ientry in range(N):
                self.Vi[:, ientry] = np.frombuffer(f.read(4 * self.Np), dtype=np.float32)
                self.Wght[:, ientry] = np.frombuffer(f.read(4 * self.Np), dtype=np.float32)
                self.T_C[:, ientry] = np.frombuffer(f.read(4 * self.NTemp), dtype=np.float32)
                self.Vcage[ientry] = struct.unpack('f', f.read(4))[0]
                self.AmbNs[ientry] = struct.unpack('f', f.read(4))[0]
                self.B0flag[ientry] = struct.unpack('i', f.read(4))[0]
                self.tstamp[ientry] = struct.unpack('q', f.read(8))[0]  # int64
                self.colcode[ientry] = struct.unpack('i', f.read(4))[0]

    
    def extractCleanData(self):
        # Validate data
        test_RGB = np.isin(self.colcode, [0b000, 0b001, 0b010, 0b011, 0b100, 0b101, 0b110, 0b111])   # Calculates Boolean Array
        test_B0flag = np.isin(self.B0flag, [0, 1])   # Calculates Boolean Array
        test_Vi = (self.Vi.max(axis=0) <= self.Vpi) & (self.Vi.min(axis=0) >= 0)   # Calculates Boolean Array
        test = test_RGB & test_B0flag & test_Vi     # Combines results from each boolean array
        good_locs = np.where(test)[0]   # Mask

        # Filter good data
        self.Vi = self.Vi[:, good_locs]
        self.Wght = self.Wght[:, good_locs]
        self.T_C = self.T_C[:, good_locs]
        self.Vcage = self.Vcage[good_locs]
        self.AmbNs = self.AmbNs[good_locs]
        self.B0flag = self.B0flag[good_locs]
        self.tstamp = self.tstamp[good_locs]
        self.colcode = self.colcode[good_locs]

        # Rearrange Weight Data
        self.Wght = self.Wght[::-1, :].T

        # Rescale Timestamp from microsescond to millisecond;
        self.tstamp = (0.001 * self.tstamp)
            
    def plotData(self):

        # Plot the pressure (weight) data
        plt.figure(1)
        t = (self.tstamp - self.tstamp[0]) / 1e6  # Convert to seconds
        plt.plot(t, 1000 * self.Wght, linestyle='none', marker='.', markersize=10)  # Weight in grams
        plt.title(self.SC_data)
        plt.gcf().set_size_inches(12, 4)
        plt.xlabel("Time (s)")
        plt.ylabel("Weight (grams)")

        # Plot the temperature data
        plt.figure(2)
        plt.plot(t, self.T_C[0, :], linestyle='none', marker='.', markersize=4, color=[0, 0.6, 0])
        if self.NTemp > 1:
            plt.plot(t, self.T_C[1, :], linestyle='none', marker='.', markersize=4, color=[0, 0, 1])
        if self.NTemp > 2:
            plt.plot(t, self.T_C[2, :], linestyle='none', marker='.', markersize=4, color=[0.9, 0, 0])


        plt.xlabel("Time (s)")
        plt.ylabel("Temperature (°C)")
        plt.show()

class ReadSC2():
    def __init__(self, filename):
        self.SC_data = filename
        print(f"Reading file: {filename}")
        self.read_file()
        self.extractCleanData()
    
    def read_file(self):
        with open(self.SC_data,'rb') as f:
            # Reading the header as before
            self.v = struct.unpack('f', f.read(4))[0]
            self.v = round(self.v,2)
            self.Np = struct.unpack('i', f.read(4))[0]
            self.NTemp = struct.unpack('i', f.read(4))[0]
            self.Nothers = struct.unpack('i', f.read(4))[0]
            self.Pconv = struct.unpack('i', f.read(4))[0]
            self.Vpi = struct.unpack('i', f.read(4))[0]
            self.Rdown = struct.unpack('i', f.read(4))[0]
            self.calib_Woffs = np.frombuffer(f.read(4 * self.Np), dtype=np.float32)
            self.calib_Wrel = np.frombuffer(f.read(4 * self.Np), dtype=np.float32)
            self.calib_Wglob = struct.unpack('f', f.read(4))[0]
            self.hdr_length = f.tell()

            # Jump over header
            f.seek(self.hdr_length,0)
            print(f"version: {self.v}")
            if (self.v == 1.0): # One time stamp
                self.entry_length = 4*(2*self.Np + 1*self.NTemp + 4) + 1*8 # Length of one data entry
            
            elif (self.v == 1.1):   # Two time stamps
                self.entry_length = 4*(2*self.Np + 1*self.NTemp + 4) + 2*8 # Length of one data entry
            
            elif (self.v==1.2): # Two time stamps, Temp for CPU (NTemp=4), and WiFi signal strength
                self.entry_length = 4*(2*self.Np + 1*self.NTemp + 5) + 2*8 
            
            f.seek(0, 2)  # Move to end of file
            file_length = f.tell()
            N = int((file_length - self.hdr_length) / self.entry_length)

            # Initalize arrays
            self.Vi = np.zeros((self.Np, N))
            self.Wght = np.zeros((self.Np, N))
            if (self.v==1.2):
                self.T_C = np.zeros((self.NTemp-1, N))
            else:
                self.T_C = np.zeros((self.NTemp, N))
            self.TCPU = np.zeros(N, dtype=np.int32)
            self.Vcage = np.zeros(N)
            self.AmbNs = np.zeros(N)
            self.B0flag = np.zeros(N, dtype=np.int32)
            self.sig = np.zeros(N, dtype=np.int32)
            self.tstamp1 = np.zeros(N, dtype=np.int64)
            self.tstamp2 = np.zeros(N, dtype=np.int64)
            self.colcode = np.zeros(N, dtype=np.int32)

            # Read all of the recorded entries
            f.seek(self.hdr_length, 0)
            for ientry in range(N):
                self.Vi[:, ientry] = np.frombuffer(f.read(4 * self.Np), dtype=np.float32)
                self.Wght[:, ientry] = np.frombuffer(f.read(4 * self.Np), dtype=np.float32)
                
                if(self.v == 1.2):
                    self.T_C[:, ientry] = np.frombuffer(f.read(4 * (self.NTemp-1)), dtype=np.float32)
                    self.TCPU[ientry] = np.frombuffer(f.read(4), dtype=np.float32)
                else:
                    self.T_C[:, ientry] = np.frombuffer(f.read(4 * self.NTemp), dtype=np.float32)
                
                
                self.Vcage[ientry] = struct.unpack('f', f.read(4))[0]
                self.AmbNs[ientry] = struct.unpack('f', f.read(4))[0]
                self.B0flag[ientry] = struct.unpack('i', f.read(4))[0]
                
                if (self.v == 1.2):
                    self.sig[ientry] = struct.unpack('i', f.read(4))[0] 
                if (self.v >= 1.1):
                    self.tstamp1[ientry] = struct.unpack('q', f.read(8))[0]  # int64
                
                self.tstamp2[ientry] = struct.unpack('q',f.read(8))[0]
                self.colcode[ientry] = struct.unpack('i', f.read(4))[0]

    def extractCleanData(self):
        # Validate data
        test_RGB = np.isin(self.colcode, [0b000, 0b001, 0b010, 0b011, 0b100, 0b101, 0b110, 0b111])   # Calculates Boolean Array
        test_B0flag = np.isin(self.B0flag, [0, 1])   # Calculates Boolean Array
        test_Vi = (self.Vi.max(axis=0) <= self.Vpi) & (self.Vi.min(axis=0) >= 0)   # Calculates Boolean Array
        test = test_RGB & test_B0flag & test_Vi     # Combines results from each boolean array
        good_locs = np.where(test)[0]   # Mask

        # Filter good data
        self.Vi = self.Vi[:, good_locs]
        self.Wght = self.Wght[:, good_locs]
        self.T_C = self.T_C[:, good_locs]
        self.Vcage = self.Vcage[good_locs]
        self.AmbNs = self.AmbNs[good_locs]
        self.B0flag = self.B0flag[good_locs]
        self.tstamp1 = self.tstamp1[good_locs]
        self.tstamp2 = self.tstamp2[good_locs]
        self.colcode = self.colcode[good_locs]

        # Rescale raspberry pi time from microseconds to 

        # Rearrange Weight Data
        #self.Wght = np.flip(self.Wght[::-1, :].T, axis=1)
        self.Wght = self.Wght[::-1, :].T
        
        # Rearrange Vi data
        #self.Vi = np.flip(self.Vi[::-1, :].T, axis=1)
        self.Vi = self.Vi[::-1, :].T


        # Rescale Timestamp from microsescond to millisecond;
        self.tstamp1 = (0.001 * self.tstamp1)
        self.tstamp2 = (0.001 * self.tstamp2)
    
    # Associate each element in the data matrix with a weighted timestamp. The logic is as follows:
    # There are 16 recordings of the ADC, tstamp1 is the timestamp before the recording starts, and tstamp2 is the timestamp after 
    # the recording ends. Multiply recording duration, dt = tstamp2 - tstamp1, with a factor, [1/32, 3/32, 5/32, 7/32 ... 31/32]. 
    # The 16-element row vector containing (dt * factor[j]) are the timestamps corresponding to the ADC channel being recorded. 
    def new_sc_timestamps(self):

        self.Nt = self.Vi.shape[0] # Nt: Number of points recorded by the Smart Cushion.
        Nd = 16 # Nd: Number of ADC Channels; 11 Pressure, 3 Temp, 1 Sound, 1 cage voltage sensors  

        # A second matrix that will hold all the timestamps of each data point recorded. 
        self.new_time = np.zeros((self.Nt, Nd),dtype='float')

        for i in range(self.Nt):
            
            tend, tstart = self.tstamp2[i], self.tstamp1[i] 
            dt = tend - tstart  # duration of recording
            
            for j in range(Nd):

                factor = (1 + (2*j)) / 32    # Compute factor, which are all odd fractions 1/32 - 31/32
                self.new_time[i,j] = tstart + (factor*dt)
                


    # The new time stamps are made for an ADC matrix whose columns are arranged: A0, A1, A2, A3, ... A15
    # We know that the pressure sensor numbering does not align with the ADC channels, so we need to re-assign each column of new_time
    # Pressure to ADC Index: 

    #  Pressure:    | P1   P2   P3   P4   P5   P6   P7   P8   P9   P10  P11  |
    #  ADC Channel: | A15  A13  A9   A4   A11  A7   A10  A8   A3   A2   A5   |
    #  Array Idx:   | V[0] V[1] V[2] V[3] V[4] V[5] V[6] V[7] V[8] V[9] V[10] |
    
    
    # Realignment 
    def realign_sc(self,value_type="Wght"):
        
        # A 3D-array structured as (Nt, Nd, T) where the third axis is the re-assigned timestamp column corresponding
        # to 'Nd'. Example: aligned_capi[:, 5, :] is all the data from P6 with the correct timestamp column associated with it
        if value_type == "Wght":
            self.rearranged_time = self.new_time[:, [15, 15, 9, 4, 11, 7, 10, 8, 3, 2, 5]]   # Numpy advanced indexing.
            self.aligned_sc = np.stack((self.Wght, self.rearranged_time),axis=2)
            print(f"Shape of new_time: {self.rearranged_time.shape}")
            print(f"Shape of Vi: {self.Wght.shape}")
        
        elif value_type == "Vi":
            self.rearranged_time = self.new_time[:, [15, 15, 9, 4, 11, 7, 10, 8, 3, 2, 5]]   # Numpy advanced indexing.
            print(f"Shape of new_time: {self.rearranged_time.shape}")
            print(f"Shape of Vi array: {self.Vi.shape}")
            self.aligned_sc = np.stack((self.Vi, self.rearranged_time),axis=2)


        '''        for i in range(self.aligned_sc.shape[1]):

            # NOTE: put this in a log, some code that confirms the time difference between each time stamp is indeed +2/32 away from each other in each row
            plt.figure(3)
            plt.plot(self.aligned_sc[:,i,1],self.aligned_sc[:,i,0]) # You can test which wghts index corresponds to ADC channel here
            print(f"aligned_sc[{i}]")
            plt.show()'''
    
    def plot(self,array):
        plt.figure(1)
        tstamp = (self.tstamp1 + self.tstamp2)/2
        t = (self.tstamp2 - self.tstamp2[0]) / 1e6  # Convert to seconds
    
        plt.plot(t, array)  # Weight in grams
        plt.title(self.SC_data)
        plt.gcf().set_size_inches(12, 4)
        plt.xlabel("Time (s)")
        plt.ylabel("value")
        plt.show()


    def plotData(self,iterative=False,scram=False,addon=None):

        if(self.v == 1.1):
            tstamp = (self.tstamp1 + self.tstamp2)/2
        
        else:
            tstamp = self.tstamp2
        
        if scram:
            Vi = self.Vi[: , [8, 7, 2, 9, 4, 6, 1, 5, 3, 0]]
        
        # Plot the pressure (weight) data
        #plt.figure(1)
        #t = (tstamp - tstamp[0]) / 1e3  # Convert milliseconds to seconds

        
        if iterative:
            for i in range(Vi.shape[1]):
                plt.plot(t, Vi[:,i])  # Weight in grams
                print(f"Vi[{i}] - {len(Vi[:,i])}")

                plt.title(self.SC_data)
                plt.gcf().set_size_inches(12, 4)
                plt.xlabel("Time (s)")
                plt.ylabel("Voltage")
                plt.show()
        
        else:

            # Plot the pressure (voltage) data
            plt.figure(1)
            t = (self.tstamp2 - self.tstamp2[0]) / 1e6  # Convert to seconds
            plt.plot(t, self.Vi)  # raw voltage
            
            plt.title(self.SC_data)
            plt.gcf().set_size_inches(12, 4)
            plt.xlabel("Time (s)")
            plt.ylabel("Voltage")


        # Plot the temperature data
        plt.figure(2)
        plt.plot(t, self.T_C[0, :], linestyle='none', marker='.', markersize=4, color=[0, 0.6, 0])
        if self.NTemp > 1:
            plt.plot(t, self.T_C[1, :], linestyle='none', marker='.', markersize=4, color=[0, 0, 1])
        if self.NTemp > 2:
            plt.plot(t, self.T_C[2, :], linestyle='none', marker='.', markersize=4, color=[0.9, 0, 0])

        print(f"Vi shape: {self.Vi.shape}")

        plt.xlabel("Time (s)")
        plt.ylabel("Temperature (°C)")
        plt.show()