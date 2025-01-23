import numpy as np

def onoffset(interval, mode):
    """
    Function calculates on/off set of QRS complex.
    
    :param interval: The ECG signal interval
    :param mode: Mode either 'on' for onset or 'off' for offset
    :return: ind - the index of the onset or offset
    """
    slope = []
    
    # Calculate the slope of the interval
    for i in range(1, len(interval) - 1):
        slope.append(interval[i + 1] - interval[i - 1])
    
    # Using MIN_SLOPE to determine onset placement
    if mode == 'on':
        ind = np.argmin(np.abs(slope))
    elif mode == 'off':
        slope_th = 0.2 * np.max(np.abs(slope))
        slope_s = np.where(np.abs(slope) >= slope_th)[0]
        ind = slope_s[0] if len(slope_s) > 0 else -1  # Return -1 if no slope_s is found
    else:
        raise ValueError("Invalid mode, please select 'on' or 'off'")
    
    return ind

def qsPeaks(ECG, Rposition, fs):
    """
    Q, S peaks detection.
    
    :param ECG: The ECG signal
    :param Rposition: Positions of R-peaks
    :param fs: Sampling frequency
    :return: ECGpeaks (detected Q, S, P, T peaks)
    """
    # Average heart beat length
    aveHB = len(ECG) / len(Rposition)
    
    # Initialize an array to store the fiducial points
    fid_pks = np.zeros((len(Rposition), 7), dtype=int)
    # fiducial points: P wave onset, Q wave onset, R wave peak, 
    # S wave onset, T wave onset, R wave offset, T wave offset
    
    # Set up the search windows (in samples)
    windowS = round(fs * 0.1)
    windowQ = round(fs * 0.05)
    windowP = round(aveHB / 3)
    windowT = round(aveHB * 2 / 3)
    windowOF = round(fs * 0.04)
    
    # Process each R-position
    for i in range(len(Rposition)):
        thisR = Rposition[i]
        
        # First
        if i == 0:
            fid_pks[i, 3] = thisR
            fid_pks[i, 5] = thisR + windowS
        # Last
        elif i == len(Rposition) - 1:
            fid_pks[i, 3] = thisR
            fid_pks[i, 1] = thisR - windowQ
        else:
            if (thisR + windowT) < len(ECG) and (thisR - windowP) >= 1:
                # Detect Q and S peaks
                fid_pks[i, 3] = thisR
                Sp = np.argmin(ECG[thisR:thisR + windowS])
                thisS = Sp + thisR
                fid_pks[i, 4] = thisS
                Qp = np.argmin(ECG[thisR - windowQ:thisR])
                thisQ = thisR - (windowQ + 1) + Qp
                fid_pks[i, 2] = thisQ
                
                # Detect QRS onset and offset
                interval_q = ECG[thisQ - windowOF:thisQ]
                thisON = thisQ - (windowOF + 1) + onoffset(interval_q, 'on')
                
                interval_s = ECG[thisS:thisS + windowOF]
                thisOFF = thisS + onoffset(interval_s, 'off') - 1
                
                fid_pks[i, 1] = thisON
                fid_pks[i, 5] = thisOFF
    
    # Detect P and T waves
    for i in range(1, len(Rposition) - 1):
        lastOFF = fid_pks[i - 1, 5]
        thisON = fid_pks[i, 1]
        thisOFF = fid_pks[i, 5]
        nextON = fid_pks[i + 1, 1]
        
        if thisON > lastOFF and thisOFF < nextON:

            
            Tzone = ECG[thisOFF:int(nextON - round((nextON - thisOFF) / 3))]
            Pzone = ECG[lastOFF + int(round(2 * (thisON - lastOFF) / 3)):thisON]

            try:
                thisT = np.argmax(Tzone)
                thisP = np.argmax(Pzone)
            except Exception as e:
                print("Error in Tzone or Pzone:", e)
                continue
            
            fid_pks[i, 0] = lastOFF + round(2 * (thisON - lastOFF) / 3) + thisP - 1
            fid_pks[i, 6] = thisOFF + thisT - 1
    
    # Filter out invalid peaks (those with 0 value)
    #ECGpeaks = []
    #for i in range(len(Rposition)):
        #if np.prod(fid_pks[i, :]) != 0:
    #    ECGpeaks.append(fid_pks[i, :])
    
    return np.array(fid_pks) #np.array(ECGpeaks)

def normalize(signal, min_val=-5, max_val=5):
    """
    Function to normalize the ECG signal.
    
    :param signal: The ECG signal
    :return: Normalized ECG signal
    """
    signal = (signal - min_val) / (max_val - min_val)
    return signal