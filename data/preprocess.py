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

    # Cambiar la inicialización de fid_pks a un diccionario
    fid_pks = {
        'R_peak': np.zeros(len(Rposition), dtype=int),
        'Q_peak': np.zeros(len(Rposition), dtype=int),
        'S_peak': np.zeros(len(Rposition), dtype=int),
        'onset': np.zeros(len(Rposition), dtype=int),
        'offset': np.zeros(len(Rposition), dtype=int),
        'P_peak': np.zeros(len(Rposition), dtype=int),
        'T_peak': np.zeros(len(Rposition), dtype=int)
    }

    windowS = round(fs * 0.1)
    windowQ = round(fs * 0.05)
    windowP = round(aveHB / 3)
    windowT = round(aveHB * 2 / 3)
    windowOF = round(fs * 0.04)

    for i in range(len(Rposition)):
        thisR = Rposition[i]
        
        # First
        if i == 0:
            fid_pks['R_peak'][i] = thisR
            fid_pks['offset'][i] = thisR + windowS
        # Last
        elif i == len(Rposition) - 1:
            fid_pks['R_peak'][i] = thisR
            fid_pks['onset'][i] = thisR - windowQ
        else:
            if (thisR + windowT) < len(ECG) and (thisR - windowP) >= 1:
                # Detect Q and S peaks
                fid_pks['R_peak'][i] = thisR
                Sp = np.argmin(ECG[thisR:thisR + windowS])
                thisS = Sp + thisR
                fid_pks['S_peak'][i] = thisS
                Qp = np.argmin(ECG[thisR - windowQ:thisR])
                thisQ = thisR - (windowQ + 1) + Qp
                fid_pks['Q_peak'][i] = thisQ
                
                # Detect QRS onset and offset
                interval_q = ECG[thisQ - windowOF:thisQ]
                thisON = thisQ - (windowOF + 1) + onoffset(interval_q, 'on')
                
                interval_s = ECG[thisS:thisS + windowOF]
                thisOFF = thisS + onoffset(interval_s, 'off') - 1
                
                fid_pks['onset'][i] = thisON
                fid_pks['offset'][i] = thisOFF

    # Detect P and T waves
    for i in range(1, len(Rposition) - 1):
        lastOFF = fid_pks['offset'][i-1]
        thisON = fid_pks['onset'][i]
        thisOFF = fid_pks['offset'][i]
        nextON = fid_pks['onset'][i + 1]
        
        if thisON > lastOFF and thisOFF < nextON:

            Tzone = ECG[thisOFF:int(nextON - round((nextON - thisOFF) / 3))]
            Pzone = ECG[lastOFF + int(round(2 * (thisON - lastOFF) / 3)):thisON]

            try:
                thisT = np.argmax(Tzone)
                thisP = np.argmax(Pzone)
            except Exception as e:
                print("Error in Tzone or Pzone:", e)
                continue
            
            fid_pks['P_peak'][i] = lastOFF + round(2 * (thisON - lastOFF) / 3) + thisP - 1
            fid_pks['T_peak'][i] = thisOFF + thisT - 1

    return fid_pks

def normalize(data, min_val=0, max_val=1):
    """
    Function to normalize the ECG signal to range [0,1].
    
    :param data: The ECG signal
    :param min_val: Minimum value for scaling (default 0)
    :param max_val: Maximum value for scaling (default 1) 
    :return: Normalized ECG signal scaled to [min_val, max_val]
    """
    #data = data - np.mean(data)
    #data = data / np.std(data)
    #return data

    data_min = np.min(data)
    data_max = np.max(data)
    data_norm = (data - data_min) / (data_max - data_min)
    data_scaled = data_norm * (max_val - min_val) + min_val
    return data_scaled