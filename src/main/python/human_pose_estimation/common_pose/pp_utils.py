# --------------------------------------------------------------------------------
# BodyFlow
# Version: 2.0
# Copyright (c) 2024 Instituto Tecnologico de Aragon (www.ita.es) (Spain)
# Date: March 2024
# Authors: Ana Caren Hernandez Ruiz                      ahernandez@ita.es
#          Angel Gimeno Valero                              agimeno@ita.es
#          Carlos Maranes Nueno                            cmaranes@ita.es
#          Irene Lopez Bosque                                ilopez@ita.es
#          Jose Ignacio Calvo Callejo                       jicalvo@ita.es
#          Maria de la Vega Rodrigalvarez Chamarro   vrodrigalvarez@ita.es
#          Pilar Salvo Ibanez                                psalvo@ita.es
#          Rafael del Hoyo Alonso                          rdelhoyo@ita.es
#          Rocio Aznar Gimeno                                raznar@ita.es
#          Pablo Perez Lazaro                               plazaro@ita.es
#          Marcos Marina Castello                           mmarina@ita.es
# All rights reserved 
# --------------------------------------------------------------------------------

import numpy as np
import pandas as pd

def kalman_filter(signal, initial_state_mean, initial_state_covariance, process_noise, measurement_noise):
    num_time_steps = len(signal)

    # Initialize arrays to store the filtered state and covariance estimates
    filtered_state_mean = np.zeros(num_time_steps)
    filtered_state_covariance = np.zeros(num_time_steps)

    # Initialize the state estimate and covariance
    state_mean = initial_state_mean
    state_covariance = initial_state_covariance

    for t in range(num_time_steps):
        # Measurement update
        kalman_gain = state_covariance / (state_covariance + measurement_noise)
        state_mean = state_mean + kalman_gain * (signal[t] - state_mean)
        state_covariance = (1 - kalman_gain) * state_covariance + process_noise

        # Time update
        filtered_state_mean[t] = state_mean
        filtered_state_covariance[t] = state_covariance

    return filtered_state_mean, filtered_state_covariance

def apply_kalman(filename):
    
    initial_state_mean = 0.0  # Initial state mean
    initial_state_covariance = 1  # Initial state covariance

    process_noise = 0.8  # Process noise (Q)
    measurement_noise = 0.8  # Measurement noise (R)
            
    
    df = pd.read_csv(filename, index_col='timestamp')
    filtered_df = df.copy()
    for person_id in range(1, df['id'].max() +1): 
        actual_person = df.loc[df['id'] == person_id ]
        actual_person_copy = actual_person.copy()
        print (person_id)
        for col in actual_person.columns:
            if 'coordinate' in col and 'bodyLandmarks2d' not in col:
                data =  actual_person[col].values
                filtered_state_mean, filtered_state_covariance = kalman_filter(data, 
                                                                            initial_state_mean,
                                                                            initial_state_covariance,
                                                                            process_noise,
                                                                            measurement_noise)
                print(col)
                actual_person_copy[col] = filtered_state_mean
                filtered_df[col] = actual_person_copy[col] 
                
    filtered_df.to_csv('{filename}_kf.csv')      
  
if __name__ == '__main__':
    filename = 'logs/Log_cpn_motionbert_gp2_006.csv'
    apply_kalman(filename)
    

    


    
 
