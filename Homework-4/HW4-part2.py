# Bayesian Multiple Linear Regression
import matplotlib.pyplot as plt 
import numpy as np  
import pandas as pd 
import pymc3 as pm
import arviz as az

timestamp_dataset = []
refSt_dataset = []
sensor_o3_dataset = []
temp_dataset = []
hum_dataset = []
index_list = []

end_time = '10/07/2017 23:30'

index = 0
with open('data.csv', 'r') as f:
    # if the first row is the header
    for row in f:
        if index == 0:
            index += 1
            continue

        row = row.split(';')
        date = row[0]
        refSt = row[1]
        sensor_o3 = row[2]
        temp = row[3]
        hum = row[4]

        index_list.append(index)
        timestamp_dataset.append(date)
        refSt_dataset.append(float(refSt))
        sensor_o3_dataset.append(float(sensor_o3))
        temp_dataset.append(float(temp))
        hum_dataset.append(float(hum))

        index += 1
        if date == end_time:
            break

x = []
for i in range(len(sensor_o3_dataset)):
    x.append([sensor_o3_dataset[i], temp_dataset[i], hum_dataset[i]])

x = np.array(x)
y = np.array(refSt_dataset)

data = pd.DataFrame({'o3': x[:,0], 'temp': x[:,1], 'hum': x[:,2], 'y': y})
data

az.style.use("arviz-darkgrid")
my_model = pm.Model()
with my_model:
    pm.glm.GLM.from_formula('y ~ o3 + temp + hum', data, family=pm.glm.families.Normal())
    trace = pm.sample(1000, return_inferencedata=False)
    ppc = pm.sample_posterior_predictive(trace, random_seed=42, progressbar=True)
    trace_updated = az.from_pymc3(trace, posterior_predictive=ppc)
    az.plot_trace(trace_updated, figsize=(12,10), legend=True)
    az.plot_posterior(trace_updated, textsize=10, kind='hist', figsize=(10,8), bins=50, color='blue', point_estimate='mean')
    az.plot_ppc(trace_updated)


# Predictive mean for each sample in the trace
def predict_sensor_reading(reading, trace):
    # Extracting samples
    theta0_samples = trace['Intercept']
    theta1_samples = trace['o3']
    theta2_samples = trace['temp']
    theta3_samples = trace['hum']
    sigma_samples = trace['sd']

    # Predictive mean for each sample in the trace
    mu_pred = theta0_samples + theta1_samples * reading[0] + theta2_samples * reading[1] + theta3_samples * reading[2]

    # Generate predictive samples
    predictive_samples = np.random.normal(mu_pred, sigma_samples)
    return predictive_samples


reading_1107_11 = [264.744, 28.97, 30.33]
pred_samples_1107_11 = predict_sensor_reading(reading_1107_11, trace)
plt.figure(figsize=(10,8))
plt.hist(pred_samples_1107_11.flatten(), bins=50, alpha=0.7, color='blue')
plt.axvline(x = 82, color = 'red', ls='--', label = 'Reference station value')
plt.title("Predictive distribution 11/07/2017 11:00")
plt.xlabel("Predictive distribution")
plt.ylabel("Frequency")
plt.legend()
plt.show()

reading_1107_11 = [498.9, 33.2, 28.3]
pred_samples_1107_11 = predict_sensor_reading(reading_1107_11, trace)
plt.figure(figsize=(10,8))
plt.hist(pred_samples_1107_11.flatten(), bins=50, alpha=0.7, color='blue')
plt.axvline(x = 164.0, color = 'red', ls='--', label = 'Reference station value')
plt.title("Predictive distribution 11/07/2017 11:00")
plt.xlabel("Predictive distribution")
plt.ylabel("Frequency")
plt.legend()
plt.show()

full_timestamp_dataset = []
full_refSt_dataset = []
full_sensor_o3_dataset = []
full_temp_dataset = []
full_hum_dataset = []
full_index_list = []

start_time = '11/07/2017 0:00'
ctrl = True

index = 0
with open('data.csv', 'r') as f:
    # if the first row is the header
    for row in f:
        if index == 0:
            index += 1
            continue

        row = row.split(';')
        date = row[0]
        if date != start_time and ctrl:
            continue
        ctrl = False
        refSt = row[1]
        sensor_o3 = row[2]
        temp = row[3]
        hum = row[4]

        full_index_list.append(index)
        full_timestamp_dataset.append(date)
        full_refSt_dataset.append(float(refSt))
        full_sensor_o3_dataset.append(float(sensor_o3))
        full_temp_dataset.append(float(temp))
        full_hum_dataset.append(float(hum))

        index += 1

x = []
for i in range(len(full_sensor_o3_dataset)):
    x.append([full_sensor_o3_dataset[i], full_temp_dataset[i], full_hum_dataset[i]])

x = np.array(x)
y = np.array(full_refSt_dataset)

full_data = pd.DataFrame({'o3': x[:,0], 'temp': x[:,1], 'hum': x[:,2], 'y': y})

predicted_values = []
for _, item in full_data.iterrows():
    reading = [item['o3'], item['temp'], item['hum']]
    predictive_samples = predict_sensor_reading(reading, trace)
    predicted_values.append(predictive_samples)

predicted_values = np.array(predicted_values)
expected_values = np.mean(predicted_values, axis=1)
lower_credibility_interval = np.percentile(predicted_values, 2.5, axis=1)
upper_credibility_interval = np.percentile(predicted_values, 97.5, axis=1)

plt.figure(figsize=(15,10))
plt.fill_between(full_data.index, lower_credibility_interval, upper_credibility_interval, color='lightgreen', alpha=0.4)
plt.plot(full_data.index, expected_values, label="Expected Value", color='blue')
plt.scatter(full_data.index, full_data['y'], color='red', label="Reference Station Readings", s=10)
plt.title("Expected value compared to reference station values on July 11th")
plt.axvline(x = 36, color = 'skyblue', ls='--', label = 'Value of 11-07 at 18:00')
plt.axvline(x = 22, color = 'violet', ls='--', label = 'Value of 11-07 at 11:00')
plt.xticks(np.arange(0, len(full_data), step=2))
plt.xlabel("Samples")
plt.ylabel("Values")
plt.legend()
plt.show()