# Resilient Cyber-physical system design using Automated Attack Generation

This repo contains a suite of simulations for our paper:
```
Zheng, Yu; Sayghe, Ali; Anubi, Olugbenga (2021): Algorithm Design for Resilient Cyber-Physical Systems using an Automated Attack Generative Model. TechRxiv. Preprint. https://doi.org/10.36227/techrxiv.17032898.v1 
```

**Goals:**
- Show how to use the proposed automated attack generator
- Show how automated attack generator improve the attack detection and localization precision
- Show a concurrent learning-based framework for improved resilient estimation 

## Simulation system - Water Distributed System
![water_distribution_sys](https://user-images.githubusercontent.com/36635562/160011533-44149d2e-43c4-482a-a2c6-857bbddde641.png)
This water distribution system is an 11-tanks system which contains 10 operating water tanks and a storage tank. The goal is to regulate all operating tanks' water levels around desired values. The magnetic valves V at the entrance pipelines of operating tanks are controlled to adjust the tank water levels. The magnetic valve at the entrance of the storage tank is fixed at a constant opening value. It is assumed that there are water level measurement sensors and pressure sensors in the pipelines. <br>
![image](https://user-images.githubusercontent.com/36635562/127782515-0093f227-5820-4f9f-833d-1b3d1a3a1ab7.png) <br>
The system has 10 states, 10 control inputs, and 61 measurements. The A,B,C,D matrices are saved in `common_fcn` folder. <br>


## Automated attack generation



## MLP training and testing


## Concurrent learning resilient estimation - Pruning algorithm
