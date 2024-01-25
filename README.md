# PVxTE web tool

The PVxTE web tool has been developed as part of the Swiss project PVTutt'Elettrico. It allows public transportation companies to assess the feasibility of transitioning toward electric buses. With growing concerns over the climate impacts 
of the transportation sector and fossil fuel prices, many public transportation companies wish to evaluate the potential to gradually 
transition away from fossil fuel-driven buses to an electric bus fleet. 
Especially in Europe, buses are just one part of the highly interconnected and integrated public transportation system. 
For example, bus schedules are highly integrated with the intercity and regional train schedules, which makes the electrification of buses a question with stringent constraints. 
Moreover, the erection of fast-charging infrastructure in public spaces is a contentious issue due to real estate availability and public opinion. 
These factors make developing an optimal planning tool for electric buses an extremely complicated endeavour. 
The PVxTE web tool, as such, is a simplified tool that we can deploy and run via an open-source web service based on Flask. 


The project PVTutt'Elettrico was financed by the [Swiss Federal Office of Transport](https://www.bav.admin.ch/bav/it/home.html) as part of the 
[SETP 2050](https://www.bav.admin.ch/bav/it/home/temi-generali/ricerca-e-innovazione/programmi-di-ricerca-e-d-innovazione/Energia2050.html) programme activities with the grant agreement 0780000887.
PVxTE is an open-source project released under [MIT license](https://github.com/supsi-dacd-isaac/pvxte-web/blob/main/LICENSE).


The rest of the documentation explains the architecture of the web tool. It also provides an easy-to-understand guide on how to use this web tool for your specific use case.  

<p align="center">
  <img src="https://github.com/supsi-dacd-isaac/pvxte-web/blob/main/docs/img/landing_page.png" alt="Landing page" width="85%"/>
</p>
<p align="center"><em>Figure 1: Landing page</em></p>

## Model basics
The workhorse of the web tool is a non-linear mixed integer (MINLP) optimization model. 
It extends the classical vehicle scheduling problem with additional time windows and battery charging and discharging constraints. 
Moreover, we also have a graph-theoretic optimization model that runs based on the results of the MINLP model. 
The purpose of the graph-theoretic optimization model is to calculate the minimum number of drivers a transportation company 
would need to execute the optimal bus schedule, which is the result of the MINLP model. 

## User Interface

### Inputs
Bus model description:
* Bus model identifier (string) – A name to uniquely identify the bus model,
* Bus length in meters (integer),
* Empty weight of a bus without battery in kg (integer),
* Maximum passenger capacity (integer),
* Minimum number of battery packs required (integer),
* Maximum number of battery packs allowed (integer),
* Size of one battery pack in kWh (integer),
* Bus lifetime in years (integer),
* Lifetime of a battery pack in years (integer),
* Investment cost of a bus in CHF (real),
* Investment cost of a battery pack in CHF (real)

Schedule, terminal, and distance information:
* Terminals file (CSV format described [here](https://github.com/supsi-dacd-isaac/pvxte-web?tab=readme-ov-file#terminals-and-distance-matrix)) providing elevation and charging infrastructure availability at each terminal location,
* Distance matrix (CSV format described [here](https://github.com/supsi-dacd-isaac/pvxte-web?tab=readme-ov-file#terminals-and-distance-matrix)) providing distances between each pair of terminal stations,
* Charging power in kW (real),
* Daily trips schedule (CSV format described [here](https://github.com/supsi-dacd-isaac/pvxte-web?tab=readme-ov-file#step1)).

Other inputs:
* Investment cost of a charger in CHF (real),
* Grid connection cost in CHF/ kW (real),
* Lifetime of a charger in years (real),
* Lifetime of the transformer in years (real),
* Interest rate as a percentage (real),
* Bus maintenance cost in CHF/km (real),
* Charging efficiency (real),
* Annual usage per single bus in km (integer),
* Energy cost in CHF/ kWh (real),
* Annual increase in energy cost in percentage,
* Demand tariff in CHF/ kW (real),
* Diesel cost in CHF/liter (real),

### Signup and login page

You can access the PVxTE web tool via [pvxte.isaac.supsi.ch](https://pvxte.isaac.supsi.ch). If you do not have a user account, you may create one by clicking “Sign up” in the top right corner of the landing page (Figure 2).
If you already have an account, you can log in to the web tool as shown in Figure 3.  

<p align="center">
  <img src="https://github.com/supsi-dacd-isaac/pvxte-web/blob/main/docs/img/new_user.png" alt="New user" width="85%"/>
</p>
<p align="center"><em>Figure 2: Create new user account</em></p>

<p align="center">
  <img src="https://github.com/supsi-dacd-isaac/pvxte-web/blob/main/docs/img/log-in.png" alt="Login" width="85%"/>
</p>
<p align="center"><em>Figure 3: Login page</em></p>

The user profile encapsulates all the information related to a particular user that includes input data, simulations, and results. 

### Company management section

The company management section is where you can manage your company specific input data. These include the type of buses you wish to simulate, the bus terminals, 
and the distances between the terminal stations. Specifying these data under company profile makes it easier to reuse the same data for multiple simulations.

As shown in Figure 4, the pre-existing bus models appears in at the top of the company management section as a table. 

shows a section of the company management section. The left menu shows the different options available under the company management section. You can add 
a new bus model by clicking “Add new”. You may also edit an existing bus model by clicking on the name of 
the existing bus model in the table.

<p align="center">
  <img src="https://github.com/supsi-dacd-isaac/pvxte-web/blob/main/docs/img/buses.png" alt="bus models" width="85%"/>
</p>
<p align="center"><em>Figure 4: Company management - Bus models</em></p>

Figure 5 shows the data that is required to set up a new bus model. These data are typically available from the electric bus manufacturers. 
The created bus models are stored in a database linked to the company profile. Therefore, you can reuse the same bus model for multiple feasibility evaluations using the same login. 
The data is not shared between user accounts.  

<p align="center">
  <img src="https://github.com/supsi-dacd-isaac/pvxte-web/blob/main/docs/img/new_bus.png" alt="new bus" width="65%"/>
</p>
<p align="center"><em>Figure 5: Create new bus model</em></p>

### Terminals and distance matrix

The two inputs that we use to characterize the public transportation network are the information pertaining to the terminals and the distance matrix.
They are uploaded in a pre-defined CSV format. We recommend the users download and edit the example file provided to ensure there will be no conflicts with the acceptable data formats (See Figure 6). 
The terminals file contains the following columns.

* `terminal_station:` Terminal station name (string), if the terminal station is the depot, then the name must be “Depot”,
* `elevation_m:` Elevation above sea level in meters (real),
* `is_charging_station:` Type of charging infrastructure available at the terminal (string, categorical), the options are “no”, “depo_charger”, and “not_depo_charger”.

The web tool is limited to simulating single-depot problems. Therefore, only one terminal station can be designated as the depot with the name "Depot" (Case sensitive). 
The depot must have charging infrastructure of the type "depo_charger". Failure to assign charging infrastructure to the depot results in model infeasibility.
Non depot chargers can be used to assign fast-charging infrastructure such as pantographs at terminal stations other than the depot.

The inability of the web tool to simulate multi-depot scenarios is not a problem for the vast majority of practical applications concerning urban and suburban bus lines. 
Even in a rare multi-depot scenario, if all buses return to the same depot they originate from, the user can disaggregate the multi-depot problem into several single-depot problems as a workaround.
The cases in which a bus may not return to the same depot it originates from is a typical use case for long-distance buses, such as FlixBus, which is beyond the scope of this project.

The distances file describes the travel distance (in km) and time (minutes) between any two terminal stations. 
If there are *k* terminal stations (including the depot) in the terminals file, then there should be a *k<sup>2</sup>* number of rows in the distance file. 
The user can manually restrict buses from transiting between two terminal stations by setting the travel distance and time between the two terminals to a very large number (i.e., 1000). 
Make sure to use the correct units for both time and distance. 

<p align="center">
  <img src="https://github.com/supsi-dacd-isaac/pvxte-web/blob/main/docs/img/terminals.png" alt="terminals" width="65%"/>
</p>
<p align="center"><em>Figure 6: Description of the public transport network using terminals and distance information </em></p>


## Create a new simulation

### Step1

A new simulation can be created by clicking “Create simulation” in the left menu.  You must choose one of the bus models you already created from the drop-down list, as shown in the following picture. Currently, we can only assign one bus type per simulation. 
Although this simplifies, it does correspond to most real-world cases where, typically, only one type of bus is used to serve a particular line. If you need to simulate several bus lines simultaneously and they use different bus types, 
we recommend you separate them into different simulations. 

![Step1](https://github.com/supsi-dacd-isaac/pvxte-web/blob/main/docs/img/create-sim-step-1.png)

Next, you  must also provide the schedule of the buses serving the line/ lines that you wish to simulate. To avoid unexpected conflicts, we recommend the users to follow the example file. The accepted timetable is a table format that has the following columns:

* 	`bus_id`: bus identifier(data type: string)
* `line_id`: bus line (data type: string)
*	`starting_city`: departure station (data type: string), it must be one of the terminal stations in the terminals file.
*	`arrival_city`: destination station (data type: string), it must be one of the terminal stations in the terminals file.
*	`departure_time`: departure time in minutes after midnight (data type: integer)
*	`arrival_time`: arrival time in minutes after midnight (data type: integer)
*	`day_type`: type of day you want to simulate (data type: string). For example, this is useful to simulate weekday and weekend schedules.

### Step2

In the 2nd step, you must choose which bus lines and the day type the simulation must include. The selection window automatically displays all the available bus lines and day types extracted from the timetable you uploaded in the first step, 
as shown in the following picture. 

![Step2 Top](https://github.com/supsi-dacd-isaac/pvxte-web/blob/main/docs/img/run_step2_top_selected.png)

If the bus line you want to simulate is interconnected or shares buses with some other lines, you must include all the interconnected bus lines in the simulation as well. 

For a detailed explanation of the _Energy model calibration parameter_ shown in the bottom of the image above, please refer to the [related section](https://github.com/supsi-dacd-isaac/pvxte-web#energy-consumption-calibration).

This step also provides you with the space to include economic parameters 
that the model uses to estimate the investment and operational costs of the electric bus fleet, as depicted in the next picture. After that, you can simulate by clicking “Launch simulation.” 

![Step2 Bottom](https://github.com/supsi-dacd-isaac/pvxte-web/blob/main/docs/img/capex-opex-section.png)

## Viewing results

Once the simulation is complete, you will get an email notifying you that your simulation is finished.

Saved simulation results can be accessed by clicking “Simulations” on the main menu, as shown in the following image (Figure 11). 
If a valid solution (a feasible solution) is not found, those instances will be highlighted as seen in the figure. In such a case, the user is advised to ensure all inputs are correct, 
especially if the trip timetable does not have discontinuities. Common causes of infeasibility are;
1.	_Maximum battery size is insufficient_: The user can choose a higher battery pack size or increase the maximum number of battery packs, if possible. 
2.	_Discontinuous timetables_: If the buses perform empty rides (trips without passengers to relocate the bus to a different station), make sure these trips are also included in the timetable.
3.	_Unable to fully recharge_: In some cases, with the maximum allowed charging power given, the buses may not be able to recharge fully. If that is the case, try with higher charging power or with the fast charging option.

![Simulations List](https://github.com/supsi-dacd-isaac/pvxte-web/blob/main/docs/img/simulations_list.png)

There might be other edge instances that result in infeasibilities that must be evaluated case-by-case with the specific user inputs. 
You can access the detailed results (including the option to download raw results files in CSV format). Accessing “details” takes you to the following window.

![Simulation Detail](https://github.com/supsi-dacd-isaac/pvxte-web/blob/main/docs/img/simulation_detail1.png)

The simulation detail page basically contains four tables. 
* The first table reports the most important inputs used in the simulation
* The second table shows the main output calculated by the simulation; among them, the number of electric buses, the minimum battery size and the corresponding number of battery packs in the “Buses and batteries sizing” table. The model assumes that all the buses are homogeneous; therefore, we must understand “minimum battery size” as the minimum battery size required for the feasible operation of every bus in the fleet. Additionally, the tool also provides the minimum number of drivers required to operate the bus fleet under the work and rest time requirements in the labor law. The model does not account for operational conveniences such as convenient driver exchange locations in its estimate. Therefore, this value must be interpreted as a lower bound on the actual driver requirement.
* The third table shows the analysis of the investment costs
* The last table reports the results related to the operating costs and the GHG emissions
  
At the bottom of the page, two graphs about the economic aspects are shown to depict to the user the comparison between the electrical and the diesel solution.

## Energy consumption calibration

We use a generic energy consumption model, taking into account many important parameters published by Hjelkrem et al. (2021) [[1]](#1) with one key simplification to improve the run time and user convenience.
Hjelkrem et al. modeled each trip as a sequence of many small trip segments, each with a known elevation profile. In reality, the transportation companies do not have such detailed elevation profiles of the roads. Consequently, in our simplified model, we only consider the elevations of the start and end points of the trip for the energy calculation. This approximation and the use of other approximate parameters, such as rolling resistance, average velocity, etc., leads to inaccuracies in the energy consumption estimates. To minimize the impact of these inaccuracies, we introduce a calibration parameter that the user can change based on their current knowledge about the energy intensity (kWH/km) of each bus line.  

First, simulate with the default calibration parameter value of 1.0 and observe the resulting bus energy efficiency (kWh/km). If it approximately matches your previous knowledge, then the energy consumption model already generates a reasonable approximation of the energy consumption of your bus line. Otherwise, either increase or decrease the calibration parameter accordingly. 

## Limitations of the web tool

The current version of the web tool that is developed within the scope of the PVTutt'Elettrico project has the following limitations.

* The model supports evaluating the one-to-one replacement of diesel buses with electric buses. Therefore, the number of electric buses must be equal to the number of diesel buses in the original fleet. Moreover, the model assumes that the electric buses are homogeneous. 
* The fleet must consist of electric buses only, i.e., the user cannot simulate a mixed fleet of diesel and electric buses in the same simulation,
* The fleet must consist of only one bus type, i.e., the user cannot simulate multiple bus types (e.g., 12m and 18m) in the same simulation.
* Multi-depot scenarios are not supported; however, the user can simulate multi-depot scenarios by disaggregating the problem into several single-depot problems.

## References
<a id="1">[1]</a> 
Hjelkrem, et al. (2021). 
A battery electric bus energy consumption model for strategic purposes: Validation of a proposed model structure with data from bus fleets in China and Norway.
Transportation Research Part D: Transport and Environment, 94, 102804.

