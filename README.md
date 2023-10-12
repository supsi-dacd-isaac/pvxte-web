# PVxTE web tool

The PVxTE web tool allows public transportation companies to assess the feasibility of transitioning toward electric buses. With growing concerns over the climate impacts 
of the transportation sector and fossil fuel prices, many public transportation companies wish to evaluate the potential to gradually 
transition away from fossil fuel-driven buses to an electric bus fleet. 
Especially in Europe, buses are just one part of the highly interconnected and integrated public transportation system. 
For example, bus schedules are highly integrated with the intercity and regional train schedules, which makes the electrification of buses a question with stringent constraints. 
Moreover, the erection of fast-charging infrastructure in public spaces is a contentious issue due to real estate availability and public opinion. 
These factors make developing an optimal planning tool for electric buses an extremely complicated endeavour. 
The PVxTE web tool, as such, is a simplified tool that we can deploy and run via an open-source web service based on Flask. 
The rest of the documentation explains the architecture of the web tool. It also provides an easy-to-understand guide on how to use this web tool for your specific use case.  


## Model basics
The workhorse of the web tool is a non-linear mixed integer (MINLP) optimization model. 
It extends the classical vehicle scheduling problem with additional time windows and battery charging and discharging constraints. 
Moreover, we also have a graph-theoretic optimization model that runs based on the results of the MINLP model. 
The purpose of the graph-theoretic optimization model is to calculate the minimum number of drivers a transportation company 
would need to execute the optimal bus schedule, which is the result of the MINLP model. 

## User Interface
You can access the PVxTE web tool via pvxte.isaac.supsi.ch. On the landing page, you must create an account using a personal username and a password (Figure 2).  
