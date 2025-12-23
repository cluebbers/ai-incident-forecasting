---
layout: post
title: "AI Incident Forecasting: December 2025 Update"
date: 2025-12-28 10:00:00 +0000
categories: forecasting incidents
---

AI Incident Forecasting

## Introduction

<hook/context/lede — 1 sentence>
As AI systems become a regular part of society, real-world problems like misinformation, misuse, and unsafe behavior are increasing rapidly, often surprising those involved.

<define the issue/need — 1 sentence>
Can we predict the next AI disaster before it strikes? To move from reacting to preventing harm, we need a systematic, data-driven approach to predict AI incidents before they occur, enabling policymakers and developers to take early action.

<what you actually did — 1 sentence>
We built a complete process to predict how many incidents will occur in the future and how they will be distributed across different risk areas, using models trained on the AI Incident Database, a historical record of reported AI Incidents.

<most interesting result — 1 sentence>
Our forecasts demonstrate the urgency: the number of recorded incidents each year could be 6 to 11 times higher by 2030, especially for misuse, misinformation, and system safety issues.

<clear statement of blogpost's purpose — 1 sentence>
This post explains the approach and results, highlights what the forecast can and cannot tell us, and outlines the next steps to make incident forecasting more useful for governance.

## Motivation

<explanation of key concept — 1-2 sentences>
An “AI incident” is vague in common use. In our research, we use a simple definition: a real-world event where an AI system causes harm or almost does.
Forecasting incidents means estimating how often these events might happen in the future and what types of incidents we’re likely to see, and cases might be recorded.

Most importantly, the AI Incidents database focuses on incidents rather than individual reports. Multiple reports are often aggregated into a single incident. Therefore, we can expect the severity and the number of people affected by each incident to be much greater than in a single report.



<state of discourse, cite sources — 2 sentences>
AI incidents are increasingly tracked in resources such as the AI Incident Database and emerging public reporting efforts (for example, the OECD’s AI Incidents and Hazards Monitor and its incident reporting framework).
While many governance frameworks prioritize tracking risks over time, most existing incident work focuses on describing the past instead of estimating future incidents. Bridging this gap is essential for proactive risk management.

## Methods

<summary of *most similar* work, cite sources — 1 sentence>

The most closely related resource is the AI Risk Repository, which provides a shared set of risk categories (a “structured list of types”) but does not attempt to forecast incident counts over time.

<why yours is different — 1-2 sentences>
Differing from previous resources, our work builds a statistical forecast using mathematical models that project both the total number of recorded incidents and how that total is likely to split across risk domains, such as misuse or safety. It reports uncertainty ranges, showing where predictions might fall, rather than offering a single best guess.
We incorporate backtests by training on earlier years and checking how well they predict later years.

<why your approach makes sense — 1 sentence>
Forecasts with uncertainty ranges aid risk-aware planning when intuition fails as trends accelerate or shift.

image.png

<reiterate goal — 1 sentence>
Our goal throughout was to build a forecast that is accurate, easy to explain, and clear about its uncertainty—so that it can guide policy decisions.

<step 1 — 1-2 sentences>
To start, we gathered incidents from the AI Incident Database and its corresponding MIT Risk Repository, which labelled each incident with a risk area, from the 1990s to 2025. 
However, the newest year is often incomplete due to reporting lag, the time delay before all relevant incidents are added. To address this, we used a simple 'so-far-this-year' adjustment, known as year-to-date assimilation, that scales partial-year counts based on typical monthly reporting patterns.

<roadblock 1 — 1 sentence>
We observed a marked increase in recorded AI-incident counts beginning around 2021. We added a hinge feature to divide the model into the time before and after the hinge.

image.png

Caption: Some of the various techniques we used are visualised.

<step 2 — 1-2 sentences>
To forecast yearly incident totals, we fit a Poisson Regression model, a statistical method for estimating the number of events (incidents) expected each year, assuming the rate may change over time (such as around 2021). We give recent years extra weight so the model can respond if the world is speeding up.
To forecast the share of incident types, we use a logistic regression model. This type of model estimates the probability of each incident type (risk domain) over time. We place greater emphasis on domains that have recently surged. This helps the forecast adapt when the type mix shifts quickly.

<step 3 — 1-2 sentences>
To represent uncertainty, we ran Monte Carlo simulations, which involve conducting many randomized trials with slightly different model assumptions. This method produces a range of po ssible future outcomes rather than just a single prediction.
From these trials, we compute a 90% prediction interval, the range within which the true value should fall about 9 times out of 10 if future patterns resemble those in the model’s training period.

<step 4 — 1-2 sentences>
We tested the method using a backtest (“pretend-it’s-the-past”) evaluation. For 2022–2024, we trained on earlier years, predicted 3 years ahead, and compared the predictions with actual outcomes.
We compared our model to a baseline that simply extends the recent average forward.

By calibrating the uncertainty parameters (such as their growth), we ensure backtest accuracy and, in turn, improve the reliability of modeling recent years after 2025.

## Results

total_incidents.jpg
Description (what to draw): A single plot with Year on the x-axis and Incident count on the y-axis; show observed values through 2024 as dots, then a dashed median line for 2025–2030, plus a shaded band for the 90% interval that widens over time. Add a small annotation near 2030: “~1,250 to ~2,500 (90% interval), median ~1,900.”
Alt text: “Line chart showing historical AI incident counts through 2024 and a forecast to 2030 with uncertainty bands that widen over time.”

<main result — 1 sentence>

Our forecast shows a 6-11x increase in the total number of incidents by 2030. 

<briefly why the main result is interesting — 1-2 sentences>
This pattern is fast, compounding growth, not just a slow increase. Compounding quickly makes it hard to keep up by only reacting. Even the lower end of our prediction range (6x) shows a significant increase in incidents.

<unexpected result — 1-2 sentences>
In our backtests, our model didn’t always outperform the baseline each year, but it performed much better in 2024, when incidents surged.
 We argue this is an important tradeoff since modeling acceleration matters most when the world shifts quickly.

calibration.jpg

<other results — 1-2 sentences>
Our forecasts by area show the fastest growth in harmful misuse, misinformation, and system safety failures.
Other domains also rise, but these are most prominent under our assumptions.

[possible supplementary results figure]

incidents_category.jpg
Title: “Forecasts for Three High-Growth Risk Domains”
Caption: Forecasted incident counts (median and 90% prediction interval) for three domains: malicious misuse, misinformation, and system safety failures.
Description (what to draw): Three small panels (or one plot with three lines if you prefer): each shows historical dots, a forecast line, and a shaded interval; label each domain clearly.
Alt text: “Three panels showing incident forecasts by risk domain with uncertainty bands.”

## Conclusion

<why should the reader care, can include insights — 2 sentences>
We predict a 6-11x increase in the number of total AI incidents, mostly within the risk domains of malicious misuse, misinformation, and system safety failures. 
A forecast is not a prophecy, but it can serve as a risk radar. 

<potential benefits — 1 sentence>
We recommend policies that particularly target these domains. Our incident forecasting enables regulators, developers, and decision-makers to allocate attention and resources to these domains, fostering a proactive, shared approach to safety. This work supports proactive rules, focused testing, and more thorough checks in fast-growing, high-risk areas.

<potential risks — 1 sentence>
If forecasts are treated as certainties rather than updated with new data and technology, they can mislead or foster false confidence.

<limitations — 1-2 sentences>
The incident database only tracks public incidents, misses unreported problems, and changes as rules and definitions change.
Our forecasts are therefore forecasts of recorded incidents, not a perfect measurement of all AI harms.

We also note that 50% of reports are included by 1 person, which may introduce systematic bias.

<assumptions — 1-2 sentences>
We assume that recent trends in the number and mix of incidents suggest what will happen soon, and we weight recent years more heavily in our model.


Our model assumes that the number of AI Incidents will continue to increase indefinitely, whereas in reality, AI would likely reach a point of full adoption, at which the incident rate would level off. 

<next steps — 1 sentence>
Next, we want to more explicitly account for reporting delays, incorporate incident severity, and extend forecasts to more granular fields, such as the affected sector or the alleged developer.

<brief recap — 2 sentences>
We built a two-part workflow that forecasts total recorded AI incident counts and then allocates them across risk domains, with uncertainty quantified through multiple randomized trial simulations. 
The results suggest rapid growth through 2030, especially in misuse and misinformation, assuming recorded-incident trends continue.

<your takeaway/why does this matter for the future — 1-2 sentence>
Anticipating failures is a prerequisite for mitigating them at scale, and incident forecasting is one practical way to support earlier action.
As AI deployment accelerates, tools that help society see emerging risks sooner become more valuable.

We recommend that governance decisions be made to mitigate the abuse from the risk domains: malicious misuse, misinformation, and system safety failures. We endorse further work on AI forecasting and within this particular database.

call-to-action — 1 sentence>
Explore the repository and reproduce the forecasts using our open-source code on GitHub.



## Future Work

Next, we want to improve model accuracy by:

* Measuring Reporting Lag: Incidents are reported and added to the AI incidents database after they occur, meaning the current database is incomplete. We can measure this and account for these unreported incidents.
* True Harm Rate: The incident database reports incidents, which may reflect the true harm rate or increased reporting. We can measure media coverage of AI incidents as a proxy for detection and then derive a harm rate.
* Analysing victims of incidents: We can use AI labelling to categorise which groups of society have been most affected by AI Incidents
* Analysing developers and publishers of incidents: We can use AI labelling to categorise which developers and publishers of AI have caused the most incidents.
* Severity/impact enrichment: Incident severity varies widely. Measuring and analysing these can aid policy recommendations. The count of reports per incident might be an indication of severity and impact.
* Policy intervention scenarios: Predict the impact of upcoming or possible policies on the number of AI incidents.

{contact info}
Contact: https://github.com/cluebbers/ai-incident-forecasting

{acknowledgements}
We thank Apart Research for hosting the AI Forecasting Hackathon that initiated this project. Apart Labs assisted in funding and supporting the research, without which this work would not have been possible. Thank you to all organisers, judges, and the community for an incredibly collaborative event. Thanks to the Responsible AI Collaborative for curating the AI Incident Database.  provided insightful feedback on our initial draft.

### References

AI Forecasting Hackathon
https://github.com/cluebbers/ai-incident-forecasting
AI Incident Database
Preventing Repeated Real World AI Failures by Cataloging Incidents: The AI Incident Database
A Comprehensive Meta-Review, Database, and Taxonomy of Risks From Artificial Intelligence

### Footnotes

* “Year-to-date adjustment” means we estimate a full-year total from partial-year data by using historical seasonality in reporting; this is a pragmatic fix for reporting lag, not a claim about when incidents truly occurred.
** “Surge” means a category’s share increased sharply in the most recent period compared to its earlier baseline; we use this to avoid underreacting to sudden shifts.

