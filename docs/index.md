---
layout: home
title: AI Incident Forecasting
---

## Forecasting real-world AI failures before they happen

This site documents the **AI Incident Forecasting** project developed during the Apart AI Forecasting Hackathon.

We build statistical models on the AI Incident Database to:
- Forecast future AI incident counts
- Break forecasts down by risk domain and category
- Quantify uncertainty using calibrated prediction intervals
- Compare performance against naive baselines

### Latest posts
{% for post in site.posts limit:5 %}
- **[{{ post.title }}]({{ post.url | relative_url }})**  
  <small>{{ post.date | date: "%Y-%m-%d" }}</small>
{% endfor %}

---

**Repository:**  
https://github.com/cluebbers/ai-incident-forecasting
