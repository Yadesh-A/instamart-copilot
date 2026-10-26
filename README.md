# Instamart Customer Intelligence & Support Copilot

## Overview
This repository contains a 4-part data and AI pipeline that transforms raw e-commerce data into a clean analytical database, segments customers, predicts churn, triages support tickets using LLMs, and deploys an autonomous ReAct refund agent.

## Reproduce
1. `python -m venv venv`
2. Activate environment: `.\venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Mac/Linux)
3. `pip install -r requirements.txt`
4. `python generate_data.py`
5. `python part_a_cleaning.py`
*(Add Part B, C, D execution commands here as you build them)*

## Part A (Data Foundation)
* **Raw Audit:** 13,084 order rows; 119 exact duplicates; 656 pending rows... *(Add the rest of your printout numbers here)*
* **Cleaning Rules:**
  * Standardized city strings via `.strip().title()` (e.g., mapping " chennai " to "Chennai").
  * Mapped irregular gender spellings to standard categories or "unknown".
  * Filled age outliers with group-based median transforms...
*(Proceed to fill out the SQL answers and EDA correlation figures once you run the cross-check script).*

## Part B (Customer Segments)
*(To be completed)*

## Part C (Ticket Triage)
*(To be completed)*

## Part D (RAG & Agent)
*(To be completed)*

## Limitations and Next Steps
* The Premium flag is a post-hoc whole-window label, making the retention list illustrative rather than a strict leakage-free evaluation.