# Set-Up instructions

This document provides a step-by-step guide to setting up and running the project, including unzipping the project files, building the necessary components, and executing the scripts.

Follow these steps to set up the project:

1. ```bash
   unzip 2021CS51004
2. ```bash
   cd 2021CS51004

3. ```bash
   bash dictcons.sh <path to doc directory> {0|1|2}

4. ```bash
   bash invidx.sh <path to doc directory> index {0|1|2}

5. ```bash
   bash tf_idf_search.sh <path to cord19-trec-covid-queries> result index.idx index.dict

