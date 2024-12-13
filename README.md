# Intermetallic database containing kagome nets

This repository provides the tools for searching ICSD intermetallic compounds that contain kagome net and the ML models/tools trained on the database to rank unexplored chemical systems. 

We adopt a two-step strategy with minimal constraint to filter the intermetallic compounds in order to provide a comprehensive database for fine-tuning use:

Step-1: Topological filter: finding the connectivity of atoms in a structure that matches the kagome connectivity. 

Step-2: Geometrical filter: analyzing the distortions of the nets using the pymatgen structure functionalities. 

The product database is used to train ML models in order to rank chemical systems that are unexplored and can potentially host kagome nets. 

The product ML models have identified chemical systems that are not seen in the ICSD but reported in the Pearson's crystal database. 
