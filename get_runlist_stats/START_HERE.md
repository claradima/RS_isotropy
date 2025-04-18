### How to use this folder

### Usage Instructions

### Documentation History

The contents of this folder are meant to compute all coverage and isotropy metrics for a given runlist. To do that, open the a terminal in the folder location and run a command like:

`python3 full_loop_optimized.py --runlist-filename "bronze_runlist_6runs.txt" --output-csv "random_coverage_stats.csv" --output-pmtmap-dir "random_list" --download-streams 6 --proc-streams 6`

Parameters:

- `--runlist-filename` → name of the .txt file that contains the runs you want to compute metrics for

- `--output-csv` → name of the .csv file you want to save metrics in; file will be created if it doesn't exist already

- `--output-pmt` → name of directory where you want to save the PMT maps for 'active' (NORMAL, on, good ECA) PMTs for the runs in the runlist; dir will be created and placed in `/home/claramariadima/SNO/RS_isotropy/get_runlist_stats/PMT_maps`

- `--download-streams` → maximum number of download streams (i.e. max number of zdabs downloaded simultaneously; 6 definitely works great, might be even better if you increase until 12)

- `--proc-streams` → maximum number of processing streams; should be $\leq$ `--download-streams`, and should have (`--download-streams`) + (`--proc-streams`) $\leq$ 20

Check the .cvs file for results at the end :)



###### Other contents of this folder and further explanations:

- **dir:** `detailed_stats_for_testing_364600` → contains histograms with the cap count distributions for run 364600; not actively used, I just made these plots when testing the `compute_run_metrics.py` script and dumped the plots in this folder. As it is now, these plots are not produced by the script anymore (would get too many plots...)

- **dir:** `PMT_maps` → contains one subfolder for each runlist; each subfolder contains the plots of the active, normal PMTs with good ECA for all the runs in the runlist

- **dir:** `phi_theta` → contains CSV files with the normalized spherical polar coordinates of many runs, one file per run. I haven't separated these into subfolders, I just dumped everything here. These are needed to run `compute_run_metrics.py` but otherwise no need to look at them

- **files:** `*_runlist.txt` → these are runlist files with all the runs in the list. To use a specific runlist, you can just make a `.txt` file of whatever runs you want and add it to the folder. The runlist you use has to be specified in the command for running the`full_loop_optimized.py` script

- **file:** `More_Nodes_Grid.csv` → CSV file that contains the coordinates for the "nodes" that I selected on the PSUP — these are the centers of the caps used to compute cap count distributions and associated metrics. The file is used by the `compute_run_metrics.py` script.

- **file:** `*_coverage_stats.csv` → CSV file that contains values for all the coverage and isotropy metrics, for all the runs in a list specified in the command for running the`full_loop_optimized.py` script . 

- **files:** `log_*.txt` → Log files produced by running `full_loop_optimized.py`.(and other previous versions of the script). I gave them random names because I didn't want previous log files to be overwritten. I'll try to keep the folder clean and remove log files (I think I added files with this format to .gitignore actually) — but if you're running this, just sort by date and look at the latest one to check.

- **script file:** `compute_run_metrics.py` → This script computes the coverage metrics for a single run and adds them to a specified `*_coverage_stats.csv` file.  It also makes a PMT map for that run and dumps it in the appropriate subfolder in `PMT_maps` (also specified in command to run) as part of a big big loop.
  
  - To run this on its own, you need to run the command as: `python3 compute_run_metrics.py --runlist-filename "bronze_runlist_6runs.txt" --output-csv "random_coverage_stats.csv" --output-pmtmap-dir "random_list"` (and replacing arguments with whatever you want to use instead). You also need the respective `364600_phi_theta.csv` file in the `phi_theta` directory. Making these is a bit weird, you need to run the DQ PMT Processor via a macro in a rat version that is old and has some added bits to make the file. `full_loop_optimized.py`(and previous versions of it) does this automatically, and `test_run_rat_in_docker.py` kinda does this as well, but you'd need to manually change the macro with the correct run number in the rat folder ... there's a bunch of hardcoded things, I doubt anyone who isn't me will ever run this so no need to go through the whole setup :)

- **script file:** `test_run_rat_in_docker.py` → this script can make `{run}_phi_theta.csv` files in rat, but it's not super useful on its own - I made it to test running the container and the macro inside the container without user input, before adding the code snippet to `full_loop_optimized.py`. Overall, not super useful on its own, but I kept it just in case.

- **script file:** `full_loop_optimized.py`→ this is the big script → it loops througha given runlist and adds one line with stats for each run in the runlist to a csv file. It does this in a pretty convoluted way. To run this, just run `python3 full_loop_optimized.py --runlist-filename "bronze_runlist_6runs.txt" --output-csv "random_coverage_stats.csv" --output-pmtmap-dir "random_list" --download-streams 6 --proc-streams 6` (with changed argument values) in a terminal opened in this folder. This script is very very dependent on my specific setup, but again, I doubt anyone else will use it. For reference, here is an outline of what it does:
  
  - It sets some path names, file names, credentials - this has my passwords hardcoded in it, I'll remove them before pushing on github
  
  - It reads the runlist txt file
  
  - There are two main steps of processing, downloading and processing, and they are both paralellized separately to use specified number of threads in parallel. The download function runs a bunch of downloads in parallel, then as soon as a file is ready, it is detected and grabbed by the processing function.
  
  - The steps that the script follows for a single run are:
    
    - First, it downloads the  .zdab file for the run from GRID. For this, it needs a singularity container for downloading GRID files, it opens it, makes a temporary `filelist_{run}.dat` file, it cleans it up outside of the singularity container, then opens the container again and runs grabber. To check progress on this step, you can go in the `downloaded` folder in `GridTools` in `rat-tools` and check the size of the file that is currently downloading - they should be 1GB total at the end of the download. The `filelist_{run}.dat` is deleted after the zdab was taken by the processing part.
    
    - The processing part checks whether the .zdab is ready.
    
    - Then, it checks whether the rat folder is in the git branch that has the specific rat version I want to use, and moves the .zdab in the rat folder. It also modifies the macro it will use later to have the correct run number.
    
    - Note: The rat version used is that right before I removed the bits of code in DQHL that were making a bunch of plots, including the PMT coverage map (which was made specifically by the DQ PMT Processor). To that version of RAT, I added some bits between the lines that make the PMT coverage map to export the coordinates to a `{run}__phi_theta.csv` file that has the formatting I wanted. The reason I did this instead of writing a simple script that just grabs the coordinates from some database for that run is because I am lazy and bad at coding, and also don't fully know where stuff is: this code already existed so I just used that instead. Not sure if it saved me time in the end ... running the macro for the processor also makes a bunch of extra stuff that I had to get rid of later, and it obviously takes a lot more computing resources, but! it all works even if it's bad :)
    
    - After the .zdab is in the rat folder, it opens a docker container with rat and, inside of the container, runs a macro that calls the DQ PMT Processor. This creates the `{run}_phi_theta.csv` file.
    
    - After the macro runs, it deletes the zdab (they take up a lot of space - that's why I'm downloading them one by one with each iteration of the loop instead of just getting all of them from the start and looping through the remaining steps), it deletes additional useless file, and moves the `{run}_phi_theta.csv` file to the `phi_theta` directory so it can be used later by `compute_run_metrics.py`
    
    - Finally, it runs `compute_run_metrics.py` for the run, saves the PMT map plot in `PMT_maps` and adds the relevant info to the specified csv file
  
  - At the end of the run, you should check the following:
    
    - the `downloaded` folder in `GridTools` doesn't have zdabs anymore
    
    - the `rat` folder is free of zdabs or any files that were created during the run - they should all be either moved or deleted
    
    - the `phi_theta` folder contains the files for the runs in the run list
    
    - the correct subfolder in `PMT_maps` contains plots of good PMTs for the runs in the run list
    
    - check the log file seems ok and nothing weird happened and nothing failed on the way
    
    - the specified csv file contains info for all the runs in the runlist; check how many runs are there and whether it matches with the total number of runs in the runlist
  
  - **script files:** `full_nickel_loop.py`, `full_nickel_loop_optimized.py` → previous versions of `full_loop_optimized.py` that had output files and some other stuff hardcoded; the first script had no paralellisation
  
  - **script file:** `compute_runlist_metrics.py` → similar to `compute_run_metrics.py` but it computes these for a full runlist. It assumes the `*_phi_theta.csv` files for each run in the list are already in the `phi_theta` dir. It has the name of the runlist .txt file hardcoded, so if you want to use this you have to go in the code and change it. Can be useful if you want to separate this step form `full_nickel_loop.py`, but it is currently integrated there.



- **SUMMARY**
  
  - `full_loop_optimized.py` loops through the runlist and calculates coverage and isotropy metrics
  
  - the code as a whole is:
    
    - ugly
    
    - inefficient
    
    - convoluted - opens a bunch of containers and does many things in different places
    
    - super setup-specific and almost impossible to transfer somewhere else
  
  - but!
    
    - it requires no user input - does all the convoluted things on its own!
    
    - it's paralellized - the inefficiency mostly comes from the fact that downloads take a while, and I wasn't able to find a place to retrieve the same info from DQHL tables without the .zdabs
    
    - it has good logging
    
    - it cleans up after itself
    
    - it reruns downloads if they failed due to timeout
    
    - it was fun to make
    
    - it works
    
    - it makes me happy



#### 16 April - Initial Setup

The contents of this folder are meant to compute all coverage and isotropy metrics for a given runlist. As it stands now, the nickel list is hard coded, as it's still a work in progress, but I'll make it more flexible soon when I'm done going through the whole process with the nickel list.

Here is what you can find in this folder:

- **dir:** `detailed_stats_for_testing_364600` → contains histograms with the cap count distributions for run 364600; not actively used, I just made these plots when testing the `compute_run_metrics.py` script and dumped the plots in this folder. As it is now, these plots are not produced by the script anymore (would get too many plots...)

- **dir:** `PMT_maps` → contains one subfolder for each runlist; each subfolder contains the plots of the active, normal PMTs with good ECA for all the runs in the runlist

- **dir:** `phi_theta` → contains CSV files with the normalized spherical polar coordinates of many runs, one file per run. I haven't separated these into subfolders, I just dumped everything here. These are needed to run `compute_run_metrics.py` but otherwise no need to look at them

- **files:** `*_runlist.txt` → these are runlist files with all the runs in the list. After I make the scripts run with runlists other than nickel, you can just make a `.txt` file of whatever runlist you want and add it to the folder. The runlist you use has to be specified in the `full_nickel_loop.py` script, which will hopefully gain a more general name once it's generalized

- **file:** `More_Nodes_Grid.csv` → CSV file that contains the coordinates for the "nodes" that I selected on the PSUP — these are the centers of the caps used to compute cap count distributions and associated metrics. The file is used by the `compute_run_metrics.py` script.

- **file:** `nickel_coverage_stats.csv` → CSV file that contains values for all the coverage and isotropy metrics, for all the runs in the nickel list. Once I make the code more general and run it for other runlists, there will be other files with similar names generated.

- **files:** `log_*.txt` → Log files produced by running `full_nickel_loop.py`. I gave them random names because I didn't want previous log files to be overwritten. I'll try to keep the folder clean and remove log files once I'm confident the code ran well for a specific runlist — but if you're running this, just sort by date and look at the latest one to check.

- **script file:** `compute_run_metrics.py` → This script computes the coverage metrics for a single run and adds them to `nickel_coverage_stats.csv`. I'll add another parsed argument to make it add to other .csv files for other run lists when we get there. It also makes a PMT map for that run and dumps it in the appropriate subfolder in `PMT_maps` (which right now it's hardcoded to be the one for the nickel list). You can run this on its own if you want, but its purpose is to be called by the `full_nickel_loop.py` script (or whatever its name will be in the future) as part of a big big loop. 
  
  - To run this on its own, in its current form, you need to run the command as: `python3 compute_run_metrics.py -run_number 364600` (and replacing with whatever run number you need). You also need the respective `364600_phi_theta.csv` file in the `phi_theta` directory. Making these is a bit weird, you need to run the DQ PMT Processor via a macro in a rat version that is old and has some added bits to make the file. `full_nickel_loop.py` does this automatically, and `test_run_rat_in_docker.py` kinda does this as well, but you'd need to manually change the macro with the correct run number in the rat folder ... there's a bunch of hardcoded things, I doubt anyone who isn't me will ever run this so no need to go through the whole setup :) 

- **script file:** `test_run_rat_in_docker.py` → this script can make `{run}_phi_theta.csv` files in rat, but it's not super useful on its own - I made it to test running the container and the macro inside the container without user input, before adding the code snippet to `full_nickel_loop.py`. Overall, not super useful on its own, but I kept it just in case.

- **script file:** `full_nickel_loop.py`→ this is the big script¬ it loops througha given runlist (currently just nickel) and adds lines to `nickel_coverage_stats.csv`. It does this in a very convoluted way. To run it, just type `python3 full_nickel_loop.py` in terminal (I'll later add an argument parser to give it whatever list you want, and make it more flexible so it puts everything in the right places in the process). This script is very very dependent on my specific setup, but again, I doubt anyone else will use it. For reference, here is an outline of what it does:
  
  - It sets some path names, file names, credentials - this has my passwords hardcoded in it, I'll remove them before pushing on github
  
  - It reads the runlist txt file
  
  - It loops through the runs in the runlist and does the following:
    
    - First, it downloads the first .zdab file for the run from GRID. For this, it needs a singularity container for downloading GRID files, it opens it, makes the filelist.dat file, it cleans it up outside of the singularity container, then opens the container again and runs grabber. To check progress on this step, you can go in the `downloaded` folder in `GridTools` in `rat-tools` and check the size of the file that is currently downloading - they should be 1GB total at the end of the download
    
    - Then, it checks whether the rat folder is in the git branch that has the specific rat version I want to use, and moves the .zdab in the rat folder. It also modifies the macro it will use later to have the correct run number.
    
    - Note: The rat version used is that right before I removed the bits of code in DQHL that were making a bunch of plots, including the PMT coverage map (which was made specifically by the DQ PMT Processor). To that version of RAT, I added some bits between the lines that make the PMT coverage map to export the coordinates to a `{run}__phi_theta.csv` file that has the formatting I wanted. The reason I did this instead of writing a simple script that just grabs the coordinates from some database for that run is because I am lazy and bad at coding, and also don't fully know where stuff is: this code already existed so I just used that instead. Not sure if it saved me time in the end ... running the macro for the processor also makes a bunch of extra stuff that I had to get rid of later, and it obviously takes a lot more computing resources, but! it all works even if it's bad :)
    
    - After the .zdab is in the rat folder, it opens a docker container with rat and, inside of the container, runs a macro that calls the DQ PMT Processor. This creates the `{run}_phi_theta.csv` file. 
    
    - After the macro runs, it deletes the zdab (they take up a lot of space - that's why I'm downloading them one by one with each iteration of the loop instead of just getting all of them from the start and looping through the remaining steps), it deletes additional useless file, and moves the `{run}_phi_theta.csv` file to the `phi_theta` directory so it can be used later by `compute_run_metrics.py`
    
    - Finally, it runs `compute_run_metrics.py` for the run, saves the PMT map plot in `PMT_maps` and adds the relevant info to `nickel_coverage_stats.csv`
  
  - At the end of the run, you should check the following:
    
    - the `downloaded` folder in `GridTools` doesn't have zdabs anymore
    
    - the `rat` folder is free of zdabs or any files that were created during the run - they should all be either moved or deleted
    
    - the `phi_theta` folder contains the files for the runs in the run list
    
    - the correct subfolder in `PMT_maps` contains plots of good PMTs for the runs in the run list
    
    - check  the log file seems ok and nothing weird happened and nothing failed on the way
    
    - the `nickel_coverage_stats.csv` contains info for all the runs in the runlist; check how many runs are there and whether it matches with the total number of runs in the runlist

- **NOTE: NEXT STEPS**
  
  - I have to add another script that looks at each coverage or isotropy metric from `nickel_coverage_stats.csv` and computes its distribution over the specified runlist 
  
  - I have to make the code more flexible so it also works for other runlists
  
  - I have to update this file after I do that 

- **SUMMARY**
  
  - `full_nickel_loop.py` loops through the runlist and calculates coverage and isotropy metrics
  
  - the code as a whole is:
    
    - ugly
    
    - inefficient
    
    - convoluted - opens a bunch of containers and does many things in different places
    
    - super setup-specific and almost impossible to transfer somewhere else
  
  - but!
    
    - it requires no user input - does all the convoluted things on its own!
    
    - it was fun to make
    
    - it works
    
    - it makes me happy



#### 18 April Updates
