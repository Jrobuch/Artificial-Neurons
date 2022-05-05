README!!!

###############################
Drexel University
CS-615: HW1
John Obuch
###############################

Submission File Contents:

The /HW1_Submision folder contains the following:

1) README.txt - overview of file contents and how to execute the code (i.e. THE CURRENT FILE YOU ARE READING!!!).
2) \yalefaces - Folder contents contain the provided GIF images
3) cost_func_J.m - Matlab cost function
4) partial_theta1.m - Matlab function that computes the partial derivative with respect to theta1 of the provided cost function.
5) partial_tehta2.m - Matlab function that computes the partial derivative with respect to theta2 of the provided cost function.
6) CS615_HW1_Q2.m - Primary matlab gradient decent file for executing part 2 matlab visualization (See source code execution instructions below).
7) CS615_HW1.py - Mother file that produces outputs for all parts of the assignment.
8) HW1_revised.pdf - The original outline of the homework assignment.
9) CS615_HW.pdf - Assignement submission write-up.


To Run The Code:

The following outlines how to run the source code to reproduce the results for each
part of the assignment.

Part 1:

See PDF write-up for mathematical approach to the theory questions.

PYTHON - Part 2 - 4:

To run the CS615_HW1.py source code file on Tux, upload the the \HW1_Submssion.zip file to Tux. 
To do this, open up the command prompt terminal on your machine and type the following command:

scp HW1_Submission.zip user_id@tux.cs.drexel.edu:~/Directory_Name

Where the user_id is your drexel uid and /Directory_Name is the directory on Tux that you will be uploading the zip file to.
Next, navigate to the Tux terminal from the command prompt via the following command:

ssh user_id@tux.cs.drexel.edu

You will be prompted to enter your Tux password credentials. 
Once in the tux envrionment, cd into the directory where you uploaded the HW1_Submission.zip file to via the following command.

cd Directory_Name

Once in the directory of interest, type the following command to unzip the file contents:

unzip HW1_Submission.zip

Once the file has been unzipped, navigate into the /HW1_Submission via the followin command:

cd HW1_Submission

Finally, once in the directory of interest (i.e. the /HW1_Submission directory), in the Tux terminal type the following command:

python3 CS615_HW1.py

The results for all parts (see CS615_HW1.pdf file contents for part 1) of the assignment will populate within the terminal. 
Note: All resulting figures will be stored in the parent directory after the script has been exicuted, namely the /HW1_Submission directory.
To ensure the resulting figures populated in the parent directory, in the terminal type the following:

ls

The images should now appear in  the parent directory after the code has been executed if they were not already present in the parent directory.
Otherwise, if the parent directory already contained the .png images, they will be updated/overwritten.

MATLAB - Part 2:

The the secondary figure produced in the PDF write-up was generated via MATLAB output. 
To run the code that generated the image (as outlined in part 2 of the PDF write-up) double click on the CS615_HW1_Q2.m file.
Assuming the user has MATLAB configured on the their machine, the MATLAB interface should populate and the file contents should appear.
Navigate to the "Run" icon in the tool ribbin located near the top of the application. Click "Run."
The code will execute and the resulting gradient decent visualization will populate.

ADDITIONAL NOTES/OBSEVATIONS/LEARNINGS:

Random seed was set at 42.
Hyperparmeters are very sensitive and play a big factor in the accuracy of the results.
Cross OS platform (e.g. Local VS. Tux) differences in results caused by randomization/seeding and package version differences.

END OF DOCUMENT.