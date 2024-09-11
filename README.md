## published-data-to-magic: Script and data to prep site-level data for upload to MagIC
This Python notebook (`2024-08-16_Published_Data_to_MagIC.ipynb`) and associated files will produce a set of MagIC formatted files in the `./magic/` directory that can then be uploaded to the MagIC database using the web interface at [https://www2.earthref.org/MagIC/upload].

# Requirements
This script requires the following packages aside from the standard ones:
>   `pmagpy`
>   `pandas`
>   `matplotlib`
>   `numpy`


# Workflow
1. Site-level data, copied from tables in published papers, are in Excel spreadsheets in the `./data/` directory. Each spreadsheet represents a paper or series of papers related to a single location or paleopole.
2. These spreadsheets have been lightly modified from the tables in the original publications. In some cases, for example, location coordinates in degrees-minutes-seconds have been converted to decimal degrees; in other cases information from elsewhere in the paper (e.g. site groups from a different table or lithologies from a figure) have been added manually.
3. I added the original papers themselves in the `./publications/` directory, in case there are questions about the modifications to the spreadsheets.
4. Each section of the notebook contains a set of comments describing how I mapped columns from the spreadsheet to columns in tables from the Magic Data Model 3.0. NOTE: In the Moulin series of papers on the Karoo lava flows, I followed Nick Jarboe's suggestion to group sites as locations in the location table (so a site group is a location containing multiple sites, and a stratigraphic section is a location containing multiple site groups). The reason for this is that the "site group" nomenclature has been removed in the latest version of the MagIC data model.
5. Running a section of the notebook will create a set of text files (_pub_`_locations.txt` and _pub_`_sites.txt`) containing information to upload to MagIC.
6. The two `.txt`files can be dragged and dropped into [https://www2.earthref.org/MagIC/upload]. You will be prompted asbout whether you want to add them to one of your existing contributions (replacing the existing _locations_ and _sites_ tables in that contribution) or whether you want to add a new contribution. A screen will appear with the mapping between the columns of the file you uploaded and the database fields. Check to make sure that is correct.
7. For some reason, programmatically creating and uploading a _pub_`_contributions.txt` file does not add the proper information to the MagIC contribution. The contribution name, DOI, and lab need to be added manually via the web interface for the contribution to be complete. Note that labs need to be chosen from a list that does not include all labs (though the list does include options such as "Not Specified"). There is still an example of a contributions file for the Bushveld data.
8. Validate the contribution using the "Validate" button on the web interface. Note that the validation has problems with values of `nan`, though these are created in most cases by the iPmag/PmagPy software. I have fixed the notebook to get rid of this.
9. Share contribution with co-authors. 
10. Once errors are fixed and co-authors are OK with it, click "Publish" to add to MagIC.

# MagIC Private Contribution Links:
- Letts et al. (2009): https://earthref.org/MagIC/20205/4ce0cf5e-9131-434a-ab9c-cd8f879677eb (Ready to publish)
- Kosterov and Perrin (1996): https://earthref.org/MagIC/20206/7985f008-61d0-46a7-ae78-b37f9855fc3c (Ready to publish)
- Moulin et al. (2011): https://earthref.org/MagIC/20214/3485d059-f168-4506-9938-65afdfc2cbdf (Need to fix errors)
- Moulin et al. (2012): (Need to update how locations are listed)
- Moulin et al. (2017): (Need to update how locations are listed)



