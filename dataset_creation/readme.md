# Directory Format
Unzip the `datasets` folder:
- there should be a folder for each subject, and an all-encompassing folder, `all_splits`
- subject folders should have either 3 or 5 trials, in both `.txt` and `.csv` format
- there should be either 3 or 5 subfolders per subject, named `SUBJECT_splits`, containing the approximate data for each trial 
- most subjects should have copies of their video prompts for reference
- `results.py` contains dictionaries for each subject's prompts per trial, which are the data labels

# Data Preprocessing
- `txt_to_csv.py` converts the original `.txt` output files to `.csv` for easier processing
- `start_script.py` crops the original recording with an approximation of the start, then saves the split recordings
- `splits_to_csv.py` assumes the correct file structure and converts it to DataFrame for modeling

# Dataset Creation Process
1. Ensure you are in a quiet, clear environment, and your subject has earbuds or equivalent
2. Generate 3 videos for your subject with N prompts in each video (ideally a multiple of 3) using `generate_prompts.ipynb`
3. Seat the subject and explain to them the prompt order:

- Physically opening/closing fists with audio and text prompt
- Imagining opening/closing fists with text only prompt
- Imagining direction with audio and text prompt
- Imagining direction with audio only prompt
- Imagining direction with text only prompt

4. Start the recording session and ensure the EEG is well positioned on the subject's head before playing the videos
5. Have the subject follow the directions on the video for the prompts, then check that the file saved correctly
6. Repeat for the next 4 prompt groups - if the subject has no earbuds, have them do only prompts 1, 2, and 5

I did a clap whenever the video generally started so a spike would show in the recordings indicating the start of data