# Job-Hunter
Part 2: As a job seeker, it can be overwhelming to apply for positions and not receive a response. One reason for this may be that companies receive a high volume of resumes and must carefully review them to identify the most qualified candidates. To increase your chances of being noticed, it's important to tailor your resume to the specific job you are applying for and highlight your relevant skills and experiences. Our job posting screener tool uses advanced techniques like sklearn's cosine similarity and TfidfVectorizer to process job descriptions and nltk to visualize and select the top postings to apply for, ensuring that your application stands out among the competition. By optimizing your resume with our tool, you can increase your chances of being considered for the positions you are interested in. Given more time, we would add a feature that would scrape for relevant and recently posted jobs and rank the best jobs to apply for.

Input: Resume and multiple job descriptions

Output: Score of the best jobs to apply to based on resume.

Dataset taken from: https://www.kaggle.com/datasets/wahib04/multilabel-resume-dataset

This dataset contains the category of the resume and the resume as a string. I renamed category to current occupation in order to add a little bit of story behind the dataset. This was the only resume dataset I found on Kaggle.

The application was deployed using Streamlit and further deployed publically using herokuapp.
