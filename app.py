# app.py
import json

import pandas as pd
from flask import Flask, jsonify, request
from flask import render_template
from flask_cors import CORS

from model import SkillRecommender

app = Flask(__name__)
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)
# Constants for file paths
USERS_JSON_PATH = 'Data/users.json'
JOBS_JSON_PATH = 'Data/jobs.json'   

# Load data from JSON files
def load_data_from_json():
    try:
        # Load users from JSON file
        with open(USERS_JSON_PATH, 'r') as f:
            users_data = json.load(f)
        users_df = pd.DataFrame(users_data)
        
        # Load jobs from JSON file
        with open(JOBS_JSON_PATH, 'r') as f:
            jobs_data = json.load(f)
        jobs_df = pd.DataFrame(jobs_data)
        
        # Process the data to ensure it has the required format
        # Ensure skills are lists, not strings
        if users_df.shape[0] > 0 and isinstance(users_df['skills'].iloc[0], str):
            users_df['skills'] = users_df['skills'].apply(lambda x: x.split(','))
        if jobs_df.shape[0] > 0 and isinstance(jobs_df['skills'].iloc[0], str):
            jobs_df['skills'] = jobs_df['skills'].apply(lambda x: x.split(','))
        
        return users_df, jobs_df
    
    except Exception as e:
        print(f"Error loading data from JSON: {str(e)}")
        # Return empty DataFrames if files don't exist yet
        users_df = pd.DataFrame(columns=['id', 'skills', 'primary_focus', 'experience_years'])
        jobs_df = pd.DataFrame(columns=['id', 'title', 'company', 'skills', 'role_type'])
        return users_df, jobs_df

# Initialize the recommender with data from JSON files
def initialize_recommender():
    users_df, jobs_df = load_data_from_json()
    
    # Initialize the recommender only if we have data
    if len(users_df) > 0 and len(jobs_df) > 0:
        recommender = SkillRecommender(users_df, jobs_df)
        return recommender
    return None

# Initialize the recommender on startup
recommender = initialize_recommender()

@app.route('/api/upload-users', methods=['POST'])
def upload_users():
    """Endpoint to upload users data from JSON"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if file and file.filename.endswith('.json'):
            # Save the file
            file.save(USERS_JSON_PATH)
            
            # Refresh recommender
            global recommender
            recommender = initialize_recommender()
            
            return jsonify({'status': 'success', 'message': 'Users data uploaded successfully'})
        else:
            return jsonify({'error': 'File must be JSON format'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload-jobs', methods=['POST'])
def upload_jobs():
    """Endpoint to upload jobs data from JSON"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if file and file.filename.endswith('.json'):
            # Save the file
            file.save(JOBS_JSON_PATH)
            
            # Refresh recommender
            global recommender
            recommender = initialize_recommender()
            
            return jsonify({'status': 'success', 'message': 'Jobs data uploaded successfully'})
        else:
            return jsonify({'error': 'File must be JSON format'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/recommend-skills', methods=['POST'])
def recommend_skills():
    data = request.json
    user_id = data.get('user_id')
    
    if not user_id:
        return jsonify({'error': 'User ID is required'}), 400
    
    if recommender is None:
        return jsonify({'error': 'Recommender not initialized. Please upload users and jobs data first.'}), 500
    
    try:
        # Get user details
        user = recommender.users_df[recommender.users_df['id'] == user_id].iloc[0]
        user_skills = user['skills']
        primary_focus = user['primary_focus']
        experience_years = user.get('experience_years', 0)
        
        # Get recommendations
        market_recommendations = recommender.recommend_skills_for_market(
            user_skills, primary_focus, experience_years, n_skills=5
        )
        
        collaborative_recommendations = recommender.recommend_skills_collaborative(
            user_id, n_skills=5
        )
        
        # Get detail analysis
        analysis_text = recommender.generate_recommendation_text(
            user_skills, primary_focus, experience_years, n_skills=5
        )
        
        # Get top matching jobs
        matching_jobs, job_scores = recommender.find_relevant_jobs(user_id, top_n=5)
        job_matches = [
            {
                'id': job['id'],
                'title': job['title'],
                'company': job['company'],
                'match_score': float(score)
            }
            for (_, job), score in zip(matching_jobs.iterrows(), job_scores)
        ]
        
        return jsonify({
            'market_recommendations': market_recommendations,
            'collaborative_recommendations': collaborative_recommendations,
            'analysis': analysis_text,
            'job_matches': job_matches
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze-skill-gaps', methods=['POST'])
def analyze_skill_gaps():
    data = request.json
    user_id = data.get('user_id')
    job_id = data.get('job_id')  # Optional
    
    if not user_id:
        return jsonify({'error': 'User ID is required'}), 400
    
    if recommender is None:
        return jsonify({'error': 'Recommender not initialized. Please upload users and jobs data first.'}), 500
    
    try:
        # Get skill gaps
        skill_gaps = recommender.analyze_skill_gaps(user_id, job_id)
        
        return jsonify({
            'skill_gaps': [
                {'skill': skill, 'importance': float(importance)}
                for skill, importance in skill_gaps[:10]  # Return top 10 skills
            ]
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/refresh-recommender', methods=['POST'])
def refresh_recommender():
    """Endpoint to refresh the recommender with updated data"""
    try:
        global recommender
        recommender = initialize_recommender()
        return jsonify({
            'status': 'success', 
            'message': 'Recommender refreshed with latest data',
            'has_data': recommender is not None
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """Endpoint to check if recommender is initialized and data is loaded"""
    if recommender is None:
        return jsonify({
            'status': 'not_initialized',
            'message': 'Recommender not initialized. Please upload users and jobs data.'
        })
    
    # Get data summary
    users_count = len(recommender.users_df)
    jobs_count = len(recommender.jobs_df)
    
    return jsonify({
        'status': 'ready',
        'users_count': users_count,
        'jobs_count': jobs_count
    })

@app.route('/api/jobs', methods=['GET'])
def get_jobs():
    """Endpoint to retrieve all jobs data"""
    try:
        if recommender is None:
            # If recommender is not initialized, try to load data directly
            _, jobs_df = load_data_from_json()
        else:
            jobs_df = recommender.jobs_df
        
        # Convert DataFrame to list of dictionaries for JSON serialization
        # Handle skills list properly
        jobs_list = []
        for _, job in jobs_df.iterrows():
            job_dict = job.to_dict()
            # Ensure skills is a list for proper JSON serialization
            if isinstance(job_dict['skills'], list):
                job_dict['skills'] = job_dict['skills']
            else:
                # Handle any unexpected format
                job_dict['skills'] = str(job_dict['skills']).split(',') if job_dict['skills'] else []
            jobs_list.append(job_dict)
        
        return jsonify({
            'count': len(jobs_list),
            'jobs': jobs_list
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/users', methods=['GET'])
def get_users():
    """Endpoint to retrieve all users data"""
    try:
        if recommender is None:
            # If recommender is not initialized, try to load data directly
            users_df, _ = load_data_from_json()
        else:
            users_df = recommender.users_df
        
        # Convert DataFrame to list of dictionaries for JSON serialization
        # Handle skills list properly
        users_list = []
        for _, user in users_df.iterrows():
            user_dict = user.to_dict()
            # Ensure skills is a list for proper JSON serialization
            if isinstance(user_dict['skills'], list):
                user_dict['skills'] = user_dict['skills']
            else:
                # Handle any unexpected format
                user_dict['skills'] = str(user_dict['skills']).split(',') if user_dict['skills'] else []
            users_list.append(user_dict)
        
        return jsonify({
            'count': len(users_list),
            'users': users_list
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Optional: Add a parameter to limit the number of results
@app.route('/api/jobs/<int:limit>', methods=['GET'])
def get_limited_jobs(limit):
    """Endpoint to retrieve a limited number of jobs"""
    try:
        if recommender is None:
            _, jobs_df = load_data_from_json()
        else:
            jobs_df = recommender.jobs_df
        
        # Apply limit
        limited_jobs_df = jobs_df.head(limit)
        
        # Convert DataFrame to list of dictionaries
        jobs_list = []
        for _, job in limited_jobs_df.iterrows():
            job_dict = job.to_dict()
            if isinstance(job_dict['skills'], list):
                job_dict['skills'] = job_dict['skills']
            else:
                job_dict['skills'] = str(job_dict['skills']).split(',') if job_dict['skills'] else []
            jobs_list.append(job_dict)
        
        return jsonify({
            'count': len(jobs_list),
            'jobs': jobs_list
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/users/<int:limit>', methods=['GET'])
def get_limited_users(limit):
    """Endpoint to retrieve a limited number of users"""
    try:
        if recommender is None:
            users_df, _ = load_data_from_json()
        else:
            users_df = recommender.users_df
        
        # Apply limit
        limited_users_df = users_df.head(limit)
        
        # Convert DataFrame to list of dictionaries
        users_list = []
        for _, user in limited_users_df.iterrows():
            user_dict = user.to_dict()
            if isinstance(user_dict['skills'], list):
                user_dict['skills'] = user_dict['skills']
            else:
                user_dict['skills'] = str(user_dict['skills']).split(',') if user_dict['skills'] else []
            users_list.append(user_dict)
        
        return jsonify({
            'count': len(users_list),
            'users': users_list
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
