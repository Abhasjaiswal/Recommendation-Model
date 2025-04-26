import pickle
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer

warnings.filterwarnings('ignore')

class SkillRecommender:
    def __init__(self, users_df, jobs_df):
        """
        Initialize the skill recommender with real user and job data
        
        Parameters:
        -----------
        users_df : pandas DataFrame
            DataFrame containing user data with columns 'id', 'skills', 'primary_focus', 'experience_years'
        jobs_df : pandas DataFrame
            DataFrame containing job data with columns 'id', 'title', 'company', 'skills', 'role_type'
        """
        self.users_df = users_df
        self.jobs_df = jobs_df
        
        # Extract all unique skills
        all_user_skills = list(set([skill for skills in users_df['skills'] for skill in skills]))
        all_job_skills = list(set([skill for skills in jobs_df['skills'] for skill in skills]))
        self.all_skills = list(set(all_user_skills + all_job_skills))
        
        # Create user-skill matrix
        self.user_skill_matrix, _ = self._create_user_skill_matrix()
        self.user_similarity = cosine_similarity(self.user_skill_matrix)
        
        # Create skill co-occurrence matrix
        self._create_cooccurrence_matrix()
    
    def _create_user_skill_matrix(self):
        """Create a matrix where each row is a user and each column is a skill"""
        user_skill_matrix = np.zeros((len(self.users_df), len(self.all_skills)))
        
        for i, user_skills in enumerate(self.users_df['skills']):
            for skill in user_skills:
                if skill in self.all_skills:
                    j = self.all_skills.index(skill)
                    user_skill_matrix[i, j] = 1
        
        return user_skill_matrix, self.all_skills
    
    def _create_cooccurrence_matrix(self):
        """Create a matrix showing how often skills co-occur in jobs"""
        # Create a binarized skill matrix for jobs
        mlb = MultiLabelBinarizer()
        job_skill_matrix = mlb.fit_transform(self.jobs_df['skills'])
        
        # Create a co-occurrence matrix
        cooccurrence_matrix = np.dot(job_skill_matrix.T, job_skill_matrix)
        
        # Convert to DataFrame
        self.cooccurrence_df = pd.DataFrame(cooccurrence_matrix, index=mlb.classes_, columns=mlb.classes_)
    
    def get_similar_users(self, user_id, n=5):
        """Find users with similar skill sets"""
        user_idx = self.users_df[self.users_df['id'] == user_id].index[0]
        sim_scores = list(enumerate(self.user_similarity[user_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:n+1]  # Skip the user itself
        similar_user_indices = [i[0] for i in sim_scores]
        similar_users = self.users_df.iloc[similar_user_indices]
        return similar_users, [i[1] for i in sim_scores]  # Return users and similarity scores
    
    def recommend_skills_collaborative(self, user_id, n_similar_users=5, n_skills=5):
        """Recommend skills based on similar users"""
        try:
            user_skills = set(self.users_df[self.users_df['id'] == user_id]['skills'].iloc[0])
        except IndexError:
            # If user_id is not found, return empty recommendations
            return []
        
        similar_users, _ = self.get_similar_users(user_id, n=n_similar_users)
        
        # Get skills of similar users
        skill_counts = Counter()
        for _, sim_user in similar_users.iterrows():
            sim_user_skills = set(sim_user['skills'])
            # Only count skills the original user doesn't have
            new_skills = sim_user_skills - user_skills
            for skill in new_skills:
                skill_counts[skill] += 1
        
        # Get top N skills
        recommended_skills = [skill for skill, _ in skill_counts.most_common(n_skills)]
        
        return recommended_skills
    
    def recommend_skills_for_market(self, user_skills, primary_focus, experience_years=0, n_skills=5):
        """Recommend skills based on market demand"""
        # Filter jobs by focus area
        focus_jobs = self.jobs_df[self.jobs_df['role_type'] == primary_focus]
        
        if len(focus_jobs) == 0:  # If no jobs match the focus, use all jobs
            focus_jobs = self.jobs_df
        
        # Count skills in these jobs
        skill_counts = Counter()
        for job_skills in focus_jobs['skills']:
            for skill in job_skills:
                if skill not in user_skills:  # Only count skills the user doesn't have
                    skill_counts[skill] += 1
        
        # Weight by co-occurrence with existing skills
        skill_scores = {}
        for skill, count in skill_counts.items():
            score = count
            # Add co-occurrence bonus
            for user_skill in user_skills:
                if user_skill in self.cooccurrence_df.index and skill in self.cooccurrence_df.columns:
                    score += self.cooccurrence_df.loc[user_skill, skill] * 0.1
            
            skill_scores[skill] = score
        
        # Sort by score
        sorted_skills = sorted(skill_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Get top N skills
        recommended_skills = [skill for skill, _ in sorted_skills[:n_skills]]
        
        return recommended_skills
    
    def find_relevant_jobs(self, user_id, top_n=5):
        """Find jobs that match a user's skills"""
        user_skills = self.users_df[self.users_df['id'] == user_id]['skills'].iloc[0]
        
        # Create a user vector
        user_vector = np.zeros(len(self.all_skills))
        for skill in user_skills:
            if skill in self.all_skills:
                idx = self.all_skills.index(skill)
                user_vector[idx] = 1
        
        # Create job vectors
        job_vectors = np.zeros((len(self.jobs_df), len(self.all_skills)))
        for i, job_skills in enumerate(self.jobs_df['skills']):
            for skill in job_skills:
                if skill in self.all_skills:
                    idx = self.all_skills.index(skill)
                    job_vectors[i, idx] = 1
        
        # Calculate similarity
        similarity = cosine_similarity([user_vector], job_vectors)[0]
        
        # Get top matching jobs
        top_indices = similarity.argsort()[-top_n:][::-1]
        top_jobs = self.jobs_df.iloc[top_indices]
        
        return top_jobs, similarity[top_indices]
    
    def get_skills_with_scores(self, user_skills, primary_focus, experience_years=0, n_skills=10):
        """Get recommended skills with their scores for more detailed output"""
        # Filter jobs by focus area
        focus_jobs = self.jobs_df[self.jobs_df['role_type'] == primary_focus]
        
        if len(focus_jobs) == 0:  # If no jobs match the focus, use all jobs
            focus_jobs = self.jobs_df
        
        # Count skills in these jobs
        skill_counts = Counter()
        total_jobs = len(focus_jobs)
        for job_skills in focus_jobs['skills']:
            for skill in job_skills:
                if skill not in user_skills:  # Only count skills the user doesn't have
                    skill_counts[skill] += 1
        
        # Calculate market demand percentage
        market_demand = {skill: count / total_jobs for skill, count in skill_counts.items()}
        
        # Weight by co-occurrence with existing skills
        skill_scores = {}
        for skill, demand in market_demand.items():
            score = demand
            # Add co-occurrence bonus
            cooccurrence_score = 0
            for user_skill in user_skills:
                if user_skill in self.cooccurrence_df.index and skill in self.cooccurrence_df.columns:
                    cooccurrence_score += self.cooccurrence_df.loc[user_skill, skill] / len(user_skills)
            
            # Normalize cooccurrence score to 0-1 range
            max_possible = self.cooccurrence_df.max().max()
            if max_possible > 0:
                cooccurrence_score = min(cooccurrence_score / max_possible, 1.0)
            
            # Combine scores (70% market demand, 30% skill relevance)
            skill_scores[skill] = {
                'market_demand': demand,
                'skill_relevance': cooccurrence_score,
                'combined_score': 0.7 * demand + 0.3 * cooccurrence_score
            }
        
        # Sort by combined score
        sorted_skills = sorted(skill_scores.items(), key=lambda x: x[1]['combined_score'], reverse=True)
        
        # Return top N skills with scores
        return sorted_skills[:n_skills]
    
    def generate_recommendation_text(self, user_skills, primary_focus, experience_years=0, n_skills=5):
        """Generate a textual recommendation analysis"""
        skill_scores = self.get_skills_with_scores(user_skills, primary_focus, experience_years, n_skills)
        
        # Create analysis text
        analysis = f"Based on your current skill set and the job market trends, here are our recommendations:\n\n"
        
        # Add recommendations for each skill
        for i, (skill, scores) in enumerate(skill_scores, 1):
            market_demand = scores['market_demand'] * 100
            relevance = scores['skill_relevance'] * 100
            
            if market_demand > 50 and relevance > 50:
                reason = f"{skill} is highly in demand ({market_demand:.0f}%) and complements your existing skills well."
            elif market_demand > 50:
                reason = f"{skill} is currently in high demand ({market_demand:.0f}%) in the job market."
            elif relevance > 50:
                reason = f"{skill} pairs well with your existing skillset and would be a natural next step."
            else:
                reason = f"{skill} would diversify your skill set and open new opportunities."
            
            analysis += f"{i}. {reason}\n"
        
        return analysis

    def analyze_skill_gaps(self, user_id, job_id=None):
        """
        Analyze skill gaps for a user against specific job or jobs in their focus area
        
        Parameters:
        -----------
        user_id : int or str
            User ID to analyze
        job_id : int or str, optional
            Specific job ID to compare against, if None will use jobs in user's focus area
            
        Returns:
        --------
        list of tuples
            (skill, importance_score) sorted by importance
        """
        user_skills = set(self.users_df[self.users_df['id'] == user_id]['skills'].iloc[0])
        
        if job_id is not None:
            # Compare against specific job
            job_skills = set(self.jobs_df[self.jobs_df['id'] == job_id]['skills'].iloc[0])
            missing_skills = job_skills - user_skills
            skill_importance = {skill: 1.0 for skill in missing_skills}
        else:
            # Find jobs relevant to user's primary focus
            user_focus = self.users_df[self.users_df['id'] == user_id]['primary_focus'].iloc[0]
            relevant_jobs = self.jobs_df[self.jobs_df['role_type'] == user_focus]
            
            if len(relevant_jobs) == 0:  # If no jobs match the focus, use all jobs
                relevant_jobs = self.jobs_df
            
            # Count skills across these jobs
            skill_counts = Counter()
            for job_skills in relevant_jobs['skills']:
                for skill in job_skills:
                    if skill not in user_skills:  # Only count skills the user doesn't have
                        skill_counts[skill] += 1
            
            # Calculate importance score (frequency in relevant jobs)
            total_jobs = len(relevant_jobs)
            skill_importance = {skill: count / total_jobs for skill, count in skill_counts.items()}
        
        # Sort by importance
        sorted_skills = sorted(skill_importance.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_skills
