import React, { useState } from 'react';
import './HealthProfile.css';

const HealthProfile = () => {
  const [healthData, setHealthData] = useState({
    age: '',
    gender: '',
    weight: '',
    height: '',
    healthConditions: '',
    dietaryPreferences: '',
    allergies: ''
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [profileSaved, setProfileSaved] = useState(false);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setHealthData(prevData => ({
      ...prevData,
      [name]: value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsSubmitting(true);
    
    try {
      const response = await fetch('http://localhost:5000/api/save-health-profile', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(healthData),
      });
      
      if (!response.ok) {
        throw new Error('Failed to save health profile');
      }
      
      setProfileSaved(true);
      setTimeout(() => setProfileSaved(false), 3000);
    } catch (error) {
      console.error('Error saving health profile:', error);
      alert('Failed to save your health profile. Please try again.');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="health-profile-container">
      <h2 className="profile-heading">Your Health Profile</h2>
      <p className="profile-description">Share your health information to get personalized food recommendations</p>
      
      <form className="health-form" onSubmit={handleSubmit}>
        <div className="form-row">
          <div className="form-group">
            <label htmlFor="age">Age</label>
            <input 
              type="number" 
              id="age" 
              name="age" 
              value={healthData.age} 
              onChange={handleChange} 
              placeholder="Years"
              required
            />
          </div>
          
          <div className="form-group">
            <label htmlFor="gender">Gender</label>
            <select 
              id="gender" 
              name="gender" 
              value={healthData.gender} 
              onChange={handleChange}
              required
            >
              <option value="">Select</option>
              <option value="male">Male</option>
              <option value="female">Female</option>
              <option value="other">Other</option>
            </select>
          </div>
        </div>
        
        <div className="form-row">
          <div className="form-group">
            <label htmlFor="weight">Weight (kg)</label>
            <input 
              type="number" 
              id="weight" 
              name="weight" 
              value={healthData.weight} 
              onChange={handleChange} 
              placeholder="kg"
              step="0.1"
              required
            />
          </div>
          
          <div className="form-group">
            <label htmlFor="height">Height (cm)</label>
            <input 
              type="number" 
              id="height" 
              name="height" 
              value={healthData.height} 
              onChange={handleChange} 
              placeholder="cm"
              required
            />
          </div>
        </div>
        
        <div className="form-group">
          <label htmlFor="healthConditions">Health Conditions</label>
          <textarea 
            id="healthConditions" 
            name="healthConditions" 
            value={healthData.healthConditions} 
            onChange={handleChange} 
            placeholder="E.g., Diabetes, Hypertension, etc."
            rows="3"
          ></textarea>
        </div>
        
        <div className="form-group">
          <label htmlFor="dietaryPreferences">Dietary Preferences</label>
          <textarea 
            id="dietaryPreferences" 
            name="dietaryPreferences" 
            value={healthData.dietaryPreferences} 
            onChange={handleChange} 
            placeholder="E.g., Vegetarian, Vegan, Keto, etc."
            rows="3"
          ></textarea>
        </div>
        
        <div className="form-group">
          <label htmlFor="allergies">Food Allergies</label>
          <textarea 
            id="allergies" 
            name="allergies" 
            value={healthData.allergies} 
            onChange={handleChange} 
            placeholder="E.g., Peanuts, Shellfish, Gluten, etc."
            rows="3"
          ></textarea>
        </div>
        
        <button 
          type="submit" 
          className="submit-button"
          disabled={isSubmitting}
        >
          {isSubmitting ? 'Saving...' : 'Save Health Profile'}
        </button>
        
        {profileSaved && (
          <div className="success-message">
            Your health profile has been saved successfully!
          </div>
        )}
      </form>
    </div>
  );
};

export default HealthProfile;