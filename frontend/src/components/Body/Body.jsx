import React, { useState, useRef } from 'react';
import './Body.css';

const Body = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [previewUrl, setPreviewUrl] = useState('/images/pngegg.png');
  const [isLoading, setIsLoading] = useState(false);
  const [foodInfo, setFoodInfo] = useState(null);
  const fileInputRef = useRef(null);
  const cameraInputRef = useRef(null);

  const handleImageUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setSelectedImage(file);
    setPreviewUrl(URL.createObjectURL(file));
    
    // Upload to backend
    await uploadImageToServer(file, 'upload');
  };

  const handleCameraCapture = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setSelectedImage(file);
    setPreviewUrl(URL.createObjectURL(file));
    
    // Upload to backend
    await uploadImageToServer(file, 'capture');
  };

  const uploadImageToServer = async (file, source) => {
    setIsLoading(true);
    setFoodInfo(null);
    
    try {
      const formData = new FormData();
      formData.append('foodImage', file);
      formData.append('source', source);
      
      const response = await fetch('http://localhost:5000/api/analyze-food', {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error('Failed to analyze food image');
      }
      
      const data = await response.json();
      setFoodInfo(data);
    } catch (error) {
      console.error('Error analyzing food:', error);
      alert('Failed to analyze the food image. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleUploadButtonClick = () => {
    fileInputRef.current.click();
  };

  const handleCaptureButtonClick = () => {
    cameraInputRef.current.click();
  };

  return (
    <div className="app-container">
      <div className="content-wrapper">
        <div className="main-content">
          <div className="text-section">
            <h2 className="heading">Health Companion</h2>
            <h3 className="subheading">Eat Healthy Food.</h3>
            
            <div className="button-container">
              <button 
                className="action-button" 
                onClick={handleCaptureButtonClick}
                disabled={isLoading}
              >
                {isLoading && selectedImage ? 'Analyzing...' : 'Click Image'}
              </button>
              <input
                type="file"
                accept="image/*"
                capture="environment"
                ref={cameraInputRef}
                onChange={handleCameraCapture}
                style={{ display: 'none' }}
              />
              
              <button 
                className="action-button" 
                onClick={handleUploadButtonClick}
                disabled={isLoading}
              >
                {isLoading && selectedImage ? 'Analyzing...' : 'Upload Image'}
              </button>
              <input
                type="file"
                accept="image/*"
                ref={fileInputRef}
                onChange={handleImageUpload}
                style={{ display: 'none' }}
              />
            </div>

            {foodInfo && (
              <div className="food-info">
                <h3>Food Analysis Results</h3>
                <p><strong>Name:</strong> {foodInfo.name}</p>
                <p><strong>Calories:</strong> {foodInfo.calories} kcal</p>
                <p><strong>Protein:</strong> {foodInfo.protein}g</p>
                <p><strong>Carbs:</strong> {foodInfo.carbs}g</p>
                <p><strong>Fat:</strong> {foodInfo.fat}g</p>
                {foodInfo.recommendations && (
                  <div className="recommendations">
                    <h4>Recommendations:</h4>
                    <p>{foodInfo.recommendations}</p>
                  </div>
                )}
              </div>
            )}
          </div>
          
          <div className="image-section">
            <div className="image-container">
              <img 
                src={previewUrl} 
                alt="Food image" 
                className="food-image"
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Body;