// Handle "Click image" button
document.getElementById("clickImageButton").addEventListener("click", () => {
  navigator.mediaDevices.getUserMedia({ video: true })
  .then((stream) => {
    // You can use the stream to display the camera feed in a video element
    console.log("Camera is accessible.");
  })
  .catch((err) => {
    alert("Camera access denied: " + err.message);
  });

  });
  
  // Handle "Upload image" button
  const uploadButton = document.getElementById("uploadImageButton");
  const fileInput = document.getElementById("imageUploadInput");
  
  uploadButton.addEventListener("click", () => {
    fileInput.click(); // Trigger the file input click
  });
  
  fileInput.addEventListener("change", (event) => {
    const file = event.target.files[0];
    if (file) {
      alert(`File uploaded: ${file.name}`);
      // Additional functionality for processing the image can be added here
    }
  });

  const video = document.getElementById("camera");
  const canvas = document.getElementById("canvas");
  const clickImageButton = document.getElementById("clickImageButton");
  
  // Flag to track if the camera stream is active
  let cameraStream = null;
  
  // Event Listener for the "Click Image" Button
  clickImageButton.addEventListener("click", () => {
    // If the camera stream is not already active, start it
    if (!cameraStream) {
      navigator.mediaDevices.getUserMedia({ video: true })
        .then((stream) => {
          // Assign the stream to the video element and store the reference
          video.srcObject = stream;
          cameraStream = stream;
  
          // Display the video feed for capturing
          video.style.display = "block";
        })
        .catch((err) => {
          alert("Unable to access the camera: " + err.message);
        });
    } else {
      // Capture the picture
      const context = canvas.getContext("2d");
      // Set the canvas size to match the video feed
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
  
      // Draw the current video frame onto the canvas
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
  
      // Hide the video feed and show the captured image
      video.style.display = "none";
      canvas.style.display = "block";
    }
  });
  