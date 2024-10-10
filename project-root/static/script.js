const video = document.getElementById('video');
const result = document.getElementById('result');
const startCameraButton = document.getElementById('start-camera');
const fileInput = document.getElementById('file-input');
const uploadedImage = document.getElementById('uploaded-image');
const uploadedVideo = document.getElementById('uploaded-video');

// Start the webcam feed
startCameraButton.addEventListener('click', () => {
  navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
      video.srcObject = stream;
      video.style.display = 'block';
      uploadedImage.style.display = 'none';
      uploadedVideo.style.display = 'none';
      video.play();
    })
    .catch(err => {
      console.error('Error accessing the camera: ', err);
    });
});

// Handle file input (image or video)
fileInput.addEventListener('change', (event) => {
  const file = event.target.files[0];
  if (file) {
    const fileType = file.type.split('/')[0];
    const reader = new FileReader();

    reader.onload = function(e) {
      if (fileType === 'image') {
        uploadedImage.src = e.target.result;
        uploadedImage.style.display = 'block';
        uploadedVideo.style.display = 'none';
        video.style.display = 'none';
      } else if (fileType === 'video') {
        uploadedVideo.src = e.target.result;
        uploadedVideo.style.display = 'block';
        uploadedImage.style.display = 'none';
        video.style.display = 'none';
        uploadedVideo.play();
      }

      // Once media is loaded, start analyzing frames
      startEmotionDetection(e.target.result, fileType);
    };

    reader.readAsDataURL(file);
  }
});

// Analyze video or image for emotion detection
function startEmotionDetection(mediaSource, mediaType) {
  if (mediaType === 'video') {
    setInterval(() => {
      analyzeFrameFromMedia(uploadedVideo);
    }, 1000); // Capture frame every second
  } else {
    analyzeFrameFromImage(mediaSource);
  }
}

// Analyze frame from image or webcam feed
function analyzeFrameFromImage(imageSource) {
  const canvas = document.createElement('canvas');
  const img = new Image();
  img.src = imageSource;
  img.onload = function() {
    canvas.width = 48;
    canvas.height = 48;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    const frame = canvas.toDataURL('image/jpeg');
    sendFrameToServer(frame);
  };
}

function analyzeFrameFromMedia(mediaElement) {
  const canvas = document.createElement('canvas');
  canvas.width = mediaElement.videoWidth;
  canvas.height = mediaElement.videoHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(mediaElement, 0, 0, canvas.width, canvas.height);

  const frame = canvas.toDataURL('image/jpeg');
  sendFrameToServer(frame);
}

// Send the frame to the server for emotion detection
function sendFrameToServer(frame) {
  fetch('/analyze-frame', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ image: frame })
  })
  .then(response => response.json())
  .then(data => {
    result.innerHTML = `Emotion Detected: ${data.result}`;
  })
  .catch(error => {
    console.error('Error:', error);
  });
}
