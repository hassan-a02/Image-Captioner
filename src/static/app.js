document.getElementById('uploadButton').addEventListener('click', async () => {
    const fileInput = document.getElementById('imageUpload');
    const file = fileInput.files[0];

    if (!file) {
        alert('Please upload an image.');
        return;
    }

    const formData = new FormData();
    formData.append('image', file);

    try {
        const response = await fetch('/generate-caption', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const data = await response.json();
        const caption = data.caption;

        const imgElement = document.getElementById('uploadedImage');
        imgElement.src = URL.createObjectURL(file);
        imgElement.style.display = 'block';

        const captionElement = document.getElementById('caption');
        captionElement.textContent = caption;
        captionElement.style.display = 'block';
    } catch (error) {
        console.error('Error:', error);
        alert('Error generating caption. Please try again.');
    }
});